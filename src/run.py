import argparse
from scrape import ScriptiebankScraper
import pandas as pd
import re
import requests
from pathlib import Path
from langdetect import detect, LangDetectException
import mu
import io
import hashlib
import ast
import json
import numpy as np
from mu import DataFrameCleaner
import pyarrow  as pa
import pyarrow.json as pj
import pyarrow.parquet as pq
import pyarrow.csv as pv
import pyarrow.compute as pc
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import os
import subprocess
import time
import atexit
import socket
import sys
import string
import binascii
import random

#TODO check if paths exist and skip steps accordingkly
#TODO relative paths
#TODO create dirs and files

#downloads
def download_raw_data(
        scriptiebank=False,
        ugent=True
):
    #create data folder + raw
    if scriptiebank == True:
        scraper = ScriptiebankScraper()
        scraper.run(gather_metadata=True,gather_urls=True,download_files=False) #download selection later
    
    if ugent == True:
        file_path = Path('data/raw_data/UG/publications.json')
        datadump_url = 'https://biblio.ugent.be/exports/publications.json'
    
        def download_ug(url, save_path):
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True,exist_ok=True)

            response = requests.get(url)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)
            print("downlaod complete")

        download_ug(datadump_url, file_path)

#cleaning
def clean_df(df_path, save_path, protected_values=None,schema=None):
    if not save_path: raise 'please input save path'
    if df_path.suffix == '.tsv':
        df = pd.read_csv(df_path,sep='\t')
    elif df_path.suffix == '.json': df = pd.read_json(df_path, lines=True)
    else: raise 'format not supported'
    cleaner = DataFrameCleaner(df)
    cleaner.run_auto_pipeline(schema=schema, protected_values=protected_values) #MAKE SURE IT HANDLES NONE
    cleaner.save_parquet(path=save_path)


def clean_ug(
    ug_path = Path('/home/gderijck/internship/data/raw_data/UG/publications.json'),
    ug_clean = Path('/home/gderijck/internship/data/silver/ug_cleaned.parquet')
):
    prot_val_ug = {
        'volume': [999,'999',9999,'9999'], #CHANGE TO REGEX
        'issue': ['999',999,'9999',9999]
    }
    clean_df(ug_path, ug_clean, protected_values=prot_val_ug)

#clean abstracts and select rows
def select_and_clean_abstracts_ug(
    input_path: Path,
    output_path: Path,
    min_year: int = 1980,
    max_year: int = 2022,
    min_char_length: int = 100,
    source_lang_tag: str = 'dut',
    detect_lang_tag: str = 'nl',
    tokenizer_lang: str = 'dutch'
) -> pa.Table:
    """
    Filters, tokenizes, and cleans abstracts from a PyArrow Parquet table.
    
    Parameters:
    - input_path: Path to the raw input Parquet file.
    - output_path: Path to write the cleaned Parquet file.
    - min_year: Minimum year (inclusive) to filter.
    - max_year: Maximum year (inclusive) to filter.
    - min_char_length: Minimum character length of raw abstract text.
    - source_lang_tag: Language tag to look for in 'abstract_full' dictionary items.
    - detect_lang_tag: Target language code for sentence-level filtering (langdetect).
    - tokenizer_lang: Target language name for NLTK sentence tokenization.
    """
    print(f"Reading input table from: {input_path}")
    table = pq.read_table(input_path)
    rows = table.to_pylist()

    filtered_data = []

    for row in rows:
        year = row.get('year')
        abstract_list = row.get('abstract_full')

        # 1. Year Filter
        if year is not None and min_year <= year <= max_year:
            if isinstance(abstract_list, list):
                for item in abstract_list:
                    # 2. Source Language Tag Filter
                    if isinstance(item, dict) and item.get('lang') == source_lang_tag:
                        text_content = item.get('text')
                        
                        # 3. Minimum Character Length Filter
                        if isinstance(text_content, str) and len(text_content) >= min_char_length and text_content.strip():
                            
                            # 4. Tokenization (Optimized for target language)
                            raw_sentences = nltk.sent_tokenize(text_content, language=tokenizer_lang)
                            cleaned_sentences = []
                            
                            for sent in raw_sentences:
                                sent = sent.strip()
                                if not sent:
                                    continue
                                
                                should_merge = False
                                if cleaned_sentences:
                                    if not re.match(r'^[A-Z]', sent):
                                        should_merge = True
                                    elif len(sent) >= 2 and sent[1] in string.punctuation:
                                        should_merge = True
                                
                                if should_merge:
                                    cleaned_sentences[-1] = cleaned_sentences[-1] + ' ' + sent
                                else:
                                    cleaned_sentences.append(sent)
                            
                            # 5. Sentence-Level Language Filter
                            dutch_sentences = []
                            for sent in cleaned_sentences:
                                try:
                                    if detect(sent) == detect_lang_tag:
                                        dutch_sentences.append(sent)
                                except LangDetectException:
                                    continue
                                    
                            # 6. Save back to row if valid sentences remain
                            if dutch_sentences:
                                # Reconstruct full abstract
                                dutch_abstract = ' '.join(dutch_sentences)
                                row['text_dut'] = dutch_abstract
                                # Save list of sentences
                                row['sent_dut'] = dutch_sentences
                                
                                filtered_data.append(row)

    print(f"Filtering complete. Kept {len(filtered_data)} rows of data.")
    
    # Load the filtered list of dicts back into a PyArrow Table
    filtered_table = pa.Table.from_pylist(filtered_data)

    # Write back to a clean Parquet file
    print(f"Writing cleaned table to: {output_path}")
    pq.write_table(filtered_table, output_path)
    print("Writing complete.")
    
    return filtered_table




#generation
#ollama server functions
    #helpers
def is_port_in_use(port: int) -> bool:
    """Checks if a local port is already active."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def shutdown_ollama_server(process: subprocess.Popen):
    """Kills the background server process on exit."""
    # Check if the process exists and is still running
    if process and process.poll() is None:
        print("\nShutting down background Ollama server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server stopped cleanly.")

def start_ollama_server(port: int = 11435, gpu_id: str = "0") -> subprocess.Popen:
    """
    Launches a private, user-space Ollama server in the background.
    Automatically registers a cleanup handler to stop the process on exit.
    
    Parameters:
    - port: The port to bind (default: 11435)
    - gpu_id: Target GPU ID as string (e.g. "0"). Pass "" or None to force CPU execution.
    """
    # Set client-side port environment variable
    os.environ["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    
    # Configure device targeting (CPU vs. specific GPU)
    if gpu_id and gpu_id.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device_label = f"GPU {gpu_id}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_label = "CPU"

    # Start the server if the port is free
    if not is_port_in_use(port):
        print(f"Starting background Ollama server on {device_label} (Port {port})...")
        
        # Clone environment settings for the background subprocess
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Launch server process
        process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for initialization loop
        for attempt in range(15):
            if is_port_in_use(port):
                print(f"Ollama server successfully launched on {device_label}.")
                break
            time.sleep(1)
        else:
            print("Error: Ollama server initialization timed out. Terminating process.")
            process.terminate()
            sys.exit(1)
            
        # Register the cleanup handler, passing the active process reference directly
        atexit.register(shutdown_ollama_server, process)
        return process
        
    else:
        print(f"Ollama server is already running on port {port}. Reusing existing instance.")
        return None

#GENERATION proper
    #save on exit helper
def save_parquet_on_exit(rows: list[dict], output_path: Path):
    """Saves the entire active rows list directly to your Parquet file on exit."""
    if rows:
        print(f"\n[Exit Handler] Auto-saving active progress to Parquet: {output_path}...")
        try:
            # Reconstruct the PyArrow Table and save
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, output_path)
            print("[Exit Handler] Progress saved successfully.")
        except Exception as e:
            print(f"[Exit Handler] Error during auto-save: {e}")

#prepare tasks for generation
def prepare_tasks(table: pa.Table, models_list: list[str], percentages: list[int] = None) -> tuple[list[dict], list[dict]]:
    """
    Scans the Parquet table for missing completions across three task types 
    and returns a flat list of pending tasks and the active rows list.
    """
    if percentages is None:
        percentages = []

    # Convert the PyArrow table to a list of standard mutable Python dicts
    rows = table.to_pylist()
    tasks = []

    # Determine unique ID key
    id_key = '_id' if '_id' in table.column_names else 'id'

    for row in rows:
        row_id = row.get(id_key)
        if row_id is None:
            continue
            
        text_dut = row.get('text_dut')
        sent_dut = row.get('sent_dut')

        # Skip rows that don't have valid text or sentence lists
        if not isinstance(text_dut, str) or not text_dut.strip():
            continue
        if not isinstance(sent_dut, list) or not sent_dut:
            continue

        num_sentences = len(sent_dut)

        for model in models_list:
            # ===============================================================
            # TASK TYPE 1: Single Sentence (Stored as a list of strings)
            # ===============================================================
            # Ensure the column exists as a list matching the length of sent_dut
            col_name = f'{model}_single'
            if col_name not in row or not isinstance(row[col_name], list) or len(row[col_name]) != num_sentences:
                row[col_name] = [None] * num_sentences

            for sent_idx, sentence in enumerate(sent_dut):
                # Only generate a task if this specific sentence is missing a rewrite
                if row[col_name][sent_idx] is None:
                    tasks.append({
                        "id": row_id,
                        "type": "sentence",
                        "model": model, #TODO ???
                        "sent_idx": sent_idx,
                        "text": sentence,
                        "context": text_dut  # Passed in case the LLM needs surrounding context #TODO needed????
                    })

            # ===============================================================
            # TASK TYPE 2: %-Based Rewrites with Context
            # ===============================================================
            for pct in percentages:
                col_name = f"{model}_{pct}" #good
                
                # Only generate if this hybrid abstract hasn't been created yet
                if col_name not in row or not row[col_name]:
                    row[col_name] = None  # Initialize
                    
                    # Deterministic, stateless seed based on ID and percentage
                    # This ensures the same sentences are tagged if you restart #TODO EXPLAIN
                    seed_str = f"{row_id}_{pct}".encode("utf-8")
                    seed = binascii.crc32(seed_str)
                    rng = random.Random(seed)
                    
                    # Select percentage of sentences to tag
                    num_to_tag = max(1, round(num_sentences * (pct / 100.0)))
                    tagged_indices = set(rng.sample(range(num_sentences), num_to_tag))
                    
                    # Annotate the abstract with XML-style target tags
                    annotated_sentences = []
                    for idx, sent in enumerate(sent_dut):
                        if idx in tagged_indices:
                            annotated_sentences.append(f"<target>{sent}</target>")
                        else:
                            annotated_sentences.append(sent)
                    annotated_abstract = " ".join(annotated_sentences)

                    tasks.append({
                        "id": row_id,
                        "type": "percentage",
                        "model": model, #TODO AGIAN
                        "percentage": pct,
                        "text": annotated_abstract
                    })

            # ===============================================================
            # TASK TYPE 3: Full Abstract Rewrite
            # ===============================================================
            col_name = f"{model}_full"
            if col_name not in row or not row[col_name]:
                row[col_name] = None  # Initialize
                
                tasks.append({
                    "id": row_id,
                    "type": "full_abstract",
                    "model": model,
                    "text": text_dut
                })

    return tasks, rows

#pass to ollama and get result
def rewrite_sentence(model_to_run, system_prompt, sentence):
    """Sends a single sentence to Ollama for rewriting."""
    try:
        # Using the official Ollama chat implementation
        response = ollama.generate(
            model=model_to_run,
            system = system_prompt,
            prompt = sentence,
            think=False,
            options={
                "seed": 42,
                #"num_thread": 4
            },
            #format = StrucResponse.model_json_schema(),
        )

        rewritten = response['response'].strip()
        rewritten = rewritten.replace('\x00','').replace('\u0000','')
        #response['done'] == True confirms it went thru
        #<channel|> gemma response?
        #if model == 'gemma...
        if not sentence.startswith('"'):
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
        return rewritten
    
    except Exception as e:
        print(f"Error calling Ollama ({model_to_run}): {e}")
        sys.exit(1)

#generation lopp
def run_generation(tasks: list[dict], rows: list[dict], system_prompt_mapping: dict, output_path: Path, save_interval: int = 100):
    """
    Iterates through the task list, executes them on the Ollama server, 
    and writes results directly back to our active rows list with progressive saving.
    """
    id_key = '_id' if '_id' in rows[0] else 'id'
    rows_map = {row[id_key]: row for row in rows}

    for i, task in enumerate(tasks):
        t_id = task["id"]
        t_type = task["type"]
        model = task["model"]
        text = task["text"]
        
        row = rows_map.get(t_id)
        if not row:
            continue

        system_prompt = system_prompt_mapping.get(t_type)
        if not system_prompt:
            print(f"Error: No system prompt found for task type '{t_type}'")
            sys.exit(1)

        print(f"\n[{model}] Processing Task {i+1}/{len(tasks)} (Type: {t_type}, ID: {t_id})...")
        
        rewritten = rewrite_sentence(model, system_prompt, text)
        
        # Route output back to our in-memory datastructure
        if t_type == "sentence":
            sent_idx = task["sent_idx"]
            row[f'{model}_single'][sent_idx] = rewritten
        elif t_type == "percentage":
            pct = task["percentage"]
            row[f"{model}_{pct}"] = rewritten
        elif t_type == "full_abstract":
            row[f"{model}_full"] = rewritten

        # progressive saving
        if (i + 1) % save_interval == 0:
            print(f"\n[Progressive Save] Auto-saving progress ({i+1}/{len(tasks)} tasks completed)...")
            try:
                table = pa.Table.from_pylist(rows)
                pq.write_table(table, output_path)
                print("[Progressive Save] Save successful.")
            except Exception as e:
                print(f"[Progressive Save] Error: {e}")

    return rows


#generation main
def generation_main(
    ug_path: Path = Path('data/silver/ug_selected.parquet'),
    models_list: list[str] = None,
    percentages_to_run: list[int] = None,
    port: int = 11435,
    gpu_id: str = "0",
    save_interval: int = 100
):
    """
    Fully parameterized main orchestrator for the LLM rewrite pipeline.
    """
    if models_list is None:
        models_list = ['qwen3.5:4b', 'gemma4:e4b']
    if percentages_to_run is None:
        percentages_to_run = [25, 50, 75]

    system_prompts = {
        "sentence": (
            "Je bent een professionele Nederlandse redacteur.\n"
            "Herschrijf de volgende zin. Bewaar cruciale informatie.\n"
            "Geef UITSLUITEND de herschreven zin terug, zonder aanhalingstekens."
        ),
        "percentage": (
            "Je bent een professionele Nederlandse redacteur.\n"
            "Je krijgt een abstract te zien waarin specifieke zinnen zijn gemarkeerd met <target>...</target> tags.\n"
            "Herschrijf UITSLUITEND de zinnen binnen deze tags. Bewaar cruciale informatie.\n"
            "Pas de omliggende tekst niet aan. Geef het volledige abstract terug inclusief je wijzigingen, maar zonder de <target> tags." #TODO check if correct sent was changed.
        ),
        "full_abstract": (
            "Je bent een professionele Nederlandse redacteur.\n"
            "Herschrijf het volledige abstract. Bewaar cruciale informatie.\n"
            "Geef UITSLUITEND het volledige, herschreven abstract terug."
        )
    }
    
    # 1. Start Ollama server
    start_ollama_server(port=port, gpu_id=gpu_id)
    global ollama  # Import globally inside local scope after process bound
    import ollama
    
    # 2. Load dataset
    print(f"Loading Parquet data from: {ug_path}")
    if not ug_path.exists():
        print(f"Error: Parquet file not found at {ug_path}")
        sys.exit(1)
    ug_table = pq.read_table(ug_path)
    
    # 3. Scan and prepare pending tasks
    tasks, rows = prepare_tasks(ug_table, models_list, percentages_to_run)
    print(f"Total pending tasks to generate: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All specified tasks are already completed in the dataset. Exiting.")
        return
        
    # 4. Register final emergency fallback exit handler
    atexit.register(save_parquet_on_exit, rows, ug_path)
    
    # 5. Run generation with progressive saving
    run_generation(tasks, rows, system_prompts, ug_path, save_interval=save_interval)
    
    # 6. Complete clean write back
    print("\nGeneration finished successfully. Writing final table...")
    save_parquet_on_exit(rows, ug_path)

def check_dataset_completion(
        table: pa.Table,
        models_list: list[str] = ['qwen3.6:27b', 'qwen:3.5:2b', 'gemma4:26b', 'gemma4:e4b'], #TODO ADD MODELS, INTEGRATE INTO HIGHER
        percentages: list[int] = [25,50,75]
    ) -> bool:
    """
    Scans the Parquet dataset to check if all desired columns are present and fully completed.
    Prints a detailed completion status report.
    Returns True if generation is still needed, False if the dataset is 100% complete.
    """
    if percentages is None:
        percentages = []

    rows = table.to_pylist()
    total_rows = len(rows)
    
    if total_rows == 0:
        print("[Warning] Dataset is empty.")
        return False

    # Initialize a statistics tracker
    stats = {}
    needs_generation = False

    # Setup the statistics schema
    for model in models_list:
        stats[f"{model}_single"] = {"completed_sents": 0, "total_sents": 0}
        for pct in percentages:
            stats[f"{model}_{pct}"] = {"completed_abstracts": 0, "total_abstracts": total_rows}
        stats[f"{model}_full"] = {"completed_abstracts": 0, "total_abstracts": total_rows}

    # Scan the dataset row-by-row
    for row in rows:
        sent_dut = row.get('sent_dut')
        num_sentences = len(sent_dut) if isinstance(sent_dut, list) else 0

        for model in models_list:
            # 1. Check [model]_single (Sentence-level lists)
            col_single = f"{model}_single"
            stats[col_single]["total_sents"] += num_sentences
            
            if col_single in row and isinstance(row[col_single], list):
                # Count non-None entries inside the list
                completed_sents = sum(1 for x in row[col_single] if x is not None)
                stats[col_single]["completed_sents"] += completed_sents
                
                # If there are any missing (None) elements or length mismatches, we need generation
                if completed_sents < num_sentences or len(row[col_single]) != num_sentences:
                    needs_generation = True
            else:
                needs_generation = True

            # 2. Check [model]_[pct] (Paragraph-level strings)
            for pct in percentages:
                col_pct = f"{model}_{pct}"
                if col_pct in row and isinstance(row[col_pct], str) and row[col_pct].strip():
                    stats[col_pct]["completed_abstracts"] += 1
                else:
                    needs_generation = True

            # 3. Check [model]_full (Paragraph-level strings)
            col_full = f"{model}_full"
            if col_full in row and isinstance(row[col_full], str) and row[col_full].strip():
                stats[col_full]["completed_abstracts"] += 1
            else:
                needs_generation = True

    # Print the Dashboard
    print("\n====================================================")
    print("           DATASET GENERATION STATUS REPORT")
    print("====================================================")
    
    for col_name, data in stats.items():
        if "total_sents" in data:
            completed = data["completed_sents"]
            total = data["total_sents"]
            unit = "sentences"
        else:
            completed = data["completed_abstracts"]
            total = data["total_abstracts"]
            unit = "abstracts"
            
        pct_complete = (completed / total * 100) if total > 0 else 0
        status_icon = "✓" if completed == total else "✗"
        
        print(f" {status_icon} {col_name:<25}: {completed:>5} / {total:<5} {unit:<10} ({pct_complete:>6.2f}%)")
        
    print("====================================================")
    
    if needs_generation:
        print(" -> Status: GENERATION IS STILL REQUIRED (some cells are missing).\n")
    else:
        print(" -> Status: 100% COMPLETE (all columns are fully generated!).\n")
        
    return needs_generation



def main(): #TODO set default and relative and variable paths + checks for skipping

    models_list = ['qwen3.6:27b', 'qwen:3.5:2b', 'gemma4:26b', 'gemma4:e4b'], #TODO ADD MODELS, INTEGRATE INTO HIGHER
    percentages = [25,50,75]

    #selection
    ug_select = Path('/home/gderijck/internship/data/silver/ug_selected.parquet')
    if not ug_select:
        table = select_and_clean_abstracts_ug(
            input_path=Path('/home/gderijck/internship/data/silver/ug_cleaned.parquet'), #TODO relative
            output_path=ug_select
        )
    if not table:
        table = pq.read_parquet(ug_select)
    
    #generation
    if check_dataset_completion(table=table):
        generation_main(
            table = table,
            models_list = models_list,
            percentages_to_run = percentages,
        ) 



if __name__ == "__main__":
    main()

