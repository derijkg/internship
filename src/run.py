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
from ollama import Client
import torch

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
    if process and process.poll() is None:
        print("\nShutting down background Ollama server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server stopped cleanly.")

def start_ollama_server(port: int = 11435, gpu_id: str = None) -> subprocess.Popen:
    """
    Launches a private, user-space Ollama server in the background.
    Uses the native GPU-enabled binary rather than the active Conda binary.
    """
    gpu_str = str(gpu_id).strip() if gpu_id is not None else ""
    
    # Configure the base environment
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    
    # Locate the GPU-enabled binary in your user-local space
    user_bin = Path.home() / ".local" / "bin" / "ollama"
    system_bin = Path("/usr/local/bin/ollama")
    system_bin_alt = Path("/usr/bin/ollama")
    
    if user_bin.exists():
        ollama_executable = str(user_bin)
    elif system_bin.exists():
        ollama_executable = str(system_bin)
    elif system_bin_alt.exists():
        ollama_executable = str(system_bin_alt)
    else:
        ollama_executable = "ollama"
        print("Warning: GPU-enabled native Ollama binary not found in standard locations.")
        print("Falling back to PATH, which might default to your Conda environment binary.")

    # Configure dynamic linking paths for CUDA
    cuda_libs = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
    
    # Point to the native CUDA runner libraries we extracted in user space
    user_libs = str(Path.home() / ".local" / "lib" / "ollama")
    
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{user_libs}:{env['LD_LIBRARY_PATH']}:{cuda_libs}"
    else:
        env["LD_LIBRARY_PATH"] = f"{user_libs}:{cuda_libs}"
    
    # Configure device targeting
    if gpu_str == "-1":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        device_label = "CPU Only"
    elif "CUDA_VISIBLE_DEVICES" in env and not gpu_str:
        device_label = f"GPU {env['CUDA_VISIBLE_DEVICES']} (Inherited from environment)"
    elif gpu_str:
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        device_label = f"GPU {gpu_str} (with CPU fallback)"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        device_label = "GPU Auto-Discovery (with CPU fallback)"

    if not is_port_in_use(port):
        print(f"Starting background Ollama server on {device_label} (Port {port})...")
        print(f"Using binary executable: {ollama_executable}")
        
        log_path = Path("ollama_server.log")
        log_file = open(log_path, "w", encoding="utf-8")
        
        process = subprocess.Popen(
            [ollama_executable, "serve"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        # Wait for initialization loop
        for attempt in range(15):
            if is_port_in_use(port):
                print(f"Ollama server successfully launched.")
                break
            time.sleep(1)
        else:
            print("Error: Ollama server initialization timed out. Terminating process.")
            process.terminate()
            log_file.close()
            sys.exit(1)
            
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


#writer to checkpoint only (add to pq at the end of all tasks)
def append_to_checkpoint(checkpoint_path: Path, task: dict, rewritten_text: str):
    """
    Appends a single successfully generated and validated rewrite to the JSONL checkpoint.
    """
    record = {
        "id": task["id"],
        "type": task["type"],
        "model": task["model"],
        "sent_idx": task.get("sent_idx"),
        "percentage": task.get("percentage"),
        'text': task.get('text'),
        "rewritten": rewritten_text
    }
    
    # Open in append mode ('a')
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


#prepare tasks 3 types of task for generation
#checks state of both pq and checkpoint and builds tasks that are needed
#validates single sentence legacy lines from checkpoint
def prepare_tasks(
    table: pa.Table, 
    checkpoint_path: Path,
    models_list: list[str], 
    percentages: list[int] = None
) -> tuple[list[dict], list[dict]]:
    """
    Unified pipeline that:
    1. Converts the Parquet table to mutable Python dicts.
    2. Parses the JSONL checkpoint (handling legacy/modern formats dynamically).
    3. Aligns sentence rewrites via strict text matching (with index fallback).
    4. Prints a detailed Checkpoint Merging & Alignment Report.
    5. Prints the Dataset Generation Status Report.
    6. Constructs and returns the final flat list of pending tasks.
    """
    if percentages is None:
        percentages = []

    # 1. Convert PyArrow table to standard Python dicts
    rows = table.to_pylist()
    id_key = '_id' if '_id' in table.column_names else 'id'
    rows_map = {row[id_key]: row for row in rows}

    # 2. Parse checkpoint and perform strict text-matching alignment
    total_loaded = 0
    text_match_count = 0
    idx_fallback_count = 0
    mismatched_discard_count = 0
    missing_row_discard_count = 0
    corrupted_count = 0

    if checkpoint_path.exists():
        print(f"Loading progress from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    total_loaded += 1
                    
                    # Extract variables based on schema (Modern vs. Legacy)
                    #new
                    if "type" in record:
                        row_id = record.get("id")
                        t_type = record.get("type")
                        model = record.get("model")
                        sent_idx = record.get("sent_idx")
                        pct = record.get("percentage")
                        orig_text = record.get("text")
                        rewritten_text = record.get("rewritten")
                    
                    #old data single sentence
                    else:
                        # Legacy fallback
                        row_id = record.get("id")
                        t_type = "sentence"
                        sent_idx = record.get("sent_idx")
                        pct = None
                        orig_text = record.get("text")
                        
                        metadata_keys = {"id", "_id", "sent_idx", "text"}
                        model_keys = [k for k in record.keys() if k not in metadata_keys] 
                        if not model_keys or sent_idx is None:
                            mismatched_discard_count += 1
                            continue
                        model = model_keys[0]
                        rewritten_text = record[model]

                    if rewritten_text is None:
                        mismatched_discard_count += 1
                        continue

                    # Strip chain-of-thought tags from raw models
                    if isinstance(rewritten_text, str) and '<channel|>' in rewritten_text:
                        rewritten_text = rewritten_text.split('<channel|>')[1].strip()

                    row = rows_map.get(row_id)
                    if not row:
                        missing_row_discard_count += 1
                        continue

                    # Execute routing and text verification logic
                    if t_type == "sentence":
                        sent_dut = row.get("sent_dut")
                        if not isinstance(sent_dut, list):
                            mismatched_discard_count += 1
                            continue

                        matched_idx = -1
                        # Step A: Perform strict text-matching verification
                        if orig_text:
                            clean_orig = orig_text.strip()
                            for idx, sent in enumerate(sent_dut):
                                if sent.strip() == clean_orig:
                                    matched_idx = idx
                                    break

                        # discard if strict text match fails
                        if matched_idx == -1:
                            mismatched_discard_count += 1
                            continue
                        else:
                            text_match_count += 1

                        task_meta = {"type": t_type, "model": model, "sent_idx": matched_idx, "percentage": pct}
                        apply_rewrite_to_row(row, task_meta, rewritten_text)
                    else:
                        # Non-sentence tasks (percentage/full) do not require index alignment
                        task_meta = {"type": t_type, "model": model, "sent_idx": sent_idx, "percentage": pct}
                        apply_rewrite_to_row(row, task_meta, rewritten_text)

                except (json.JSONDecodeError, ValueError, TypeError):
                    corrupted_count += 1
                    continue

    # Print Report 1: Alignment Summary
    if total_loaded > 0:
        print("\n====================================================")
        print("          CHECKPOINT MERGING & ALIGNMENT REPORT")
        print("====================================================")
        print(f" Total records loaded from checkpoint: {total_loaded}")
        print(f" Successfully aligned (Exact Text Matches) : {text_match_count}")
        print(f" Successfully aligned (Index Fallbacks)    : {idx_fallback_count}")
        print(f" Discarded (Unmatched sentence text/index) : {mismatched_discard_count}")
        print(f" Discarded (Missing row IDs)               : {missing_row_discard_count}")
        print(f" Discarded (Corrupted/Malformed lines)     : {corrupted_count}")
        print("====================================================\n")

    # 4. Scan rows and build the final task queue
    tasks = []
    for row in rows:
        row_id = row.get(id_key)
        text_dut = row.get('text_dut')
        sent_dut = row.get('sent_dut')

        if not isinstance(text_dut, str) or not text_dut.strip() or not isinstance(sent_dut, list) or not sent_dut:
            continue

        num_sentences = len(sent_dut)

        for model in models_list:
            # TASK TYPE 1: Single Sentence
            col_name = f'{model}_single'
            if col_name not in row or not isinstance(row[col_name], list) or len(row[col_name]) != num_sentences:
                row[col_name] = [None] * num_sentences

            for sent_idx, sentence in enumerate(sent_dut):
                if row[col_name][sent_idx] is None:
                    tasks.append({
                        "id": row_id,
                        "type": "sentence",
                        "model": model,
                        "sent_idx": sent_idx,
                        "text": sentence,
                        "context": text_dut
                    })

            # TASK TYPE 2: %-Based Rewrites with Context
            for pct in percentages:
                col_name = f"{model}_{pct}"
                if col_name not in row or not row[col_name]:
                    row[col_name] = None
                    seed_str = f"{row_id}_{pct}".encode("utf-8")
                    seed = binascii.crc32(seed_str)
                    rng = random.Random(seed)
                    num_to_tag = max(1, round(num_sentences * (pct / 100.0)))
                    tagged_indices = set(rng.sample(range(num_sentences), num_to_tag))
                    
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
                        "model": model,
                        "percentage": pct,
                        "text": annotated_abstract,
                        'tagged_indices': list(tagged_indices)
                    })

            # TASK TYPE 3: Full Abstract Rewrite
            col_name = f"{model}_full"
            if col_name not in row or not row[col_name]:
                row[col_name] = None
                tasks.append({
                    "id": row_id,
                    "type": "full_abstract",
                    "model": model,
                    "text": text_dut
                })

    return tasks, rows

#checkpoint -> pq
def apply_rewrite_to_row(row: dict, task: dict, rewritten: str):
    """Helper to route the rewritten string to the correct column in-memory."""
    if not row:
        return
    t_type = task["type"]
    model = task["model"]
    
    if t_type == "sentence":
        sent_idx = task["sent_idx"]
        num_sentences = len(row['sent_dut'])
        if f'{model}_single' not in row or not isinstance(row[f'{model}_single'], list) or len(row[f'{model}_single']) != num_sentences:
            row[f'{model}_single'] = [None] * num_sentences
            
        # Ensure our target index sits strictly inside our allocated boundaries
        if 0 <= sent_idx < num_sentences:
            row[f'{model}_single'][sent_idx] = rewritten
    elif t_type == "percentage":
        pct = task["percentage"]
        row[f"{model}_{pct}"] = rewritten
    elif t_type == "full_abstract":
        row[f"{model}_full"] = rewritten


#pass to ollama and get result
def rewrite_sentence(model_to_run, system_prompt, sentence, seed=42):
    """Sends text to Ollama for rewriting."""
    try:
        # Retrieve the dynamically set OLLAMA_HOST from the environment
        # Fall back to localhost:11435 if not set
        host_env = os.environ.get("OLLAMA_HOST", "127.0.0.1:11435")
        
        # Ensure the string has the HTTP prefix required by the Client library
        if not host_env.startswith("http://"):
            host_env = f"http://{host_env}"
            
        # Instantiate a client on the exact active port with a 5-minute timeout
        client = Client(host=host_env, timeout=300)

        # Using the official Ollama chat implementation
        response = client.generate(
            model=model_to_run,
            system = system_prompt,
            prompt = sentence,
            think=False,
            options={
                "seed": seed,
                #'temperature': 0.0,
                #"num_thread": 4
            },
            #format = StrucResponse.model_json_schema(),
        )

        rewritten = response['response'].strip()

        #remove null
        rewritten = rewritten.replace('\x00','').replace('\u0000','')
        #response['done'] == True confirms it went thru

        #remove gemma thinking
        if isinstance(rewritten, str) and '<channel|>' in rewritten:
            rewritten = rewritten.split('<channel|>')[1].strip()

        #remove added quotes
        if not sentence.startswith('"'):
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
        return rewritten
    
    except Exception as e:
        print(f"Error calling Ollama ({model_to_run}): {e}")
        sys.exit(1)

#unload model helper
def unload_model(model_name: str):
    """
    Sends an empty generate request with keep_alive=0 to explicitly
    unload the model from GPU VRAM.
    """
    print(f"\nUnloading model '{model_name}' from GPU memory...")
    try:
        host_env = os.environ.get("OLLAMA_HOST", "127.0.0.1:11435")
        if not host_env.startswith("http://"):
            host_env = f"http://{host_env}"
            
        # Initialize client pointing to the active user-space server
        client = Client(host=host_env, timeout=30)
        
        # An empty prompt with keep_alive=0 tells Ollama to release the model's memory allocation
        client.generate(model=model_name, prompt="", keep_alive=0)
        print(f"Successfully unloaded '{model_name}'.")
        time.sleep(2)  # Give the system driver a brief moment to stabilize and reclaim VRAM
    except Exception as e:
        print(f"Warning: Could not explicitly unload '{model_name}': {e}")

#generation lopp
def run_generation(
    tasks: list[dict],
    rows: list[dict],
    system_prompt_mapping: dict,
    checkpoint_path: Path,
    debug_mode: bool = False
):
    """
    Iterates through the task list and executes them on Ollama.
    """
    id_key = '_id' if rows and '_id' in rows[0] else 'id' 
    rows_map = {row[id_key]: row for row in rows}

    current_model = None

    for i, task in enumerate(tasks):
        t_id = task["id"]
        t_type = task["type"]
        model = task["model"]
        text = task["text"]

        if current_model is not None and model != current_model:
            unload_model(current_model)
        current_model = model

        row = rows_map.get(t_id)
        if not row:
            continue

        system_prompt = system_prompt_mapping.get(t_type)
        if not system_prompt:
            print(f"Error: No system prompt found for task type '{t_type}'")
            sys.exit(1)

        print(f"\n[{model}] Processing Task {i+1}/{len(tasks)} (Type: {t_type}, ID: {t_id})...")

        # --- Generation & Validation Retries ---
        rewritten = None
        max_attempts = 3
        for attempt in range(max_attempts):    
            current_seed = 42+attempt
            candidate_rewrite = rewrite_sentence(model, system_prompt, text,seed=current_seed)
            
            if t_type == 'percentage':
                    original_sentences = row['sent_dut']
                    tagged_indices = set(task['tagged_indices'])
                    
                    is_valid, reason, stitched_text = validate_percentage_rewrite(
                        original_sentences, 
                        candidate_rewrite,  # Pass raw response with tags intact
                        tagged_indices
                    )
                    
                    if is_valid:
                        # Save the clean, stitched abstract (tags are already removed inside the validator)
                        rewritten = stitched_text 
                        print(f'ORIGINAL: {text}\n')
                        print(f'REWRITTEN: {rewritten}')
                        break
                    else:
                        print(f"  [Warning - Attempt {attempt+1}/{max_attempts} Failed] {reason}.")
                        # --- DIAGNOSTIC PRINTS ---
                        print(f"  [DEBUG - Prompt Sent to LLM]:\n{text}")
                        print(f"  [DEBUG - Raw Candidate Response]:\n{candidate_rewrite}")
                        print("-" * 50)
                        if attempt < max_attempts - 1:
                            print("  Retrying generation...")
                        else:
                            print(f"  [Warning] Task {t_id} failed validation. Writing sentinel.")
                            rewritten = "FAILED_VALIDATION"
                            break
                        
            elif t_type in ('sentence', 'full_abstract'):
                # Validation check: Ensure the candidate rewrite is structurally different from the input
                is_valid = candidate_rewrite.strip() != text.strip()
                
                if is_valid:
                    rewritten = candidate_rewrite
                    print(f'ORIGINAL: {text}\n')
                    print(f'REWRITTEN: {rewritten}')
                    break
                else:
                    #let through identical short sentences
                    word_count = len(text.strip().split())
                    char_count = len(text.strip())
                    is_short = word_count <= 6 or char_count <= 40
                    if is_short:
                        print(f'ORIGINAL: {text}\n')
                        print(f'REWRITTEN: {rewritten} [Accepted identical output due to short sentence length]')
                        rewritten = candidate_rewrite
                        break

                    reason = "The model's rewrite is identical to the original input."
                    print(f"  [Warning - Attempt {attempt+1}/{max_attempts} Failed] {reason}.")
                    # --- DIAGNOSTIC PRINTS ---
                    print(f"  [DEBUG - Prompt Sent to LLM]:\n{text}")
                    print(f"  [DEBUG - Raw Candidate Response]:\n{candidate_rewrite}")
                    print("-" * 50)
                    if attempt < max_attempts - 1:
                        print("  Retrying generation...")
                    else:
                        print(f"  [Warning] Task {t_id} failed validation. Writing sentinel.")
                        rewritten = "FAILED_VALIDATION"
                        break

        # Apply to in-memory structure
        apply_rewrite_to_row(row, task, rewritten)

        # Append to checkpoint so we don't repeat this on subsequent runs
        if not debug_mode:
            append_to_checkpoint(checkpoint_path, task, rewritten)

    # Clean up the final model when the entire task loop finishes
    if current_model is not None:
        unload_model(current_model)
    return rows


#check if target sentences in % rewrite were correctly rewritten
def normalize_text(text: str) -> str:
    """
    Standardizes whitespaces, newlines, and quotation marks to prevent 
    validation failures caused by minor LLM formatting normalizations.
    """
    if not text:
        return ""
    # Normalize smart quotes and backticks to standard straight typewriter quotes
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"').replace("`", "'")
    # Compress all sequences of whitespace (tabs, newlines, multiple spaces) into a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def validate_percentage_rewrite(
    original_sents: list[str], 
    raw_output_text: str, 
    tagged_indices: set[int]
) -> tuple[bool, str, str | None]:
    """
    Parses the LLM's raw output containing <target> tags.
    Extracts the edited target sentences, validates them, and stitches them
    back into the original sentence list to build a pristine final abstract.
    
    Returns:
        (is_valid, reason, stitched_abstract_or_None)
    """
    if not raw_output_text:
        return False, "Empty response from the model", None

    # Use a case-insensitive regex to find all segments wrapped in <target> tags
    target_pattern = re.compile(r'<target>(.*?)</target>', re.DOTALL | re.IGNORECASE)
    extracted_targets = target_pattern.findall(raw_output_text)
    
    expected_count = len(tagged_indices)
    actual_count = len(extracted_targets)
    
    # 1. Verify that the tag count matches expectations
    if actual_count != expected_count:
        return (
            False, 
            f"Tag count mismatch. Expected {expected_count} target tags, but found {actual_count}.", 
            None
        )
        
    # Create a copy of our original unedited sentences
    stitched_sentences = list(original_sents)
    sorted_target_indices = sorted(list(tagged_indices))
    
    # 2. Iterate and validate each rewrite
    for idx, rewrite_text in zip(sorted_target_indices, extracted_targets):
        rewrite_clean = rewrite_text.strip()
        original_clean = original_sents[idx].strip()
        
        # Verify the sentence wasn't deleted
        if not rewrite_clean:
            return False, f"Target sentence at index {idx} was returned empty.", None
            
        # Verify the sentence was actually modified (using normalized comparison)
        if normalize_text(rewrite_clean) == normalize_text(original_clean):
            return False, f"Target sentence at index {idx} was not modified by the model.", None
            
        # Stitch the validated rewrite back into the pristine sentence index
        stitched_sentences[idx] = rewrite_clean
        
    # 3. Construct the final stitched abstract without any remaining XML tags
    final_abstract = " ".join(stitched_sentences)
    
    return True, "Success", final_abstract

#calc distribution helper
def get_models_list():
    CALC_MODEL_MAPPING = {
        'calc12': ['qwen3.6:27b','qwen3.5:4b'],
        'calc11': ['gemma4:e4b','gemma4:26b'],
    }
    try:
        current_host = socket.gethostname().split('.')[0]
        if current_host not in CALC_MODEL_MAPPING: #ADDED: Replaced inline else assignment containing print and sys.exit(1) statement. In your previous implementation, if the current host was missing, the inline "and" condition evaluated to None, which did not exit the process. Instead, it set default_calc to None, triggering a downstream KeyError on CALC_MODEL_MAPPING[None].
            print(f"Host '{current_host}' not found in configuration mappings.")
            sys.exit(1)
        default_calc = current_host
    except Exception as e:
        print(f'Error identifying host config: {e}')
        sys.exit(1)

    selected_models = CALC_MODEL_MAPPING[default_calc]
    print(f'selected config: {current_host} -> models_list: {selected_models}')
    return selected_models

#generation main
def generation_main(
    table: pa.Table = None,
    ug_path: Path = Path('data/silver/ug_selected.parquet'),
    checkpoint_path: Path = None,
    models_list: list[str] = None,
    percentages_to_run: list[int] = [25, 50, 75],
    port: int = 11435,
    gpu_id: str = None,
    debug_mode: bool = False,
    debug_count: int = 5,
    exclude_percentage: bool = False
):
    """
    Fully parameterized main orchestrator for the LLM rewrite pipeline.
    """
    if checkpoint_path is None:
        checkpoint_path = ug_path.parent / "checkpoint_rewrites.jsonl"
    else:
        checkpoint_path = Path(checkpoint_path)
        print(f'Adding to existing checkpoint path found at {checkpoint_path}')

    system_prompts = {
        "sentence": (
            "You are a professional Dutch editor.\n"
            "Rewrite the following sentence to make it better while preserving all crucial information.\n"
            "Provide ONLY the rewritten sentence."
        ),
        "percentage": (
            "You are a professional Dutch editor.\n"
            "You will be given a Dutch abstract where specific sentences are enclosed within <target>...</target> tags.\n\n"
            "YOUR TASKS:\n"
            "1. Rewrite ONLY the sentences inside the <target>...</target> tags.\n"
            "2. Make sure the sentences inside the tags are actually edited and different from the original input.\n"
            "3. Keep the <target> and </target> tags exactly where they are, enclosing your newly rewritten sentences.\n"
            "4. Output the full abstract including both the untargeted text and your newly edited target sentences.\n\n"
            "CRITICAL RULES:\n"
            "- It is absolutely mandatory to preserve the <target> and </target> tags. Do not remove, alter, or misspell the tags themselves.\n"
            "- If the abstract begins immediately with a <target> tag, you must still rewrite that first sentence. Do not leave the first sentence unedited if it is tagged.\n"
            "- Do NOT add any introductory or concluding text (e.g., do not say 'Here is your rewrite:'). Output ONLY the final abstract.\n\n"
            "EXAMPLE:\n"
            "Input: Dit is de eerste zin. <target>Deze zin moet anders.</target> Dit is de derde zin.\n"
            "Output: Dit is de eerste zin. <target>Deze specifieke zin dient aangepast te worden.</target> Dit is de derde zin."
        ),
        "full_abstract": (
            "You are a professional Dutch editor.\n"
            "Rewrite the entire abstract in Dutch to make it better, while preserving all crucial information.\n"
            "Provide ONLY the fully rewritten abstract."
        )
    }
    
    # 1. Start Ollama server
    start_ollama_server(port=port, gpu_id=gpu_id)
    global ollama
    import ollama
    
    # 2. Load dataset
    if table is not None:
        ug_table = table
    else:
        print(f"Loading Parquet data from: {ug_path}")
        if not ug_path.exists():
            print(f"Error: Parquet file not found at {ug_path}")
            sys.exit(1)
        ug_table = pq.read_table(ug_path)
            
    # 3. Unified checkpoint load, alignment verification, completion check, and task preparation
    tasks, rows = prepare_tasks(ug_table, checkpoint_path, models_list, percentages_to_run)

    #exclude percentage option
    if exclude_percentage:
        print('EXCLUDING PERCENTAGE TASKS')
        tasks = [t for t in tasks if t['type']!='percentage']

    #ordering model -> sent -> abs
    if not models_list:
        # Fallback: extract unique models in order of appearance if none provided
        models_list = list(dict.fromkeys([t["model"] for t in tasks]))
        
    model_order = {model: idx for idx, model in enumerate(models_list)}
    type_order = {"sentence": 0, "percentage": 2, "full_abstract": 1}
    
    tasks.sort(key=lambda x: (
        model_order.get(x["model"], 99),
        type_order.get(x["type"], 99)
    ))
    
    if debug_mode:
        print("\n[DEBUG MODE ACTIVE] Filtering tasks: Keeping exactly 5 tasks of each unique combination of model, task type, and percentage.")
        counts = {}
        debug_tasks = []
        for task in tasks:
            key = (task.get("model"), task.get("type"), task.get("percentage"))
            current_count = counts.get(key, 0)
            if current_count < debug_count:
                counts[key] = current_count + 1
                debug_tasks.append(task)
        tasks = debug_tasks
        
    print(f"Total pending tasks to generate: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All specified tasks are already completed in the dataset. Exiting.")
        #if not debug_mode and checkpoint_path.exists(): TODO check if all are inside pq and clean up checkpoints
        return
        
    # 4. Register final emergency fallback exit handler
    #if not debug_mode:
    #    atexit.register(save_parquet_on_exit, rows, ug_path)
    else: 
        print('Debug mode active, not saving on exit.')
    
    # 5. Run generation
    run_generation(tasks, rows, system_prompts, checkpoint_path, debug_mode=debug_mode)
    
    # save when done
    if not debug_mode:
        try:
            print("\nGeneration finished successfully. Writing final table to Parquet...")
            #save_parquet_on_exit(rows, ug_path) #TODO change to write to row func
            
        except Exception as e:
            print(f"Error during final Parquet save: {e}. Checkpoint remains preserved for recovery.")
    else: 
        print('Debug complete, not saving final output.')








#script execution
def main(): #TODO set default and relative and variable paths + checks for skipping
    #TODO download section ---------------------------------------------------------------------------------
    pass

    #TODO cleaning section
    pass

    #selection ---------------------------------------------------------------------------------
    ug_select = Path('/home/gderijck/internship/data/silver/ug_selected.parquet') #TODO make relative
    table = None 
    
    if not ug_select.exists():
        table = select_and_clean_abstracts_ug(
            input_path=ug_select.parent / 'ug_cleaned.parquet', 
            output_path=ug_select
        )
    if table is None:
        table = pq.read_table(ug_select)


    #generation ---------------------------------------------------------------------------------
    checkpoint_path = ug_select.parent / "checkpoint_rewrites.jsonl"
    models_list = ['qwen3.6:27b', 'qwen3.5:4b', 'gemma4:26b', 'gemma4:e4b']
    #models_list = ['qwen3.5:4b', 'gemma4:26b', 'gemma4:e4b']
    percentages = [25,50,75]
    tasks, rows = prepare_tasks(
        table=table,
        checkpoint_path=checkpoint_path,
        models_list=models_list,
        percentages=percentages, 
    )
    
    active_models_list = get_models_list()


    if tasks:
        print('Starting llm gen')
        generation_main(
            table = table,
            ug_path = ug_select,
            checkpoint_path=checkpoint_path,
            models_list = active_models_list,
            percentages_to_run = percentages,
            debug_mode = False,
            #debug_count = 1,
            exclude_percentage=True #TODO fix percentage plssss
        )
    else:
        print('Dataset already populated')

    #---------------------------------------------------------------------------------


if __name__ == "__main__":
    main()