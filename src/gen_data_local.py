from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import json
import os
import subprocess
import time
import atexit
import socket
import argparse
import sys
import re
import string
from langdetect import detect, LangDetectException

PORT = 11435

# Force Ollama to ignore the broken GPU drivers entirely
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Empty string disables GPU/CUDA entirely

def is_port_in_use(port):
    """Checks if our private server is already active on this node."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

ollama_process = None

if not is_port_in_use(PORT):
    print(f"Starting background Ollama server on CPU (Port {PORT})...")
    
    # Clone the environment and apply settings to the background server
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU fallback
    env["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
    
    # Launch Ollama as a background subprocess
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for the server to successfully boot up
    for attempt in range(15):
        if is_port_in_use(PORT):
            print("Ollama server successfully launched on CPU.")
            break
        time.sleep(1)
    else:
        print("Warning: Ollama server initialization timed out. Exiting.")
        sys.exit(1)
else:
    print(f"Ollama server is already running on port {PORT}. Reusing existing instance.")

def shutdown_ollama_server():
    """Kills the background server process when Python exits."""
    global ollama_process
    if ollama_process:
        print("\nShutting down background Ollama server...")
        ollama_process.terminate()
        try:
            ollama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ollama_process.kill()
        print("Server stopped cleanly.")

# Register cleanup
atexit.register(shutdown_ollama_server)

import ollama

CALC_MODEL_MAPPING = {
    'calc12': ['qwen3.6:27b'],
    'calc11': ['qwen3.5:4b'],
    'calc10': ['gemma4:e4b']
}

def get_models_list():
    try:
        current_host = socket.gethostname().split('.')[0]
        default_calc = current_host if current_host in CALC_MODEL_MAPPING else 'calc12'
    except Exception:
        default_calc = 'calc12'

    parser = argparse.ArgumentParser(description='run llm rewrites on speicifc cluster node')
    parser.add_argument(
        '--calc',
        type=str,
        choices=list(CALC_MODEL_MAPPING.keys()),
        default=default_calc,
        help='specify calc'
    )
    args = parser.parse_args()
    selected_models = CALC_MODEL_MAPPING[args.calc]
    print(f'selected config: {args.calc} -> models_list: {selected_models}')
    return selected_models

models_list = ['gemma4:26b']






#-------------
ug_path = Path(r'E:\code\dta\internship\data\ug_selected.parquet')
checkpoint_path_final = Path(r'E:\code\dta\internship\data\checkpoint_rewrites_LOCAL_final.jsonl')
checkpoint_path = Path(r'E:\code\dta\internship\data\checkpoint_rewrites_LOCAL.jsonl')

ug = pq.read_table(ug_path)

def prepare_tasks(table: pa.Table) -> list:
    """Explodes the PyArrow table abstracts into indexed sentence tasks."""
    tasks = []
    
    if '_id' in table.column_names:
        ids = table['_id'].to_pylist()
    else:
        ids = list(range(len(table)))
        
    abstracts = table['text_dut'].to_pylist()
    
    for row_id, abstract in zip(ids, abstracts):
        if isinstance(abstract, str) and abstract.strip():
            # 1. Initial NLTK sentence tokenization (optimized for Dutch)
            raw_sentences = nltk.sent_tokenize(abstract, language='dutch')
            
            # 2. Tokenization Cleanup (Punctuation and Capitalization fixes)
            cleaned_sentences = []
            for sent in raw_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                should_merge = False
                if cleaned_sentences:
                    # Check 1: Does not start with a capital letter A-Z
                    if not re.match(r'^[A-Z]', sent):
                        should_merge = True
                    
                    # Check 2: Starts with A-Z but has a punctuation mark in the first 5 characters
                    elif len(sent) >= 2 and sent[1] in string.punctuation:
                        should_merge = True
                
                if should_merge:
                    # Append to the previous sentence with a space
                    cleaned_sentences[-1] = cleaned_sentences[-1] + " " + sent
                else:
                    cleaned_sentences.append(sent)
            
            # 3. Language Filtering (Keep only Dutch 'nl')
            dutch_sentences = []
            for sent in cleaned_sentences:
                try:
                    if detect(sent) == 'nl':
                        dutch_sentences.append(sent)
                except LangDetectException:
                    continue
            
            # 4. Generate task list with continuous sent_idx
            for sent_idx, sentence in enumerate(dutch_sentences):
                tasks.append({
                    "id": row_id,
                    "sent_idx": sent_idx,
                    "text": sentence
                })
    return tasks

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
        
        # Extract the text content from the Ollama response dictionary
        rewritten = response['response'].strip()
        rewritten = rewritten.replace('\x00','').replace('\u0000','')
        #response['done'] == True confirms it went thru
        
        # Clean up stray quotes if the LLM adds them
        if not sentence.startswith('"'):
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
        return rewritten
    
    except Exception as e:
        print(f"Error calling Ollama ({model_to_run}): {e}")
        sys.exit(1)


def run_generation(models_to_run, system_prompt, tasks, load_file:Path, save_file:Path):
    tasks_map = {(task['id'], task['text']): task for task in tasks}
    completed_runs = set()
    if load_file.exists():
        print(f'Loading existing progress from {load_file}')
        with open(load_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    t_id = record['id']
                    text = record['text']
                    if text:
                        for model in models_to_run:
                            if model in record:
                                rewritten = record[model]
                                completed_runs.add((t_id, text, model))
                                if (t_id, text) in tasks_map:
                                    tasks_map[(t_id, text)][model] = rewritten
        print(f'Loaded {len(completed_runs)} completed runs. Skipping all of those.')
    
    #adds completed gens from active file
    if save_file.exists() and save_file != load_file:
        print(f'Loading active run progress from: {save_file}')
        with open(save_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    t_id = record['id']
                    text = record.get('text')
                    if text:
                        for model in models_to_run:
                            if model in record:
                                rewritten = record[model]
                                completed_runs.add((t_id, text, model))
                                if (t_id, text) in tasks_map:
                                    tasks_map[(t_id, text)][model] = rewritten
        print(f'Synchronized active run progress with master cache.')

    for model in models_to_run:
        print(f'\nStarting model: {model}')
        for i, task in enumerate(tasks):
            t_id = task['id']
            s_idx = task['sent_idx']
            text = task['text']
    
            if (t_id, text, model) in completed_runs:
                continue

            print(f'\n[{model}] Rewriting sentence {i+1}/{len(tasks)} (ID: {t_id}, Sent#: {s_idx}, Sentence:)')
            print(f'{text}')
            rewritten = rewrite_sentence(model, system_prompt=system_prompt, sentence=text)
            print(f'{rewritten}')
            task[model] = rewritten
            checkpoint_record = {
                'id': t_id,
                'sent_idx': s_idx,
                'text': text,
                model: rewritten
            }
            with open(save_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(checkpoint_record, ensure_ascii=False) + '\n')
            completed_runs.add((t_id, text, model))
    return tasks




system_prompt = (
    "Je bent een professionele Nederlandse tekstverwerker."
    "Jouw taak is: Herschrijf de volgende zin. Bewaar cruciale informatie"
    "Geef UITSLUITEND de herschreven zin terug."
)
tasks = prepare_tasks(ug)

output = run_generation(
    models_to_run=models_list,
    system_prompt=system_prompt,
    tasks=tasks,
    load_file=checkpoint_path_final,
    save_file = checkpoint_path
)