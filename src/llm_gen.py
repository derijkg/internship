#TODO remove download libs
#TODO change to use final df from datasetcontrstuction, 
    #  change #GOLD.. 
    #change inputs for gen_main (and others?)
    # change or remove consolidate checkpoints move to using only 1 in gold
    #
#TODO set argparse options to prio one source? 
#TODO

#TODO integreer beststaande cp_rwr_ug -> als de exacte zin er al in staat skip (verwijderen van task)
#TODO read from csv abstract sentences '|' or ' | ' 

import re
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from langdetect import detect, LangDetectException
import string
import random
import argparse
import json
import pyarrow as pa
import pyarrow.json as pj
import pyarrow.parquet as pq
import pyarrow.csv as pv
import pyarrow.compute as pc
import os
import subprocess
import atexit
import socket
import binascii
from ollama import Client
import unicodedata
#from pydantic import BaseModel, Field
#import itertools

#TODO add different methods for generating the uuhhhh percentage task thru argparse
#TODO read from csv: abs_sent =  ' | ' joined sentences


# GLOBAL VARS

BASE_DIR = Path(__file__).resolve().parent.parent #this script is in /home/gderijck/internship/src/llm_gen.py
INPUT_DATA_CSV = BASE_DIR / "data" / "gold" / "merged_publications.csv"
INPUT_DATA_PARQUET = BASE_DIR / "data" / "gold" / "merged_publications.parquet"
OUTPUT_DATA_CSV = BASE_DIR / "data" / "gold" / "llm_added.csv"
OUTPUT_DATA_PARQUET = BASE_DIR / "data" / "gold" / "llm_added.parquet"
CHECKPOINT_PATH = BASE_DIR / "data" / "gold" / "checkpoint_rewrites.jsonl"

# generation
# ollama server functions
def kill_process_on_port(port: int):
    """
    Attempts to find and terminate any process listening on the specified port.
    Safely targets only your specific port to avoid affecting other shared processes.
    """
    import subprocess
    import os
    import signal
    import time
    
    try:
        # 'lsof -t' returns only the raw PID(s) bound to the port
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True, check=False)
        pids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        
        for pid_str in pids:
            pid = int(pid_str)
            print(f"Found active process (PID {pid}) on port {port}. Force-terminating...")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError as e:
                print(f"Could not kill PID {pid}: {e}")
        
        if pids:
            time.sleep(1.5)  # Give the operating system a brief moment to release the port socket
    except Exception as e:
        print(f"Warning: Port cleanup helper failed: {e}")

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def shutdown_ollama_server(process: subprocess.Popen):
    if process and process.poll() is None:
        print("\nShutting down background Ollama server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server stopped cleanly.")

def start_ollama_server(port: int = 11435, gpu_id: str = None) -> subprocess.Popen:
    kill_process_on_port(port)
    gpu_str = str(gpu_id).strip() if gpu_id is not None else ""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    
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
        print("Falling back to PATH.")

    cuda_libs = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
    user_libs = str(Path.home() / ".local" / "lib" / "ollama")
    
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{user_libs}:{env['LD_LIBRARY_PATH']}:{cuda_libs}"
    else:
        env["LD_LIBRARY_PATH"] = f"{user_libs}:{cuda_libs}"
    
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
        
        log_path = BASE_DIR / "ollama_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        
        process = subprocess.Popen(
            [ollama_executable, "serve"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        for attempt in range(15):
            if is_port_in_use(port):
                print("Ollama server successfully launched.")
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

def unload_model(model_name: str):
    print(f"\nUnloading model '{model_name}' from GPU memory...")
    try:
        host_env = os.environ.get("OLLAMA_HOST", "127.0.0.1:11435")
        if not host_env.startswith("http://"):
            host_env = f"http://{host_env}"
            
        client = Client(host=host_env, timeout=30)
        client.generate(model=model_name, prompt="", keep_alive=0)
        print(f"Successfully unloaded '{model_name}'.")
        time.sleep(2)
    except Exception as e:
        print(f"Warning: Could not explicitly unload '{model_name}': {e}")



# GENERATION proper
#after generation add to checkpoint.jsonl
def append_to_checkpoint(checkpoint_path: Path, task: dict, rewritten_text: str):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": task["id"],
        "type": task["type"],
        "model": task["model"],
        "sent_idx": task.get("sent_idx"),
        "percentage": task.get("percentage"),
        'text': task.get('text'),
        "rewritten": rewritten_text
    }
    
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


#pydantic 
#class TargetRewrite(BaseModel):
#    target_id: int = Field(
#        description="The matching numeric ID of the target block (e.g., 1 for target_1, 2 for target_2)."
#    )
#    #thought_process: str = Field(
#    #    description="A brief planning note analyzing how to rewrite this block to improve flow and grammar while matching the rest of the abstract's context."
#    #)
#    rewritten_text: str = Field(
#        description="The edited Dutch text for this target block." # Do NOT include any XML or target tags in this output."
#    )#
#
#class PercentageRewrites(BaseModel):
#    rewrites: list[TargetRewrite] = Field(
#        description="The list of edits for each numbered target block in sequential order."
#    )



#check if % rewrite task was succesfully completed
#TODO % task robust validate or change task
def validate_percentage_rewrite(
    original_sents: list[str], 
    raw_output_text: str, 
    tagged_groups: list[list[int]]
) -> tuple[bool, str, str | None]:
    if not raw_output_text:
        return False, "Empty response from the model", None

    # #AD: Parser updated to extract numbered target blocks using backreferences from raw output instead of loading JSON schemas
    matches = re.findall(r'<target_(\d+)>(.*?)</target_\1>', raw_output_text, re.DOTALL | re.IGNORECASE)
    
    # #AD: Defensive map lookup dictionary handles string-to-int conversion securely to prevent ValueError execution crashes
    rewrites_map = {}
    for t_id_str, text in matches:
        text_clean = text.strip()
        if t_id_str and text_clean:
            try:
                parsed_id = int(t_id_str)
                rewrites_map[parsed_id] = text_clean
            except (ValueError, TypeError):
                continue

    stitched_sentences = list(original_sents)
    
    for idx, group in enumerate(tagged_groups, 1):
        rewrite_text = rewrites_map.get(idx)
        if not rewrite_text:
            return False, f"Missing target rewrite for target ID {idx} (Verify tag syntax in LLM output)", None

        # Reconstruct the original contiguous block of sentences
        original_block = " ".join([original_sents[s_idx].strip() for s_idx in group])
        original_clean = original_block.strip()
        
        # Verify changes were made compared to the normalized original text
        if normalize_text(rewrite_text) == normalize_text(original_clean):
            return False, f"Target block {idx} was not modified by the model.", None
            
        # Place the rewrite in the first index, and clear out subsequent indices in the group
        stitched_sentences[group[0]] = rewrite_text
        for subsequent_idx in group[1:]:
            stitched_sentences[subsequent_idx] = ""

    #AD: Stitch the original untagged sentences and new edits into a single abstract
    final_abstract = " ".join([sent for sent in stitched_sentences if sent])
    return True, "Success", final_abstract


#TODO maybe change %task
def prepare_tasks(
    table: pa.Table, 
    checkpoint_path: Path,
    models_list: list[str], 
    percentages: list[int] = None
) -> tuple[list[dict], list[dict]]:
    checkpoint_path = Path(checkpoint_path)
    if percentages is None:
        percentages = []

    rows = table.to_pylist()
    
    # Dynamic key fallback resolution if the dataset has unconventional naming schemas
    if '_id' in table.column_names:
        id_key = '_id'
    elif 'id' in table.column_names: #This one
        id_key = 'id'
    elif 'page_link' in table.column_names:
        id_key = 'page_link'
    else:
        id_key = 'synthetic_id'
        for idx, r in enumerate(rows):
            r[id_key] = str(idx)

    rows_map = {str(row[id_key]): row for row in rows}

    # Initialize counters (including full_abstract and percentage tracking)
    total_loaded = 0
    text_match_count = 0
    idx_fallback_count = 0
    full_abstract_match_count = 0
    percentage_match_count = 0
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
                    
                    if "type" in record:
                        row_id = record.get("id")
                        t_type = record.get("type")
                        model = record.get("model")
                        sent_idx = record.get("sent_idx")
                        pct = record.get("percentage")
                        orig_text = record.get("text")
                        rewritten_text = record.get("rewritten")
                    else:
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

                    if isinstance(rewritten_text, str) and '<channel|>' in rewritten_text:
                        rewritten_text = rewritten_text.split('<channel|>')[1].strip()

                    row = rows_map.get(str(row_id))
                    if not row:
                        missing_row_discard_count += 1
                        continue

                    if t_type == "sentence":
                        # Changed 'sent_dut' to 'abstract_sentence' to align with the new schema
                        abstract_sentence = row.get("abstract_sentence")
                        if not isinstance(abstract_sentence, list):
                            mismatched_discard_count += 1
                            continue

                        matched_idx = -1
                        if orig_text:
                            clean_orig = orig_text.strip()
                            for idx, sent in enumerate(abstract_sentence):
                                if sent.strip() == clean_orig:
                                    matched_idx = idx
                                    text_match_count += 1
                                    break

                        if matched_idx == -1 and sent_idx is not None:
                            try:
                                sent_idx_int = int(sent_idx)
                                if 0 <= sent_idx_int < len(abstract_sentence):
                                    matched_idx = sent_idx_int
                                    idx_fallback_count += 1
                            except (ValueError, TypeError):
                                pass

                        if matched_idx == -1:
                            mismatched_discard_count += 1
                            continue

                        task_meta = {"type": t_type, "model": model, "sent_idx": matched_idx, "percentage": pct}
                        apply_rewrite_to_row(row, task_meta, rewritten_text)
                    else:
                        task_meta = {"type": t_type, "model": model, "sent_idx": sent_idx, "percentage": pct}
                        apply_rewrite_to_row(row, task_meta, rewritten_text)
                        
                        # Increment correct tracker based on task type
                        if t_type == "full_abstract":
                            full_abstract_match_count += 1
                        elif t_type == "percentage":
                            percentage_match_count += 1

                except (json.JSONDecodeError, ValueError, TypeError):
                    corrupted_count += 1
                    continue

    # Improved alignment report printing
    if total_loaded > 0:
        total_aligned = text_match_count + idx_fallback_count + full_abstract_match_count + percentage_match_count
        print("\n====================================================")
        print("          CHECKPOINT MERGING & ALIGNMENT REPORT")
        print("====================================================")
        print(f" Total records loaded from checkpoint: {total_loaded}")
        print("----------------------------------------------------")
        print(f" Successfully aligned (Total)              : {total_aligned}")
        print(f"   - Sentence (Exact Text Matches)         : {text_match_count}")
        print(f"   - Sentence (Index Fallbacks)            : {idx_fallback_count}")
        print(f"   - Full Abstract Rewrites                : {full_abstract_match_count}")
        print(f"   - Percentage-based Rewrites             : {percentage_match_count}")
        print("----------------------------------------------------")
        print(f" Discarded (Unmatched sentence text/index) : {mismatched_discard_count}")
        print(f" Discarded (Missing row IDs)               : {missing_row_discard_count}")
        print(f" Discarded (Corrupted/Malformed lines)     : {corrupted_count}")
        print("====================================================\n")


    #CREATE TASKS FOR REMAINING
    tasks = []
    for row in rows:
        row_id = row.get(id_key)
        # ADDED: Changed 'text_dut' -> 'abstract' and 'sent_dut' -> 'abstract_sentence'
        abstract = row.get('abstract')
        abstract_sentence = row.get('abstract_sentence')

        if not isinstance(abstract, str) or not abstract.strip() or not isinstance(abstract_sentence, list) or not abstract_sentence:
            continue

        num_sentences = len(abstract_sentence)

        for model in models_list:
            col_name = f'{model}_single'
            if col_name not in row or not isinstance(row[col_name], list) or len(row[col_name]) != num_sentences:
                row[col_name] = [None] * num_sentences

            for sent_idx, sentence in enumerate(abstract_sentence):
                if row[col_name][sent_idx] is None:
                    tasks.append({
                        "id": row_id,
                        "type": "sentence",
                        "model": model,
                        "sent_idx": sent_idx,
                        "text": sentence,
                        "context": abstract # ADDED: Changed context references to new abstract column
                    })

            for pct in percentages:
                col_name = f"{model}_{pct}"
                if col_name not in row or not row[col_name]:
                    row[col_name] = None
                    seed_str = f"{row_id}_{pct}".encode("utf-8")
                    seed = binascii.crc32(seed_str)
                    rng = random.Random(seed)
                    num_to_tag = max(1, round(num_sentences * (pct / 100.0)))
                    tagged_indices = set(rng.sample(range(num_sentences), num_to_tag))

                    groups = []
                    sorted_indices = sorted(list(tagged_indices))
                    if sorted_indices:
                        current_group = [sorted_indices[0]]
                        for idx in sorted_indices[1:]:
                            if idx == current_group[-1] + 1:
                                current_group.append(idx)
                            else:
                                groups.append(current_group)
                                current_group = [idx]
                        groups.append(current_group)
                    
                    annotated_parts = []
                    group_starts = {g[0]: g for g in groups}
                    group_to_id = {g[0]: idx + 1 for idx, g in enumerate(groups)}
                    
                    i = 0
                    while i < num_sentences:
                        if i in group_starts:
                            group = group_starts[i]
                            t_id = group_to_id[i]
                            block_text = " ".join([abstract_sentence[idx] for idx in group]) # ADDED: Changed from sent_dut
                            annotated_parts.append(f"<target_{t_id}>{block_text}</target_{t_id}>")
                            i += len(group)
                        else:
                            annotated_parts.append(abstract_sentence[i]) # ADDED: Changed from sent_dut
                            i += 1
                    
                    annotated_abstract = " ".join(annotated_parts)

                    tasks.append({
                        "id": row_id,
                        "type": "percentage",
                        "model": model,
                        "percentage": pct,
                        "text": annotated_abstract,
                        'tagged_groups': groups  # List of index groups
                    })

            col_name = f"{model}_full"
            if col_name not in row or not row[col_name]:
                row[col_name] = None
                tasks.append({
                    "id": row_id,
                    "type": "full_abstract",
                    "model": model,
                    "text": abstract # ADDED: Changed from text_dut
                })

    return tasks, rows



def apply_rewrite_to_row(row: dict, task: dict, rewritten: str):
    if not row:
        return
    t_type = task["type"]
    model = task["model"]
    
    if t_type == "sentence":
        sent_idx = task["sent_idx"]
        num_sentences = len(row['abstract_sentence'])
        if f'{model}_single' not in row or not isinstance(row[f'{model}_single'], list) or len(row[f'{model}_single']) != num_sentences:
            row[f'{model}_single'] = [None] * num_sentences
            
        if 0 <= sent_idx < num_sentences:
            row[f'{model}_single'][sent_idx] = rewritten
    elif t_type == "percentage":
        pct = task["percentage"]
        row[f"{model}_{pct}"] = rewritten
    elif t_type == "full_abstract":
        row[f"{model}_full"] = rewritten


def rewrite_sentence(client, model_to_run, system_prompt, sentence, seed=42, response_format=None):
    """Sends text to Ollama for rewriting with optional JSON schema constraints."""
    response = client.generate(
        model=model_to_run,
        system=system_prompt,
        prompt=sentence,
        think=False,
        options={
            "seed": seed,
        },
    )

    rewritten = response['response'].strip()
    rewritten = rewritten.replace('\x00','').replace('\u0000','')

    if isinstance(rewritten, str) and '<channel|>' in rewritten:
        rewritten = rewritten.split('<channel|>')[1].strip()

    if not sentence.startswith('"'):
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        
    return normalize_text(rewritten)


def run_generation(
    tasks: list[dict],
    rows: list[dict],
    system_prompt_mapping: dict,
    checkpoint_path: Path,
    debug_mode: bool = False
):
    checkpoint_path = Path(checkpoint_path)
    host_env = os.environ.get("OLLAMA_HOST", "127.0.0.1:11435")
    if not host_env.startswith("http://"):
        host_env = f"http://{host_env}"
    client = Client(host=host_env, timeout=300)

    id_key = next((k for k in ['_id', 'id', 'page_link', 'synthetic_id'] if rows and k in rows[0]), None)
    rows_map = {row[id_key]: row for row in rows} if id_key else {}

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

        rewritten = None
        max_attempts = 5

        for attempt in range(max_attempts):    
            current_seed = 42 + attempt
                
            try:
                candidate_rewrite = rewrite_sentence(
                    client, 
                    model, 
                    system_prompt, 
                    text, 
                    seed=current_seed,
                )
            except Exception as e:
                print(f"  [Error - Attempt {attempt+1}/{max_attempts}] Calling Ollama failed: {e}")
                if attempt < max_attempts - 1:
                    print("  Waiting 5 seconds before retrying...")
                    time.sleep(5)
                    continue
                else:
                    print(f"  [Error] Task {t_id} failed after {max_attempts} attempts. Writing sentinel.")
                    rewritten = "FAILED_GENERATION"
                    break
            
            if t_type == 'percentage':
                original_sentences = row['abstract_sentence']
                tagged_groups = task['tagged_groups']
                
                try:
                    is_valid, reason, stitched_text = validate_percentage_rewrite(
                        original_sentences, 
                        candidate_rewrite,  
                        tagged_groups
                    )
                except Exception as eval_err:
                    is_valid = False
                    reason = f"Internal evaluation parser raised an unhandled exception: {eval_err}"
                    stitched_text = None

                if is_valid:
                    rewritten = stitched_text
                    print(f'ORIGINAL: {text}\n')
                    print(f'REWRITTEN: {rewritten}')
                    break
                else:
                    print(f"  [Warning - Attempt {attempt+1}/{max_attempts} Failed] {reason}.")
                    print(f"  [DEBUG - Prompt Sent to LLM]:\n{text}")
                    print(f"  [DEBUG - Raw Candidate Response]:\n{candidate_rewrite}")
                    print("-" * 50)
                    
                    if attempt < max_attempts - 1:
                        print("  Retrying generation...")
                    else:
                        print(f"  [Warning] Task {t_id} failed validation. Writing sentinel: FAILED_VALIDATION")
                        rewritten = "FAILED_VALIDATION"
                        break
                        
            elif t_type in ('sentence', 'full_abstract'):
                is_valid = candidate_rewrite.strip() != text.strip()
                
                if is_valid:
                    rewritten = candidate_rewrite
                    print(f'ORIGINAL: {text}\n')
                    print(f'REWRITTEN: {rewritten}')
                    break
                else:
                    word_count = len(text.strip().split())
                    char_count = len(text.strip())
                    is_short = word_count <= 6 or char_count <= 40
                    if is_short:
                        rewritten = candidate_rewrite
                        print(f'ORIGINAL: {text}\n')
                        print(f'REWRITTEN: {rewritten} [Accepted identical output due to short sentence length]')
                        break

                    reason = "The model's rewrite is identical to the original input."
                    print(f"  [Warning - Attempt {attempt+1}/{max_attempts} Failed] {reason}.")
                    print(f"  [DEBUG - Prompt Sent to LLM]:\n{text}")
                    print(f"  [DEBUG - Raw Candidate Response]:\n{candidate_rewrite}")
                    print("-" * 50)
                    
                    if attempt < max_attempts - 1:
                        print("  Retrying generation...")
                    else:
                        print(f"  [Warning] Task {t_id} failed validation. Writing sentinel: FAILED_VALIDATION")
                        rewritten = "FAILED_VALIDATION"
                        break

        apply_rewrite_to_row(row, task, rewritten)

        if not debug_mode:
            append_to_checkpoint(checkpoint_path, task, rewritten)

    if current_model is not None:
        unload_model(current_model)
    return rows


def normalize_text(text):
    if not isinstance(text, str):
        return text
    
    # 1. Normalize unicode (compatibility decomposition followed by canonical composition)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Standardize common quotation marks and dashes (optional but recommended)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-') # Standardize em/en dashes to hyphens
    
    # 3. Collapse multiple whitespaces and strip leading/trailing spaces
    text = " ".join(text.split())
    
    return text


#select models for differnet calcs
def get_models_list():
    CALC_MODEL_MAPPING = {
        'calc12': ['gemma4:e4b', 'qwen3.5:4b'],
        'calc11': ['qwen3.6:27b','gemma4:26b'], #['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']
    }
    try:
        current_host = socket.gethostname().split('.')[0]
        if current_host not in CALC_MODEL_MAPPING:
            print(f"Host '{current_host}' not found in configuration mappings.")
            sys.exit(1)
        default_calc = current_host
    except Exception as e:
        print(f'Error identifying host config: {e}')
        sys.exit(1)

    selected_models = CALC_MODEL_MAPPING[default_calc]
    print(f'selected config: {current_host} -> models_list: {selected_models}')
    return selected_models


def generation_main(
    table: pa.Table,
    models_list: list[str] = None,
    percentages_to_run: list[int] = [25, 50, 75],
    port: int = 11435,
    gpu_id: str = None,
    debug_mode: bool = False,
    debug_pct_only: bool = False,
    debug_count: int = 5,
    exclude_percentage: bool = False,
    task_steps: List = None,
    priority: bool = False
) -> list[dict]: # ADDED: Returns updated rows list
    system_prompts = {
        "sentence": (
            "You are a professional Dutch editor.\n"
            "Rewrite the following sentence to make it better while preserving all crucial information.\n"
            "Provide ONLY the rewritten sentence."
        ),
        "percentage": (
            "You are a professional Dutch editor.\n"
            "Your task is to rewrite only the text segments enclosed in numbered target tags (e.g., <target_1>...</target_1>, <target_3>...</target_3>) so as to improve them.\n"
        ),
        "full_abstract": (
            "You are a professional Dutch editor.\n"
            "Rewrite the entire abstract in Dutch to make it better, while preserving all crucial information.\n"
            "Provide ONLY the fully rewritten abstract."
        )
    }
    
    start_ollama_server(port=port, gpu_id=gpu_id)
    global ollama
    import ollama
    
    # ADDED: Removed directory search and explicitly parse using the global CHECKPOINT_PATH
    tasks, rows = prepare_tasks(table, CHECKPOINT_PATH, models_list, percentages_to_run)

    if task_steps:
        # Map cli choices to internal task type identifiers
        type_mapping = {
            'sentence': 'sentence',
            'full': 'full_abstract',
            'percentage': 'percentage'
        }
        allowed_types = [type_mapping[t] for t in task_steps if t in type_mapping]
        tasks = [t for t in tasks if t['type'] in allowed_types]
        
    if debug_pct_only:
        print("\n[DEBUG PCT ONLY ACTIVE] Filtering out non-percentage tasks.")
        tasks = [t for t in tasks if t['type'] == 'percentage']
        debug_mode = True  # Force debug count verification below

    if exclude_percentage:
        print('EXCLUDING PERCENTAGE TASKS')
        tasks = [t for t in tasks if t['type'] != 'percentage']


    if priority:
        print("\n[PRIORITY MODE ACTIVE] Filtering tasks to exactly 1 random model 'full_abstract' rewrite per abstract.")
        print("Using deterministic hashing to split the workload across calc11 and calc12 models.")
        import binascii
        import random
        
        # Define the complete pool of models across both systems
        all_possible_models = ['gemma4:26b', 'gemma4:e4b', 'qwen3.6:27b', 'qwen3.5:4b']
        
        # Create a mapping of row IDs to their row data to inspect columns
        id_key = next((k for k in ['_id', 'id', 'page_link', 'synthetic_id'] if rows and k in rows[0]), None)
        rows_map = {str(row[id_key]): row for row in rows} if id_key else {}
        
        prioritized_tasks = []
        processed_row_ids = set()
        
        for t in tasks:
            if t["type"] == "full_abstract":
                row_id_str = str(t["id"])
                if row_id_str in processed_row_ids:
                    continue
                
                row = rows_map.get(row_id_str)
                
                # Check if this row already has a completed full rewrite for ANY model
                already_has_rewrite = False
                if row:
                    for col_name, col_val in row.items():
                        if col_name.endswith("_full") and col_val:
                            already_has_rewrite = True
                            break
                
                if already_has_rewrite:
                    processed_row_ids.add(row_id_str)
                    continue
                
                # Deterministically choose the assigned model for this row ID
                seed_str = f"priority_split_{row_id_str}".encode("utf-8")
                row_seed = binascii.crc32(seed_str)
                row_rng = random.Random(row_seed)
                selected_model = row_rng.choice(all_possible_models)
                
                # Search through the pending tasks prepared for the current host's models
                matching_task = None
                for candidate_t in tasks:
                    if (candidate_t["type"] == "full_abstract" and 
                        str(candidate_t["id"]) == row_id_str and 
                        candidate_t["model"] == selected_model):
                        matching_task = candidate_t
                        break
                
                # If this host is responsible for the selected model, queue it
                if matching_task:
                    prioritized_tasks.append(matching_task)
                
                processed_row_ids.add(row_id_str)
                
        tasks = prioritized_tasks

    if not models_list:
        models_list = list(dict.fromkeys([t["model"] for t in tasks]))
        
    model_order = {model: idx for idx, model in enumerate(models_list)}
    type_order = {"sentence": 0, "percentage": 2, "full_abstract": 1}
    
    tasks.sort(key=lambda x: (
        model_order.get(x["model"], 99),
        type_order.get(x["type"], 99)
    ))
    
    if debug_mode:
        print(f"\n[DEBUG MODE ACTIVE] Filtering tasks: Keeping exactly {debug_count} tasks of each combination.")
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
        print("All specified tasks are already completed in the dataset.")
        return rows
        
    # ADDED: Passes global CHECKPOINT_PATH and returns the updated rows structure
    updated_rows = run_generation(tasks, rows, system_prompts, CHECKPOINT_PATH, debug_mode=debug_mode)
    return updated_rows


# script execution
def main():
    parser = argparse.ArgumentParser(description="Multi-source NLP pipeline orchestrator")
    parser.add_argument(
        '--source', 
        type=str,
        nargs='+', 
        default=['UG', 'HBO'], 
        choices=['UG', 'HBO', 'SB'], 
        help="Source dataset to process (UG: UGent, HBO: HBO Kennisbank, SB: Scriptiebank). If not specified, processes all available."
    )
    parser.add_argument(
        '--priority',
        action='store_true',
        help='Prioritize 1 full abstract rewrite per row with random model, grouped by model.'
    )
    parser.add_argument(
        '--tasks', 
        type=str, 
        nargs='+', 
        default=['sentence','full'], 
        choices=['sentence', 'percentage', 'full'], 
        help="Type of tasks to perform. Default: sentence and full"
    )
    parser.add_argument('--debug', action='store_true', help="Run the LLM generation in debug mode (fewer tasks)")
    parser.add_argument(
        '--format', 
        type=str, 
        default='both', 
        choices=['csv', 'parquet', 'both'], 
        help="Output save format. Default: both"
    )

    args = parser.parse_args()

    # ADDED: Load previous output if it exists to preserve current progress across runs, otherwise fallback to base dataset
    if OUTPUT_DATA_PARQUET.exists():
        print(f"Loading existing output parquet: {OUTPUT_DATA_PARQUET}")
        table = pq.read_table(OUTPUT_DATA_PARQUET)
    elif OUTPUT_DATA_CSV.exists():
        print(f"Loading existing output csv: {OUTPUT_DATA_CSV}")
        table = pa.Table.from_pandas(pd.read_csv(OUTPUT_DATA_CSV))
    elif INPUT_DATA_PARQUET.exists():
        print(f"Loading base publications parquet: {INPUT_DATA_PARQUET}")
        table = pq.read_table(INPUT_DATA_PARQUET)
    elif INPUT_DATA_CSV.exists():
        print(f"Loading base publications csv: {INPUT_DATA_CSV}")
        table = pa.Table.from_pandas(pd.read_csv(INPUT_DATA_CSV))
    else:
        print("Error: No input or output data datasets located in the gold directory.")
        sys.exit(1)

    df = table.to_pandas()
    for col in df.columns:
        if df[col].dtype == object:
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, str) and first_val.strip().startswith('['):
                try:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                except Exception:
                    pass

    if args.source:
        print(f"Filtering dataset for source: {args.source}")
        source_mask = df['source'].isin(args.source)
        active_df = df[source_mask].copy()
        inactive_df = df[~source_mask].copy()
    else:
        print("Processing all sources found within the dataset.")
        active_df = df.copy()
        inactive_df = pd.DataFrame()

    if active_df.empty:
        print(f"No records available to process matching source: {args.source}")
        sys.exit(0)

    active_table = pa.Table.from_pandas(active_df)

    percentages = []
    if 'percentage' in args.tasks:
        percentages = [25, 50, 75]

    if args.debug:
        print("[DEBUG CONFIG] Allocating complete multi-model test suite.")
        active_models_list = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']
    else:
        active_models_list = get_models_list()


    print("Starting LLM generation pipeline...")
    updated_rows = generation_main(
        table=active_table,
        models_list=active_models_list,
        percentages_to_run=percentages,
        debug_mode=args.debug,
        task_steps=args.tasks,
        exclude_percentage=('percentage' not in args.tasks),
        priority=args.priority
    )

    #TODO remove this and run seperate script for creating or adding to llm_added.parquet / .csv
    if not args.debug:
        updated_active_df = pd.DataFrame(updated_rows)
        
        # Combine modifications back with unprocessed entries
        if not inactive_df.empty:
            final_df = pd.concat([updated_active_df, inactive_df], ignore_index=True)
        else:
            final_df = updated_active_df

        print("\nAll tasks completed. Saving final aggregated dataset...")
        OUTPUT_DATA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        
        # ADDED: Export to CSV and/or Parquet according to selection (default: both)
        if args.format in ('parquet', 'both'):
            final_table = pa.Table.from_pandas(final_df)
            pq.write_table(final_table, OUTPUT_DATA_PARQUET)
            print(f"Successfully saved Parquet: {OUTPUT_DATA_PARQUET}")
            
        if args.format in ('csv', 'both'):
            final_df.to_csv(OUTPUT_DATA_CSV, index=False)
            print(f"Successfully saved CSV: {OUTPUT_DATA_CSV}")
    else:
        print("\n[DEBUG] Pipeline completed execution. File system changes skipped in debug mode.")

if __name__ == "__main__":
    main()


'''
import unicodedata

def preprocess_text(text):
    if not isinstance(text, str):
        return text
    
    # 1. Normalize unicode (compatibility decomposition followed by canonical composition)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Standardize common quotation marks and dashes (optional but recommended)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-') # Standardize em/en dashes to hyphens
    
    # 3. Collapse multiple whitespaces and strip leading/trailing spaces
    text = " ".join(text.split())
    
    return text

# Apply this function to both datasets
human_abstracts_cleaned = [preprocess_text(t) for t in human_abstracts]
llm_abstracts_cleaned = [preprocess_text(t) for t in existing_llm_abstracts]

'''