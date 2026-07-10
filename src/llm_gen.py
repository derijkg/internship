#TODO remove download libs
#TODO change to use final df from datasetcontrstuction, 
    #  change #GOLD.. 
    #change inputs for gen_main (and others?)
    # change or remove consolidate checkpoints move to using only 1 in gold
    #
#TODO set argparse options to prio one source? 
#TODO

#TODO integreer beststaande cp_rwr_ug -> als de exacte zin er al in staat skip (verwijderen van task)

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
#from pydantic import BaseModel, Field
#import itertools

#TODO add different methods for generating the uuhhhh percentage task thru argparse
#TODO 


# GLOBAL VARS
# - Path(__file__).resolve() points to 'internship/src/run.py'
# - .parent.parent resolves to the root folder 'internship/'
BASE_DIR = Path(__file__).resolve().parent.parent

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
#not used
def save_parquet_on_exit(rows: list[dict], output_path: Path):
    if rows:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[Exit Handler] Auto-saving active progress to Parquet: {output_path}...")
        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, output_path)
            print("[Exit Handler] Progress saved successfully.")
        except Exception as e:
            print(f"[Exit Handler] Error during auto-save: {e}")

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
    elif 'id' in table.column_names:
        id_key = 'id'
    elif 'page_link' in table.column_names:
        id_key = 'page_link'
    else:
        id_key = 'synthetic_id'
        for idx, r in enumerate(rows):
            r[id_key] = str(idx)

    rows_map = {str(row[id_key]): row for row in rows}

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
                        sent_dut = row.get("sent_dut")
                        if not isinstance(sent_dut, list):
                            mismatched_discard_count += 1
                            continue

                        matched_idx = -1
                        if orig_text:
                            clean_orig = orig_text.strip()
                            for idx, sent in enumerate(sent_dut):
                                if sent.strip() == clean_orig:
                                    matched_idx = idx
                                    text_match_count += 1
                                    break

                        if matched_idx == -1 and sent_idx is not None:
                            try:
                                sent_idx_int = int(sent_idx)
                                if 0 <= sent_idx_int < len(sent_dut):
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

                except (json.JSONDecodeError, ValueError, TypeError):
                    corrupted_count += 1
                    continue

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

    tasks = []
    for row in rows:
        row_id = row.get(id_key)
        text_dut = row.get('text_dut')
        sent_dut = row.get('sent_dut')

        if not isinstance(text_dut, str) or not text_dut.strip() or not isinstance(sent_dut, list) or not sent_dut:
            continue

        num_sentences = len(sent_dut)

        for model in models_list:
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
                            block_text = " ".join([sent_dut[idx] for idx in group])
                            annotated_parts.append(f"<target_{t_id}>{block_text}</target_{t_id}>")
                            i += len(group)
                        else:
                            annotated_parts.append(sent_dut[i])
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
                    "text": text_dut
                })

    return tasks, rows




def apply_rewrite_to_row(row: dict, task: dict, rewritten: str):
    if not row:
        return
    t_type = task["type"]
    model = task["model"]
    
    if t_type == "sentence":
        sent_idx = task["sent_idx"]
        num_sentences = len(row['sent_dut'])
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
        
    return rewritten


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
                original_sentences = row['sent_dut']
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

#TODO '\n \r? 
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"').replace("`", "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



#select models for differnet calcs
def get_models_list():
    CALC_MODEL_MAPPING = {
        'calc12': ['gemma4:26b'],
        'calc11': ['qwen3.6:27b'],
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
    source_name: str,
    table: pa.Table = None,
    selected_path: Path = None,
    checkpoint_path: Path = None,
    models_list: list[str] = None,
    percentages_to_run: list[int] = [25, 50, 75],
    port: int = 11435,
    gpu_id: str = None,
    debug_mode: bool = False,
    debug_pct_only: bool = False,
    debug_count: int = 5,
    exclude_percentage: bool = False,
    task_steps: List = None
):
    if selected_path is None:
        selected_path = BASE_DIR / 'data' / 'silver' / source_name / f"{source_name.lower()}_selected.parquet"
    else:
        selected_path = Path(selected_path)
        
    if checkpoint_path is None:
        checkpoint_path = selected_path.parent / f"checkpoint_rewrites_{source_name.lower()}.jsonl"
    else:
        checkpoint_path = Path(checkpoint_path)

    # Ensure parent directories exist before processing
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
    
    if table is not None:
        ug_table = table
    else:
        print(f"Loading Parquet data from: {selected_path}")
        if not selected_path.exists():
            print(f"Error: Parquet file not found at {selected_path}")
            sys.exit(1)
        ug_table = pq.read_table(selected_path)
            
    tasks, rows = prepare_tasks(ug_table, checkpoint_path, models_list, percentages_to_run)

    #TODO make subset based on args.tasks
    if debug_pct_only:
        print("\n[DEBUG PCT ONLY ACTIVE] Filtering out non-percentage tasks.")
        tasks = [t for t in tasks if t['type'] == 'percentage']
        debug_mode = True  # Force debug count verification below

    if exclude_percentage:
        print('EXCLUDING PERCENTAGE TASKS')
        tasks = [t for t in tasks if t['type'] != 'percentage']

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
        print("All specified tasks are already completed in the dataset. Exiting.")
        return
        
    run_generation(tasks, rows, system_prompts, checkpoint_path, debug_mode=debug_mode)


#CONSTRUCTION OF FINAL DATASET
#TODO remove but keep logic for loading checkpoint (also from func above), and uhh adding cp to df, modify for source and shit
def consolidate_checkpoints(silver_dir: Union[str, Path] = BASE_DIR / 'data' / 'silver') -> pd.DataFrame:
    """
    Traverses the silver directory, parses all source checkpoint JSONL files,
    and aggregates them into a single, structured Pandas DataFrame.
    """
    silver_path = Path(silver_dir)
    
    # 1. Discover all checkpoint files matching the pattern
    # Looks in the silver directory and any subfolders (e.g., silver/UG/)
    checkpoint_files = list(silver_path.glob("**/checkpoint_rewrites_*.jsonl"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {silver_path}")
        return pd.DataFrame()

    # We use a nested dictionary to aggregate tasks by (source, doc_id)
    # Key: (source, doc_id) -> Value: aggregated document record
    aggregated_data = {}
    
    # Keep track of all models and percentages discovered dynamically
    discovered_models = set()
    discovered_percentages = set()

    # 2. Parse checkpoint files
    for file_path in checkpoint_files:
        match = re.search(r"checkpoint_rewrites_([a-zA-Z0-9_-]+)\.jsonl$", file_path.name)
        if not match:
            continue
        source = match.group(1).upper()
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                doc_id = str(record.get("id"))
                t_type = record.get("type")
                model = record.get("model")
                rewritten = record.get("rewritten")
                orig_text = record.get("text")
                
                if not doc_id:
                    continue
                
                key = (source, doc_id)
                if key not in aggregated_data:
                    aggregated_data[key] = {
                        "id": doc_id,
                        "source": source,
                        "abstract": None,
                        "abstract_sent_dict": {},  # temp storage to order by index
                        "models_single_dict": {},  # {model: {sent_idx: rewritten}}
                        "models_pct": {},          # {(model, pct): rewritten}
                        "models_full": {}          # {model: rewritten}
                    }
                
                entry = aggregated_data[key]
                
                if model:
                    discovered_models.add(model)
                
                # --- Task parsing ---
                if t_type == "full_abstract":
                    if orig_text:
                        entry["abstract"] = orig_text
                    if model and rewritten:
                        entry["models_full"][model] = rewritten
                        
                elif t_type == "percentage":
                    pct = record.get("percentage")
                    if pct is not None:
                        discovered_percentages.add(pct)
                        if model and rewritten:
                            entry["models_pct"][(model, pct)] = rewritten
                            
                elif t_type == "sentence":
                    sent_idx = record.get("sent_idx")
                    if sent_idx is not None:
                        try:
                            idx = int(sent_idx)
                            if orig_text:
                                entry["abstract_sent_dict"][idx] = orig_text
                            if model and rewritten:
                                if model not in entry["models_single_dict"]:
                                    entry["models_single_dict"][model] = {}
                                entry["models_single_dict"][model][idx] = rewritten
                        except (ValueError, TypeError):
                            pass

    # 3. Reconstruct into flat rows for the final DataFrame
    flat_rows = []
    for (source, doc_id), entry in aggregated_data.items():
        row = {
            "id": entry["id"],
            "source": entry["source"],
            "abstract": entry["abstract"]
        }
        
        # Sort and construct the original sentence list
        sent_dict = entry["abstract_sent_dict"]
        if sent_dict:
            max_idx = max(sent_dict.keys())
            abstract_sent = [sent_dict.get(i) for i in range(max_idx + 1)]
        else:
            abstract_sent = []
        row["abstract_sent"] = abstract_sent
        
        # Build dynamic model columns
        for model in discovered_models:
            # {model}_single: Reconstruct list of rewritten sentences aligned with indices
            single_dict = entry["models_single_dict"].get(model, {})
            if single_dict or sent_dict:
                max_single_idx = max(list(single_dict.keys()) + list(sent_dict.keys()))
                model_single_list = [single_dict.get(i) for i in range(max_single_idx + 1)]
            else:
                model_single_list = []
            row[f"{model}_single"] = model_single_list
            
            # {model}_{pct}
            for pct in discovered_percentages:
                row[f"{model}_{pct}"] = entry["models_pct"].get((model, pct))
                
            # {model}_full
            row[f"{model}_full"] = entry["models_full"].get(model)
            
        flat_rows.append(row)
        
    return pd.DataFrame(flat_rows)



# script execution
#TODO clean up after remove download etc
def main():
    parser = argparse.ArgumentParser(description="Multi-source NLP pipeline orchestrator")
    parser.add_argument(
        '--source', 
        type=str, 
        default=['UG', 'HBO'], 
        choices=['UG', 'HBO', 'SB'], 
        help="Source dataset to process (UG: UGent, HBO: HBO Kennisbank, SB: Scriptiebank). Default: UG and HBO"
    )
    
    #TODO integrate into main
    parser.add_argument(
        '--tasks', 
        type=str, 
        nargs='+', 
        default=['sentence','full'], 
        choices=['sentence', 'percentage', 'full'], 
        help="Type of tasks to perform. Default: no percentage (sent, full)"
    )

    parser.add_argument('--debug', action='store_true', help="Run the LLM generation in debug mode (fewer tasks)")

    args = parser.parse_args()



    # 4. Generation Section
    #TODO question do we keep the ug cp and add to it or do we handle it seperately
    #checkpoint_path = selected_path.parent / f"checkpoint_rewrites_{source_name.lower()}.jsonl" #TODO make generic
    #checkpoint_path = BASE_DIR / 'data' / 'gold' / 'checkpoint_rewrites.jsonl'
    
    percentages = [25, 50, 75]
    


    if args.debug:
        print("[DEBUG CONFIG] Allocating complete multi-model test suite.")
        active_models_list = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']
    else:
        active_models_list = get_models_list()

    print(f'Starting LLM generation pipeline for source: {source_name}')
    generation_main(
        source_name=source_name,
        table=table, #TODO empty keyword lists will not be recognized as null but thats fine for now
        selected_path=selected_path, #TODO should always be gold merged
        checkpoint_path=checkpoint_path, #TODO create new generic non source included cp file and add seperate check list of paths of older cp
        models_list=active_models_list,
        percentages_to_run=percentages,
        debug_mode=args.debug,
        task_steps = args.tasks,
        exclude_percentage=False #TODO unuse
    )

    # GOLD DATAFRAME
    if not args.debug:
        gold_df_path = BASE_DIR / 'data' / 'gold' / 'gold.csv'
        pass
        #TODO save new gen data added csv and or pq to gold

if __name__ == "__main__":
    main()