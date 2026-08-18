import os
import sys
import json
import time
import socket
import random
import re
import binascii
import unicodedata
import argparse
from pathlib import Path
from typing import List
from collections import defaultdict
from ollama import Client
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import atexit

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
# after generation add to checkpoint.jsonl
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


# MODIFIED: Redesigned validator for continuous segment percentage rewriting and context leakage prevention
def validate_percentage_rewrite(
    original_sents: list[str], 
    candidate_rewrite: str, 
    start_idx: int,
    end_idx: int
) -> tuple[bool, str, str | None]:
    if not candidate_rewrite or not candidate_rewrite.strip():
        return False, "Empty response from the model", None

    candidate_clean = candidate_rewrite.strip()
    target_block = " ".join([original_sents[i].strip() for i in range(start_idx, end_idx)]).strip()

    # 1. Verify changes were made compared to the normalized original block
    if normalize_text(candidate_clean) == normalize_text(target_block):
        return False, "Target segment was not modified by the model.", None

    # 2. Check for context leakage (untargeted surrounding sentences included in output)
    prefix_sents = original_sents[:start_idx]
    suffix_sents = original_sents[end_idx:]
    norm_candidate = normalize_text(candidate_clean)

    for prefix in prefix_sents:
        clean_pref = normalize_text(prefix)
        if len(clean_pref) >= 15 and clean_pref in norm_candidate:
            return False, f"Output leaked untargeted preceding sentence: '{prefix[:30]}...'", None

    for suffix in suffix_sents:
        clean_suff = normalize_text(suffix)
        if len(clean_suff) >= 15 and clean_suff in norm_candidate:
            return False, f"Output leaked untargeted succeeding sentence: '{suffix[:30]}...'", None

    # 3. Check for full-abstract rewrite over-generation when only a partial segment was targeted
    total_sentences = len(original_sents)
    targeted_sentences = end_idx - start_idx
    if targeted_sentences < total_sentences:
        full_abstract_len = len(" ".join(original_sents))
        if len(candidate_clean) > 0.85 * full_abstract_len and len(candidate_clean) > 2.0 * len(target_block):
            return False, "Output appears to rewrite the entire abstract instead of just the target segment.", None

    # ADDED: Stitch original untargeted sentences and new rewritten segment into final abstract
    stitched_sentences = list(original_sents[:start_idx]) + [candidate_clean] + list(original_sents[end_idx:])
    final_abstract = " ".join([sent.strip() for sent in stitched_sentences if sent.strip()])
    return True, "Success", final_abstract


ALL_MODELS = ['gemma4:e4b', 'qwen3.5:4b', 'qwen3.6:27b', 'gemma4:26b']

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
                        
                        if t_type == "full_abstract":
                            full_abstract_match_count += 1
                        elif t_type == "percentage":
                            percentage_match_count += 1

                except (json.JSONDecodeError, ValueError, TypeError):
                    corrupted_count += 1
                    continue

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


    # CREATE TASKS FOR REMAINING
    tasks = []
    for row in rows:
        row_id = str(row.get(id_key))
        abstract = row.get('abstract')
        abstract_sentence = row.get('abstract_sentence')

        if not isinstance(abstract, str) or not abstract.strip() or not isinstance(abstract_sentence, list) or not abstract_sentence:
            continue

        num_sentences = len(abstract_sentence)

        # 1. Sentence tasks (if sentence tasks are enabled)
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
                        "context": abstract
                    })

        # 2. Percentage tasks: DETERMINISTICALLY ASSIGN EXACTLY 1 MODEL PER (ROW, PCT)
        for pct in percentages:
            seed_str = f"{row_id}_{pct}".encode("utf-8")
            seed = binascii.crc32(seed_str)
            
            # Pick exactly ONE model deterministically based on seed
            assigned_model = ALL_MODELS[seed % len(ALL_MODELS)]
            col_name = f"{assigned_model}_{pct}"

            if col_name not in row or not row[col_name]:
                row[col_name] = None
                rng = random.Random(seed)
                
                num_to_tag = max(1, round(num_sentences * (pct / 100.0)))
                num_to_tag = min(num_to_tag, num_sentences)
                max_start_idx = num_sentences - num_to_tag
                start_idx = rng.randint(0, max_start_idx)
                end_idx = start_idx + num_to_tag

                target_segment = " ".join([abstract_sentence[i].strip() for i in range(start_idx, end_idx)])

                tasks.append({
                    "id": row_id,
                    "type": "percentage",
                    "model": assigned_model, # Only create task for assigned model
                    "percentage": pct,
                    "text": target_segment,
                    "full_context": abstract,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })

        # 3. Full abstract tasks (if full tasks are enabled)
        for model in models_list:
            col_name = f"{model}_full"
            if col_name not in row or not row[col_name]:
                row[col_name] = None
                tasks.append({
                    "id": row_id,
                    "type": "full_abstract",
                    "model": model,
                    "text": abstract
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
                # MODIFIED: Format user prompt dynamically for percentage rewrite vs standard sentence/full abstract
                if t_type == "percentage":
                    prompt_input = (
                        f"FULL ABSTRACT (FOR CONTEXT ONLY):\n{task['full_context']}\n\n"
                        f"SEGMENT TO REWRITE:\n{task['text']}"
                    )
                else:
                    prompt_input = text

                candidate_rewrite = rewrite_sentence(
                    client, 
                    model, 
                    system_prompt, 
                    prompt_input, 
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
                start_idx = task['start_idx']
                end_idx = task['end_idx']
                
                try:
                    # MODIFIED: Updated validator arguments for bounds-based validation
                    is_valid, reason, stitched_text = validate_percentage_rewrite(
                        original_sentences, 
                        candidate_rewrite,  
                        start_idx,
                        end_idx
                    )
                except Exception as eval_err:
                    is_valid = False
                    reason = f"Internal evaluation parser raised an unhandled exception: {eval_err}"
                    stitched_text = None

                if is_valid:
                    rewritten = stitched_text
                    print(f'ORIGINAL SEGMENT ({start_idx}:{end_idx}): {text}\n')
                    print(f'REWRITTEN STITCHED ABSTRACT: {rewritten}')
                    break
                else:
                    print(f"  [Warning - Attempt {attempt+1}/{max_attempts} Failed] {reason}.")
                    print(f"  [DEBUG - Prompt Sent to LLM]:\n{prompt_input}")
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
    
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = " ".join(text.split())
    
    return text


def get_models_list():
    CALC_MODEL_MAPPING = {
        'calc12': ['gemma4:e4b', 'qwen3.5:4b'],
        'calc11': ['gemma4:e4b', 'qwen3.5:4b', 'qwen3.6:27b', 'gemma4:26b'],
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


def generate_priority(tasks: list[dict], rows: list[dict], models_list: list[str]) -> list[dict]:
    if not tasks:
        return []

    # Filter tasks assigned to models active on this host
    host_tasks = [t for t in tasks if t["model"] in models_list]

    model_order = {model: idx for idx, model in enumerate(models_list)}

    def get_column_rank(task: dict) -> int:
        t_type = task.get("type")
        pct = task.get("percentage")
        if t_type == "percentage":
            if pct == 25:
                return 1
            elif pct == 50:
                return 2
            elif pct == 75:
                return 3
            return 4
        elif t_type == "sentence":
            return 5
        elif t_type == "full_abstract":
            return 6
        return 99

    ordered_tasks = sorted(
        host_tasks,
        key=lambda t: (
            model_order.get(t["model"], 999),
            get_column_rank(t),
            str(t.get("id")),
            t.get("sent_idx") if t.get("sent_idx") is not None else -1
        )
    )

    print(f"\n[DETERMINISTIC BALANCED PRIORITY] Re-ordered {len(ordered_tasks)} tasks on host:")
    seen_groups = []
    for t in ordered_tasks:
        group_key = f"{t['model']} (Type: {t['type']}, Pct: {t.get('percentage')})"
        if not seen_groups or seen_groups[-1][0] != group_key:
            seen_groups.append([group_key, 1])
        else:
            seen_groups[-1][1] += 1

    for group, count in seen_groups:
        print(f"  --> {group}: {count} tasks")

    return ordered_tasks


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
) -> list[dict]:
    # MODIFIED: Updated system prompt instructions for percentage rewrites
    system_prompts = {
        "sentence": (
            "You are a professional Dutch editor.\n"
            "Rewrite the following sentence to make it better while preserving all crucial information.\n"
            "Provide ONLY the rewritten sentence."
        ),
        "percentage": (
            "You are a professional Dutch editor.\n"
            "Your task is to rewrite ONLY the target segment of the abstract provided to improve its phrasing, flow, and quality.\n"
            "You are provided with the full abstract for contextual background only.\n"
            "CRITICAL: Output ONLY the rewritten target segment. Do NOT output any other sentences, conversational filler, or tags."
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
    
    tasks, rows = prepare_tasks(table, CHECKPOINT_PATH, models_list, percentages_to_run)

    if task_steps:
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
        debug_mode = True

    if exclude_percentage:
        print('EXCLUDING PERCENTAGE TASKS')
        tasks = [t for t in tasks if t['type'] != 'percentage']

    if priority:
        print("\n[PRIORITY MODE ACTIVE] Re-ordering tasks to dynamically rebalance dataset and minimize model swaps...")
        tasks = generate_priority(tasks, rows, models_list)

    if not models_list:
        models_list = list(dict.fromkeys([t["model"] for t in tasks]))
        
    if not priority:
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
        
    updated_rows = run_generation(tasks, rows, system_prompts, CHECKPOINT_PATH, debug_mode=debug_mode)
    return updated_rows


def main():
    parser = argparse.ArgumentParser(description="Multi-source NLP pipeline orchestrator")
    parser.add_argument(
        '--source', 
        type=str,
        nargs='+', 
        default=['UG', 'HBO'], 
        choices=['UG', 'HBO', 'SB'], 
        help="Source dataset to process."
    )
    parser.add_argument(
        '--priority',
        action='store_true',
        help='Prioritize rewrites using deficit balancing, grouped into continuous model blocks.'
    )
    parser.add_argument(
        '--tasks', 
        type=str, 
        nargs='+', 
        default=['sentence','full'], 
        choices=['sentence', 'percentage', 'full'], 
        help="Type of tasks to perform."
    )
    parser.add_argument('--debug', action='store_true', help="Run the LLM generation in debug mode (fewer tasks)")
    parser.add_argument(
        '--format', 
        type=str, 
        default='both', 
        choices=['csv', 'parquet', 'both'], 
        help="Output save format."
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()