import json
import re
import string
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import nltk
from langdetect import detect, LangDetectException

# --- CONFIGURATION (Paths on your local machine) ---
ug_path = Path('/home/gderijck/internship/data/silver/ug_selected.parquet')
local_checkpoint_path = Path('/home/gderijck/internship/data/silver/checkpoint_rewrites_LOCAL.jsonl')
server_checkpoint_path = Path('/home/gderijck/internship/data/silver/checkpoint_rewrites.jsonl')
final_checkpoint_path = Path('/home/gderijck/internship/data/silver/checkpoint_rewrites_final.jsonl') # Overwrites your local active file

# Backups for safety before overwriting
backup_local_path = Path('/home/gderijck/internship/data/silver/checkpoint_rewrites_LOCAL_BACKUP.jsonl')
backup_server_path = Path('/home/gderijck/internship/data/silver/checkpoint_rewrites_BACKUP.jsonl')

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


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


def run_consolidation_pipeline():
    print("====================================================")
    print("Starting Consolidated Migration Pipeline")
    print("====================================================\n")
    
    # --- STEP 1: PARSE NEW TASK STRUCTURE ---
    if not ug_path.exists():
        print(f"[ERROR] Parquet file not found at {ug_path}")
        return
    print(f"Reading Parquet file: {ug_path}")
    table = pq.read_table(ug_path)
    new_tasks = prepare_tasks(table)
    print(f"Generated {len(new_tasks)} clean tasks from Parquet.\n")

    # --- STEP 2: BACK UP INPUT FILES FOR SAFETY ---
    if local_checkpoint_path.exists():
        print(f"Backing up local checkpoint to: {backup_local_path}")
        if backup_local_path.exists():
            backup_local_path.unlink()
        local_checkpoint_path.rename(backup_local_path)
        
    if server_checkpoint_path.exists():
        print(f"Backing up server checkpoint to: {backup_server_path}")
        if backup_server_path.exists():
            backup_server_path.unlink()
        server_checkpoint_path.rename(backup_server_path)
    print("Backups completed successfully.\n")

    # --- STEP 3: CONSOLIDATE SERVER CHECKPOINT (Merge multi-line duplicates) ---
    master_rewrites = {}
    
    if backup_server_path.exists():
        print(f"Consolidating server checkpoints from {backup_server_path}...")
        server_lines_parsed = 0
        with open(backup_server_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    t_id = record['id']
                    text = record.get('text')
                    
                    if text:
                        # Extract only model key-value pairs from server file
                        model_data = {
                            k: v for k, v in record.items() 
                            if k not in ('id', 'sent_idx', 'text')
                        }
                        
                        # Initialize if first time seeing this sentence, then update/merge
                        if (t_id, text) not in master_rewrites:
                            master_rewrites[(t_id, text)] = {}
                        master_rewrites[(t_id, text)].update(model_data)
                        server_lines_parsed += 1
                        
        print(f"Parsed {server_lines_parsed} lines from server checkpoint. Consolidated into {len(master_rewrites)} unique sentences.\n")
    else:
        print("[INFO] No server checkpoint file found. Skipping server merge.\n")

    # --- STEP 4: MERGE LOCAL CHECKPOINT INTO MASTER (ONLY gemma4:26b) ---
    if backup_local_path.exists():
        print(f"Merging local checkpoints from {backup_local_path} into master (only importing 'gemma4:26b')...")
        local_lines_parsed = 0
        with open(backup_local_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    t_id = record['id']
                    text = record.get('text')
                    
                    if text:
                        # FIXED: Extract ONLY the 'gemma4:26b' key-value pair from the local file
                        model_data = {}
                        if 'gemma4:26b' in record:
                            model_data['gemma4:26b'] = record['gemma4:26b']
                        
                        # Only merge if this record actually contains Gemma 26B data
                        if model_data:
                            if (t_id, text) not in master_rewrites:
                                master_rewrites[(t_id, text)] = {}
                            master_rewrites[(t_id, text)].update(model_data)
                            local_lines_parsed += 1
                        
        print(f"Parsed {local_lines_parsed} lines from local checkpoint containing 'gemma4:26b' and merged them with master.\n")
    else:
        print("[INFO] No local checkpoint file found. Skipping local merge.\n")

    # --- STEP 5: RESTRUCTURE AND EXPORT TO FINAL CHECKPOINT ---
    print(f"Writing final consolidated checkpoint to: {final_checkpoint_path}")
    preserved_count = 0
    discarded_count = 0
    
    with open(final_checkpoint_path, 'w', encoding='utf-8') as f:
        for task in new_tasks:
            t_id = task['id']
            text = task['text']
            s_idx = task['sent_idx']
            
            # Match strictly by original text identity
            if (t_id, text) in master_rewrites:
                # Construct a new clean record with the corrected sent_idx
                # and all model key-value pairs from both server and local runs
                new_record = {
                    "id": t_id,
                    "sent_idx": s_idx,
                    "text": text,
                    **master_rewrites[(t_id, text)]
                }
                f.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                preserved_count += 1
            else:
                # Any sentence that has changed, been merged, or was dropped
                # is skipped here, so your generator can handle it cleanly later.
                discarded_count += 1
                
    print("\n====================================================")
    print("Pipeline Execution Complete!")
    print(f"Total Unique Sentences Preserved: {preserved_count}")
    print(f"Total Sentences Dropped/Impacted (To Be Redone): {discarded_count}")
    print(f"Final output saved to: {final_checkpoint_path}")
    print("====================================================")


if __name__ == "__main__":
    run_consolidation_pipeline()