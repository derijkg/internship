import json
from pathlib import Path

# --- CONFIGURATION (Adjust filenames as needed) ---
consolidated_path = Path('/home/gderijck/internship/data/silver/checkpoint_rewrites_final.jsonl')
active_path = Path('/home/gderijck/internship/data/silver/checkpoint_rewrites.jsonl')  # Your running, append-only file


def sync_checkpoints(consolidated_file: Path, active_file: Path):
    if not consolidated_file.exists():
        print(f"[ERROR] Consolidated master file not found at {consolidated_file}")
        return
    if not active_file.exists():
        print(f"[ERROR] Active generation file not found at {active_file}")
        return

    # 1. Load the existing master consolidated data (preserving insertion order)
    print(f"Reading consolidated master file: {consolidated_file}")
    consolidated_data = {}
    order = []
    
    with open(consolidated_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                key = (record['id'], record['text'])
                consolidated_data[key] = record
                order.append(key)

    print(f"Loaded {len(consolidated_data)} unique sentences from the master file.")

    # 2. Parse the active file and merge new, missing generations
    print(f"Reading and reconciling updates from: {active_file}")
    added_keys_count = 0
    new_sentences_count = 0
    
    with open(active_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                t_id = record['id']
                text = record.get('text')
                
                if text:
                    # Extract only the model completions
                    model_data = {
                        k: v for k, v in record.items() 
                        if k not in ('id', 'sent_idx', 'text')
                    }
                    
                    key = (t_id, text)
                    if key in consolidated_data:
                        # Reconcile: Add model keys ONLY if they do not already exist in the master file
                        for model, rewritten in model_data.items():
                            if model not in consolidated_data[key]:
                                consolidated_data[key][model] = rewritten
                                added_keys_count += 1
                    else:
                        # Handle case where a completely new sentence is found
                        new_record = {
                            "id": t_id,
                            "sent_idx": record.get('sent_idx', 0),
                            "text": text,
                            **model_data
                        }
                        consolidated_data[key] = new_record
                        order.append(key)
                        new_sentences_count += 1

    print(f"Successfully merged {added_keys_count} missing model-rewrite pairs into master.")
    if new_sentences_count > 0:
        print(f"Added {new_sentences_count} completely new sentences to the master file.")

    # 3. Overwrite the master file with the fully updated, clean data
    print(f"Saving updated master checkpoint to: {consolidated_file}")
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        for key in order:
            f.write(json.dumps(consolidated_data[key], ensure_ascii=False) + '\n')
            
    print("Synchronization pipeline complete!")


if __name__ == "__main__":
    sync_checkpoints(consolidated_path, active_path)