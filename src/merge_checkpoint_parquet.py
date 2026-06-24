import json
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# --- CONFIGURATION (Paths on your local Windows machine) ---
data_dir = Path('/home/gderijck/internship/data/silver')
ug_path = data_dir / 'ug_selected.parquet'
parquet_backup_path = data_dir / 'ug_selected_BACKUP.parquet'

# Explicitly define the two target files to prevent any sorting or globbing errors
consolidated_path = data_dir / 'checkpoint_rewrites_final.jsonl'
active_path = data_dir / 'checkpoint_rewrites.jsonl'


def main():
    print("====================================================")
    print("Starting Checkpoint-to-Parquet Consolidation Pipeline")
    print("====================================================\n")

    # 1. Verify existence of all required files
    if not ug_path.exists():
        print(f"[ERROR] Parquet file not found at {ug_path}")
        return
    if not consolidated_path.exists():
        print(f"[ERROR] Consolidated file not found at {consolidated_path}")
        return
    if not active_path.exists():
        print(f"[ERROR] Active file not found at {active_path}")
        return

    print("Targeting Files:")
    print(f" -> File A (Consolidated Master): {consolidated_path.name}")
    print(f" -> File B (Active Flat Run): {active_path.name}\n")

    # 2. Back up your Parquet file before making modifications for absolute safety
    print(f"Creating a safety backup of your Parquet file to: {parquet_backup_path.name}")
    shutil.copy(ug_path, parquet_backup_path)
    print("Backup completed successfully.\n")

    # 3. Read and merge both JSONL files chronologically
    master_cache = {}  # Key: (id_str, original_text_stripped) -> Value: {model_single: rewritten_text}
    
    # We read consolidated first, then active, so active overwrites/updates duplicates
    files_to_read = [consolidated_path, active_path]
    
    for file_path in files_to_read:
        print(f"Parsing and loading data from: {file_path.name}")
        lines_parsed = 0
        corrupt_lines_count = 0
        line_num = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line.strip():
                    try:
                        record = json.loads(line)
                        
                        # Fetch the ID under 'id' or '_id', fallback to None
                        raw_id = record.get('id') if 'id' in record else record.get('_id')
                        text = record.get('text')
                        
                        if raw_id is not None and text:
                            # CRITICAL FIX: Force ID to a stripped string to prevent string-vs-int mismatches [5]
                            t_id = str(raw_id).strip()
                            clean_text = text.strip()
                            
                            # Extract all model keys
                            model_data = {}
                            for k, v in record.items():
                                if k not in ('id', '_id', 'sent_idx', 'text') and v is not None:
                                    # Map {model} -> {model}_single
                                    target_col = f"{k}_single" if not k.endswith("_single") else k
                                    
                                    # DATA RESCUE: Extract only the actual rewrite if <channel|> is in the text
                                    cleaned_v = v
                                    if isinstance(v, str) and "<channel|>" in v:
                                        cleaned_v = v.split("<channel|>")[-1].strip()
                                    
                                    model_data[target_col] = cleaned_v
                            
                            key = (t_id, clean_text)
                            if key not in master_cache:
                                master_cache[key] = {}
                            # Update (newer files/entries will overwrite older duplicates automatically)
                            master_cache[key].update(model_data)
                            lines_parsed += 1
                    except json.JSONDecodeError as e:
                        corrupt_lines_count += 1
                        print(f" [Warning] Skipped corrupt line {line_num} in {file_path.name}: {e}")
                        print(f"   Line Content: {repr(line[:100])}...")
                        continue
        print(f" -> Successfully loaded {lines_parsed} records from {file_path.name} (Skipped {corrupt_lines_count} corrupt lines).\n")

    print(f"Total unique sentence completions compiled in memory: {len(master_cache)}\n")

    # 4. Load your Parquet file and map the flat sentences back to their correct lists
    print(f"Reading Parquet file: {ug_path}")
    table = pq.read_table(ug_path)
    rows = table.to_pylist()

    # Determine what ID column is used inside the Parquet table
    id_key = '_id' if '_id' in table.column_names else 'id'
    discovered_models = set()
    mapped_count = 0

    for row in rows:
        raw_row_id = row.get(id_key)
        sent_dut = row.get('sent_dut')
        
        # Skip rows that don't have valid text or sentence lists
        if raw_row_id is None or not isinstance(sent_dut, list) or not sent_dut:
            continue
            
        # CRITICAL FIX: Normalize Parquet row ID to a stripped string [5]
        row_id = str(raw_row_id).strip()
        num_sentences = len(sent_dut)
        
        # Loop through each sentence in this abstract's tokenized list
        for sent_idx, sentence in enumerate(sent_dut):
            clean_sentence = sentence.strip()
            key = (row_id, clean_sentence)
            
            # If we have a completed rewrite for this exact sentence text
            if key in master_cache:
                for model_single, rewritten in master_cache[key].items():
                    discovered_models.add(model_single)
                    
                    # Ensure the list-column exists and is initialized
                    if model_single not in row or not isinstance(row[model_single], list) or len(row[model_single]) != num_sentences:
                        row[model_single] = [None] * num_sentences
                        
                    # Insert the rewritten sentence into the correct index slot
                    row[model_single][sent_idx] = rewritten
                    mapped_count += 1

    # --- 5. AUTOMATIC DIAGNOSTIC CHECK IF MAPPED COUNT IS 0 ---
    if mapped_count == 0 and len(master_cache) > 0:
        print("\n[DIAGNOSTIC] Mapped 0 rows. Let's compare keys to find the mismatch:")
        print("-" * 60)
        print("First 3 keys in your checkpoint cache:")
        for k in list(master_cache.keys())[:3]:
            print(f"  ID: {repr(k[0])} (Type: {type(k[0])}) | Text snippet: {repr(k[1][:50])}...")
        
        print("\nFirst 3 keys in your Parquet file:")
        sample_printed = 0
        for row in rows:
            r_id = row.get(id_key)
            s_list = row.get('sent_dut')
            if r_id is not None and s_list:
                # Show types to diagnose string-vs-integer mismatches
                print(f"  ID: {repr(r_id)} (Type: {type(r_id)}) | Text snippet: {repr(s_list[0][:50])}...")
                sample_printed += 1
                if sample_printed >= 3:
                    break
        print("-" * 60)

    # 6. Save the final processed rows list back into the Parquet file
    print(f"\nMapped {mapped_count} sentence completions back to their correct abstract indexes.")
    print(f"Identified models to save: {list(discovered_models)}")
    
    print(f"Overwriting original Parquet with updated columns: {ug_path}")
    output_table = pa.Table.from_pylist(rows)
    pq.write_table(output_table, ug_path)
    
    print("\n====================================================")
    print("Consolidation Pipeline Completed Successfully!")
    print(f"Your Parquet file has been updated and is ready.")
    print("====================================================")


if __name__ == "__main__":
    main()