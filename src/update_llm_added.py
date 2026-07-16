import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Define paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_TABLE = BASE_DIR / 'data' / 'gold' / 'merged_publications.parquet'
CHECKPOINT_FILE = BASE_DIR / 'data' / 'gold' / 'checkpoint_rewrites.jsonl'
OUTPUT_PARQUET = BASE_DIR / 'data' / 'gold' / 'llm_added.parquet'
OUTPUT_CSV = BASE_DIR / 'data' / 'gold' / 'llm_added.csv'


def safe_literal_eval(val):
    """Safely converts string representations of lists/dicts back to Python objects."""
    if pd.isna(val) or not isinstance(val, str):
        return val
    val_stripped = val.strip()
    if (val_stripped.startswith('[') and val_stripped.endswith(']')) or \
       (val_stripped.startswith('{') and val_stripped.endswith('}')):
        try:
            return ast.literal_eval(val_stripped)
        except (ValueError, SyntaxError):
            try:
                return json.loads(val_stripped)
            except json.JSONDecodeError:
                return val
    return val


def is_already_applied(row: dict, task: dict) -> bool:
    """Checks if the rewrite in the task has already been applied to the row."""
    t_type = task["type"]
    model = task["model"]
    
    if t_type == "sentence":
        sent_idx = task["sent_idx"]
        col_name = f'{model}_single'
        if col_name not in row:
            return False
        
        val_list = row[col_name]
        if isinstance(val_list, str):
            val_list = safe_literal_eval(val_list)
            
        if not isinstance(val_list, list) or sent_idx >= len(val_list):
            return False
            
        val = val_list[sent_idx]
        return not pd.isna(val) and val is not None and val != ""
        
    elif t_type == "percentage":
        pct = task["percentage"]
        col_name = f"{model}_{pct}"
        if col_name not in row:
            return False
        val = row[col_name]
        return not pd.isna(val) and val is not None and val != ""
        
    elif t_type == "full_abstract":
        col_name = f"{model}_full"
        if col_name not in row:
            return False
        val = row[col_name]
        return not pd.isna(val) and val is not None and val != ""
        
    return False


def apply_rewrite_to_row(row: dict, task: dict, rewritten: str):
    """Applies the rewrite to the record dictionary safely."""
    if not row:
        return
    t_type = task["type"]
    model = task["model"]
    
    if t_type == "sentence":
        sent_idx = task["sent_idx"]
        
        # Ensure abstract_sentence is parsed as a list
        abstract_sentence = row.get('abstract_sentence')
        if isinstance(abstract_sentence, str):
            abstract_sentence = safe_literal_eval(abstract_sentence)
        if not isinstance(abstract_sentence, list):
            abstract_sentence = []
            
        num_sentences = len(abstract_sentence)
        col_name = f'{model}_single'
        
        # Resolve target model column representation
        val = row.get(col_name)
        if isinstance(val, str):
            val = safe_literal_eval(val)
            
        if not isinstance(val, list):
            if hasattr(val, 'tolist'):
                val = val.tolist()
            else:
                val = [None] * num_sentences
        
        # Align list length with num_sentences
        if len(val) < num_sentences:
            val = val + [None] * (num_sentences - len(val))
        elif len(val) > num_sentences:
            val = val[:num_sentences]
            
        if 0 <= sent_idx < num_sentences:
            val[sent_idx] = rewritten
            
        row[col_name] = val
        
    elif t_type == "percentage":
        pct = task["percentage"]
        row[f"{model}_{pct}"] = rewritten
        
    elif t_type == "full_abstract":
        row[f"{model}_full"] = rewritten


def clean_id(val) -> str:
    """Standardizes row IDs to string format for reliable mapping."""
    if pd.isna(val) or val is None:
        return ""
    if isinstance(val, float):
        return str(int(val))
    return str(val).strip()


def main():
    parser = argparse.ArgumentParser(description="Update dataset with checkpoint rewrites.")
    parser.add_argument("--input", type=Path, default=INPUT_TABLE, help="Path to base input table.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_FILE, help="Path to checkpoints jsonl file.")
    parser.add_argument("--output-parquet", type=Path, default=OUTPUT_PARQUET, help="Path to output Parquet file.")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="Path to output CSV file.")
    args = parser.parse_args()

    # 1. Load the most current version of the dataset
    if args.output_parquet.exists():
        print(f"Loading existing Parquet dataset: {args.output_parquet}")
        df = pd.read_parquet(args.output_parquet)
    elif args.output_csv.exists():
        print(f"Loading existing CSV dataset: {args.output_csv}")
        df = pd.read_csv(args.output_csv)
    else:
        print(f"No existing output found. Initializing from base table: {args.input}")
        df = pd.read_parquet(args.input)

    # 2. Parse any serialized list structures in existing columns
    list_cols = ['abstract_sentence', 'keywords']
    for col in df.columns:
        if col.endswith('_single') or col in list_cols:
            df[col] = df[col].apply(safe_literal_eval)

    # 3. Read the checkpoint task records
    tasks = []
    if args.checkpoint.exists():
        print(f"Reading checkpoint file: {args.checkpoint}")
        with open(args.checkpoint, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tasks.append(json.loads(line))
                    except Exception as e:
                        print(f"Skipping malformed JSON line: {line}. Error: {e}")
    else:
        print(f"Checkpoint file not found: {args.checkpoint}")
        return

    # Convert dataframe to records for quicker modifications
    records = df.to_dict('records')
    record_map = {clean_id(r.get('_id')): r for r in records if r.get('_id') is not None}

    # Track distinct models in the workspace to construct missing columns later
    models = set()
    for col in df.columns:
        if col.endswith('_single'):
            models.add(col[:-7])
        elif col.endswith('_full'):
            models.add(col[:-5])
        elif any(col.endswith(f'_{p}') for p in [25, 50, 75]):
            parts = col.split('_')
            models.add('_'.join(parts[:-1]))

    # Add any models introduced by checkpoints
    for task in tasks:
        if 'model' in task:
            models.add(task['model'])

    # 4. Perform updates
    updates_count = 0
    skipped_count = 0
    not_found_count = 0

    for task in tasks:
        task_id = clean_id(task.get("id"))
        if not task_id or task_id not in record_map:
            not_found_count += 1
            continue
            
        row = record_map[task_id]
        rewritten = task.get("rewritten")
        
        # Skip if the rewrite target cell is already populated
        if is_already_applied(row, task):
            skipped_count += 1
            continue
            
        apply_rewrite_to_row(row, task, rewritten)
        updates_count += 1

    print(f"Processed checkpoint file: {updates_count} updates applied, "
          f"{skipped_count} skipped (already applied), {not_found_count} skipped (IDs not in dataset).")

    # 5. Convert records back to DataFrame
    df_updated = pd.DataFrame(records)

    # 6. Initialize missing model columns explicitly (single, full, percentages)
    for model in sorted(models):
        col_single = f"{model}_single"
        if col_single not in df_updated.columns:
            df_updated[col_single] = df_updated.apply(
                lambda r: [None] * len(r['abstract_sentence']) if isinstance(r.get('abstract_sentence'), list) else [], 
                axis=1
            )
            
        col_full = f"{model}_full"
        if col_full not in df_updated.columns:
            df_updated[col_full] = None
            
        for pct in [25, 50, 75]:
            col_pct = f"{model}_{pct}"
            if col_pct not in df_updated.columns:
                df_updated[col_pct] = None

    # Re-order columns so original order is preserved and new model columns are sorted at the end
    original_cols = [c for c in df.columns if c in df_updated.columns]
    new_cols = [c for c in df_updated.columns if c not in df.columns]
    df_updated = df_updated[original_cols + sorted(new_cols)]

    # 7. Write back to Parquet and CSV
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving updated dataset to {args.output_parquet}...")
    df_updated.to_parquet(args.output_parquet, index=False)
    
    print(f"Saving updated dataset to {args.output_csv}...")
    df_updated.to_csv(args.output_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    main()