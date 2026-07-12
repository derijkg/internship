import os
import re
import json
import ast
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import unicodedata
from tqdm import tqdm

# Set base directory (same structure as your main script)
BASE_DIR = Path(__file__).resolve().parent.parent

# Define file paths to scan and retrofit
SILVER_DIR = BASE_DIR / 'data' / 'silver'
GOLD_DIR = BASE_DIR / 'data' / 'gold'

FILES_CONFIG = {
    # --- Silver Selected Files ---
    'UG_selected_pq': {
        'path': SILVER_DIR / 'UG' / 'ug_selected.parquet',
        'type': 'parquet',
        'cols_to_normalize': {'text_dut': 'scalar', 'sent_dut': 'silver_list'}
    },
    'UG_selected_csv': {
        'path': SILVER_DIR / 'UG' / 'ug_selected.csv',
        'type': 'csv',
        'cols_to_normalize': {'text_dut': 'scalar', 'sent_dut': 'silver_list'}
    },
    'HBO_selected_pq': {
        'path': SILVER_DIR / 'HBO' / 'hbo_selected.parquet',
        'type': 'parquet',
        'cols_to_normalize': {'text_dut': 'scalar', 'sent_dut': 'silver_list'}
    },
    'HBO_selected_csv': {
        'path': SILVER_DIR / 'HBO' / 'hbo_selected.csv',
        'type': 'csv',
        'cols_to_normalize': {'text_dut': 'scalar', 'sent_dut': 'silver_list'}
    },
    # --- Gold Merged Files ---
    'Gold_merged_pq': {
        'path': GOLD_DIR / 'merged_publications.parquet',
        'type': 'parquet',
        'cols_to_normalize': {'abstract': 'scalar', 'abstract_sentence': 'gold_list'}
    },
    'Gold_merged_csv': {
        'path': GOLD_DIR / 'merged_publications.csv',
        'type': 'csv',
        'cols_to_normalize': {'abstract': 'scalar', 'abstract_sentence': 'gold_list'}
    }
}

CHECKPOINT_FILE = GOLD_DIR / 'checkpoint_rewrites.jsonl'


# =====================================================================
# THE EXACT NORMALIZATION LOGIC
# =====================================================================
def normalize_text(text: str) -> str:
    """
    Applies NFKC normalization, standardizes smart punctuation (quotes/dashes),
    and collapses duplicate whitespaces or hidden layout breaks.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. NFKC normalization (standardizes ligatures, accents, and symbol variants)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Convert curly/smart punctuation and non-standard dashes to standard ASCII
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    
    # 3. Collapse multiple whitespaces and convert carriage returns/tabs into standard spaces
    text = " ".join(text.split())
    
    return text


def clean_list_element(val, col_type: str):
    """
    Normalizes scalar text, standard lists, or stringified list structures 
    depending on the storage format (CSV vs Parquet).
    """
    if val is None:
        return val

    # 1. Intercept array/list-like types first to prevent scalar checks (like pd.isna) from throwing ValueErrors
    if isinstance(val, (list, np.ndarray, tuple)):
        return [normalize_text(x) for x in val]
        
    # 2. Now that we know it is a scalar, we can safely run pd.isna()
    if pd.isna(val):
        return val

    # Case A: Standard scalar string column
    if col_type == 'scalar':
        if isinstance(val, str):
            return normalize_text(val)
        return val

    # Case B: Stringified representations inside CSV files
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return val
        
        if col_type == 'gold_list':
            # Gold CSV strings are joined by " | "
            parts = val_str.split(' | ')
            return ' | '.join([normalize_text(x) for x in parts])
            
        elif col_type == 'silver_list':
            # Silver CSV strings are python list literal strings e.g. "['Zin 1', 'Zin 2']"
            if val_str.startswith('[') and val_str.endswith(']'):
                try:
                    parsed = ast.literal_eval(val_str)
                    if isinstance(parsed, list):
                        return str([normalize_text(x) for x in parsed])
                except Exception:
                    pass
            # Fallback to general split/normalization if list evaluation fails
            return normalize_text(val_str)

    return val

def values_are_different(before, after) -> bool:
    """Helper to detect if normalization altered the string or structural values."""
    # Safe check for null/NaN values on both sides (prevents float('nan') != float('nan') false-positives)
    before_null = before is None or (not isinstance(before, (list, np.ndarray, tuple)) and pd.isna(before))
    after_null = after is None or (not isinstance(after, (list, np.ndarray, tuple)) and pd.isna(after))
    
    if before_null and after_null:
        return False
    if before_null != after_null:
        return True
        
    if type(before) != type(after):
        return True
        
    if isinstance(before, (list, np.ndarray, tuple)):
        if len(before) != len(after):
            return True
        return any(b != a for b, a in zip(before, after))
        
    return before != after

# =====================================================================
# PROCESSING PIPELINE
# =====================================================================
def process_dataframe_file(file_key: str, config: dict) -> dict:
    file_path = config['path']
    file_type = config['type']
    cols_to_normalize = config['cols_to_normalize']
    
    report = {
        'file_key': file_key,
        'path': str(file_path.relative_to(BASE_DIR) if BASE_DIR in file_path.parents else file_path),
        'status': 'Skipped (File Not Found)',
        'rows_processed': 0,
        'changes_detected': 0
    }
    
    if not file_path.exists():
        return report
    
    # 1. Load File
    if file_type == 'parquet':
        df = pd.read_parquet(file_path)
    else:  # CSV
        df = pd.read_csv(file_path)
        
    report['status'] = 'Processing'
    report['rows_processed'] = len(df)
    
    changes = 0
    
    # 2. Apply Normalization and track changes
    for col, col_type in cols_to_normalize.items():
        if col not in df.columns:
            print(f'didnt find {col}')
            continue
            
        original_col_data = df[col].copy()
        df[col] = df[col].apply(lambda x: clean_list_element(x, col_type))
        
        # Calculate row-level modifications
        for orig, norm in zip(original_col_data, df[col]):
            if values_are_different(orig, norm):
                changes += 1
                
    report['changes_detected'] = changes
    
    # 3. Save Cleaned Data back to disk
    if changes > 0:
        if file_type == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)
        report['status'] = f'Success (Updated)'
    else:
        report['status'] = 'Success (No Changes Needed)'
        
    return report


def process_checkpoint_jsonl() -> dict:
    report = {
        'file_key': 'checkpoint_rewrites',
        'path': str(CHECKPOINT_FILE.relative_to(BASE_DIR) if BASE_DIR in CHECKPOINT_FILE.parents else CHECKPOINT_FILE),
        'status': 'Skipped (File Not Found)',
        'rows_processed': 0,
        'changes_detected': 0
    }
    
    if not CHECKPOINT_FILE.exists():
        return report
        
    report['status'] = 'Processing'
    
    temp_lines = []
    changes = 0
    total_lines = 0
    
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                modified = False
                
                # Check and normalize structural text fields
                for field in ['text', 'rewritten']:
                    if field in data and isinstance(data[field], str):
                        original_val = data[field]
                        normalized_val = normalize_text(original_val)
                        if original_val != normalized_val:
                            data[field] = normalized_val
                            modified = True
                
                if modified:
                    changes += 1
                temp_lines.append(json.dumps(data, ensure_ascii=False))
            except Exception as e:
                # Fallback to keep raw line if parsing errors occur
                temp_lines.append(line.strip())
                
    report['rows_processed'] = total_lines
    report['changes_detected'] = changes
    
    if changes > 0:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            for line in temp_lines:
                f.write(line + '\n')
        report['status'] = f'Success (Updated)'
    else:
        report['status'] = 'Success (No Changes Needed)'
        
    return report


# =====================================================================
# MAIN RUNNER & REPORT PRINT
# =====================================================================
def main():
    print("=" * 80)
    print("      RETROACTIVE UNICODE & FORMATTING NORMALIZER FOR NLP DATASETS")
    print("=" * 80)
    print(f"Base Directory Identified: {BASE_DIR}\n")
    
    reports = []
    
    # 1. Process Silver and Gold Datasets
    for file_key, config in tqdm(FILES_CONFIG.items(), desc="Retrofitting DataFrames"):
        rep = process_dataframe_file(file_key, config)
        reports.append(rep)
        
    # 2. Process Gold Checkpoint JSONL File
    print("Retrofitting Gold Checkpoint JSONL...")
    checkpoint_rep = process_checkpoint_jsonl()
    reports.append(checkpoint_rep)
    
    # 3. Print Detailed Report Printout
    print("\n" + "=" * 105)
    print(f"{'FILE KEY / DATASET':<25} | {'FILE PATH (RELATIVE)':<45} | {'STATUS':<20} | {'RECORDS':<8} | {'MUTATIONS':<8}")
    print("=" * 105)
    
    for r in reports:
        print(f"{r['file_key']:<25} | {r['path']:<45} | {r['status']:<20} | {r['rows_processed']:<8} | {r['changes_detected']:<8}")
        
    print("=" * 105)
    print("Process Complete. All text assets in the selection, gold, and generation layers are synchronized.\n")


if __name__ == '__main__':
    main()