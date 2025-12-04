
import pandas as pd
import numpy as np
from pathlib import Path
from mu import DataFrameCleaner
import ast
from langdetect import detect, LangDetectException

# HARDCODE ID
LAST_ID = '4180'

# PATHS
PATH_UGENT_CLEAN = Path('data/ugent_clean.parquet')
PATH_SB_CLEAN = Path('data/metadata_clean.parquet')
PATH_MERGED = Path('data/metadata.parquet')

# LOAD DATA (Using filters to save memory)
print("Loading datasets...")
df1 = pd.read_parquet(
    PATH_SB_CLEAN, 
    engine='pyarrow', dtype_backend='pyarrow',
    filters=[('year', '>=', 2000), ('year', '<=', 2022)]
)

df2 = pd.read_parquet(
    PATH_UGENT_CLEAN, 
    engine='pyarrow', dtype_backend='pyarrow',
    filters=[('year', '>=', 2000), ('year', '<=', 2022)]
)

# --- HELPERS ---

def get_open_access_link(file_list):
    """
    Returns URL if access is open AND it is the full text.
    Fixes the issue of non-downloadable texts making it in.
    """
    if isinstance(file_list, (list, np.ndarray)):
        for f in file_list:
            if isinstance(f, dict):
                # Strict check for 'open' AND 'fullText'
                if f.get('access') == 'open' and f.get('kind') == 'fullText':
                    return f.get('url')
    return None

def check_metadata_dutch(lang_list):
    """Returns True if 'nl', 'dut', or 'nld' is in the language list."""
    if isinstance(lang_list, (list, np.ndarray)):
        # Clean list and check
        return any(str(l).lower() in ['nl', 'dut', 'nld'] for l in lang_list)
    return False

def safe_detect_lang(text):
    """Wrapper for langdetect."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None

def process_authors(auth_list):
    """Extracts author names from list of dicts."""
    if not isinstance(auth_list, (list, np.ndarray)):
        return [] # Return empty list for arrow consistency
    result = []
    for a in auth_list:
        if isinstance(a, dict):
            fn = a.get('first_name') or ""
            ln = a.get('last_name') or ""
            full = f"{fn} {ln}".strip()
            # Fallback if name is split differently or just 'name' exists
            if not full: 
                full = a.get('name', '').strip()
            if full:
                result.append(full)
    return result

def process_affiliation(aff_list):
    """Extracts the 'name' from the first affiliation dictionary."""
    if isinstance(aff_list, (list, np.ndarray)) and len(aff_list) > 0:
        first_item = aff_list[0]
        if isinstance(first_item, dict):
            return first_item.get('name')
    return None

# --- LOGIC HANDLERS ---

def get_best_abstract(row):
    """
    Returns tuple: (selected_abstract_text, detected_language_code)
    Logic:
    1. Check abstract_full for explicit Dutch ('dut', 'nl').
    2. If not, check generic 'abstract' and run langdetect.
    3. If detected as Dutch, return it.
    4. Else, return default abstract and its detected language.
    """
    # 1. Check explicit abstract_full
    abs_full = row.get('abstract_full')
    if isinstance(abs_full, (list, np.ndarray)):
        for item in abs_full:
            if isinstance(item, dict):
                lang = item.get('lang', '').lower()
                if lang in ['nl', 'dut', 'nld']:
                    return item.get('text', ''), 'nl'

    # 2. Get generic abstract text
    raw_abstract = ""
    ab_main = row.get('abstract')
    
    # Handle List<String> vs String
    if isinstance(ab_main, (list, np.ndarray)) and len(ab_main) > 0:
        if isinstance(ab_main[0], str): raw_abstract = ab_main[0]
    elif isinstance(ab_main, str) and len(ab_main) > 100:
        raw_abstract = ab_main

    if not raw_abstract:
        return None, None

    # 3. Detect Language
    detected = safe_detect_lang(raw_abstract[:500]) # Check first 500 chars for speed
    
    return raw_abstract, detected

# --- MAIN PROCESS ---

print("Processing UGent DataFrame...")
df_work = df2.copy()

# 1. Calculate Metadata Columns
# Determine if metadata says it's dutch
df_work['temp_language_full'] = df_work['language'] # Keep original list
df_work['is_meta_dutch'] = df_work['language'].map(check_metadata_dutch) # dutch in list (language) y/n -> t/f

# Check File Access (Strict)
df_work['download_link'] = df_work['file'].map(get_open_access_link) # if kind==fullText and access==open -> url
df_work['has_open_access'] = df_work['download_link'].notna()

# 2. Abstract & Language Detection Logic
# This returns a tuple series, we expand it next
print("  > Running abstract extraction and language detection (this takes time)...")
abstract_results = df_work.apply(get_best_abstract, axis=1, result_type='expand')
df_work['final_abstract'] = abstract_results[0]
df_work['language_abstract'] = abstract_results[1]

# 3. Define Keep Mask
# Rule: Keep if (Metadata is Dutch AND Open Access) OR (Abstract is explicitly Dutch)
# Note: You might want to verify if you really want rows with NO abstract but Dutch Metadata? 
# Usually you want at least an abstract or a file.
is_abstract_dutch = df_work['language_abstract'].isin(['nl', 'dut', 'nld'])

keep_mask = (df_work['is_meta_dutch'] & df_work['has_open_access']) | is_abstract_dutch

# 4. Filter
df_filtered = df_work[keep_mask].copy()
print(f"  > Filtered {len(df2)} -> {len(df_filtered)} rows.")

if not df_filtered.empty:
    # 5. Final Cleanups
    df_filtered['final_authors'] = df_filtered['author'].map(process_authors)
    df_filtered['final_affiliation'] = df_filtered['affiliation'].map(process_affiliation)
    
    #ids
    start_new_id = int(LAST_ID) +1
    end_new_id = start_new_id + len(df_filtered)
    new_id_range = range(start_new_id, end_new_id)

    df_filtered['id_temp'] = list(new_id_range)
    
    # 6. Construct Final DataFrame
    # Note: we cast year to Int64 to allow NaNs if necessary, though filters prevent it
    df_ugent_final = pd.DataFrame({
        'id': df_filtered['id_temp'].astype(str),
        'source': 'ugent',
        'title': df_filtered['title'],
        'year': df_filtered['year'].astype('Int64'),
        'page_link': df_filtered['handle'],
        'download_link': df_filtered['download_link'], 
        'authors': df_filtered['final_authors'],
        'affiliation': df_filtered['final_affiliation'], 
        'themes': df_filtered['subject'],              
        'keywords': df_filtered['keyword'],              
        'abstract': df_filtered['final_abstract'],
        'language_abstract': df_filtered['language_abstract'], # New Col
        'temp_language_full': df_filtered['temp_language_full'], # New Col
        'downloaded': False,
        # Empty columns to match DF1 if needed
        'chapters': None,
        'pages': None,
        'type': df_filtered['type']
    })

    cleaner = DataFrameCleaner(df_ugent_final)
    ugent_schema = {
    'id': 'string',
    }
    prot_values = {
        'id': ['9999',9999,999,'999']
    }
    cleaner.run_auto_pipeline(schema=ugent_schema,protected_values=prot_values)

    df1_ready = df1.rename(columns={
    'college': 'affiliation', 
    'language_full': 'temp_language_full',
    })

    df_combined = pd.concat([df1_ready, cleaner.df], axis=0, ignore_index=True)

    cleaner = DataFrameCleaner(df_combined)
    ugent_schema = {
        'id': 'string',
    }
    prot_values = {
        'id': ['9999',9999,'999',999]
    }
    cleaner.run_auto_pipeline(schema=ugent_schema,protected_values=prot_values)
    cleaner.save_parquet(PATH_MERGED)