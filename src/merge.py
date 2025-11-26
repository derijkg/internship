import pandas as pd
import numpy as np
from pathlib import Path
from mu import DataFrameCleaner
from rapidfuzz import process, fuzz

# HARDCODE ID
LAST_ID = '4180'

# PATHS
PATH_UGENT_CLEAN = Path('data/ugent_datadump/ugent_cleaned.parquet')
PATH_SB_CLEAN = Path('data/metadata_clean.parquet')

PATH_MERGED = Path('data/metadata.parquet')

# MERGE
    #TODO
        # some non downloadable english abstract texts were making it in, maybe download type is wrong? check fullText condition
''' selecting from ugent and merging datasets '''

df1 = pd.read_parquet(
    PATH_SB_CLEAN, 
    engine='pyarrow', 
    dtype_backend='pyarrow',
    filters=[
        ('year', '>=', 2000),
        ('year', '<=', 2022)
    ]
)
df2 = pd.read_parquet(
    PATH_UGENT_CLEAN, 
    engine='pyarrow', 
    dtype_backend='pyarrow',
    filters=[
        ('year', '>=', 2000),
        ('year', '<=', 2022)
    ]
)

# --- HELPERS ---

def get_open_access_link(file_list):
    """Returns URL if access is open, else None."""
    if isinstance(file_list, (list, np.ndarray)):
        for f in file_list:
            if isinstance(f, dict) and f.get('access') == 'open': #TODO and f.get(typeofcontent) == 'fullText' something
                return f.get('url')
    return None

def check_metadata_dutch(lang_list):
    """Returns True if 'nl', 'dut', or 'nld' is in the language list."""
    if isinstance(lang_list, (list, np.ndarray)):
        clean = [l for l in lang_list if isinstance(l, str)]
        return any(l in ['nl', 'dut', 'nld'] for l in clean)
    return False

def extract_dutch_from_full(abs_full):
    """Scans abstract_full for Dutch text."""
    if isinstance(abs_full, (list, np.ndarray)):
        for item in abs_full:
            if isinstance(item, dict):
                lang = item.get('lang', '').lower()
                if lang in ['nl', 'dut', 'nld']:
                    return item.get('text', '') #TODO if empty just forget it
    return None

def get_fallback_abstract(row):
    """Fallback logic for abstracts."""
    # 1. Main Abstract
    ab_main = row.get('abstract')
    if isinstance(ab_main, (list, np.ndarray)) and len(ab_main) > 0:
        if isinstance(ab_main[0], str) and ab_main[0].strip():
            return ab_main[0]
    elif isinstance(ab_main, str) and ab_main.strip():
        return ab_main
        
    # 2. Any abstract_full
    ab_full = row.get('abstract_full')
    if isinstance(ab_full, (list, np.ndarray)) and len(ab_full) > 0:
        if isinstance(ab_full[0], dict):
            return ab_full[0].get('text', '')
    return ""

def process_authors(auth_list):
    """Extracts author names from list of dicts."""
    if not isinstance(auth_list, (list, np.ndarray)):
        return []
    result = []
    for a in auth_list:
        if isinstance(a, dict):
            fn = a.get('first_name') or ""
            ln = a.get('last_name') or ""
            full = f"{fn} {ln}".strip()
            if full:
                result.append(full)
    return result


# unused 
def process_affiliation(aff_list):
    """
    Extracts the 'name' from the first affiliation dictionary.
    Input sample: [{'name': 'Department of...', 'ugent_id': '...'}]
    """
    if isinstance(aff_list, (list, np.ndarray)) and len(aff_list) > 0:
        first_item = aff_list[0]
        if isinstance(first_item, dict):
            return first_item.get('name')
    return None

# --- MAIN PROCESS ---

# 1. FILTER YEAR
df_work = df2.copy()
df_work['year_clean'] = pd.to_numeric(df_work['year'], errors='coerce')
df_work = df_work[df_work['year_clean'].between(2000, 2022)]

if df_work.empty:
    print("No entries found in year range.")
else:
    # 2. CALCULATE CONDITIONS
    df_work['download_link'] = df_work['file'].map(get_open_access_link)
    is_open_access = df_work['download_link'].notna() 
    is_meta_dutch = df_work['language'].map(check_metadata_dutch)
    
    df_work['dutch_full_text'] = df_work['abstract_full'].map(extract_dutch_from_full)
    has_dutch_full = df_work['dutch_full_text'].notna() & (df_work['dutch_full_text'] != "")
    
    # 3. APPLY SELECTION CRITERIA
    keep_mask = (is_meta_dutch & is_open_access) | has_dutch_full
    df_work = df_work[keep_mask].copy()
    
    if df_work.empty:
        print("No rows matched the selection criteria.")
    else:
        # 4. PREPARE FINAL COLUMNS
        
        # Complex Transformations
        df_work['final_authors'] = df_work['author'].map(process_authors)
        df_work['final_affiliation'] = df_work['affiliation'].map(process_affiliation)
        
        df_work['final_abstract'] = df_work.apply(
            lambda r: r['dutch_full_text'] if (pd.notna(r['dutch_full_text']) and r['dutch_full_text'] != "") 
            else get_fallback_abstract(r), 
            axis=1
        )
        
        # 5. BUILD RESULT DATAFRAME
        df_final = pd.DataFrame({
            'id': df_work['_id'].combine_first(df_work['biblio_id']), #TODO
            'title': df_work['title'],
            'year': df_work['year_clean'].astype(int),
            'page_link': df_work['handle'],
            'download_link': df_work['download_link'], 
            'authors': df_work['final_authors'],
            'affiliation': df_work['affiliation'], 
            'themes': df_work['subject'],              
            'keywords': df_work['keyword'],              
            'abstract': df_work['final_abstract'],
            'language': 'nl',
            'source': 'ugent_biblio',
            'downloaded': False,

        })

        # 6. PREPARE DF1
        df1_clean = df1.rename(columns={'college': 'affiliation'})
        
        # 7. MERGE
        start_idx = df1_clean.index[-1] + 1 if not df1_clean.empty else 0
        df_final.index = range(start_idx, start_idx + len(df_final))
        


        df_final['id'] = df_final.index
        
        df_combined = pd.concat([df1_clean, df_final], axis=0)
        
        print(f"Success! Added {len(df_final)} rows.")
        print(f"Columns in new dataframe: {df_combined.columns.tolist()}")