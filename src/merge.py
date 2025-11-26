import pandas as pd
import numpy as np
import requests
from pathlib import Path
from mu import DataFrameCleaner
import gc
import ast
from rapidfuzz import process, fuzz

# PATHS
PATH_UGENT_JSON = Path('data/ugent_datadump/publications.json')
PATH_UGENT_CLEAN = Path('data/ugent_datadump/ugent_cleaned.parquet')

PATH_SB_CSV = Path('data/metadata copy.csv')
PATH_SB_CLEAN = Path('data/metadata_clean.parquet')

PATH_MERGED = Path('data/metadata.parquet')


# UGENT -----------------------------------------------------------------------------------

# DOWNLOAD UGENT JSON IF DOESNT EXIST
datadump_url = 'https://biblio.ugent.be/exports/publications.json'

if not PATH_UGENT_JSON.exists():
    print(f'Downloading ugent datadump from {PATH_UGENT_JSON}')

    try:
        PATH_UGENT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(datadump_url, stream=True) as response:
            response.raise_for_status()
            with open(PATH_UGENT_JSON, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'Sucessfully downloaded to {PATH_UGENT_JSON}')
    except requests.exceptions.RequestException as e:
        print(f'error during download {e}')
else:
    print(f'File already exists at: {PATH_UGENT_JSON}')



# CLEANING JSON -> PARQUET (arrow dtypes)
if not PATH_UGENT_CLEAN.exists():
    cleaner = DataFrameCleaner(pd.read_json(PATH_UGENT_JSON, lines=True))

    schema ={

    }

    protected_values = {
        'volume': [999,'999'],
        'issue': ['999',999]
    }
    cleaner.run_auto_pipeline(protected_values=protected_values)
    cleaner.save_parquet(path=PATH_UGENT_CLEAN)
    del cleaner
    gc.collect()



# SCRIPTIEBANK --------------------------------------------------------------------------------------------------------------------------
#TODO
    # limit years (2000-2022)


if not PATH_SB_CLEAN.exists():
    df_s = pd.read_csv(PATH_SB_CSV)

    # add future rows (might delete later)
    df_s['abstract'] = None
    df_s['language_abstract'] = None
    df_s['language_full'] = None
    df_s['pages'] = None #
    df_s['chapters'] = None
    df_s['type'] = None

    # merge authors
    df_s['authors'] = [[f"{first} {last}"] for first, last in zip(df_s['first_name'], df_s['last_name'])]
    df_s = df_s.drop(columns=['first_name', 'last_name'])


    # fix homepage
        # (list -> string)
        # delete unusable -> na
    def parse_and_flatten(val):
        if isinstance(val, list):
            return val[0] if len(val) > 0 else None
            
        if isinstance(val, str) and val.strip().startswith('['):
            try:
                actual_list = ast.literal_eval(val)
                return actual_list[0] if isinstance(actual_list, list) and len(actual_list) > 0 else None
            except (ValueError, SyntaxError):
                return None         
        return val
    df_s['text_homepage'] = df_s['text_homepage'].apply(parse_and_flatten)    


    # auto cleaner
    cleaner = DataFrameCleaner(df_s)

    schema_definition = {
        # --- Identity & Source (Strings) ---
        'id': 'string',           # Convert int -> string for safety
        'source': 'string',
        'college': 'string',      # Will likely map to 'publisher' or 'affiliation' later
        'type': 'string',         # Empty now, but strictly string
        
        # --- Content (Strings) ---
        'title': 'string',
        'abstract': 'string',     
        'language_full': 'string', 
        'language_abstract': 'string',
        'text_homepage': 'string',
        
        # --- URLs (Strings) ---
        'page_link': 'string',
        'download_link': 'string',

        # --- Numeric (Ints) ---
        'year': 'int',
        'pages': 'int',           
        'chapters': 'int',        

        # --- Boolean ---
        'downloaded': 'bool',

        # --- Lists (Strings inside Lists) ---
        'authors': 'list',        # Note: Main DB uses 'author' (list of dicts). Keep as list for now.
        'keywords': 'list',
        'themes': 'list',
        'promoter': 'list',
    }


    protected_values = {
        'id': ['999',999,0,'0']
    }

    cleaner.run_auto_pipeline(schema=schema_definition, protected_values=protected_values)
    


    # DELETE DUPLICATE ENTRIES
    # --- CONFIGURATION ---
    TITLE_THRESHOLD = 60   # Similarity to consider titles a match
    AUTHOR_THRESHOLD = 80  # Similarity to confirm same author
    # ---------------------


    def solve_duplicates(df):
        # Sets to track our decisions
        keep_idx = set()
        remove_idx = set()
        
        # Helper: Calculate completeness (count of non-null columns)
        # We calculate this once for the whole dataframe for speed
        df['_completeness_score'] = df.notna().sum(axis=1)

        # ---------------------------------------------------------
        # STEP 1: Identify Candidate Pairs (Title Based)
        # ---------------------------------------------------------
        print("Step 1: Identifying candidate title pairs...")
        
        # Map titles to list of indices: {'Title A': [0, 5], 'Title B': [2]}
        title_map = df.groupby('title').groups
        unique_titles = list(title_map.keys())
        
        candidate_pairs = [] # List of (index_1, index_2)

        # A. Fuzzy Matches (Using your logic)
        # Note: limit=5 might be too low if you have many matches, 
        # but for "near matches" per string it is okay.
        for t1 in unique_titles:
            matches = process.extract(t1, unique_titles, scorer=fuzz.token_sort_ratio, limit=5)
            for t2, score, _ in matches:
                if t1 == t2: continue # Skip exact self-match (handled in B)
                if score >= TITLE_THRESHOLD:
                    # We found two DIFFERENT strings that look alike.
                    # We must compare every row with Title 1 against every row with Title 2
                    idxs_1 = title_map[t1]
                    idxs_2 = title_map[t2]
                    
                    for i1 in idxs_1:
                        for i2 in idxs_2:
                            # Store sorted tuple to avoid checking (1,2) and (2,1)
                            pair = tuple(sorted((i1, i2)))
                            candidate_pairs.append(pair)

        # B. Exact Matches (Same title appearing in multiple rows)
        for title, indices in title_map.items():
            if len(indices) > 1:
                # Create pairs for every combination of indices with this title
                indices = list(indices)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        candidate_pairs.append((indices[i], indices[j]))

        # Deduplicate pairs
        candidate_pairs = list(set(candidate_pairs))
        print(f" -> Found {len(candidate_pairs)} candidate row pairs based on Title.")

        # ---------------------------------------------------------
        # STEP 2: Verify Author & Resolve (Completeness)
        # ---------------------------------------------------------
        print("Step 2: Verifying Authors and Resolving...")
        
        processed_indices = set() # Track indices we have made a decision on

        for idx1, idx2 in candidate_pairs:
            # Skip if we already removed one of these (greedy approach)
            # remove this check if you want to evaluate all overlaps
            if idx1 in remove_idx or idx2 in remove_idx:
                continue

            author1 = str(df.loc[idx1, 'authors']) # Ensure string
            author2 = str(df.loc[idx2, 'authors'])

            # Check Author Similarity
            auth_score = fuzz.token_sort_ratio(author1, author2)
            
            if auth_score >= AUTHOR_THRESHOLD:
                # --- CONFIRMED DUPLICATE ---
                processed_indices.add(idx1)
                processed_indices.add(idx2)
                
                # Check Completeness
                score1 = df.loc[idx1, '_completeness_score']
                score2 = df.loc[idx2, '_completeness_score']
                
                if score1 >= score2:
                    keep_idx.add(idx1)
                    remove_idx.add(idx2)
                else:
                    keep_idx.add(idx2)
                    remove_idx.add(idx1)
            else:
                # Authors look different, likely not a duplicate
                pass

        print(f" -> Marked {len(remove_idx)} rows for removal.")
        print(f" -> Marked {len(keep_idx)} rows to keep (as the 'better' versions).")

        # ---------------------------------------------------------
        # STEP 3: Check "Unaccounted" Repeated Authors
        # ---------------------------------------------------------
        print("\nStep 3: Checking Unaccounted Author Duplicates...")
        
        # All indices involved in the duplicate process (either kept or removed)
        accounted_indices = keep_idx | remove_idx
        
        # Find authors that appear > 1 time in the WHOLE dataframe
        # (We cast to str to ensure hashability if list)
        author_counts = df['authors'].astype(str).value_counts()
        repeated_authors = author_counts[author_counts > 1].index
        
        unaccounted_results = []

        for auth in repeated_authors:
            # Get all indices for this author
            # Note: This filtering might be slow on huge DFs, usually better to iterate groups
            mask = df['authors'].astype(str) == auth
            author_indices = set(df.index[mask])
            
            # Subtract indices we already handled in Step 2
            unaccounted = author_indices - accounted_indices
            
            #TODO further check: year? promoter?
            if len(unaccounted) > 1:
                # These are rows with the same author that were NOT flagged as title duplicates
                titles = df.loc[list(unaccounted), 'title'].tolist()
                unaccounted_results.append({
                    'author': auth,
                    'count_left': len(unaccounted),
                    'titles': titles # Show first 3 titles
                })
        df.drop(columns=['_completeness_score'], inplace=True, errors='ignore')
        
        return keep_idx, remove_idx


    keep, remove = solve_duplicates(cleaner.df)
    cleaner.df.drop(index=list(remove),inplace=True)
    cleaner.run_auto_pipeline() # runs again to remove new NA values   PROBLEM: new ugent entries already downloaded. index != id == filename...

    
    
    #TODO remove duplicates from archive and marker_ouput








    # remove garbage text_homepage values TODO assume len < x is garbage
    cleaner.drop_short_strings(column='text_homepage', min_chars=100)
    
    # final
    cleaner.save_parquet(path=PATH_SB_CLEAN)
    del cleaner
    del df_s
    gc.collect()





# MERGE
    #TODO
        # some non downloadable english abstract texts were making it in, maybe download type is wrong? check fullText condition
''' selecting from ugent and merging datasets '''

df1 = pd.read_parquet(PATH_SB_CLEAN, engine='pyarrow', dtype_backend='pyarrow')
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
            if isinstance(f, dict) and f.get('access') == 'open':
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
                    return item.get('text', '')
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