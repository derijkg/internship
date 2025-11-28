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
PATH_UGENT_CLEAN = Path('data/ugent_clean.parquet')

PATH_SB_CSV = Path('data/metadata copy.csv')
PATH_SB_CLEAN = Path('data/metadata_clean.parquet')

#PATH_MERGED = Path('data/metadata.parquet')


# FUNCTIONS
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
    author_counts = df['authors'].apply(str).value_counts()
    repeated_authors = author_counts[author_counts > 1].index
    
    unaccounted_results = []

    for auth in repeated_authors:
        # Get all indices for this author
        # Note: This filtering might be slow on huge DFs, usually better to iterate groups
        mask = df['authors'].apply(str) == auth
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



# EXECUTION ---------------------------------------------------------------------------------

# UGENT -----------------------------------------------------------------------------------

# DOWNLOAD UGENT JSON IF DOESNT EXIST
datadump_url = 'https://biblio.ugent.be/exports/publications.json'
print('Attempting ugent datadump download')
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
    # CHECK FOR LATER OPERATION
    last_idx_pre = len(df_s)
    last_id_pre = df_s['id'].iloc[-1]

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
        'authors': 'list',   
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


 
    
    keep, remove = solve_duplicates(cleaner.df)
    cleaner.df.drop(index=list(remove),inplace=True)
    cleaner.df.reset_index()
    
    
    #TODO remove duplicates from archive and marker_ouput








    # remove garbage text_homepage values TODO assume len < x is garbage
    cleaner.drop_short_strings(column='text_homepage', min_chars=100)
    cleaner.convert_data_types_arrow() # runs again to remove new NA values   PROBLEM: new ugent entries already downloaded. index != id == filename...

    # final
    cleaner.save_parquet(path=PATH_SB_CLEAN)

    # CHECK
    last_idx_post = len(cleaner.df)
    last_id_post = cleaner.df['id'].iloc[-1]

    print(f'idx: {last_idx_pre, last_idx_post}')
    print(f'id: {last_id_pre, last_id_post}')
    if int(last_id_pre) != int(last_id_post):
        print('\nwarning last id got removed or things moved i think'*10)

    del cleaner
    del df_s
    gc.collect()