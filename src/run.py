# main script to run all
import argparse
from scrape import ScriptiebankScraper
import pandas as pd
import re
import requests
from pathlib import Path
from langdetect import detect, LangDetectException
import mu
import io
import hashlib
import ast
import json
import numpy as np
from mu import DataFrameCleaner


def main():
    #path_df = Path('data2/')


    #parser = argparse.ArgumentParser(description="A simple script to demonstrate argparse.")
    #parser.add_argument("input_file", type=str, help="Path to the input file.")
    #parser.add_argument("-o", "--output", type=str, default="output.txt", help="Path to the output file (default: output.txt).")
    #parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    #args = parser.parse_args()

    #print(f"Input file: {args.input_file}")
    #print(f"Output file: {args.output}")

    #if args.verbose:
    #    print("Verbose mode enabled.")
    
    run_all = False
    
    sb_scrape = False
    ug_dl = False
    clean_metadata = True


    if run_all == True:
        sb_scrape = True
        ug_dl = True
        clean_metadata = True

    #STEP 1 BRONZE: RAW DATA COLLECTION: SCRIPTIEBANK HTML + UGENT JSONLINES DATADUMP
    if sb_scrape == True:
        scraper = ScriptiebankScraper()
        scraper.run(gather_metadata=True,gather_urls=True,download_files=False) #download selection later


    if ug_dl == True:
        file_path = Path('data/raw_data/UG/publications.json')
        datadump_url = 'https://biblio.ugent.be/exports/publications.json'
    
        def download_ug(url, save_path):
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True,exist_ok=True)

            response = requests.get(url)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)
            print("downlaod complete")

        #loading into pandas
        # df = pd.read_json(save_path, lines = True)
            
        download_ug(datadump_url, file_path)






    #STEP 2: SILVER
    # CLEAN METADATA



    if clean_metadata == True:
        ug_path = Path('/home/gderijck/internship/data/raw_data/UG/publications.json')
        sb_path = Path('/home/gderijck/internship/data/raw_data/SB/SB_metadata_raw.tsv')

        ug_clean = Path()
        sb_clean = Path()
        
        #1 SEPARATE CLEAN
        #SB PREP
        sb_df = pd.read_csv(sb_path,sep='\t')
        sb_df['year'] = sb_df['year'].astype('Int64')
        sb_df['authors'] = [[f"{f} {l}"] for f, l in zip(sb_df['first_name'], sb_df['last_name'])]
        sb_df = sb_df.drop(columns=['first_name', 'last_name'])
        sb_df['text_homepage'] = sb_df['text_homepage'].apply(parse_and_flatten)


    # fix homepage
        # (list -> string)
        # delete unusable -> na
    

        def clean_df(df_path, save_path, protected_values=None,schema=None):
            if not save_path: raise 'please input save path'
            if df_path.suffix == '.tsv':
                df = pd.read_csv(df_path,sep='\t')
            elif df_path.suffix == '.json': df = pd.read_json(df_path, lines=True)
            else: raise 'format not supported'
            if not df: raise 'empty dataframe or something else went wrong'
            cleaner = DataFrameCleaner(df)
            cleaner.run_auto_pipeline(schema=schema, protected_values=protected_values) #MAKE SURE IT HANDLES NONE
            cleaner.save_parquet(path=save_path)


        #UG
        prot_val_ug = {
            'volume': [999,'999',9999,'9999'], #CHANGE TO REGEX
            'issue': ['999',999]
        }
        
        clean_df(ug_path, ug_clean, protected_values=prot_val_ug)

        #SB
        schema_sb = {
            # --- Identity & Source (Strings) --- #how does it handle non-existant cols
            'id': 'string',
            'source': 'string',
            'college': 'string',
            'type': 'string',
            
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

        prot_val_sb = {
            
        }

        clean_df(ug_path, ug_clean, protected_values=prot_val_sb,schema=schema_sb)
        #2 FLAT MERGE
        #3 SELECTION MERGE


        def generate_hash(title, year):
            """The deterministic fingerprint used by both cleaners."""
            if pd.isna(title) or pd.isna(year):
                return "unknown_key"
            combined = f"{str(title).strip().lower()}_{str(year).strip()}"
            return hashlib.sha256(combined.encode()).hexdigest()

        def run_silver_pipeline(tsv_path, jsonl_path, output_parquet):
            """
            The Master Pipeline: 
            1. Load & Audit via DataFrameCleaner
            2. Domain-specific Cleaning
            3. Outer Merge & Coalesce
            4. Save to Parquet
            """
            
            # --- STEP 1: PROCESS TSV (SB DATA) ---
            print("\n>>> STAGE 1: Processing TSV (SB)...")
            # Initialize with your class
            sb_cleaner = DataFrameCleaner(tsv_path)
            
            # Use your class's built-in cleaning
            sb_cleaner.clean_column_names().drop_duplicates()
            
            # Extract the cleaned DF
            df_sb = sb_cleaner.df.copy()

            # Domain-specific transformations for SB
            # 1. Fix the 'stringified' lists (e.g., "['econ']")
            for col in ['themes', 'keywords']:
                if col in df_sb.columns:
                    df_sb[col] = df_sb[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
                    )
                    df_sb[col] = df_sb[col].apply(lambda x: "; ".join(x) if isinstance(x, list) else "")

            # 2. Create Join Key components
            df_sb['author_combined'] = df_sb['first_name'].astype(str) + " " + df_sb['last_name'].astype(str)
            df_sb['join_key'] = df_sb.apply(lambda r: generate_hash(r['title'], r['year']), axis=1)

            # --- STEP 2: PROCESS JSONL (UG DATA) ---
            print("\n>>> STAGE 2: Processing JSONL (UG)...")
            # Initialize with your class (make sure you updated __init__ to support .jsonl)
            ug_cleaner = DataFrameCleaner(jsonl_path)
            
            # Use your class's built in cleaning
            ug_cleaner.clean_column_names().drop_duplicates()
            
            # Extract the cleaned DF
            df_ug = ug_cleaner.df.copy()

            # Domain-specific transformations for UG (Robust Extractors)
            def extract_authors_ug(auth_list):
                if not isinstance(auth_list, list): return ""
                return "; ".join([a.get('name', '') if isinstance(a, dict) else str(a) for a in auth_list if a])

            def extract_abstract_ug(val):
                if isinstance(val, list) and len(val) > 0:
                    return val[0].get('text', '') if isinstance(val[0], dict) else str(val[0])
                return str(val) if isinstance(val, str) else ""

            df_ug['author_combined'] = df_ug['author'].apply(extract_authors_ug)
            df_ug['abstract_text'] = df_ug['abstract'].apply(extract_abstract_ug)
            df_ug['doi_clean'] = df_ug['doi'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x))
            df_ug['pdf_url'] = df_ug['file'].apply(lambda x: x[0].get('url', '') if isinstance(x, list) and len(x) > 0 else "")
            df_ug['keyword_clean'] = df_ug['keyword'].apply(lambda x: "; ".join(x) if isinstance(x, list) else "")
            
            # The "Payload" - keep everything else
            df_ug['library_payload'] = df_ug.apply(lambda x: x.to_json(), axis=1)

            # Create Join Key components
            df_ug['join_key'] = df_ug.apply(lambda r: generate_hash(r['title'], r['year']), axis=1)

            # --- STEP 3: THE MERGE (The Convergence) ---
            print("\n>>> STAGE 3: Merging Datasets...")
            
            # Outer join keeps everything from both sides
            merged = pd.merge(df_sb, df_ug, on='join_key', how='outer', suffixes=('_sb', '_ug'))

            # 1. Source Logic
            has_sb = merged['title_sb'].notna()
            has_ug = merged['title_ug'].notna()
            conditions = [(has_sb & has_ug), (has_sb & ~has_ug), (~has_sb & has_ug)]
            choices = ['Both', 'SB', 'UG']
            merged['source'] = np.select(conditions, choices, default='Unknown')

            # 2. Coalesce (Unified Columns)
            # We merge the columns from both sources into one master set
            merged['title'] = merged['title_sb'].fillna(merged['title_ug'])
            merged['author'] = merged['author_combined_sb'].fillna(merged['author_combined_ug'])
            merged['year'] = merged['clean_year_sb'].fillna(merged['clean_year_ug'])
            merged['abstract'] = merged['abstract_text'].fillna("")
            merged['doi'] = merged['doi_clean'].fillna("")
            merged['pdf_url'] = merged['pdf_url'].fillna("")
            merged['keywords'] = merged['keywords'].fillna(merged['keyword_clean']).fillna("")
            merged['library_payload'] = merged['library_payload'].fillna("{}")
            merged['scraped_url'] = merged['download_link'].fillna("")

            # 3. Final Column Selection (The Silver Schema)
            final_cols = [
                'source', 'join_key', 'title', 'author', 'year', 
                'abstract', 'doi', 'pdf_url', 'keywords', 
                'scraped_url', 'library_payload'
            ]
            # Only select columns that actually exist
            final_df = merged[[c for c in final_cols if c in merged.columns]].copy()

            # --- STEP 4: FINAL CLEANING & STORAGE ---
            print("\n>>> STAGE 4: Finalizing & Saving...")
            
            # Use your class logic one last time on the unified data
            # This ensures the merged dataframe is perfectly typed
            final_df = DataFrameCleaner(final_df)
            final_df.convert_data_types_pandas() # The "Goldilocks" zone of types
            
            # Save to Parquet
            final_df.df.to_parquet(output_parquet, engine='pyarrow', index=False)
            print(f"SUCCESS! Silver Master saved to: {output_parquet}")
            print(f"Final Record Count: {len(final_df)}")
            print(f"Source breakdown:\n{final_df['source'].value_counts()}")

        if __name__ == "__main__":
            # Config
            TSV_IN = Path('data/raw_data/SB/SB_metadata_raw.tsv')
            JSONL_IN = Path('data/raw_data/UG/publications.json')
            PARQUET_OUT = Path('data/silver/master_publications.parquet')

            run_silver_pipeline(TSV_IN, JSONL_IN, PARQUET_OUT)

    # ENFORCE SCHEMA
    # MERGE


    #STEP 3: GOLD
    # SELECT RELEVANT DATA
    # DOWNLOAD FILES & ASSOCIATE IDS
    # RUN CONVERSION TO STANDARD FORMAT



#SAVE DATA RAW

    #STEP 2 CHANGE ID TO source_downloadtoken, change year to int
    #code

    #STEP 3 DOWNLOAD JSON LINES DATADUMP FROM UGENT


    #generate more robust id: source_name+download_link_token
    #df = pd.read_csv()
    #download_pattern = re.compile(r"token=([a-zA-Z0-9_-]+)")
    #df['download_token'] = df['download_link'].str.extract(download_pattern)[0]
    #df['id'] = df['source'] + '_' + df['download_token']

    # Display the updated DataFrame
    #print(df)




    # json -> tsv from ugent
    


#for gold
    # add future rows (might delete later)
    #df_s['abstract'] = None
    #df_s['language_full'] = None
    #df_s['chapters'] = None
    #abstract found
    #...

if __name__ == "__main__":
    main()

