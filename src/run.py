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
import pyarrow  as pa
import pyarrow.json as pj
import pyarrow.parquet as pq
import pyarrow.csv as pv


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
    clean_metadata = False
    merge = True
    selection = True
    download_files = True
    marker_conversion = True
    markdown_conversion = True


    if run_all == True:
        sb_scrape = True
        ug_dl = True
        clean_metadata = True
        merge = True
        selection = True
        download_files = True
        marker_conversion = True
        markdown_conversion = True


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

        ug_clean = Path('/home/gderijck/internship/data/silver/ug_cleaned.parquet')
        sb_clean = Path('/home/gderijck/internship/data/silver/sb_cleaned.parquet')
        
        #sb preclean
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
        
        #1 SEPARATE CLEAN
        #SB PREP
        try:
            sb_df = pd.read_csv(sb_path,sep='\t')
            sb_df['year'] = sb_df['year'].astype('Int64')
            sb_df['authors'] = [[f"{f} {l}"] for f, l in zip(sb_df['first_name'], sb_df['last_name'])]
            sb_df = sb_df.drop(columns=['first_name', 'last_name'])
            sb_df['text_homepage'] = sb_df['text_homepage'].apply(parse_and_flatten)
        
            sb_df.to_csv(Path('/home/gderijck/internship/data/silver/sb_temp.tsv'), sep='\t', index=False)
            sb_path = Path('/home/gderijck/internship/data/silver/sb_temp.tsv')
        except: 
            sb_path = Path('/home/gderijck/internship/data/silver/sb_temp.tsv')

    # fix homepage
        # (list -> string)
        # delete unusable -> na
    

        def clean_df(df_path, save_path, protected_values=None,schema=None):
            if not save_path: raise 'please input save path'
            if df_path.suffix == '.tsv':
                df = pd.read_csv(df_path,sep='\t')
            elif df_path.suffix == '.json': df = pd.read_json(df_path, lines=True)
            else: raise 'format not supported'
            cleaner = DataFrameCleaner(df)
            cleaner.run_auto_pipeline(schema=schema, protected_values=protected_values) #MAKE SURE IT HANDLES NONE
            cleaner.save_parquet(path=save_path)


        #UG
            #change to using parquet? 
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
        clean_df(sb_path, sb_clean, protected_values=prot_val_sb,schema=schema_sb)

        #delete sb_temp.tsv

    if merge == True:
        #2 FLAT MERGE
        #take from ug what we need
            #download link : heuristic : open access?
            #abstract, add to main frame: dict: {lang: text, ...}
            #
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
    
        def merge_ug_to_sb():
            return row_to_add
        




    #STEP 3: GOLD
    if selection = True:
        pass
    if download_files = True:
        pass
    if marker_conversion = True:
        pass
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

