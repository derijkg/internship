import pandas as pd
import requests
from pathlib import Path
from langdetect import detect, LangDetectException

''' download ugent datadump and save at file_path '''

file_path = Path('data/ugent_datadump/publications.csv')
datadump_url = 'https://biblio.ugent.be/exports/publications.csv'

if not file_path.exists():
    print(f'Downloading ugent datadump from {file_path}')

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(datadump_url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'Sucessfully downloaded to {file_path}')
    except requests.exceptions.RequestException as e:
        print(f'error during download {e}')
else:
    print(f'File already exists at: {file_path}')

df = pd.read_csv(file_path)



''' select wanted data '''

df_filtered = df.copy()

df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce')
df_filtered['abstract'] = df_filtered['abstract'].fillna('')


# min length? 
def is_dutch(text: str) -> bool:
    if not text.strip():
        return False
    try:
        return detect(text) == 'nl'
    except LangDetectException:
        return False

'''
def is_dutch_abstact(row): # no abstract_full col?
    abstract_full_data = row['abstract_full']
    try:
        if isinstance(abstract_full_data, list) and abstract_full_data:
            if abstract_full_data[0].get('lang') == 'dut':
                return True
            #else: print(abstract_data[0].get('lang'))
            if abstract_full_data[0].get('lang') is not None:
                return False
        #else: print(abstract_data.type)
    except (TypeError, KeyError, IndexError):
        print('ERROR')
        return False
    
    abstract_text = row['abstract']
    if pd.notna(abstract_text) and isinstance(abstract_text, str) and len(abstract_text.split())>5:
        print('second branch active')
        try:
            if detect(abstract_text) == 'nl':
                return True
        except LangDetectException:
            return False
    return False
''' 

year_mask = df_filtered['year'].between(2000,2022)
language_mask = df_filtered['abstract'].apply(is_dutch)
final_mask = year_mask & language_mask
df_dutch = df_filtered[final_mask]

df_dutch.head()


#save new df
target_data_path = Path('data/ugent_datadump/target_data.csv')
df_dutch.to_csv(target_data_path, index=False, encoding='utf-8')
