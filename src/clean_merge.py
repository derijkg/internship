import pandas as pd
import requests
from pathlib import Path
from mu import DataFrameCleaner

# PATHS
UGENT_JSON_PATH = Path('data/ugent_datadump/publications.json')


# DOWNLOAD UGENT JSON IF DOESNT EXIST
datadump_url = 'https://biblio.ugent.be/exports/publications.json'

if not UGENT_JSON_PATH.exists():
    print(f'Downloading ugent datadump from {UGENT_JSON_PATH}')

    try:
        UGENT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(datadump_url, stream=True) as response:
            response.raise_for_status()
            with open(UGENT_JSON_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'Sucessfully downloaded to {UGENT_JSON_PATH}')
    except requests.exceptions.RequestException as e:
        print(f'error during download {e}')
else:
    print(f'File already exists at: {UGENT_JSON_PATH}')



# CLEANING JSON -> PARQUET??

cleaner = DataFrameCleaner(pd.read_json(Path('data/ugent_datadump/publications.json'), lines=True))
#dt_cols = ['date_created','date_updated','date_from']
df_j = (cleaner
        .drop_missing_cols()
        .standardize_missing_values()
        #.convert_to_datetime(columns=dt_cols)
        .convert_data_types()
        .unify_na_values()
        .get_df()
        )
#cleaner.summarize()

df_j.to_parquet('data/ugent_datadump/ugent_cleaned.parquet', engine='pyarrow', compression='snappy')




# CLEANING CSV

