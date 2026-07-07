import argparse
from pathlib import Path
import re
import string
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from langdetect import detect, LangDetectException
import nltk
from tqdm import tqdm

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from mu import DataFrameCleaner
from scrape import HBOScraper, ScriptiebankScraper

BASE_DIR = Path(__file__).resolve().parent.parent

# Centralized mapping for all datasets and their expected step outputs
SOURCES_CONFIG = {
    'UG': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'UG' / 'publications.json',
        'clean_file': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_cleaned.parquet',
        'selected_file': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_selected.parquet',
    },
    'HBO': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'HBO' / 'HBO_metadata.csv',
        'clean_file': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_cleaned.parquet',
        'selected_file': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_selected.parquet',
    },
    'SB': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'SB' / 'SB_metadata.csv',
        'clean_file': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_cleaned.parquet',
        'selected_file': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_selected.parquet',
    }
}


def download_raw_data(source_name: str, output_path: Path):
    """
    Downloads raw data from scrapers or remote endpoints.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if source_name == 'SB':
        print("[SB] Starting Scriptiebank scraper...")
        scraper = ScriptiebankScraper(base_folder=BASE_DIR / 'data')
        scraper.run(gather_metadata=True, gather_urls=True, download_files=False)
        
    elif source_name == 'HBO':
        print("[HBO] Starting HBO scraper...")
        scraper = HBOScraper(base_folder=BASE_DIR / 'data')
        scraper.run(gather_metadata=True, gather_urls=True, download_files=False)
        
    elif source_name == 'UG':
        datadump_url = 'https://biblio.ugent.be/exports/publications.json'
        print(f"[UG] Downloading UGent datadump from {datadump_url}...")
        response = requests.get(datadump_url)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        print("[UG] Download complete.")


def clean_source(source_name: str, input_path: Path, output_path: Path, schema=None):
    """
    Reads a raw file, applies source-specific configuration overrides (e.g. UGent protected
    values, HBO column mergers), runs DataFrameCleaner, and saves to silver as parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read raw dataset
    if input_path.suffix == '.tsv':
        df = pd.read_csv(input_path, sep='\t')
    elif input_path.suffix == '.json': 
        df = pd.read_json(input_path, lines=True)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    else: 
        raise ValueError(f"Format {input_path.suffix} is not supported.")

    # 1. Source override: protected values (UGent only)
    protected_values = None
    if source_name == 'UG':
        protected_values = {
            'volume': [99, '99', 999, '999', 9999, '9999'],
            'issue': [99, '99', '999', 999, '9999', 9999]
        }

    # 2. Source override: column harmonization (HBO only)
    if source_name == 'HBO':
        print("[HBO] Harmonizing columns: merging 'jaar' -> 'year' and 'partners' -> 'partner'...")
        
        # Merge 'jaar' into 'year'
        if 'jaar' in df.columns:
            if 'year' in df.columns:
                df['year'] = df['year'].fillna(df['jaar'])
            else:
                df['year'] = df['jaar']
            df = df.drop(columns=['jaar'])
            
        # Merge 'partners' into 'partner'
        if 'partners' in df.columns:
            if 'partner' in df.columns:
                df['partner'] = df['partner'].fillna(df['partners'])
            else:
                df['partner'] = df['partners']
            df = df.drop(columns=['partners'])

    # Execute main cleaning pipeline
    print(f"[{source_name}] Cleaning: {input_path} -> {output_path}")
    cleaner = DataFrameCleaner(df)
    cleaner.run_auto_pipeline(schema=schema, protected_values=protected_values)
    cleaner.save_parquet(path=output_path)


def select_and_clean_abstracts(
    source_name: str,
    input_path: Path,
    output_path: Path,
    min_year: int = 1980,
    max_year: int = 2022,
    min_char_length: int = 100,
    source_lang_tag: str = 'dut',
    detect_lang_tag: str = 'nl',
    tokenizer_lang: str = 'dutch'
):
    """
    Extracts, tokenizes, filters, and standardizes abstracts based on language and dates.
    Tracks statistics for all excluded rows and displays progress via a loading meter.
    """
    print(f"[{source_name}] Parsing abstracts from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    table = pq.read_table(input_path)
    rows = table.to_pylist()
    filtered_data = []

    # Initialize statistics dictionary
    removed_stats = {
        "missing_year": 0,
        "year_out_of_bounds": 0,
        "invalid_year_format": 0,
        "missing_text_content": 0,
        "text_too_short": 0,
        "no_dutch_sentences_detected": 0
    }

    # Process each row with tqdm loading meter
    for row in tqdm(rows, desc=f"[{source_name}] Processing abstracts"):
        year = row.get('year')

        # 1. Filter out missing years
        if year is None:
            removed_stats["missing_year"] += 1
            continue

        # 2. Check year validity
        try:
            year_int = int(year)
            if not (min_year <= year_int <= max_year):
                removed_stats["year_out_of_bounds"] += 1
                continue
        except (ValueError, TypeError):
            removed_stats["invalid_year_format"] += 1
            continue

        # 3. Retrieve text content
        text_content = None
        abstract_list = row.get('abstract_full')
        if isinstance(abstract_list, list):
            for item in abstract_list:
                if isinstance(item, dict) and item.get('lang') == source_lang_tag:
                    text_content = item.get('text')
                    break
        elif isinstance(row.get('abstract'), str):
            text_content = row.get('abstract')
        elif isinstance(row.get('text_homepage'), str): #TODO CHANGE FOR SB col: text_homepage DICT {title: [paragraphs]}
            text_content = row.get('text_homepage')

        # 4. Filter out missing text
        if not text_content:
            removed_stats["missing_text_content"] += 1
            continue

        # 5. Filter out short text
        if len(text_content) < min_char_length:
            removed_stats["text_too_short"] += 1
            continue

        # 6. Clean and filter sentences
        raw_sentences = nltk.sent_tokenize(text_content, language=tokenizer_lang)
        cleaned_sentences = []
        
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            should_merge = False
            if cleaned_sentences:
                if not re.match(r'^[A-Z]', sent):
                    should_merge = True
                elif len(sent) >= 2 and sent[1] in string.punctuation:
                    should_merge = True
            
            if should_merge:
                cleaned_sentences[-1] += ' ' + sent
            else:
                cleaned_sentences.append(sent)
        
        dutch_sentences = []
        for sent in cleaned_sentences:
            try:
                if detect(sent) == detect_lang_tag:
                    dutch_sentences.append(sent)
            except LangDetectException:
                continue
                
        # 7. Filter if no Dutch sentences found
        if not dutch_sentences:
            removed_stats["no_dutch_sentences_detected"] += 1
            continue

        # If row passes all filters, save updates and append
        row['text_dut'] = ' '.join(dutch_sentences)
        row['sent_dut'] = dutch_sentences
        filtered_data.append(row)

    # Print final processing statistics
    print(f"\n[{source_name}] Process complete.")
    print(f"  Total records read:    {len(rows)}")
    print(f"  Total records retained: {len(filtered_data)}")
    print("  Exclusion counts by cause:")
    for cause, count in removed_stats.items():
        formatted_cause = cause.replace('_', ' ').capitalize()
        print(f"    - {formatted_cause}: {count}")
    print()

    # Save to parquet
    filtered_table = pa.Table.from_pylist(filtered_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered_table, output_path)
    print(f"[{source_name}] Saved filtered table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-source NLP pipeline orchestrator")
    parser.add_argument(
        '--sources', 
        type=str, 
        nargs='+', 
        default=['UG', 'HBO', 'SB'], 
        choices=['UG', 'HBO', 'SB'], 
        help="Source dataset(s) to process. Default: all (UG, HBO, SB)"
    )
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        default=['download', 'clean', 'select'],
        choices=['download', 'clean', 'select'],
        help="Pipeline steps to execute. Default: all steps"
    )
    parser.add_argument('--force', action='store_true', help="Force run all processes (ignores cache)")
    parser.add_argument('--force-download', action='store_true', help="Force run the download step")
    parser.add_argument('--force-clean', action='store_true', help="Force run the cleaning step")
    parser.add_argument('--force-select', action='store_true', help="Force run the selection/filtering step")
    
    args = parser.parse_args()

    sources = [s.upper() for s in args.sources]
    steps = [step.lower() for step in args.steps]

    for source in sources:
        print(f"\n--- Processing source: {source} ---")
        config = SOURCES_CONFIG[source]

        # 1. Download Step
        if 'download' in steps:
            raw_exists = config['raw_file'].exists()
            if not raw_exists or args.force or args.force_download:
                download_raw_data(source, config['raw_file'])
            else:
                print(f"[{source}] Raw dataset already exists. Skipping download.")

        # 2. Clean Step
        if 'clean' in steps:
            if not config['raw_file'].exists():
                print(f"[{source}] Missing raw file {config['raw_file']}. Skipping cleaning step.")
                continue

            clean_exists = config['clean_file'].exists()
            if not clean_exists or args.force or args.force_clean:
                clean_source(source, config['raw_file'], config['clean_file'])
            else:
                print(f"[{source}] Cleaned dataset already exists. Skipping cleaning.")

        # 3. Select Step
        if 'select' in steps:
            if not config['clean_file'].exists():
                print(f"[{source}] Missing cleaned file {config['clean_file']}. Skipping selection step.")
                continue

            selected_exists = config['selected_file'].exists()
            if not selected_exists or args.force or args.force_select:
                select_and_clean_abstracts(
                    source_name=source,
                    input_path=config['clean_file'],
                    output_path=config['selected_file']
                )
            else:
                print(f"[{source}] Selected dataset already exists. Skipping selection.")

#TODO add final merge

if __name__ == "__main__":
    main()