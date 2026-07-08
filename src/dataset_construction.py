import argparse
import hashlib
import json
from pathlib import Path
import re
import string
import uuid
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
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


def extract_sb_homepage_text(val) -> Optional[str]:
    """
    Parses SB 'text_homepage' from either a JSON string, dictionary, or raw string,
    unifying list-based paragraph elements into a single cohesive block.
    """
    if pd.isna(val) or not val:
        return None
        
    # Attempt to decode string representation of dict
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('{') and val.endswith('}'):
            try:
                val = json.loads(val)
            except Exception:
                pass
                
    if isinstance(val, dict):
        all_paragraphs = []
        for k, v in val.items():
            if isinstance(v, list):
                all_paragraphs.extend([str(p).strip() for p in v if p])
            elif isinstance(v, str):
                all_paragraphs.append(v.strip())
        return " ".join(all_paragraphs) if all_paragraphs else None
        
    return str(val)

# TODO set subset or excl for deduplication columns per source
def clean_source(source_name: str, input_path: Path, output_path: Path, schema=None):
    """
    Reads a raw file, applies source-specific configuration overrides (e.g. UGent protected
    values, HBO column mergers, Scriptiebank year conversions), runs DataFrameCleaner, 
    and saves to silver as parquet.
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
        #subset for deduplicating
        subset = []
        excl = []

        protected_values = {
            'volume': [99, '99', 999, '999', 9999, '9999'],
            'issue': [99, '99', '999', 999, '9999', 9999]
        }

    # 2. Source override: Scriptiebank specific cleans (Float year to Int)
    if source_name == 'SB':
        #subset for deduplicating
        subset = []
        excl = []

        print("[SB] Casting float 'year' column to Integer type...")
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').round().astype('Int64')

    # 3. Source override: column harmonization & title merger (HBO only)
    if source_name == 'HBO':   
        #subset for deduplicating
        subset = []
        excl = []

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
            
        # Combine 'title' and 'subtitle'
        if 'title' in df.columns and 'subtitle' in df.columns:
            print("[HBO] Merging 'title' and 'subtitle' columns...")
            df['title'] = df.apply(
                lambda r: f"{r['title']}: {r['subtitle']}" 
                if pd.notna(r['title']) and pd.notna(r['subtitle']) and str(r['subtitle']).strip()
                else r['title'], 
                axis=1
            )
            df = df.drop(columns=['subtitle'])

    # Execute main cleaning pipeline
    print(f"[{source_name}] Cleaning: {input_path} -> {output_path}")
    cleaner = DataFrameCleaner(df)
    cleaner.run_auto_pipeline(schema=schema, protected_values=protected_values, dedupe_subset=subset, dedupe_exclude=excl)
    cleaner.save(path=output_path)


def clean_abstract(
    abstract: str,
    min_char_length: int = 100,
    tokenizer_lang: str = 'dutch',
    detect_lang_tag: str = 'nl',
    heading_words: Optional[List[str]] = None,
    logger = None
) -> Tuple[str, List[str]]:
    """
    Cleans and filters abstract text. Returns a tuple containing:
    (joined_clean_string, list_of_clean_sentences)
    """
    if heading_words is None:
        heading_words = [
            "achtergrond", "inleiding", "doelstelling", "methode", "methoden", 
            "resultaat", "resultaten", "conclusie", "conclusies", "discussie", 
            "aanbeveling", "aanbevelingen", "samenvatting", "abstract", 
            "trefwoorden", "kernwoorden"
        ]
        
    headings_list = []
    for h in heading_words:
        headings_list.extend([h.lower(), h.capitalize(), h.upper()])
    headings_pattern = '|'.join(set(headings_list))

    def _strip_layout_headers(sent: str) -> tuple[str, Optional[str]]:
        orig = sent
        sent_cleaned = re.sub(r'[*_]{1,2}', '', orig).strip()
        
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})([A-Z])', r'\1', sent_cleaned)
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})[\s]*[:.-]+[\s]*', '', sent_cleaned)
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})\s+([A-Z])', r'\1', sent_cleaned)
        
        if re.match(rf'^(?:{headings_pattern})$', sent_cleaned):
            sent_cleaned = ""

        if sent_cleaned != orig:
            if not sent_cleaned:
                removed = orig
            else:
                idx = orig.find(sent_cleaned)
                if idx != -1:
                    removed = orig[:idx]
                else:
                    removed = f"'{orig}' -> '{sent_cleaned}'"
            return sent_cleaned, removed
            
        return orig, None

    dutch_abstract = ""
    dutch_sentences = []
    
    if isinstance(abstract, str) and len(abstract) >= min_char_length and abstract.strip():
        abstract = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', abstract)
        
        raw_sentences = nltk.sent_tokenize(abstract, language=tokenizer_lang)
        cleaned_sentences = []
        
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            cleaned_sent, removed = _strip_layout_headers(sent)
            
            if removed and logger:
                logger.debug(f"Stripped layout header: {repr(removed)}")
            
            if not cleaned_sent:
                continue
            
            sent = cleaned_sent

            should_merge = False
            if cleaned_sentences:
                if not re.match(r'^[A-Z]', sent):
                    should_merge = True
                elif len(sent) >= 2 and sent[1] in string.punctuation:
                    should_merge = True
            
            if should_merge:
                cleaned_sentences[-1] = cleaned_sentences[-1] + ' ' + sent
            else:
                cleaned_sentences.append(sent)
        
        for sent in cleaned_sentences:
            try:
                if detect(sent) == detect_lang_tag:
                    dutch_sentences.append(sent)
            except LangDetectException:
                continue
                
        if dutch_sentences:
            dutch_abstract = ' '.join(dutch_sentences)
            
    return dutch_abstract, dutch_sentences


def select_and_clean_abstracts(
    source_name: str,
    input_path: Path,
    output_path: Path,
    min_year: int = 1980,
    max_year: int = 2022,
    min_char_length: int = 100,
    source_lang_tag: str = 'dut'
):
    """
    Extracts, tokenizes, filters, and standardizes abstracts based on language and dates.
    Tracks statistics for all excluded rows and generates the new text_dut and sent_dut columns.
    """
    print(f"[{source_name}] Parsing abstracts from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_parquet(input_path)
    filtered_rows = []

    removed_stats = {
        "missing_year": 0,
        "year_out_of_bounds": 0,
        "invalid_year_format": 0,
        "missing_text_content": 0,
        "text_too_short": 0,
        "no_dutch_sentences_detected": 0
    }

    # Convert the dataframe to records for fast row-by-row tqdm iteration
    records = df.to_dict(orient='records')

    for row in tqdm(records, desc=f"[{source_name}] Processing abstracts"):
        year = row.get('year')

        # 1. Filter out missing years
        if pd.isna(year):
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

        # 3. Standardize and retrieve raw text content per source
        text_content = None
        if source_name == 'UG':
            abstract_full = row.get('abstract_full')
            if isinstance(abstract_full, (list, np.ndarray)):
                for item in abstract_full:
                    if isinstance(item, dict) and item.get('lang') == source_lang_tag:
                        text_content = item.get('text')
                        break
            # Fallback to general abstract if JSON list extraction is missing
            if not text_content and isinstance(row.get('abstract'), str):
                text_content = row.get('abstract')

        elif source_name == 'SB':
            # Check 'abstract' first, fallback to 'text_homepage' parsing
            if isinstance(row.get('abstract'), str) and row.get('abstract').strip():
                text_content = row.get('abstract')
            elif row.get('text_homepage') is not None:
                text_content = extract_sb_homepage_text(row.get('text_homepage'))

        else: # HBO
            if isinstance(row.get('abstract'), str):
                text_content = row.get('abstract')

        # 4. Filter out missing text
        if not text_content or not str(text_content).strip():
            removed_stats["missing_text_content"] += 1
            continue

        # 5. Filter out short text
        if len(text_content) < min_char_length:
            removed_stats["text_too_short"] += 1
            continue

        # 6 & 7. Clean and structure NLP sentences
        text_dut, sent_dut = clean_abstract(text_content, min_char_length=min_char_length)
        
        if not sent_dut:
            removed_stats["no_dutch_sentences_detected"] += 1
            continue

        # Append processing results
        row['text_dut'] = text_dut
        row['sent_dut'] = sent_dut
        filtered_rows.append(row)

    # Convert results back to DataFrame
    filtered_df = pd.DataFrame(filtered_rows)

    # Apply duplicate dropping based on standardized abstract text (keep first)
    if not filtered_df.empty:
        initial_length = len(filtered_df)
        filtered_df = filtered_df.drop_duplicates(subset=['text_dut'], keep='first')
        dropped_duplicates = initial_length - len(filtered_df)
    else:
        dropped_duplicates = 0

    print(f"\n[{source_name}] Process complete.")
    print(f"  Total records read:         {len(df)}")
    print(f"  Total records retained:     {len(filtered_df)}")
    print(f"  Duplicate entries dropped:  {dropped_duplicates}")
    print("  Exclusion counts by cause:")
    for cause, count in removed_stats.items():
        formatted_cause = cause.replace('_', ' ').capitalize()
        print(f"    - {formatted_cause}: {count}")
    print()

    # Save selection to silver parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not filtered_df.empty:
        filtered_df.to_parquet(output_path, index=False)
    else:
        # Save empty table structure if no records passed
        pd.DataFrame(columns=df.columns.tolist() + ['text_dut', 'sent_dut']).to_parquet(output_path, index=False)
        
    print(f"[{source_name}] Saved filtered table to: {output_path}")


def _generate_robust_id(source: str, row: pd.Series) -> str:
    """
    Generates a unique and deterministic ID using MD5 hashing of 
    available content (abstract, title, or year) combined with a source prefix.
    """
    content_parts = []
    
    # Prioritize standardized abstract text
    for col in ['text_dut', 'abstract', 'abstract_full']:
        val = row.get(col)
        if val is not None and str(val).strip():
            content_parts.append(str(val))
            break
            
    # Include title for further uniqueness
    title_val = row.get('title')
    if title_val is not None and str(title_val).strip():
        content_parts.append(str(title_val))
        
    # Include year
    year_val = row.get('year')
    if year_val is not None:
        content_parts.append(str(year_val))
        
    combined_content = "|".join(content_parts)
    
    # Fallback to random identifier if identifying content is entirely missing
    if not combined_content.strip():
        combined_content = str(uuid.uuid4())
        
    content_hash = hashlib.md5(combined_content.encode('utf-8', errors='ignore')).hexdigest()
    return f"{source}_{content_hash[:16]}"


def _parse_semicolon_keywords(val) -> list:
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        return _parse_list_keywords(val)
    if pd.isna(val) or not isinstance(val, str):
        return []
    return [k.strip() for k in val.split(';') if k.strip()]


def _parse_list_keywords(val) -> list:
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        return [str(item).strip() for item in val if pd.notna(item) and str(item).strip()]
    if pd.isna(val):
        return []
    if isinstance(val, str):
        return [val.strip()]
    return []


def merge(sources: list, output_format: str = 'csv', force: bool = False):
    """
    Merges the final {source}_selected files according to the source-specific 
    mapping schema, generates robust IDs where missing, and normalizes columns.
    """
    gold_dir = BASE_DIR / 'data' / 'gold'
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_output = gold_dir / 'merged_publications.parquet'
    csv_output = gold_dir / 'merged_publications.csv'
    
    outputs_exist = (
        (output_format in ['parquet', 'both'] and parquet_output.exists()) and
        (output_format in ['csv', 'both'] and csv_output.exists())
    )
    if outputs_exist and not force:
        print("[Merge] Merged outputs already exist. Skipping merge. Use --force or --force-merge to overwrite.")
        return

    merged_dfs = []

    for source in sources:
        config = SOURCES_CONFIG.get(source)
        if not config:
            continue
        
        selected_path = config['selected_file']
        if not selected_path.exists():
            print(f"[Merge] Selected file not found for {source} at: {selected_path}. Skipping.")
            continue

        print(f"[Merge] Standardizing and processing {source}...")
        df = pd.read_parquet(selected_path)
        
        if df.empty:
            print(f"[Merge] Selected data for {source} is empty. Skipping.")
            continue

        processed_df = pd.DataFrame()

        # --- Source-Specific Column Mapping Logic ---
        if source == 'UG':
            if '_id' in df.columns:
                processed_df['id'] = df['_id'].astype(str)
            else:
                processed_df['id'] = df.apply(lambda r: _generate_robust_id(source, r), axis=1)

            processed_df['source'] = 'UG'

            if 'keyword' in df.columns:
                processed_df['keywords'] = df['keyword'].apply(_parse_list_keywords)
            else:
                processed_df['keywords'] = [[] for _ in range(len(df))]

        elif source in ['HBO', 'SB']:
            processed_df['id'] = df.apply(lambda r: _generate_robust_id(source, r), axis=1)
            processed_df['source'] = source

            if 'keywords' in df.columns:
                processed_df['keywords'] = df['keywords'].apply(_parse_semicolon_keywords)
            else:
                processed_df['keywords'] = [[] for _ in range(len(df))]

        # --- Shared Column Standardization ---
        if 'year' in df.columns:
            processed_df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        else:
            processed_df['year'] = pd.Series([None] * len(df), dtype='Int64')

        # Simply grab processed NLP results directly from the Silver Selection files
        processed_df['abstract'] = df['text_dut'] if 'text_dut' in df.columns else None
        processed_df['abstract_sentence'] = df['sent_dut'] if 'sent_dut' in df.columns else None

        # Enforce target gold schema order
        target_cols = ['id', 'source', 'keywords', 'year', 'abstract', 'abstract_sentence']
        
        for col in target_cols:
            if col not in processed_df.columns:
                if col in ['keywords', 'abstract_sentence']:
                    processed_df[col] = [[] for _ in range(len(processed_df))]
                else:
                    processed_df[col] = None

        processed_df = processed_df[target_cols]
        merged_dfs.append(processed_df)

    if not merged_dfs:
        print("[Merge] No data sources were successfully compiled. Skipping merge output.")
        return

    final_df = pd.concat(merged_dfs, ignore_index=True)

    # --- Write Outputs ---
    if output_format in ['parquet', 'both']:
        print(f"[Merge] Saving Parquet merged data to: {parquet_output}")
        final_df.to_parquet(parquet_output, index=False)

    if output_format in ['csv', 'both']:
        print(f"[Merge] Saving CSV merged data to: {csv_output}")
        csv_df = final_df.copy()
        if 'keywords' in csv_df.columns:
            csv_df['keywords'] = csv_df['keywords'].apply(lambda x: ';'.join(x) if isinstance(x, list) else x)
        if 'abstract_sentence' in csv_df.columns:
            csv_df['abstract_sentence'] = csv_df['abstract_sentence'].apply(lambda x: ' | '.join(x) if isinstance(x, list) else x)
            
        csv_df.to_csv(csv_output, index=False)

    print("[Merge] Processing completed.")


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
        default=['download', 'clean', 'select', 'merge'],
        choices=['download', 'clean', 'select', 'merge'],
        help="Pipeline steps to execute. Default: all steps (download, clean, select, merge)"
    )
    parser.add_argument('--force', action='store_true', help="Force run all processes (ignores cache)")
    parser.add_argument('--force-download', action='store_true', help="Force run the download step")
    parser.add_argument('--force-clean', action='store_true', help="Force run the cleaning step")
    parser.add_argument('--force-select', action='store_true', help="Force run the selection/filtering step")
    parser.add_argument('--force-merge', action='store_true', help="Force run the merge step")
    parser.add_argument(
        '--output-format',
        type=str,
        default='parquet',
        choices=['parquet', 'csv', 'both'],
        help="Export file format for the merged step. Default: parquet. Options: parquet, csv, both"
    )
    
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

    # 4. Merge Step
    if 'merge' in steps:
        print("\n--- Running final merge step ---")
        merge(
            sources=sources,
            output_format=args.output_format,
            force=args.force or args.force_merge
        )

if __name__ == "__main__":
    main()