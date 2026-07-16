import json
import re
import pandas as pd
from langdetect import detect_langs, DetectorFactory
from pathlib import Path

# Set seed for reproducible language detection results
DetectorFactory.seed = 0

def is_dutch_text(text, threshold=0.85, min_char_length=30):
    """
    Detects if a text segment contains Dutch.
    Splits text to identify Dutch if mixed with other languages.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    
    # Split by paragraphs or sentence boundaries to find mixed languages
    segments = [s.strip() for s in re.split(r'\n+|\.\s+', text) if len(s.strip()) > min_char_length]
    
    if not segments and len(text.strip()) >= 10:
        segments = [text.strip()]

    for segment in segments:
        try:
            predictions = detect_langs(segment)
            for pred in predictions:
                if pred.lang == 'nl' and pred.prob >= threshold:
                    return True
        except Exception:
            continue
            
    return False

def analyze_dataset_filtered(file_path):
    # Load dataset
    try:
        df = pd.read_json(file_path, lines=True)
    except ValueError:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    total_unfiltered = len(df)
    
    # 1. Filter dataset by year (1980 to 2022 inclusive)
    if 'year' in df.columns:
        # Convert to numeric, turning non-numeric values to NaN safely
        df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')
        filtered_df = df[(df['year_numeric'] >= 1980) & (df['year_numeric'] <= 2022)].copy()
    else:
        print("Warning: 'year' column not found. Running analysis on the entire dataset.")
        filtered_df = df.copy()

    total_records = len(filtered_df)
    
    # Expanded list of variations for Dutch language codes/labels
    dutch_codes = {
        'nl', 'dut', 'nld', 'dutch', 'nederlands', 'nederlandse', 
        'flemish', 'vlaams', 'vlaamse', 'flamand'
    }

    stats = {
        'metadata_nl': 0,               # 'language' column matches Dutch codes
        'abstract_detected_nl': 0,      # 'abstract' column contains Dutch text
        'abstract_full_meta_nl': 0,     # 'abstract_full' struct has lang matching Dutch codes
        'abstract_full_detected_nl': 0, # 'abstract_full' text contains Dutch
        'any_dutch_found': 0            # Record has Dutch in any of the checked areas
    }

    # List to store examples of detected abstracts
    dutch_abstract_examples = []

    for idx, row in filtered_df.iterrows():
        has_dutch_metadata = False
        has_dutch_in_abstract = False
        has_dutch_in_full_meta = False
        has_dutch_in_full_detected = False

        # 1. Check 'language' column
        lang_meta = str(row.get('language', '')).strip().lower()
        if lang_meta in dutch_codes:
            has_dutch_metadata = True
            stats['metadata_nl'] += 1

        # 2. Check 'abstract' column (string)
        abstract_str = row.get('abstract', '')
        if isinstance(abstract_str, str) and abstract_str:
            if is_dutch_text(abstract_str):
                has_dutch_in_abstract = True
                stats['abstract_detected_nl'] += 1
                
                # Keep a few examples for the final print report
                if len(dutch_abstract_examples) < 3:
                    dutch_abstract_examples.append({
                        'year': row.get('year'),
                        'metadata_lang': lang_meta,
                        'abstract_text': abstract_str
                    })

        # 3. Check 'abstract_full' column
        abstract_full = row.get('abstract_full')
        if isinstance(abstract_full, list):
            for item in abstract_full:
                if not isinstance(item, dict):
                    continue
                
                item_lang = str(item.get('lang', '')).strip().lower()
                if item_lang in dutch_codes:
                    has_dutch_in_full_meta = True
                
                item_text = item.get('text', '')
                if item_text and is_dutch_text(item_text):
                    has_dutch_in_full_detected = True

        if has_dutch_in_full_meta:
            stats['abstract_full_meta_nl'] += 1
        if has_dutch_in_full_detected:
            stats['abstract_full_detected_nl'] += 1

        if any([has_dutch_metadata, has_dutch_in_abstract, has_dutch_in_full_meta, has_dutch_in_full_detected]):
            stats['any_dutch_found'] += 1

    # Print Report
    print("\n==================================================")
    print("      FILTERED DUTCH ABSTRACT REPORT (1980-2022)  ")
    print("==================================================")
    print(f"Total records in dataset:        {total_unfiltered}")
    print(f"Records in subset (1980-2022):   {total_records}\n")
    
    if total_records == 0:
        print("No records found in the specified year range.")
        return

    print("1. Record Metadata ('language' column):")
    print(f"   - Labeled Dutch:            {stats['metadata_nl']} ({stats['metadata_nl']/total_records*100:.2f}%)")
    print(f"     (Checked: {', '.join(sorted(list(dutch_codes)))})")
    
    print("\n2. 'abstract' Column (Plain String):")
    print(f"   - Detected Dutch text:      {stats['abstract_detected_nl']} ({stats['abstract_detected_nl']/total_records*100:.2f}%)")
    
    print("\n3. 'abstract_full' Column (Structured List):")
    print(f"   - Labeled Dutch ('lang'):   {stats['abstract_full_meta_nl']} ({stats['abstract_full_meta_nl']/total_records*100:.2f}%)")
    print(f"   - Detected Dutch text:      {stats['abstract_full_detected_nl']} ({stats['abstract_full_detected_nl']/total_records*100:.2f}%)")
    
    print("\n4. Overall Summary (Within year subset):")
    print(f"   - Total records with any Dutch trace: {stats['any_dutch_found']} ({stats['any_dutch_found']/total_records*100:.2f}%)")
    print("==================================================")

    # Print Examples
    print("\n==================================================")
    print("       EXAMPLES OF DETECTED DUTCH ABSTRACTS       ")
    print("==================================================")
    if not dutch_abstract_examples:
        print("No Dutch abstracts were detected to display as examples.")
    else:
        for i, example in enumerate(dutch_abstract_examples, 1):
            text = example['abstract_text']
            # Limit printed abstract size to 400 characters for readability
            truncated_text = text if len(text) < 400 else text[:400] + "... [TRUNCATED]"
            print(f"Example {i} | Year: {example['year']} | Metadata Lang Value: '{example['metadata_lang']}'")
            print(f"Text:\n{truncated_text}")
            print("-" * 50)
    print("==================================================")

# To run:


analyze_dataset_filtered(Path('/home/gderijck/internship/data/bronze/UG/publications.json'))