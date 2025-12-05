import re
import argparse
import zipfile
import json
from pathlib import Path
from langdetect import detect, LangDetectException

# --- CONSTANTS ---
# Keywords to identify an abstract section (multilingual support based on your data)
ABSTRACT_KEYWORDS = {
    'abstract', 'summary', 'samenvatting', 'résumé', 'resume', 
    'zusammenfassung', 'overview', 'synopsis'
}

def clean_marker_markdown(text):
    """
    Cleans Marker-specific artifacts and Markdown syntax from a text block.
    """
    # 1. Remove HTML tags (Marker puts <span id="page-x"> tags everywhere)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Remove Markdown bold/italic (**text**, *text*)
    text = re.sub(r'\*\*|__', '', text) # Remove bold markers
    text = re.sub(r'\*', '', text)      # Remove italic markers
    
    # 3. Remove Links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 4. Remove Images ![]()
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 5. Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_language_safe(text):
    """
    Robust wrapper for language detection. 
    Returns 'unknown' if text is too short or detection fails.
    """
    if not text or len(text.strip()) < 10:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def parse_markdown_content(md_content):
    """
    Parses raw Markdown content into:
    - Language (Overall)
    - Abstract (Text + Language)
    - Body Paragraphs (List of strings)
    """
    
    # 1. Split into Blocks (Double newline is the standard MD paragraph separator)
    #    Marker adheres to this strictly.
    raw_blocks = md_content.split('\n\n')
    
    paragraphs = []
    abstract_lines = []
    
    # State tracking
    in_abstract_section = False
    
    # 2. Iterate Blocks
    for block in raw_blocks:
        clean_block = clean_marker_markdown(block)
        
        # Skip empty blocks
        if not clean_block:
            continue
            
        # Check for Headers
        # Marker headers start with #. 
        # Note: We check the RAW block for '#' because clean_marker_markdown strips formatting, 
        # but usually we want to know if it was a header.
        is_header = block.strip().startswith('#')
        
        if is_header:
            # Check if this header starts the Abstract
            # Remove '#' and space, convert to lower for matching
            header_text = block.lstrip('#').strip().lower()
            
            # Simple keyword match
            if any(keyword in header_text for keyword in ABSTRACT_KEYWORDS):
                in_abstract_section = True
            else:
                # If we hit a new header and were in abstract, we are done with abstract
                if in_abstract_section:
                    in_abstract_section = False
            
            # We generally don't treat headers as "Paragraphs" of content, 
            # but you can add them if you want structural text.
            continue
            
        # Detect Tables (Marker uses pipes | for tables)
        if block.strip().startswith('|'):
            continue # Skip tables for text extraction
            
        # Store Content
        if in_abstract_section:
            abstract_lines.append(clean_block)
        else:
            paragraphs.append(clean_block)

    # 3. Assemble Results
    full_text_sample = " ".join(paragraphs[:5]) # Use first 5 paragraphs for doc lang
    abstract_text = " ".join(abstract_lines)
    
    return {
        "doc_language": detect_language_safe(full_text_sample),
        "abstract": abstract_text,
        "abstract_language": detect_language_safe(abstract_text),
        "paragraphs": paragraphs
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, default='data/output_marker.zip')
    # Change type=bool to action='store_true' for correct flag behavior
    parser.add_argument("--test", action='store_true', help="Run on a subset of files") 
    args = parser.parse_args()

    results = {}
    
    if not Path(args.dir).exists():
        print(f"Error: Zip file {args.dir} not found.")
        return

    print(f"Reading from {args.dir}...")

    with zipfile.ZipFile(args.dir, 'r') as z:
        # Get list of .md files
        md_files = [f for f in z.namelist() if f.endswith('.md')]
        
        if args.test:
            print("Running in TEST mode (processing first 10 files)")
            md_files = md_files[:10]
        
        for md_filename in md_files:
            # file ID is usually the filename stem (e.g., 457.md -> 457)
            file_id = Path(md_filename).stem
            
            # Read content
            with z.open(md_filename) as f:
                content = f.read().decode('utf-8')
            
            # Process
            data = parse_markdown_content(content)
            
            # Store Result
            results[file_id] = data
            
            # Print specific output as requested
            print(f"ID: {file_id} | Doc Lang: {data['doc_language']} | Abs Lang: {data['abstract_language']}")
            if data['abstract']:
                print(f"   -> Abstract found ({len(data['abstract'])} chars)")
            else:
                print(f"   -> No abstract found.")

    # Optional: Save results to json
    with open("extraction_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()