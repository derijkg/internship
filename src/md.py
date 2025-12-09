import re
import argparse
import zipfile
import json
from pathlib import Path
from langdetect import detect, LangDetectException
import tempfile

'''
--

'''




class MDParser:
    def __init__(
            self,
            path,

        ):

        # patterns
        self.abstract_keywords = {
            'abstract',
            'summary',
            'samenvatting',
            'résumé',
            'resume',
            'zusammenfassung',
            'overview',
            'synopsis'
        }
        self.header_pattern = re.compile(r'^#{1,6}\s')
        self.html_pattern = re.compile()
        self.level_pattern = re.compile()

        self.paths_to_process = self._process_path(path)

        # operation
        result = []
        for p in self.paths_to_process:
            content = self._read_md(p)
            result_single = self.process_single(content)
            result.append()
        return result

    
    def _process_path(self, input_path):
        path = Path(input_path)
        paths_to_process = set()

        # We match the path object, using guards to check types/extensions
        match path:
            
            # Case 1: Valid Markdown file
            case p if p.is_file() and p.suffix == '.md':
                paths_to_process.add(p)

            # Case 2: Zip File
            case p if p.is_file() and p.suffix == '.zip':
                temp_dir = Path(tempfile.mkdtemp(prefix="markdown_extract_"))
                
                try:
                    with zipfile.ZipFile(p, 'r') as z:
                        z.extractall(temp_dir)
                    paths_to_process.update(temp_dir.rglob("*.md"))
                except zipfile.BadZipFile:
                    raise ValueError(f"File is not a valid zip: {p}")

            # Case 3: Directory
            case p if p.is_dir():
                paths_to_process.update(p.rglob("*.md"))

            # Case 4: Invalid input (Catch-all)
            case _:
                if not path.exists():
                    raise FileNotFoundError(f"Path does not exist: {path}")
                raise ValueError(f"Unsupported input type (must be .md, .zip, or dir): {path}")

        if not paths_to_process:
            raise ValueError(f"No markdown files found in: {path}")

        return paths_to_process
    def _read_md(self, path): #load md file
        content = path.read_text(encoding='utf-8', errors='replace')
        return content
    def _process_header(
            self,
            clean_block,
        ):

        match = re.match(self.header_pattern, clean_block)
        if match:
            hashes = match.group()
            level = len(hashes)
            header_text = clean_block.lstrip('#').strip()

            return (header_text, level)
        else:
            raise ValueError('clean block started with # but no re match')

    def clean(
            self,
            block
        ):
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

    def process_single(
            self,
            md_content
        ):

        raw_blocks = md_content.split('\n\n')

        in_abstract_section = False
        
        # 2. Iterate Blocks
        text = {
            'title':title,
            'paragraphs': [paragraphs],

        }
        for i, block in enumerate(raw_blocks):
            clean_block = self.clean(block) # edit
            if not clean_block:
                continue
            
            is_header = block.strip().startswith('#')
            if is_header:
                # need to check how many we have and store count
                header_text, level = self._process_header(clean_block)
                header_text = block.lstrip('#').strip().lower()
                
                if any(keyword in header_text for keyword in self.abstract_keywords):
                    in_abstract_section = True
                else:
                    # continue searching for other lang abstracts
                    # if i > len(raw_blocks)/2: continue # if its beyond the half way point its not an abstract / summary..., maybe its a conclusion
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
        result = {
            'id': id,
            'page_count': page_count,
            'chapter_count': chapter_count,
            'text': text
        }
        return result
    
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

    def split_headers(
            
        ):
        return (title, count, text)
    
    def split_paragraphs():
        pass
    
    def check_nested_headers(
            header_block1,
            header_block2
        ):
        return True
        return False
    
    def check_nested_headers_doc():
        return True
        return False
    
    def remove_html():
        pass

    def page_count(): # <span 
        return page_count

    def chapter_count():
        return chapter_count









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