import re
import argparse
import zipfile
import json
from pathlib import Path
from langdetect import detect, LangDetectException
import tempfile






class MDParser:
    def __init__(self, path=None, keep_footnotes=False, keep_tables=False, debug=True):

        # patterns
        self.abstract_keywords = {
            'abstract',
            'inleiding',
            'introduction',
            'summary',
            'samenvatting',
            'résumé',
            'resume',
            'zusammenfassung',
            'overview',
            'synopsis'
        }
        
        # settings
        self.keep_footnotes = keep_footnotes
        self.keep_tables = keep_tables
        self.debug = debug

        # general
        self.header_pattern = re.compile(r'^#{1,6}')
        self.page_tag_pattern = re.compile(r'<span id="page-(\d+)"[^>]*>')
        #self.html_pattern = re.compile()
        #self.level_pattern = re.compile()
        if path: self.path = path

        
    def run(self, path=None):
        if not path and self.path: path = self.path
        # operation
        result = []
        if not path: raise ValueError('provide path at .run or init')
        self.paths_to_process = self._process_path(path)
        for p in self.paths_to_process:
            result_single = self.generate_tree_single(p)
            result.append(result_single)
        return result

    
    def _process_path(self, input_path):
        p = Path(input_path)
        paths_to_process = set()
        if self.debug:
            temp_dir = Path(tempfile.mkdtemp(prefix="markdown_extract_"))
            try:
                with zipfile.ZipFile(p, 'r') as z: #only extract relevant files?
                    md_files = [f for f in z.namelist() if f.endswith('.md')]
                    md_files = md_files[:10]
                    if md_files: z.extractall(path = temp_dir, members = md_files)
                paths_to_process.update(temp_dir.rglob("*.md"))
            except zipfile.BadZipFile:
                raise ValueError(f"File is not a valid zip: {p}")
            return paths_to_process
        
        match p:
            # md file
            case p if p.is_file() and p.suffix == '.md':
                paths_to_process.add(p)

            # zip
            case p if p.is_file() and p.suffix == '.zip':
                temp_dir = Path(tempfile.mkdtemp(prefix="markdown_extract_"))
                try:
                    with zipfile.ZipFile(p, 'r') as z: #only extract relevant files?
                        z.extractall(temp_dir)
                    paths_to_process.update(temp_dir.rglob("*.md"))
                except zipfile.BadZipFile:
                    raise ValueError(f"File is not a valid zip: {p}")

            # dir
            case p if p.is_dir():
                paths_to_process.update(p.rglob("*.md"))

            # invalid
            case _:
                if not p.exists():
                    raise FileNotFoundError(f"Path does not exist: {p}")
                raise ValueError(f"Unsupported input type (must be .md, .zip, or dir): {p}")

        if not paths_to_process:
            raise ValueError(f"No markdown files found in: {p}")
        return paths_to_process
    
    def _process_header(
            self,
            clean_block,
        ):

        match = re.match(self.header_pattern, clean_block)
        if match:
            hashes = match.group()
            level = len(hashes) #supplement with additional logic conc explicit numbering 1 -> 1.1 -> 1.2
            header_text = clean_block.lstrip('#').strip()

            return (header_text, level)
        else:
            raise ValueError('clean block started with # but no re match') # must have been the wind

    #TODO check if complete, as of now doesnt remove #
        # delete text between html tags
    def _clean(self, text): 
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

    def generate_tree_single(self, path):
        # TODO check weird codes \U1223 etcc
        md_content = path.read_text(encoding='utf-8', errors='replace')
        filename = path.stem

        # TODO find number of pages
            # from filename -> filename(_meta).json
            # find "page_id": 4 -> max num or last instance
        
        raw_blocks = md_content.split('\n\n')
        header_blocks = [self._clean(block) for block in raw_blocks if block.strip().startswith('#')]
        abstracts = []



        #TODO if rawblock startswith <sup> its a footnote prob. -> skip? optional
        root = []
        stack = []
        for i, raw_block in enumerate(raw_blocks):
            clean_block = self._clean(raw_block)
            if not clean_block:
                continue

            # HEADER
            if clean_block.strip().startswith('#'):
                header_text, level = self._process_header(clean_block)
                
                new_node = {
                    "header": header_text,
                    "text": [],
                    "level": level,
                    "children": []
                }
                while stack and stack[-1]['level'] >= level:
                    stack.pop()

                if stack:
                    stack[-1]['node']['children'].append(new_node)
                else:
                    root.append(new_node)


                stack.append({'node': new_node, 'level': level})
                # TODO ADD MANUAL CHECK FOR 1.1.2...
            
            # TABLES
            elif clean_block.startswith('|'):
                if self.keep_tables and stack:
                    stack[-1]['node']['text'].append(clean_block)

            #FOOTNOTE
            elif clean_block.startswith('<sup') or clean_block.startswith('['):
                if self.keep_footnotes and stack:
                    stack[-1]['node']['text'].append(clean_block)
            
            # RUNNING TEXT
            else:
                if stack:
                    stack[-1]['node']['text'].append(clean_block)
                # TODO check cases for encountering free text without title
                else: continue

        result = {
            'filename': filename,
            'tree': root
        }
        return result

    ''' abstract logic
                # ABSTRACT HEADER
                # TODO CHECK LANGUAGES OF ALL PARAGRAPHS IN ABS, maybe after making tree structure
                if any(keyword in header_text.lower() for keyword in self.abstract_keywords):
                    in_abstract = True
                    abstract = {
                        'lang': abs_lang,
                        'text': abs_text_paragraphs
                    }
                    abstracts.append(abstract)
                else:
                    # continue searching for other lang abstracts
                    # if i > len(raw_blocks)/2: continue # if its beyond the half way point its not an abstract / summary..., maybe its a conclusion
                
                # We generally don't treat headers as "Paragraphs" of content, 
                # but you can add them if you want structural text.
                    continue
                    

    result logic
    
        # 3. Assemble Results
        full_text_sample = " ".join(paragraphs[:5]) # Use first 5 paragraphs for doc lang
        abstract_text = " ".join(abstract_lines)
        text = {
            'inferred_level': inferred_level,
            'explicit_level': explicit_level,
            'title':title,
            'paragraphs': [paragraphs]
        }

        result = {
            'id': id,
            'page_count': page_count,
            'chapter_count': chapter_count,
            'text': text
        }

    '''



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