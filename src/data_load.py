import os
import zipfile
from io import BytesIO

def load_from_zip(
        path=r'data\scriptiebank\scriptiebank_archive.zip',
):
    if not os.path.exists(path): return None
    all_texts = {}
    with zipfile.ZipFile(path, 'r') as zipf:
        pdf_files = [f for f in zipf.namelist() if f.lower().endswith('.pdf')]
        if not pdf_files: return None
        print(f'Found {len(pdf_files)} pdf files')
        for i, pdf_filename in enumerate(pdf_files):
            print(f'Processing {i+1}/{len(pdf_files)}: {pdf_filename}')
            with zipf.open(pdf_filename) as pdf_file_zip:
                pass
                #process with GROBID
        #word files too
        word_files = [f for f in zipf.namelist() if f.lower().endswith('.docx')]
        if not word_files: return None
