
'''
import os
import requests
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

# --- Configuration ---
#GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
#ROOT_DATA_DIR = 'data'
#OUTPUT_FILE = 'academic_dataset_by_chapter.jsonl' # New output file name



def parse_xml_grobid(xml_content):
    """
    Parses the TEI XML from GROBID to extract paper-level metadata and
    a structured list of all sections (chapters).

    Args:
        xml_content (str): The XML string returned by GROBID.

    Returns:
        dict: A dictionary containing paper title, abstract, and a list of sections.
              Each section is a dict with 'section_number', 'section_title', and 'section_text'.
    """
    # Register the TEI namespace to properly find elements
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    try:
        root = ET.fromstring(xml_content)
        
        # --- Extract Paper-level Metadata ---
        title_element = root.find('.//tei:titleStmt/tei:title', ns)
        paper_title = title_element.text.strip() if title_element is not None else ""
        
        abstract_element = root.find('.//tei:profileDesc/tei:abstract/tei:p', ns)
        abstract = abstract_element.text.strip() if abstract_element is not None else ""
        
        # --- Extract Sections from the Body ---
        sections = []
        # Find all <div> elements within the <body> that represent sections
        body = root.find('.//tei:body', ns)
        if body is None:
            return None # No body content found

        # We look for <div> elements that have a <head> child, which signifies a titled section
        for div in body.findall('.//tei:div[tei:head]', ns):
            head = div.find('./tei:head', ns)
            if head is None or head.text is None:
                continue

            section_title = head.text.strip()
            section_number = head.get('n', '') # Get the section number attribute 'n' if it exists

            # Find all paragraphs <p> within this specific section <div>
            # We use .// to find them even if nested in figures, etc.
            paragraphs = div.findall('.//tei:p', ns)
            section_text = "\n\n".join([p.text.strip() for p in paragraphs if p.text is not None])
            
            if section_text: # Only add sections that have text content
                sections.append({
                    'section_number': section_number,
                    'section_title': section_title,
                    'section_text': section_text
                })
        
        return {
            'paper_title': paper_title,
            'abstract': abstract,
            'sections': sections
        }
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None

def parse_pdf_grobid(root_dir, output_path):
    """
    Processes PDFs using GROBID, extracts text by chapter/section,
    and saves the structured output to a JSONL file.
    """
    label_map = {'human_written': 0, 'llm_generated': 1}

    with open(output_path, 'a', encoding='utf-8') as f_out:
        for class_name, label in tqdm(label_map.items(), desc="Processing classes"):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found for class '{class_name}': {class_dir}")
                continue

            pdf_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.pdf')]
            for filename in tqdm(pdf_files, desc=f"Processing {class_name}", leave=False):
                pdf_path = os.path.join(class_dir, filename)
                
                try:
                    with open(pdf_path, 'rb') as pdf_file:
                        response = requests.post(GROBID_URL, files={'inputFile': pdf_file}, timeout=60)
                    
                    if response.status_code == 200:
                        structured_data = parse_xml_grobid(response.text)
                        
                        if structured_data and structured_data['sections']:
                            record = {
                                'source_file': filename,
                                'label': label,
                                'paper_title': structured_data['paper_title'],
                                'abstract': structured_data['abstract'],
                                'sections': structured_data['sections'] # The list of sections
                            }
                            f_out.write(json.dumps(record) + '\n')
                        else:
                            print(f"Warning: Could not extract sections from {filename}")
                    else:
                        print(f"Error processing {filename}. Status: {response.status_code}, Body: {response.text[:100]}")
                except requests.exceptions.RequestException as e:
                    print(f"API request failed for {filename}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred with {filename}: {e}")

    print(f"\nProcessing complete. Dataset appended to {output_path}")

if __name__ == '__main__':
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    # Make sure your GROBID Docker container is running!
    parse_pdf_grobid(ROOT_DATA_DIR, OUTPUT_FILE)
'''

''' using pypdfium2 to transform pdf to bitmap and nougat small to extract text and layout into markdown'''
# if it goes wrong:
# increase scale (resolution of bitmap)
# use bigger model (nougat-base) or other model like microsoft/layoutlmv3-base
# use batch processing: rewrite extract_text_from_images (batch_size = 4)


import pypdfium2 as pdfium
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
import re
from pathlib import Path

# --- Part 1: PDF to Image Conversion ---
def convert_pdf_to_images(pdf_path):
    """Converts each page of a PDF into a PIL Image."""
    images = []
    pdf = pdfium.PdfDocument(pdf_path)
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        bitmap = page.render(scale=2) #scale 3 or higher for better resolution
        pil_image = bitmap.to_pil()
        images.append(pil_image)
    return images

# --- Part 2: Model Inference (Image to Markdown) ---
print("Loading Nougat model and processor...")
processor = NougatProcessor.from_pretrained("facebook/nougat-small", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# batch processing
'''def extract_text_from_images(images, batch_size=4): # Added batch_size parameter
    """Uses Nougat to extract structured text (Markdown) from a list of images."""
    all_markdown = ""
    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        print(f"Processing pages {i+1}-{i+len(batch_images)}/{len(images)}...")

        # The processor can handle a list of images directly
        pixel_values = processor(batch_images, return_tensors="pt").pixel_values
        
        normalized_pixel_values = pixel_values.to(torch.float32) / 255.0

        outputs = model.generate(
            normalized_pixel_values.to(device),
            min_length=1,
            max_new_tokens=3584,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )
        
        # Decode each sequence in the batch
        sequences = processor.batch_decode(outputs, skip_special_tokens=True)
        for sequence in sequences:
            processed_sequence = processor.post_process_generation(sequence, fix_markdown=True)
            all_markdown += processed_sequence + "\n"
            
    return all_markdown'''

def extract_text_from_images(images):
    """Uses Nougat to extract structured text (Markdown) from a list of images."""
    all_markdown = ""
    for i, image in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}...")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        normalized_pixel_values = pixel_values.to(torch.float32)/255.0
        outputs = model.generate(
            normalized_pixel_values.to(device),
            min_length=1,
            max_new_tokens=3584,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )
        sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        sequence = processor.post_process_generation(sequence, fix_markdown=True)
        all_markdown += sequence + "\n"
    return all_markdown

# --- Part 3: Parse Markdown to Extract Chapters ---
def extract_chapters_from_markdown(markdown_text):
    """Parses a markdown string to extract chapters based on Level 1 headings."""
    chapters = {}
    current_chapter_title = "Introduction"
    chapter_heading_pattern = re.compile(r"^\s*#\s+(.*)", re.MULTILINE)
    parts = chapter_heading_pattern.split(markdown_text)
    
    if parts:
        intro_content = parts[0].strip()
        if intro_content:
            chapters[current_chapter_title] = intro_content
    
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i+1].strip()
        chapters[title] = content
        
    return chapters


if __name__ == '__main__':
    path_pdf = Path('data/test/test_nl.pdf')
    path_output = Path('data/test/output.md')
    path_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting '{path_pdf}' to images...")
    page_images = convert_pdf_to_images(path_pdf)
    print("Extracting text with Nougat...")
    full_markdown_content = extract_text_from_images(page_images)

    with open(path_output, "w", encoding="utf-8") as f:
        f.write(full_markdown_content)
    print("\nFull markdown content saved to 'output.md'")
    
    print("Parsing markdown to extract chapters...")
    extracted_chapters = extract_chapters_from_markdown(full_markdown_content)
    
    print("\n--- Extracted Chapters ---")
    for title, content in extracted_chapters.items():
        print(f"\n✅ CHAPTER: {title}")
        print(content[:300] + "...")
