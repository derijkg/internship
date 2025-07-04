import os
import requests
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

# --- Configuration ---
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
ROOT_DATA_DIR = 'data'
OUTPUT_FILE = 'academic_dataset_by_chapter.jsonl' # New output file name

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

# --- Run the Script ---
if __name__ == '__main__':
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    # Make sure your GROBID Docker container is running!
    parse_pdf_grobid(ROOT_DATA_DIR, OUTPUT_FILE)