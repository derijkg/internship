import os
import requests
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

# --- Configuration ---
# The URL of your running GROBID service
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
# Define your folder structure and output file
ROOT_DATA_DIR = 'data'  # This folder should contain 'human_written' and 'llm_generated'
OUTPUT_FILE = 'academic_dataset.jsonl'

def parse_grobid_xml(xml_content):
    """
    Parses the TEI XML output from GROBID to extract structured text.

    Args:
        xml_content (str): The XML string returned by GROBID.

    Returns:
        dict: A dictionary containing the title, abstract, and body text.
    """
    # Register the TEI namespace to properly find elements
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    try:
        root = ET.fromstring(xml_content)
        
        # --- Extract Title ---
        title_element = root.find('.//tei:titleStmt/tei:title', ns)
        title = title_element.text.strip() if title_element is not None else ""
        
        # --- Extract Abstract ---
        abstract_element = root.find('.//tei:profileDesc/tei:abstract/tei:p', ns)
        abstract = abstract_element.text.strip() if abstract_element is not None else ""
        
        # --- Extract Body Text ---
        # Concatenate all paragraphs <p> within the <body> section
        body_paragraphs = root.findall('.//tei:body//tei:p', ns)
        body_text = "\n\n".join([p.text for p in body_paragraphs if p.text])
        
        return {
            'title': title,
            'abstract': abstract,
            'body_text': body_text
        }
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None

def process_pdf_corpus_with_grobid(root_dir, output_path):
    """
    Processes all PDFs in a directory structure using GROBID and saves
    the structured output to a JSONL file.
    """
    label_map = {
        'human_written': 0,
        'llm_generated': 1
    }

    # Open the output file in append mode ('a')
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
                        # Send the PDF to the GROBID API
                        response = requests.post(
                            GROBID_URL,
                            files={'inputFile': pdf_file},
                            timeout=30 # Add a timeout
                        )
                    
                    if response.status_code == 200:
                        # Parse the XML response
                        structured_data = parse_grobid_xml(response.text)
                        
                        if structured_data and structured_data['body_text']:
                            # Create the JSON object for this document
                            record = {
                                'source_file': filename,
                                'label': label,
                                'title': structured_data['title'],
                                'abstract': structured_data['abstract'],
                                'text': structured_data['body_text'] # The main content for your detector
                            }
                            
                            # Write the JSON object as a single line in the output file
                            f_out.write(json.dumps(record) + '\n')
                        else:
                            print(f"Warning: Could not extract body text from {filename}")

                    else:
                        print(f"Error processing {filename}. Status: {response.status_code}, Body: {response.text[:100]}")

                except requests.exceptions.RequestException as e:
                    print(f"API request failed for {filename}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred with {filename}: {e}")

    print(f"\nProcessing complete. Dataset appended to {output_path}")

# --- Run the Script ---
if __name__ == '__main__':
    # Important: Clear the file if you want to start fresh, or comment this out to append
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    process_pdf_corpus_with_grobid(ROOT_DATA_DIR, OUTPUT_FILE)