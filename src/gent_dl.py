import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from mu import timed_request

# --- Configuration ---
# 1. Path to your JSON data dump file.
#    (Replace 'biblio_data.json' with the actual path to your file)
JSON_FILE_PATH = Path('data/ugent_datadump/publications.json')

# 2. Directory where you want to save the downloaded files.
DOWNLOAD_DIR = Path('data/ugent_datadump/ugent_dl')

# 3. List of specific publication IDs to download. 
#    - Leave this list EMPTY to process ALL publications in the JSON file.
#    - Example: TARGET_IDS = ['8627198', '8511993']
TARGET_IDS = []

# --- Main Script ---

# selection
pass


# exec
def download_files_from_json():
    """
    Parses the JSON data dump and downloads open access files.
    """

    # Create the download directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created download directory: '{DOWNLOAD_DIR}'")

    # Load the JSON data from the file
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            publications = json.load(f)
        print(f"Successfully loaded {len(publications)} publications from '{JSON_FILE_PATH}'.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not read or parse the JSON file. {e}")
        return

    # Create a set for faster lookups if a target list is provided
    target_id_set = set(TARGET_IDS)
    processed_count = 0

    # Iterate through each publication in the loaded data
    for pub in publications:
        pub_id = pub.get('_id')
        if not pub_id:
            continue # Skip records without an ID

        # If a target list is specified, skip publications not in the list
        if target_id_set and pub_id not in target_id_set:
            continue
        
        processed_count += 1
        print(f"\nProcessing publication ID: {pub_id}")

        # Check for an open access file
        open_access_file_info = None
        if 'file' in pub and isinstance(pub['file'], list):
            for file_info in pub['file']:
                if file_info.get('access') == 'open':
                    open_access_file_info = file_info
                    break # Found the first open access file, stop searching

        if open_access_file_info:
            file_id = open_access_file_info.get('_id')
            file_name = open_access_file_info.get('name', '')
            
            if not file_id:
                print(f"  -> Found an open access file entry, but it's missing a file ID.")
                continue

            # Construct the download URL
            download_url = f"https://biblio.ugent.be/publication/{pub_id}/file/{file_id}"
            
            # Determine the file extension to save the file correctly
            _, file_extension = os.path.splitext(file_name)
            save_path = os.path.join(DOWNLOAD_DIR, f"{pub_id}{file_extension or '.pdf'}")

            print(f"  -> Found open access file '{file_name}'.")
            print(f"  -> Downloading from: {download_url}")

            # Download the file
            try:
                with urllib.request.urlopen(download_url) as response:
                    if response.status == 200:
                        with open(save_path, 'wb') as f_out:
                            f_out.write(response.read())
                        print(f"  -> Successfully saved to '{save_path}'")
                    else:
                        print(f"  -> Failed to download. Status code: {response.status}")
            except urllib.error.URLError as e:
                print(f"  -> An error occurred during download: {e.reason}")
        else:
            print("  -> No open access file found for this publication.")

    print(f"\nDownload process finished. Processed {processed_count} publications.")

# --- Run the script ---
if __name__ == "__main__":
    download_files_from_json()