import os
import subprocess
import zipfile
import shutil
import sys
from pathlib import Path

# --- Configuration ---
ZIP_INPUT = Path("data/archive.zip")

TEMP_OUTPUT = Path("data/temp/marker_output")
TEMP_INPUT = Path("data/temp/temp_extraction")

ZIP_OUTPUT = Path("data/output_marker.zip")

# my specifically problematic files
IGNORE_LIST = ['1058',
 '1191',
 '1215',
 '1374',
 '1445',
 '1524',
 '1592',
 '1595',
 '1821',
 '1846',
 '2070',
 '2188',
 '2206',
 '2210',
 '2242',
 '241',
 '2509',
 '2533',
 '2600',
 '2696',
 '3409',
 '3477',
 '3806',
 '3831',
 '3943',
 '446',
 '533',
 '580',
 '662',
 '689',
 '691',
 '692',
 '820',
 '841'] # constant failures

# MARKER CONFIG
workers = '1' # currently unused, see command = [...]
detection_batch_size = '14'
layout_batch_size = '14'
pdftext_workers = '12'
# ---------------------

def zip_output_files(source_dir, zip_path):
    """
    Appends new files from source_dir to zip_path.
    """
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    
    print(f"\nUpdating zip archive at: {zip_path}")

    # Open in append mode 'a'
    with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
        existing_files = set(zipf.namelist())
        new_files_count = 0

        for item in source_dir.iterdir():
            if item.is_dir():
                if item.name in IGNORE_LIST:
                    continue
                for file_path in item.rglob('*'):
                    if file_path.is_file():
                        # Create arcname (e.g. "document_name/document_name.md")
                        arcname = f"{item.name}{file_path.suffix}"
                        
                        if arcname not in existing_files:
                            zipf.write(file_path, arcname=arcname)
                            existing_files.add(arcname)
                            new_files_count += 1

    print(f"Successfully added {new_files_count} new files (Total: {len(existing_files)}).")

def get_processed_stems(output_dir):
    """
    Returns a set of folder names (stems) found in the output directory.
    """
    if not output_dir.exists():
        return set()
    # Marker creates a folder for every processed file. We look for those folders.
    return {item.name for item in output_dir.iterdir() if item.is_dir()}

def main():
    # --- Step 1: Setup & Scan ---
    if not os.path.exists(ZIP_INPUT):
        print(f"Error: The file '{ZIP_INPUT}' was not found.")
        sys.exit(1)

    os.makedirs(TEMP_OUTPUT, exist_ok=True)
    
    print(f"Checking processed status in {TEMP_OUTPUT}...")
    processed_stems = get_processed_stems(TEMP_OUTPUT)
    print(f"Found {len(processed_stems)} previously processed items.")

    # Identify files to process without extracting yet
    files_to_process = []
    
    with zipfile.ZipFile(ZIP_INPUT, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue
            
            p = Path(file_info.filename)
            
            # Ignore hidden files
            if p.name.startswith('.'):
                continue
            
            if p.stem in IGNORE_LIST:
                continue

            # Check if the stem exists as a folder in output
            if p.stem not in processed_stems:
                files_to_process.append(file_info)

    count = len(files_to_process)

    # --- Step 2: Prompt User ---
    if count == 0:
        print("\nNo new files found to process.")
    else:
        print(f"\nFound {count} new files in the zip that have not been processed.")
        user_input = input("Do you want to proceed with extraction and Marker processing? (y/n): ").strip().lower()
        
        if user_input != 'y':
            print("Operation cancelled by user. Exiting.")
            sys.exit(0)

        # --- Step 3: Extraction ---
        if TEMP_INPUT.exists():
            shutil.rmtree(TEMP_INPUT)
        os.makedirs(TEMP_INPUT)

        print(f"\nExtracting {count} files to {TEMP_INPUT}...")
        with zipfile.ZipFile(ZIP_INPUT, 'r') as zip_ref:
            for file_info in files_to_process:
                zip_ref.extract(file_info, TEMP_INPUT)

        # --- Step 4: Run Marker ---
        try:
            command = [
                "marker",
                #'--debug_print',
                #'--workers','1',
                '--disable_image_extraction',
                '--detection_batch_size', detection_batch_size, 
                '--layout_batch_size', layout_batch_size,
                '--pdftext_workers', pdftext_workers,
                '--output_dir', str(TEMP_OUTPUT),
                str(TEMP_INPUT)
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0" 
            env['TORCH_DEVICE'] = 'cuda'

            print("\nStarting Marker conversion...")
            subprocess.run(command, check=True, env=env)
            print("\nMarker conversion completed.")

        except FileNotFoundError:
            print("\nError: 'marker' command not found.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"\nError: Marker process failed with exit code {e.returncode}.")
            sys.exit(1)

        # --- Step 5: Cleanup Input ---
        if TEMP_INPUT.exists():
            shutil.rmtree(TEMP_INPUT)
            print("Cleaned up temporary input files.")

    # --- Step 6: Zip Results ---
    zip_output_files(TEMP_OUTPUT, ZIP_OUTPUT)

if __name__ == "__main__":
    main()