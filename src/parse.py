import os
import subprocess
import zipfile
import shutil
import sys
from pathlib import Path
import torch

# --- Configuration ---
ZIP_FILE_PATH = Path("data/archive.zip")
OUTPUT_ZIP = Path("data/output_md.zip")

TEMP_MD = Path("data/temp/marker_output")
TEMP_EXTRACTION_DIR = Path("data/temp/temp_pdf_extraction")

LIMIT_FILES_TO = 5
'''
MARKER_CONFIG = {
    "extract_images": False,
    "output_formats": ['markdown','json'],
    #"output_dir": 
}
'''
# ---------------------

def zip_output_files(source_dir, zip_path):
    """
    Finds all .md and .json files in the source_dir and adds them to a zip archive.

    Args:
        source_dir (Path): The directory containing the output files.
        zip_path (Path): The path for the final output zip file.
    """
    print(f"\nCreating zip archive at: {zip_path}")
    found_files = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Use pathlib's rglob for a cleaner way to find files
        for file_path in source_dir.rglob("*.md"):
            zipf.write(file_path, arcname=file_path.name)
            found_files += 1
        for file_path in source_dir.rglob("*.json"):
            zipf.write(file_path, arcname=file_path.name)
            found_files += 1
            
    print(f"Successfully added {found_files} markdown and json files to the zip archive.")


def main():
    """
    Main function to orchestrate the PDF processing and zipping workflow.
    """
    # --- Step 1: Validate Inputs and Set Up Directories ---
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"Error: The file '{ZIP_FILE_PATH}' was not found.")
        sys.exit(1)

    # Clean up previous runs to ensure a fresh start
    for path in [TEMP_MD, TEMP_EXTRACTION_DIR]:
        if os.path.exists(path):
            print(f"Warning: Directory '{path}' already exists. Cleaning it up.")
            shutil.rmtree(path)
    
    os.makedirs(TEMP_MD)
    os.makedirs(TEMP_EXTRACTION_DIR)
    print("Setup complete. Directories are ready.")

    try:
        # --- Step 2: Extract a Subset or All of the PDF Files ---
        print(f"Opening '{ZIP_FILE_PATH}' to find PDFs...")
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            # Get a list of all PDF files, ignoring other file types
            pdf_files_in_zip = [f for f in zip_ref.namelist() if f.lower().endswith(".pdf")]

            if LIMIT_FILES_TO and LIMIT_FILES_TO > 0:
                print(f"LIMIT_FILES_TO is set to {LIMIT_FILES_TO}. Extracting a subset of files.")
                files_to_extract = pdf_files_in_zip[:LIMIT_FILES_TO]
            else:
                print("Processing all PDF files in the archive.")
                files_to_extract = pdf_files_in_zip

            if not files_to_extract:
                print("Error: No PDF files found in the zip archive.")
                sys.exit(1)

            print(f"Extracting {len(files_to_extract)} PDF(s) to '{TEMP_EXTRACTION_DIR}'...")
            for pdf_file in files_to_extract:
                zip_ref.extract(pdf_file, TEMP_EXTRACTION_DIR)
        print("Extraction complete.")

        # --- Step 3: Construct and Run the Marker Command ---
        command = [
            "marker",
            '--disable_image_extraction',
            #'--workers', '2',
            #'--batch_multiplier 3',
            '--detection_batch_size', '3',
            '--layout_batch_size', '3',
            '--pdftext_workers', '10'
            '--output_dir',
            str(TEMP_MD),
            str(TEMP_EXTRACTION_DIR)
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env['TORCH_DEVICE'] = 'cuda'
        env['OUTPUT_DIR'] = str(TEMP_MD)

        print("\nStarting Marker conversion...")
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, env=env)
        print("\nMarker conversion successfully completed!")

        # --- Step 4: Zip the Markdown Results ---
        zip_output_files(TEMP_MD, OUTPUT_ZIP)

    except FileNotFoundError:
        print("\nError: 'marker' command not found.")
        print("Please ensure you have activated the correct Conda environment (e.g., 'conda activate marker_env').")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Marker process failed with exit code {e.returncode}.")
        sys.exit(1)


    finally:
        # --- Step 5: Clean Up All Intermediate Files ---
        # This 'finally' block ensures that all temporary directories are
        # removed, leaving only the final zip archive.
        print("\nCleaning up intermediate directories...")
        for path in [TEMP_EXTRACTION_DIR, TEMP_MD]:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Removed '{path}'.")
        print("Cleanup complete.")


if __name__ == "__main__":
    main()