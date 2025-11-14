import os
import subprocess
import zipfile
import shutil
import sys
from pathlib import Path
import torch

# --- Configuration ---
ZIP_INPUT = Path("data/archive.zip")
ZIP_OUTPUT = Path("data/output_marker.zip")

TEMP_OUTPUT = Path("data/temp/marker_output")
TEMP_INPUT = Path("data/temp/temp_pdf_extraction")

#TEMP_RUN_DIR = pass

LIMIT_FILES_TO = 0

# ---------------------
def state():
    pass

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
    if not os.path.exists(ZIP_INPUT):
        print(f"Error: The file '{ZIP_INPUT}' was not found.")
        sys.exit(1)

    # Clean up previous runs to ensure a fresh start
    for path in [TEMP_OUTPUT, TEMP_INPUT]:
        if os.path.exists(path):
            print(f"Warning: Directory '{path}' already exists. Cleaning it up.")
            shutil.rmtree(path)
    
    os.makedirs(TEMP_OUTPUT)
    os.makedirs(TEMP_INPUT)
    


    # --- Check for successfully processed files in the output zip ---
    successfully_processed_this_run = set()
    successfully_processed_basenames = set()
    files_to_preserve = []  # A list of valid, non-empty files to keep from the old zip

    if os.path.exists(ZIP_OUTPUT):
        print(f"Found existing archive: '{ZIP_OUTPUT}'. Validating completed files...")
        
        found_files_map = {}
        with zipfile.ZipFile(ZIP_OUTPUT, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                basename = Path(zip_info.filename).stem
                extension = Path(zip_info.filename).suffix
                if basename not in found_files_map:
                    found_files_map[basename] = {}
                if extension in ['.md', '.json']:
                    found_files_map[basename][extension] = zip_info
        
        # Validate each found file pair based on the markdown file's existence and size
        for basename, files in found_files_map.items():
            # A successful conversion must have a markdown file larger than 50 bytes
            if '.md' in files and files['.md'].file_size > 50:
                successfully_processed_basenames.add(basename)
                # If valid, mark all associated files for preservation
                for zip_info in files.values():
                    files_to_preserve.append(zip_info.filename)
            else:
                print(f"  -> Invalid or empty result for '{basename}'. It will be re-processed.")

        if files_to_preserve:
            print(f"Found and validated {len(successfully_processed_basenames)} previously processed files. Extracting them to preserve.")
            with zipfile.ZipFile(ZIP_OUTPUT, 'r') as zip_ref:
                # Only extract the files that passed validation
                zip_ref.extractall(TEMP_OUTPUT, members=files_to_preserve)

    print("Setup complete. Directories are ready.")

    try:
        # --- Step 2: Extract only the necessary PDF Files ---
        print(f"Opening '{ZIP_INPUT}' to find PDFs...")
        with zipfile.ZipFile(ZIP_INPUT, 'r') as zip_ref:
            pdf_files_in_zip = [f for f in zip_ref.namelist()]
            
            files_to_process = []
            for pdf_file in pdf_files_in_zip:
                pdf_basename = Path(pdf_file).stem
                if pdf_basename not in successfully_processed_basenames:
                    files_to_process.append(pdf_file)

            if LIMIT_FILES_TO and LIMIT_FILES_TO > 0:
                print(f"LIMIT_FILES_TO is set to {LIMIT_FILES_TO}. Limiting subset of NEW files to process.")
                files_to_process = files_to_process[:LIMIT_FILES_TO]

            if not files_to_process:
                print("\nAll PDF files have already been successfully processed. Nothing to do.")
            else:
                print(f"Found {len(files_to_process)} new or failed PDF(s) to process.")
                
                # --- MODIFIED: Only extract files that don't already exist ---
                files_to_extract = []
                for pdf_file in files_to_process:
                    # Check against the full path in the extraction dir
                    destination_path = TEMP_INPUT / Path(pdf_file).name
                    if not destination_path.exists():
                        files_to_extract.append(pdf_file)
                
                if files_to_extract:
                    print(f"Extracting {len(files_to_extract)} PDF(s) to '{TEMP_INPUT}'...")
                    for pdf_file in files_to_extract:
                        zip_ref.extract(pdf_file, TEMP_INPUT)
                    print("Extraction complete.")
                else:
                    print("All PDFs to be processed are already present in the temporary directory.")

        # --- Step 3: Construct and Run the Marker Command ---
        command = [
            "marker",
            '--disable_image_extraction',
            #'--workers', '2',
            #'--batch_multiplier 3',
            '--detection_batch_size', '5',
            '--layout_batch_size', '5',
            '--pdftext_workers', '12',
            '--output_dir',
            str(TEMP_OUTPUT),
            str(TEMP_INPUT)
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env['TORCH_DEVICE'] = 'cuda'
        env['OUTPUT_DIR'] = str(TEMP_OUTPUT)

        print("\nStarting Marker conversion...")
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, env=env)
        print("\nMarker conversion successfully completed!")


        # Update the set of successfully processed files for this run
        for pdf_file in files_to_process:
            successfully_processed_this_run.add(Path(pdf_file).stem)


        # --- Step 4: Zip the Markdown Results ---
        zip_output_files(TEMP_OUTPUT, ZIP_OUTPUT)

    except FileNotFoundError:
        print("\nError: 'marker' command not found.")
        print("Please ensure you have activated the correct Conda environment (e.g., 'conda activate marker_env').")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Marker process failed with exit code {e.returncode}.")
        sys.exit(1)

'''
    finally:
            # --- MODIFIED Step 5: Clean Up ONLY Successfully Processed Files ---
            print("\nCleaning up intermediate files...")
            if os.path.exists(TEMP_EXTRACTION_DIR) and successfully_processed_this_run:
                print(f"Cleaning up {len(successfully_processed_this_run)} successfully processed PDF(s)...")
                for basename in successfully_processed_this_run:
                    # Find the corresponding PDF file (could have different casings)
                    for pdf_file in TEMP_EXTRACTION_DIR.glob(f"{basename}.*"):
                        try:
                            pdf_file.unlink()
                            # print(f"  -> Removed '{pdf_file.name}'")
                        except OSError as e:
                            print(f"Error removing file {pdf_file}: {e}")
            
            # Always clean up the temporary markdown/json output dir
            if os.path.exists(TEMP_MD):
                shutil.rmtree(TEMP_MD)
                print(f"Removed temporary output directory '{TEMP_MD}'.")
            
            # Optional: remove the temp PDF dir if it's empty
            if os.path.exists(TEMP_EXTRACTION_DIR) and not any(TEMP_EXTRACTION_DIR.iterdir()):
                shutil.rmtree(TEMP_EXTRACTION_DIR)
                print(f"Removed empty temporary PDF directory '{TEMP_EXTRACTION_DIR}'.")

            print("Cleanup complete.")
'''


if __name__ == "__main__":
    main()