import os
import subprocess
import zipfile
import shutil
import sys
from pathlib import Path

# --- Configuration ---
ZIP_FILE_PATH = Path("../data/archive.zip")
OUTPUT_ZIP = Path("path/to/your/final_markdown_archive.zip")

TEMP_MD = Path("../data/marker_output")
TEMP_EXTRACTION_DIR = Path("temp_pdf_extraction")
# ---------------------

def zip_markdown_output(source_dir, zip_path):
    """
    Finds all .md files in the source_dir and adds them to a zip archive.

    Args:
        source_dir (str): The directory containing the markdown files.
        zip_path (str): The path for the final output zip file.
    """
    print(f"\nCreating zip archive at: {zip_path}")
    found_files = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    # The 'arcname' argument prevents storing the full directory structure.
                    zipf.write(file_path, arcname=file)
                    found_files += 1
    print(f"Successfully added {found_files} markdown files to the zip archive.")


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
        # --- Step 2: Extract the ZIP File ---
        print(f"Extracting PDFs from '{ZIP_FILE_PATH}' to '{TEMP_EXTRACTION_DIR}'...")
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(TEMP_EXTRACTION_DIR)
        print("Extraction complete.")

        # --- Step 3: Construct and Run the Marker Command ---
        command = ["marker", TEMP_EXTRACTION_DIR, TEMP_MD]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        print("\nStarting Marker conversion...")
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, env=env)
        print("\nMarker conversion successfully completed!")

        # --- Step 4: Zip the Markdown Results ---
        zip_markdown_output(TEMP_MD, OUTPUT_ZIP)

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