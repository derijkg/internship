import os
import subprocess
import zipfile
import shutil
import sys
from pathlib import Path

# --- Configuration ---
ZIP_FILE_PATH = Path("../data/archive.zip")
OUTPUT_DIR = Path("../data/marker_output")
# ---------------------
TEMP_EXTRACTION_DIR = "temp_pdf_extraction"


def main():
    """
    Main function to orchestrate the PDF processing workflow.
    """
    # --- Step 1: Validate Inputs and Set Up Directories ---
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"Error: The file '{ZIP_FILE_PATH}' was not found.")
        sys.exit(1)

    # Create the final output directory if it doesn't exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create the temporary directory for extraction.
    if os.path.exists(TEMP_EXTRACTION_DIR):
        print(f"Warning: Temporary directory '{TEMP_EXTRACTION_DIR}' already exists. Cleaning it up.")
        shutil.rmtree(TEMP_EXTRACTION_DIR)
    os.makedirs(TEMP_EXTRACTION_DIR)

    print(f"Setup complete. Output will be saved to: {OUTPUT_DIR}")

    try:
        # --- Step 2: Extract the ZIP File ---
        print(f"Extracting PDFs from '{ZIP_FILE_PATH}' to '{TEMP_EXTRACTION_DIR}'...")
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(TEMP_EXTRACTION_DIR)
        print("Extraction complete.")

        # --- Step 3: Construct and Run the Marker Command ---
        command = [
            "marker",
            TEMP_EXTRACTION_DIR,
            OUTPUT_DIR
        ]

        # Create a copy of the current environment variables.
        # This is crucial because we only want to modify it for this specific command.
        env = os.environ.copy()
        
        # Set the CUDA_VISIBLE_DEVICES environment variable.
        # This restricts the command to only see and use the first GPU (GPU #0).
        env["CUDA_VISIBLE_DEVICES"] = "0"

        print("\nStarting Marker conversion...")
        print(f"Running command: {' '.join(command)}")
        
        # The `marker` (folder) command inherently outputs Markdown, so no extra flags are needed.
        # We run the command using subprocess.run().
        # `check=True` will raise an exception if the command fails (returns a non-zero exit code).
        # `env=env` passes our modified environment to the subprocess.
        result = subprocess.run(command, check=True, env=env)

        print("\nMarker conversion successfully completed!")

    except FileNotFoundError:
        print("\nError: 'marker' command not found.")
        print("Please ensure you have activated the correct Conda environment (e.g., 'conda activate marker_env').")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Marker process failed with exit code {e.returncode}.")
        if e.stderr:
            print(f"Error output:\n{e.stderr}")
        sys.exit(1)
    finally:
        # --- Step 4: Clean Up ---
        # This 'finally' block ensures that the temporary directory is always
        # removed, even if an error occurred during the process.
        print(f"Cleaning up temporary directory: '{TEMP_EXTRACTION_DIR}'...")
        if os.path.exists(TEMP_EXTRACTION_DIR):
            shutil.rmtree(TEMP_EXTRACTION_DIR)
        print("Cleanup complete.")

    


if __name__ == "__main__":
    main()