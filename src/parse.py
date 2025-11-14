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
TEMP_INPUT = Path("data/temp/temp_extraction") # Changed name for clarity

# The new state file
DONE_FILES_LOG = Path("data/temp/done_files.txt")

LIMIT_FILES_TO = 0

# ---------------------

def zip_output_files(source_dir, zip_path):
    """
    Finds all .md and .json files in the source_dir and adds them to a zip archive.
    This function is now designed to append to the output, preserving old results.
    """
    print(f"\nUpdating zip archive at: {zip_path}")
    # Create a temporary zip file for the new results
    temp_zip_path = zip_path.with_suffix(".temp.zip")
    
    # Zip the new output from TEMP_OUTPUT
    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as temp_zipf:
        for item in source_dir.iterdir():
            if item.is_dir():
                for file_path in item.rglob('*'):
                    # arcname should be flat, e.g., "basename.md"
                    arcname = f"{item.name}{file_path.suffix}"
                    temp_zipf.write(file_path, arcname=arcname)

    # Merge the new zip with the old one if it exists
    if zip_path.exists():
        # This part requires a bit more logic to merge zips, for simplicity we will re-create it
        # from the preserved files and the new files.
        pass # For now, we assume a full re-zip is acceptable. A more complex merge is possible.

    # For simplicity in this script, we just create the full zip from scratch
    # from the combination of preserved files and new files in TEMP_OUTPUT
    print(f"Creating final zip from all content in {TEMP_OUTPUT}...")
    final_files_count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as final_zipf:
        for item in TEMP_OUTPUT.iterdir():
            if item.is_dir():
                 for file_path in item.rglob('*'):
                    arcname = f"{item.name}{file_path.suffix}"
                    final_zipf.write(file_path, arcname=arcname)
                    final_files_count += 1
    
    print(f"Successfully added {final_files_count} files to the zip archive.")


def main():
    """
    Main function to orchestrate the file processing and zipping workflow.
    """
    # --- Step 1: Initial Setup ---
    if not os.path.exists(ZIP_INPUT):
        print(f"Error: The file '{ZIP_INPUT}' was not found.")
        sys.exit(1)

    os.makedirs(TEMP_OUTPUT, exist_ok=True)

    # --- ACTION 1: Extract all files if the temp input folder doesn't exist ---
    if not os.path.exists(TEMP_INPUT):
        print(f"Temporary input cache '{TEMP_INPUT}' not found. Extracting all files from archive...")
        os.makedirs(TEMP_INPUT)
        with zipfile.ZipFile(ZIP_INPUT, 'r') as zip_ref:
            zip_ref.extractall(TEMP_INPUT)
        print("Extraction complete.")
    else:
        print(f"Using existing input cache at '{TEMP_INPUT}'.")

    # --- ACTION 2 & 3: Create/Update the 'done' file ---
    done_basenames = set()
    if DONE_FILES_LOG.exists():
        with open(DONE_FILES_LOG, 'r') as f:
            done_basenames = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(done_basenames)} records from existing 'done' file.")

    # Check TEMP_OUTPUT for any results from a prior crashed run
    for item in TEMP_OUTPUT.iterdir():
        if item.is_dir():
            done_basenames.add(item.name)

    # Check final zip for results
    if ZIP_OUTPUT.exists():
        try:
            with zipfile.ZipFile(ZIP_OUTPUT, 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    done_basenames.add(Path(filename).stem)
        except zipfile.BadZipFile:
            print("Warning: Output zip is corrupted.")

    # --- ACTION 4: Validate the 'done' file entries ---
    validated_done_basenames = set()
    print("Validating 'done' file records...")
    for basename in done_basenames:
        is_valid = False
        # Check in TEMP_OUTPUT first
        output_dir = TEMP_OUTPUT / basename
        if output_dir.is_dir():
            dir_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
            if dir_size > 50:
                is_valid = True
        
        # If not found or invalid in temp, check the final zip
        if not is_valid and ZIP_OUTPUT.exists():
            try:
                with zipfile.ZipFile(ZIP_OUTPUT, 'r') as zip_ref:
                    md_file = f"{basename}.md"
                    if md_file in zip_ref.namelist():
                        if zip_ref.getinfo(md_file).file_size > 50:
                            is_valid = True
            except (zipfile.BadZipFile, KeyError):
                pass # Ignore errors, just means file isn't valid

        if is_valid:
            validated_done_basenames.add(basename)

    if len(done_basenames) != len(validated_done_basenames):
        print(f"Validation complete. Removed {len(done_basenames) - len(validated_done_basenames)} invalid records.")
    
    # --- ACTION 5: Remove already-done files from the input folder ---
    files_to_process_paths = []
    print(f"Cleaning {len(validated_done_basenames)} already processed files from '{TEMP_INPUT}'...")
    for item in TEMP_INPUT.iterdir():
        if item.is_file():
            if item.stem in validated_done_basenames:
                item.unlink() # Delete the file
            else:
                files_to_process_paths.append(item)
    
    print(f"Found {len(files_to_process_paths)} files to process in '{TEMP_INPUT}'.")

    # --- ACTION 6: Run the marker command ---
    if not files_to_process_paths:
        print("\nNo new files to process.")
    else:
        try:
            command = [
                "marker",
                '--disable_image_extraction',
                '--detection_batch_size', '17', #increase
                '--layout_batch_size', '17', #increase
                '--pdftext_workers', '10', #fine
                #'--skip_existing',
                '--output_dir', str(TEMP_OUTPUT),
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

            # --- ACTION 7: Final update ---
            newly_processed_basenames = {p.stem for p in files_to_process_paths}
            print(f"Updating 'done' file with {len(newly_processed_basenames)} new records.")
            final_done_set = validated_done_basenames.union(newly_processed_basenames)
            with open(DONE_FILES_LOG, 'w') as f:
                for basename in sorted(list(final_done_set)):
                    f.write(f"{basename}\n")
            
            # Re-zip the entire valid output directory
            zip_output_files(TEMP_OUTPUT, ZIP_OUTPUT)

            print(f"Cleaning up {len(newly_processed_basenames)} source files from input cache...")
            for path in files_to_process_paths:
                if path.exists():
                    path.unlink()

        except FileNotFoundError:
            print("\nError: 'marker' command not found.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"\nError: Marker process failed with exit code {e.returncode}.")
            sys.exit(1)
        '''
        finally:
            # Clean up the marker output dir, as its contents are now in the zip
            if os.path.exists(TEMP_OUTPUT):
                shutil.rmtree(TEMP_OUTPUT)
                print(f"Removed temporary output directory '{TEMP_OUTPUT}'.")
        '''
if __name__ == "__main__":
    main()