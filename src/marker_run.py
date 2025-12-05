import os
import subprocess
import zipfile
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
ZIP_INPUT = Path("data/archive.zip")
TEMP_OUTPUT = Path("data/temp/marker_output")
TEMP_INPUT = Path("data/temp/temp_extraction")
ZIP_OUTPUT = Path("data/output_marker.zip")

# my specifically problematic files
IGNORE_LIST = {'1058',
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
 '841'} # constant failures

# MARKER CONFIG
workers = '1' # currently unused, see command = [...]
detection_batch_size = '14'
layout_batch_size = '14'
pdftext_workers = '12'
# ---------------------

def stage_for_processing(
        zip_input=ZIP_INPUT,
        temp_input=TEMP_INPUT,
        temp_output=TEMP_OUTPUT,
        zip_output=ZIP_OUTPUT,
        ignore_set=IGNORE_LIST
    ):
    """
    Extracts files from Archive Zip to Temp Input folder IF:
    1. They are not in Temp Output (already processed)
    2. They are not in Marker Zip (already packed)
    """
    print("Staging files for processing...")

    if temp_input.exists():
        shutil.rmtree(temp_input)
    temp_input.mkdir(parents=True, exist_ok=True)
    
    # Get list of already completed IDs
    completed_ids = set()
    
    # Check Temp Output (Folders)
    if temp_output.exists():
        completed_ids.update([p.name for p in temp_output.iterdir() if p.is_dir()])
    
    # Check Marker Zip
    if zip_output.exists():
        with zipfile.ZipFile(zip_output, 'r') as z:
            completed_ids.update([Path(f).stem for f in z.namelist()])

    # Iterate Archive and Extract needed
    extracted_count = 0
    if not zip_input.exists():
        print("Archive zip not found.")
        return 0

    with zipfile.ZipFile(zip_input, 'r') as z:
        # Filter file list: Valid ID AND Not Completed
        files_to_extract = []
        for f in z.namelist():
            if f.endswith('/'):
                continue
            fid = Path(f).stem
            if fid in completed_ids:
                continue
            if fid in ignore_set:
                continue
                
            files_to_extract.append(f)
        if not files_to_extract:
            print(' -> no new files need processing.')
            return 0
        
        # Extract
        for f in tqdm(files_to_extract, desc="Extracting to Temp"):
            z.extract(f, temp_input)
            extracted_count += 1

    print(f"Staged {extracted_count} files to {temp_input}")
    return extracted_count

    # =========================================================================
    # 5. SUGGESTED: PACK LOGIC
    # =========================================================================
def pack_processed_output(
        temp_output=TEMP_OUTPUT,
        zip_output=ZIP_OUTPUT
    ):
    """
    Moves processed folders from Temp Output into the flat Marker Zip.
    """
    print("Packing processed files...")
    
    processed_folders = [p for p in temp_output.iterdir() if p.is_dir()]
    
    if not processed_folders:
        print("No folders in temp output to pack.")
        return

    with zipfile.ZipFile(zip_output, 'a', compression=zipfile.ZIP_DEFLATED) as z:
        existing_in_zip = set(z.namelist())
        packed_count = 0
        
        for folder in tqdm(processed_folders, desc="Packing to Zip"):
            # Marker output structure: folder 458/ -> 458.md, 458.json, meta.json
            # We want flat structure in zip: 458.md, 458.json
            
            for file_path in folder.iterdir():
                if file_path.is_dir(): continue
                if file_path.name in existing_in_zip:
                    continue
                
                # Only add if it matches the ID (ignore random metadata logs if needed)
                if folder.name in file_path.name: 
                    z.write(file_path, arcname=file_path.name)
                    existing_in_zip.add(file_path.name)
                    packed_count += 1
    
    print(f"Packing complete. Added {packed_count} files")
def remove_empty_folders(directory_path: Path):
    """
    Scans the directory and removes any subdirectories that are empty.
    Returns the number of folders removed.
    """
    if not directory_path.exists():
        return 0

    removed_count = 0
    # List all subdirectories
    folders = [f for f in directory_path.iterdir() if f.is_dir()]

    for folder in folders:
        # Check if folder is empty (next() raises StopIteration if empty)
        try:
            next(folder.iterdir())
        except StopIteration:
            # It is empty, remove it
            folder.rmdir()
            removed_count += 1
            
    if removed_count > 0:
        print(f"  -> Cleaned up {removed_count} empty folders in {directory_path.name}.")
    
    return removed_count





def main():
    # --- Step 1: Setup & Scan ---
    count = stage_for_processing(ZIP_INPUT,TEMP_INPUT,TEMP_OUTPUT,ZIP_OUTPUT,IGNORE_LIST)

    # --- Step 2: Prompt User ---
    if count == 0:
        print("\nNo new files found to process.")
        remove_empty_folders(TEMP_OUTPUT)
        pack_processed_output(TEMP_OUTPUT,ZIP_OUTPUT)
        
        return
    
    print(f"\nFound {count} new files in the zip that have not been processed.")

    '''
    user_input = input("Do you want to proceed with extraction and Marker processing? (y/n): ").strip().lower()
    
    if user_input != 'y':
        print("Operation cancelled by user. Exiting.")
        if TEMP_INPUT.exists(): shutil.rmtree(TEMP_INPUT)
        sys.exit(0)
    '''

    # --- Step 4: Run Marker ---
    try:
        command = [
            "marker",
            '--debug_print',
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


    # zip, remove failed conversions, delete extraction folder
    remove_empty_folders(TEMP_OUTPUT)
    pack_processed_output(TEMP_OUTPUT,ZIP_OUTPUT)
    


    if TEMP_INPUT.exists():
        shutil.rmtree(TEMP_INPUT)
        print("Cleaned up temporary input files.")

if __name__ == "__main__":
    main()




# unused
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
# ^ stop unused