import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
TOTAL_WORKERS = 3  # Must match what you plan to run (calc12 + calc11 + calc11)

# Old Paths (Where your data is now)
OLD_ZIP = Path("data/output_marker.zip")
OLD_TEMP_OUTPUT = Path("data/temp/marker_output")

# Base New Paths (Where data needs to go)
NEW_ZIP_BASE = Path("data/output_marker") # becomes _part_0.zip
NEW_TEMP_BASE = Path("data/temp/marker_output") # becomes _0


def pack_processed_output(temp_output, zip_output):
    if not temp_output.exists(): return
    processed_folders = [p for p in temp_output.iterdir() if p.is_dir()]
    if not processed_folders: return

    print(f"Packing {len(processed_folders)} items into {zip_output.name}...")

    with zipfile.ZipFile(zip_output, 'a', compression=zipfile.ZIP_DEFLATED) as z:
        existing_in_zip = set(z.namelist())
        packed_count = 0
        
        for folder in processed_folders:
            for file_path in folder.iterdir():
                if file_path.is_dir(): continue
                if file_path.name in existing_in_zip: continue
                
                if folder.name in file_path.name: 
                    z.write(file_path, arcname=file_path.name)
                    existing_in_zip.add(file_path.name)
                    packed_count += 1
    
    print(f"Packing complete. Added {packed_count} files.")

def remove_empty_folders(directory_path: Path):
    if not directory_path.exists(): return 0
    removed_count = 0
    folders = [f for f in directory_path.iterdir() if f.is_dir()]
    for folder in folders:
        try:
            next(folder.iterdir())
        except StopIteration:
            folder.rmdir()
            removed_count += 1
    return removed_count

def get_worker_index(file_id, total_workers):
    try:
        return int(file_id) % total_workers
    except ValueError:
        return hash(file_id) % total_workers

def migrate():
    print(f"--- MIGRATING DATA FOR {TOTAL_WORKERS} WORKERS ---")

    # 1. MIGRATE THE ZIP FILE
    if OLD_ZIP.exists():
        print(f"Splitting {OLD_ZIP} into {TOTAL_WORKERS} parts...")
        
        # We open all 3 target zips at once
        target_zips = {}
        for i in range(TOTAL_WORKERS):
            p = Path(f"{NEW_ZIP_BASE}_part_{i}.zip")
            target_zips[i] = zipfile.ZipFile(p, 'a', zipfile.ZIP_DEFLATED)
            
        with zipfile.ZipFile(OLD_ZIP, 'r') as source_zip:
            file_list = source_zip.namelist()
            
            for fname in tqdm(file_list, desc="Migrating Zip"):
                # fname is likely "123.md" or "123.json"
                fid = Path(fname).stem
                
                worker_idx = get_worker_index(fid, TOTAL_WORKERS)
                
                # Copy content from old zip to specific new zip
                content = source_zip.read(fname)
                
                # Check if already exists in target to avoid duplicates
                if fname not in target_zips[worker_idx].namelist():
                    target_zips[worker_idx].writestr(fname, content)

        # Close all zips
        for z in target_zips.values():
            z.close()
        print("Zip migration complete.")
    else:
        print("No legacy zip found to migrate.")

    # 2. MIGRATE THE TEMP FOLDERS
    if OLD_TEMP_OUTPUT.exists():
        print(f"Distributing temp folders from {OLD_TEMP_OUTPUT}...")
        
        # Ensure destination folders exist
        for i in range(TOTAL_WORKERS):
            Path(f"{NEW_TEMP_BASE}_{i}").mkdir(parents=True, exist_ok=True)

        # Get all folder names (IDs)
        folders = [f for f in OLD_TEMP_OUTPUT.iterdir() if f.is_dir()]
        
        moved_count = 0
        for folder in tqdm(folders, desc="Moving Temp Folders"):
            fid = folder.name
            worker_idx = get_worker_index(fid, TOTAL_WORKERS)
            
            dest_parent = Path(f"{NEW_TEMP_BASE}_{worker_idx}")
            dest_path = dest_parent / fid
            
            # Move the folder
            if dest_path.exists():
                # If it already exists in the new location, we assume the new one is valid
                # and we delete the old one to clean up
                shutil.rmtree(folder)
            else:
                shutil.move(str(folder), str(dest_path))
                moved_count += 1
                
        print(f"Moved {moved_count} temp folders to their new worker locations.")
        
        # Clean up the old empty parent folder
        try:
            OLD_TEMP_OUTPUT.rmdir()
            print("Removed empty legacy temp folder.")
        except OSError:
            print("Legacy temp folder not empty (maybe contains unknown files), left in place.")

    print("\n--- MIGRATION COMPLETE ---")
    print(f"You now have output_marker_part_0.zip, _1.zip, _2.zip")
    print(f"You can safely run the distributed script now.")

if __name__ == "__main__":

    # remove empty folders in temp output_marker
    remove_empty_folders(OLD_TEMP_OUTPUT)

    # zip to marker_output.zip
    pack_processed_output(OLD_TEMP_OUTPUT, OLD_ZIP)

    # split
    migrate()