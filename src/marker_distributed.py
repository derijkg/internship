import os
import subprocess
import zipfile
import shutil
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# --- Configuration (Base Paths) ---
ZIP_INPUT = Path("data/archive.zip")
# These will be modified dynamically based on worker index
BASE_TEMP_OUTPUT = Path("data/temp/marker_output")
BASE_TEMP_INPUT = Path("data/temp/temp_extraction")
BASE_ZIP_OUTPUT = Path("data/output_marker") # Will become output_marker_part_0.zip

# Changed to SET for faster lookup
IGNORE_LIST = {
    '1058', '1191', '1215', '1374', '1445', '1524', '1592', '1595', '1821', 
    '1846', '2070', '2188', '2206', '2210', '2242', '241', '2509', '2533', 
    '2600', '2696', '3409', '3477', '3806', '3831', '3943', '446', '533', 
    '580', '662', '689', '691', '692', '820', '841'
}

# MARKER CONFIG
detection_batch_size = '14'
layout_batch_size = '14'
pdftext_workers = '12'

def stage_for_processing(zip_input, temp_input, temp_output, zip_output, ignore_set, worker_idx, total_workers):
    """
    Extracts files assigned to THIS worker using modulo math.
    """
    print(f"Staging files for Worker {worker_idx}/{total_workers}...")


    if temp_output.exists():
        remove_empty_folders(temp_output)
        pack_processed_output(temp_output, zip_output)
    if temp_input.exists():
        shutil.rmtree(temp_input)
    temp_input.mkdir(parents=True, exist_ok=True)
    
    # 1. Check what THIS worker has already finished
    # We only look at this worker's specific zip file
    completed_ids = set()
    if zip_output.exists():
        with zipfile.ZipFile(zip_output, 'r') as z:
            completed_ids.update([Path(f).stem for f in z.namelist()])
            
    # Also check temp output for this worker
    if temp_output.exists():
        completed_ids.update([p.name for p in temp_output.iterdir() if p.is_dir()])

    # 2. Iterate Archive
    extracted_count = 0
    if not zip_input.exists():
        print("Archive zip not found.")
        return 0

    with zipfile.ZipFile(zip_input, 'r') as z:
        files_to_extract = []
        for f in z.namelist():
            if f.endswith('/'): continue
            
            fid = Path(f).stem
            
            # --- LOGIC GATES ---
            if fid in ignore_set: continue
            if fid in completed_ids: continue
            
            # --- SHARDING LOGIC (The Magic) ---
            # Try to convert ID to int. If fails, use hash.
            try:
                num_id = int(fid)
            except ValueError:
                num_id = hash(fid)
            
            # If the ID doesn't belong to this worker, skip it
            if num_id % total_workers != worker_idx:
                continue
            # ----------------------------------

            files_to_extract.append(f)
            
        if not files_to_extract:
            return 0
        
        for f in tqdm(files_to_extract, desc=f"Worker {worker_idx} Extracting"):
            z.extract(f, temp_input)
            extracted_count += 1

    print(f"Worker {worker_idx}: Staged {extracted_count} files.")
    return extracted_count

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    if args.final == True:
        print('running final branch')
        zip_paths = []
        for i in range(3):
            remove_empty_folders(Path(f'data/temp/marker_output_{i}'))
            output_zip = Path(f'data/output_marker_part_{i}')
            zip_paths.append(str(output_zip))
            pack_processed_output(temp_output= Path(f'data/temp/marker_output_{i}'), zip_output= output_zip)
        
        final_zip_name = 'data/marker_output_zip_full.zip'
        command = ['zip', final_zip_name] + zip_paths
        subprocess.run(command, check=True)
        print('Final zip finished')
        return

    parser.add_argument("--worker_index", type=int, required=True, help="0, 1, or 2")
    parser.add_argument("--total_workers", type=int, required=True, help="Total number of GPUs being used (e.g. 3)")
    args = parser.parse_args()



    # --- DYNAMIC PATHS ---
    # Each worker gets its own sandboxed folders
    my_temp_input = Path(f"{BASE_TEMP_INPUT}_{args.worker_index}")
    my_temp_output = Path(f"{BASE_TEMP_OUTPUT}_{args.worker_index}")
    my_zip_output = Path(f"{str(BASE_ZIP_OUTPUT)}_part_{args.worker_index}.zip")

    '''
    # adapt load for titan
    if args.worker_index in [1,2]:
        detection_batch_size = '1'
        layout_batch_size = '1'
        pdftext_workers = '1'
    '''

    # --- Step 1: Stage ---
    count = stage_for_processing(ZIP_INPUT, my_temp_input, my_temp_output, my_zip_output, IGNORE_LIST, args.worker_index, args.total_workers)

    # --- Step 2: Prompt / Check ---
    if count == 0:
        print(f"Worker {args.worker_index}: No new files.")
        remove_empty_folders(my_temp_output)
        pack_processed_output(my_temp_output, my_zip_output)
        
        return

    # Auto-accept for cluster usage (remove input() prompt)
    print(f"Worker {args.worker_index}: Proceeding with {count} files.")

    # --- Step 3: Run Marker ---
    try:
        command = [
            "marker",
            '--disable_image_extraction',
            '--detection_batch_size', detection_batch_size, 
            '--layout_batch_size', layout_batch_size,
            '--pdftext_workers', pdftext_workers,
            '--output_dir', str(my_temp_output),
            str(my_temp_input)
        ]
        
        env = os.environ.copy()
        # CRITICAL: Inherit the GPU visible setting from the shell command
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        env['TORCH_DEVICE'] = 'cuda'

        '''
        if args.worker_index in [1,2]:
            
            # manual edit in settings.py files of surya and marker in marker_titan env
            env['TORCH_DTYPE'] = 'float32'
            env['SURYA_DTYPE'] = 'float32' 
        '''
        
        print(f"Worker {args.worker_index}: Starting Marker on GPU {env['CUDA_VISIBLE_DEVICES']}...")
        subprocess.run(command, check=True, env=env)

    except subprocess.CalledProcessError as e:
        print(f"[!] Worker {args.worker_index} encountered Marker error {e.returncode}. Packing what succeeded...")

    # --- Step 4: Finalize ---
    remove_empty_folders(my_temp_output)
    pack_processed_output(my_temp_output, my_zip_output)
    if my_temp_input.exists():
        shutil.rmtree(my_temp_input)

if __name__ == "__main__":
    main()