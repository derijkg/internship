import re
import os
from collections import Counter
import logging
import random
import requests
from requests import Session
import time
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Generator
from contextlib import contextmanager
from tqdm import tqdm
import mimetypes
    
# ==============================================================================
#  IO
# ==============================================================================

def set_path(path_str: str) -> Path:
    """
    Ensures a directory exists from a string path, creating it if necessary.

    This is intended for creating permanent directories (e.g., for final outputs).

    Args:
        path_str (str): The directory path to create.

    Returns:
        Path: The created (or already existing) path object.
    """
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Path ensured: '{path}'")
    return path

@contextmanager
def temp_path(path_str: str, temporary: bool = True) -> Generator[Path, None, None]:
    """
    A context manager to create a directory from a string path and optionally
    clean it up afterward. The recommended way to handle temporary directories.

    Args:
        path_str (str): The directory path to manage.
        temporary (bool): If True, the directory and all its contents will be
                          deleted upon exiting the context. If False, the
                          directory will be created but not deleted.
    
    Yields:
        Path: The path object for the created directory.
    """
    path = Path(path_str)
    # This is the "enter" part of the context
    path.mkdir(parents=True, exist_ok=True)
    print(f"Managed path created: '{path}'{' (temporary)' if temporary else ''}")
    
    try:
        # Yield the path object to be used inside the 'with' block
        yield path
    finally:
        # This is the "exit" part, which runs no matter what
        if temporary and path.exists():
            shutil.rmtree(path)
            print(f"Temporary path cleaned up: '{path}'")

def to_zip(
    input_dir: Path,
    output_dir: Path,
    extensions: Optional[List[str]] = None,
    flatten: bool = True
):
    """
    Compresses files from a source directory into a zip archive, with an
    optional filter for file extensions.

    Args:
        input_dir (Path): The directory containing the files to zip.
        output_dir (Path): The full path for the output zip file to be created.
        extensions (Optional[List[str]]): A list of file extensions to include
            (e.g., ['.md', '.json']). If None, all files will be included.
        flatten (bool): If True, the directory structure within the zip file is
            flattened, so all files appear at the root. If False, the original
            directory structure is preserved.
    """
    if not input_dir.is_dir():
        print(f"Error: Source directory '{input_dir}' does not exist.")
        return

    # Normalize extensions to ensure they start with a dot and are lowercase
    if extensions:
        normalized_exts = {f".{ext.lstrip('.').lower()}" for ext in extensions}
        print(f"Zipping files with extensions: {', '.join(normalized_exts)}")
    else:
        print("Zipping all files in the directory.")

    found_files = 0
    with zipfile.ZipFile(output_dir, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # rglob('*') finds all files in all subdirectories
        for file_path in input_dir.rglob('*'):
            if file_path.is_file():
                # Apply the extension filter if it exists
                if extensions and file_path.suffix.lower() not in normalized_exts:
                    continue  # Skip files that don't match

                # Determine the name of the file inside the zip archive
                if flatten:
                    arcname = file_path.name
                else:
                    arcname = file_path.relative_to(input_dir)

                zipf.write(file_path, arcname=arcname)
                found_files += 1

    print(f"Successfully added {found_files} files to '{output_dir}'.")


def from_zip(
    input_dir: Path,
    output_dir: Path,
    extensions: Optional[List[str]] = None
):
    """
    Extracts files from a zip archive to a destination directory, with an
    optional filter for file extensions.

    Args:
        input_dir (Path): The path to the zip file to be extracted.
        output_dir (Path): The directory where files will be extracted.
        extensions (Optional[List[str]]): A list of file extensions to extract
            (e.g., ['.pdf']). If None, all files will be extracted.
    """
    if not input_dir.is_file():
        print(f"Error: Zip file '{input_dir}' not found.")
        return

    # Ensure the destination directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize extensions for filtering
    if extensions:
        normalized_exts = {f".{ext.lstrip('.').lower()}" for ext in extensions}
        print(f"Extracting files with extensions: {', '.join(normalized_exts)}")
    else:
        print("Extracting all files from the archive.")

    extracted_count = 0
    with zipfile.ZipFile(input_dir, 'r') as zip_ref:
        # Get a list of all files in the zip archive
        file_list = zip_ref.namelist()

        for file_name in file_list:
            # Check if the file is a directory (ends with '/')
            if file_name.endswith('/'):
                continue

            # Apply the extension filter if it exists
            if extensions and Path(file_name).suffix.lower() not in normalized_exts:
                continue # Skip files that don't match

            zip_ref.extract(file_name, output_dir)
            extracted_count += 1
            
    print(f"Successfully extracted {extracted_count} files to '{output_dir}'.")


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    # Characters forbidden in filenames on Windows, Linux, and macOS
    # \/:*?"<>| are Windows-specific; \x00 is a null byte (Unix/Windows)
    valid_chars = r'[A-Za-z0-9-_ ËéèêëôöûüàâäîïçÏ]'
    for i in filename:
        if not re.match(valid_chars, i):
            filename = filename.replace(i, replacement)
    
    # Remove leading/trailing whitespace and dots (Windows restriction)
    sanitized = filename.strip().strip('.')
    
    # Avoid empty filenames
    return sanitized if sanitized else "untitled"


def sanitize_filename_new(filename: str) -> str:
    """
    Takes a string and returns a valid, safe filename for all major OSes.

    1. Replaces all invalid characters with an underscore.
    2. Checks against a list of reserved Windows names.
    3. Trims leading/trailing spaces and dots.
    4. Limits the length to a reasonable maximum.
    """
    # 1. Define invalid characters (a combination of Windows and Unix restrictions)
    # The regex `[\\/:"*?<>|]` will match any of the characters inside the brackets.
    # The `\\` is to escape the backslash in the regex pattern.
    invalid_chars = r'[\\/:"*?<>|]'
    sanitized = re.sub(invalid_chars, '_', filename)

    # 2. Define reserved names on Windows (case-insensitive)
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }

    # Split name and extension to check the base name
    name_part, ext_part = os.path.splitext(sanitized)
    if name_part.upper() in reserved_names:
        name_part = '_' + name_part  # Prepend an underscore if it's a reserved name
    
    sanitized = name_part + ext_part

    # 3. Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')

    # 4. Limit the length of the filename to a reasonable value (e.g., 200)
    # Note: this is for the filename itself, not the whole path.
    #max_len = 200
    #if len(sanitized) > max_len:
    #    name_part, ext_part = os.path.splitext(sanitized)
    #    # Truncate the name part, not the extension
    #    name_part = name_part[:max_len - len(ext_part) - 1]
    #    sanitized = name_part + ext_part
        
    # Ensure the filename is not empty after sanitization
    if not sanitized:
        return "_empty_filename_"

    return sanitized

def ensure_dir_for_file(file_path: str):
    """Ensures that the directory for a given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# ==============================================================================
#  data structures
# ==============================================================================

def flatten_list(list_of_lists):
    for item in list_of_lists:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item

def dedupe_list(input_list):
    return list(dict.fromkeys(input_list))

def list_dupes(
        input_list,
        answer = None,
    ):
    if len(input_list)!=len(set(input_list)):
        if answer == None:
            answer = input('continue, indices, dedupe or last two?(c/i/d/a):')
        if answer == 'c':
            return
        elif answer == 'i':
            counts = Counter(input_list)
            dupes = [item for item, count in counts.items() if count > 1]
            indices = [index for index, value in enumerate(input_list) if value in dupes]
            return dupes, indices
        elif answer == 'd':
            return list(dict.fromkeys(input_list))
        elif answer == 'a':
            counts = Counter(input_list)
            dupes = [item for item, count in counts.items() if count > 1]
            indices = [index for index, value in enumerate(input_list) if value in dupes]
            new_list = list(dict.fromkeys(input_list))
            return new_list, indices
        else:
            print('invalid answer')
            return
    else:
        print('no dupes')
        return
        
def single(func):
    def wrapper(*args, **kwargs):
        pass

def dict_dupes(list_of_dicts, keys):
    """
    Identifies duplicate dictionaries in a list based on a subset of keys.

    Args:
        list_of_dicts (list): The list of dictionaries to check.
        keys (list): A list of key names to use for identifying duplicates.

    Returns:
        list: A list of dictionaries that are considered duplicates.
    """
    seen = set()
    duplicates = []
    for d in list_of_dicts:
        # Create a tuple of the values for the keys to check.
        # A tuple is used because it's hashable and can be added to a set.
        identifier = tuple(d.get(key) for key in keys)
        if identifier in seen:
            duplicates.append(d)
        else:
            seen.add(identifier)
    return duplicates


class CorpusManager:
    def __init__(self, 
                 path_merged: Path, 
                 path_archive: Path, 
                 path_temp_input: Path, 
                 path_temp_output: Path, 
                 path_marker_zip: Path):
        
        self.p_meta = Path(path_merged)
        self.p_archive = Path(path_archive)
        self.p_temp_in = Path(path_temp_input)
        self.p_temp_out = Path(path_temp_output)
        self.p_marker_zip = Path(path_marker_zip)

        # Ensure directories exist
        self.p_temp_in.mkdir(parents=True, exist_ok=True)
        self.p_temp_out.mkdir(parents=True, exist_ok=True)

        # Load Data
        self.df = pd.read_parquet(self.p_meta, dtype_backend='pyarrow')
        
        # Ensure IDs
        if self.df.id.duplicated().any():
            duplicate_rows = self.df[self.df['id'].duplicated(keep=False)]
            print(f'\nWARNING: DUPLICATE ID DETECTED:')
            print(duplicate_rows.sort_values(by='id'))
            raise ValueError(f'Init aborted: duplicate ids')
        
        self.valid_ids = set(self.df['id'].unique())

    def save_metadata(self):
        """Saves the dataframe back to parquet."""
        self.df.to_parquet(self.p_meta, engine='pyarrow')
        print(f"Metadata saved to {self.p_meta}")

    # =========================================================================
    # 1. AUDIT FUNCTIONALITY
    # =========================================================================
    def audit_consistency(self):
        """
        Points out inconsistencies across all 4 containers.
        Returns a DataFrame containing the status of every ID.
        """
        print("Auditing containers...")
        
        # 1. Map Archive Content
        archive_map = set()
        if self.p_archive.exists():
            with zipfile.ZipFile(self.p_archive, 'r') as z:
                # Store IDs found in zip (stripping extensions)
                archive_map = {Path(f).stem for f in z.namelist()}

        # 2. Map Temp Output (Marker Folders)
        # Looking for folders named 'id'
        temp_out_map = {p.name for p in self.p_temp_out.iterdir() if p.is_dir()}

        # 3. Map Marker Zip (Final Output)
        marker_zip_map = set()
        if self.p_marker_zip.exists():
            with zipfile.ZipFile(self.p_marker_zip, 'r') as z:
                # We expect flat files like 457.md, 457.json. 
                # We check if ID exists in any form
                marker_zip_map = {Path(f).stem for f in z.namelist()}

        # Build Status
        results = []
        for _, row in self.df.iterrows():
            uid = row['id']
            results.append({
                'id': uid,
                'in_metadata': True, # Obviously
                'downloaded_flag': row['downloaded'], # What the DB thinks
                'in_archive_zip': uid in archive_map, # What exists physically
                'in_temp_output': uid in temp_out_map,
                'in_marker_zip': uid in marker_zip_map
            })
        
        audit_df = pd.DataFrame(results)
        
        # Print Summary of Inconsistencies
        # Example: Flagged downloaded but not in Zip
        missing_physical = audit_df[(audit_df['downloaded_flag'] == True) & (audit_df['in_archive_zip'] == False)]
        if not missing_physical.empty:
            print(f"\n[!] CRITICAL: {len(missing_physical)} rows claim to be downloaded but are missing from {self.p_archive.name}")
            print(missing_physical['id'].head().tolist())

        # Example: In output but not in final zip
        pending_pack = audit_df[(audit_df['in_temp_output'] == True) & (audit_df['in_marker_zip'] == False)]
        if not pending_pack.empty:
            print(f"\n[i] INFO: {len(pending_pack)} items are processed but not yet packed into {self.p_marker_zip.name}")

        # SEE COMBINATIONS OF ALL 
        cols = ['in_metadata', 'downloaded_flag', 'in_archive_zip', 'in_temp_output', 'in_marker_zip']

        # 1. General Request: See counts for EVERY combination
        # This creates a table showing which patterns exist and how frequent they are
        combination_counts = audit_df[cols].value_counts().reset_index(name='count')

        print("--- All Status Combinations ---")
        print(combination_counts.to_string(index=False))

        return audit_df
    
    def sync_metadata_with_archive(self):
        """
        Updates self.df to set 'downloaded' = True for any file 
        that physically exists in the archive.zip.
        """
        print("Syncing metadata with physical archive...")
        
        if not self.p_archive.exists():
            print("Archive zip does not exist.")
            return

        # 1. Get all IDs currently in the zip
        with zipfile.ZipFile(self.p_archive, 'r') as z:
            # strip extensions to get the raw ID
            ids_in_zip = {Path(f).stem for f in z.namelist()}
        
        # 2. Identify rows that are WRONG (Zip says yes, DF says no)
        # We look for IDs in the zip where the dataframe thinks it's NOT downloaded
        mask = (self.df['id'].isin(ids_in_zip)) & (self.df['downloaded'] != True)
        
        count_to_fix = mask.sum()
        
        if count_to_fix == 0:
            print("  -> Metadata is already in sync with archive.")
        else:
            # 3. Update the DataFrame
            self.df.loc[mask, 'downloaded'] = True
            self.save_metadata()
            print(f"  -> FIXED: Updated {count_to_fix} rows to 'downloaded=True' based on zip content.")

    # =========================================================================
    # 2. PRUNE FUNCTIONALITY (Remove extraneous files)
    # =========================================================================
    def prune_extraneous_files(self):
        """
        Removes files from Zips and Folders if their ID is not in the DataFrame.
        """
        print("Pruning extraneous files...")
        
        # A. Prune Archive Zip (Requires Rebuild)
        if self.p_archive.exists():
            self._prune_zip_file(self.p_archive, "Archive")

        # B. Prune Marker Zip (Requires Rebuild)
        if self.p_marker_zip.exists():
            self._prune_zip_file(self.p_marker_zip, "Marker Output")

        # C. Prune Temp Input Folder
        self._prune_folder(self.p_temp_in)
        
        # D. Prune Temp Output Folder
        self._prune_folder(self.p_temp_out)

    def _prune_zip_file(self, zip_path: Path, label: str):
        """Helper to rebuild a zip file excluding invalid IDs."""
        temp_zip_path = zip_path.with_suffix('.tmp.zip')
        removed_count = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zin, zipfile.ZipFile(temp_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for item in tqdm(zin.infolist(), desc=f"Pruning {label} Zip"):
                file_id = Path(item.filename).stem
                if file_id in self.valid_ids:
                    zout.writestr(item, zin.read(item.filename))
                else:
                    removed_count += 1
        
        if removed_count > 0:
            zip_path.unlink() # Delete old
            temp_zip_path.rename(zip_path) # Move new to old
            print(f"  -> Removed {removed_count} files from {zip_path.name}")
        else:
            temp_zip_path.unlink() # Clean up temp if nothing changed
            print(f"  -> {zip_path.name} is clean.")

    def _prune_folder(self, folder_path: Path):
        """Helper to delete files/folders not in ID list."""
        removed = 0
        for item in folder_path.iterdir():
            file_id = item.stem 
            if file_id not in self.valid_ids:
                if item.is_file(): item.unlink()
                else: shutil.rmtree(item)
                removed += 1
        if removed > 0:
            print(f"  -> Removed {removed} items from {folder_path.name}")

    # =========================================================================
    # 3. DOWNLOAD LOGIC
    # =========================================================================
    def download_missing(self):
        """
        Downloads missing files and appends them to the archive zip.
        """
        # 1. Identify rows to download
        mask = self.df['downloaded'] != True
        to_download_indices = self.df[mask].index

        if to_download_indices.empty:
            print("  -> No files to download.")
            return

        print(f"  -> Found {len(to_download_indices)} items to download.")

        # 2. Pre-scan existing files in Zip (Read Mode)
        # We do this before opening in 'append' mode to ensure we have a clean list
        existing_ids_in_zip = set()
        if self.p_archive.exists():
            try:
                with zipfile.ZipFile(self.p_archive, 'r') as z:
                    existing_ids_in_zip = {Path(f).stem for f in z.namelist()}
            except zipfile.BadZipFile:
                print(f"  [!] Warning: {self.p_archive} seems corrupted. It might be overwritten or cause errors.")

        success_count = 0
        batch_count = 0

        # 3. Open Zip ONCE in Append Mode
        with zipfile.ZipFile(self.p_archive, 'a', zipfile.ZIP_DEFLATED) as z_out:
            
            for idx in tqdm(to_download_indices, desc="Downloading Files"):
                row = self.df.loc[idx]
                uid = row['id']
                url = row['download_link']

                # no dl link -> downloaded = false
                if pd.isna(row['download_link']):
                    self.df.loc[idx, 'downloaded'] = False
                    continue

                # Safety Check: Skip if ID already exists physically in zip
                if uid in existing_ids_in_zip:
                    self.df.loc[idx, 'downloaded'] = True
                    continue

                # Pass the OPEN zip handle (z_out) to the helper
                if self._download_file(url, uid, z_out):
                    self.df.loc[idx, 'downloaded'] = True
                    existing_ids_in_zip.add(uid) # Add to local set to prevent dups in same batch
                    success_count += 1
                else:
                    self.df.loc[idx, 'downloaded'] = False

                # Periodic Metadata Save (Every 50 items)
                batch_count += 1
                if batch_count >= 50:
                    self.save_metadata()
                    batch_count = 0

        # Final Save
        self.save_metadata()
        print(f"  -> Download process complete. {success_count} new files added.")

    def _download_file(self, download_url: str, filename_in_zip: str, open_zip_handle: zipfile.ZipFile):
        """
        Downloads a file and writes it to the provided open zip handle.
        """
        mime_to_extension = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/octet-stream": "bin"
        }
        
        # 1. Download
        response = timed_request(download_url, timeout=30)
        if not response:
            return False

        # 2. Determine Extension
        content_type = response.headers.get("Content-Type", "").lower().split(';')[0].strip()
        extension = None

        # Priority A: Content-Type Header
        for mime, ext in mime_to_extension.items():
            if mime == content_type:
                extension = ext
                break
        
        # Priority B: URL Guess (Fallback)
        if not extension or extension == "bin":
            guessed = mimetypes.guess_extension(content_type)
            if guessed:
                extension = guessed.lstrip('.')
            elif '.' in download_url:
                # Try to grab extension from url path
                possible_ext = download_url.split('?')[0].split('.')[-1].lower()
                if possible_ext in ['pdf', 'docx']:
                    extension = possible_ext

        if not extension:
            print(f"  -> [Skip] Unknown file type '{content_type}' for {download_url}")
            return False

        # 3. Write to Open Zip Handle
        full_filename = f"{filename_in_zip}.{extension}"
        try:
            open_zip_handle.writestr(full_filename, response.content)
            return True
        except Exception as e:
            print(f"  -> [Error] Writing to zip failed for {full_filename}: {e}")
            return False

    # =========================================================================
    # 4. EXTRACTION (STAGING) LOGIC
    # =========================================================================
    def stage_for_processing(self):
        """
        Extracts files from Archive Zip to Temp Input folder IF:
        1. They are not in Temp Output (already processed)
        2. They are not in Marker Zip (already packed)
        """
        print("Staging files for processing...")
        
        # Get list of already completed IDs
        completed_ids = set()
        
        # Check Temp Output (Folders)
        completed_ids.update([p.name for p in self.p_temp_out.iterdir()])
        
        # Check Marker Zip
        if self.p_marker_zip.exists():
            with zipfile.ZipFile(self.p_marker_zip, 'r') as z:
                completed_ids.update([Path(f).stem for f in z.namelist()])

        # Iterate Archive and Extract needed
        extracted_count = 0
        if not self.p_archive.exists():
            print("Archive zip not found.")
            return

        with zipfile.ZipFile(self.p_archive, 'r') as z:
            # Filter file list: Valid ID AND Not Completed
            files_to_extract = []
            for f in z.namelist():
                fid = Path(f).stem
                if fid in self.valid_ids and fid not in completed_ids:
                    files_to_extract.append(f)
            
            # Extract
            for f in tqdm(files_to_extract, desc="Extracting to Temp"):
                z.extract(f, self.p_temp_in)
                extracted_count += 1

        print(f"Staged {extracted_count} files to {self.p_temp_in}")

    # =========================================================================
    # 5. SUGGESTED: PACK LOGIC
    # =========================================================================
    def pack_processed_output(self):
        """
        Moves processed folders from Temp Output into the flat Marker Zip.
        """
        print("Packing processed files...")
        
        processed_folders = [p for p in self.p_temp_out.iterdir() if p.is_dir()]
        
        if not processed_folders:
            print("No folders in temp output to pack.")
            return

        with zipfile.ZipFile(self.p_marker_zip, 'a', compression=zipfile.ZIP_DEFLATED) as z:
            existing_in_zip = set(z.namelist())
            
            for folder in tqdm(processed_folders, desc="Packing to Zip"):
                # Marker output structure: folder 458/ -> 458.md, 458.json, meta.json
                # We want flat structure in zip: 458.md, 458.json
                
                for file_path in folder.iterdir():
                    if file_path.name in existing_in_zip:
                        continue
                    
                    # Only add if it matches the ID (ignore random metadata logs if needed)
                    if folder.name in file_path.name: 
                        z.write(file_path, arcname=file_path.name)
        
        print("Packing complete.")



#  DATAFRAMECLEANER
    # TODO
        # checken voor type outliers
        # include exclude list voor placeholders + specific placeholders
        # per colom # unique waarden
        # 
import pandas as pd
import numpy as np
import re
from collections import Counter
import pyarrow as pa
import ast

class SchemaEnforcer:
    def __init__(self, df, regex_patterns, protected_values=None):
        self.df = df
        self.garbage_regex = re.compile(regex_patterns)
        
        # Track detailed statistics
        # Structure: {col: {'total': 0, 'placeholders': 0, 'mismatch': 0, 'examples': []}}
        self.stats = {} 
        
        self.protected_values = {k: set(map(str, v)) for k, v in (protected_values or {}).items()}
        self.true_vals = {'y', 'yes', 't', 'true', '1', 'on'}
        self.false_vals = {'n', 'no', 'f', 'false', '0', 'off'}

    def _is_garbage(self, val, col_name=None):
        """Row-level check used inside apply/map functions."""
        if val is None or val is pd.NA: return True
        if isinstance(val, (float, int, np.float64, np.int64)) and np.isnan(val): return True

        s_val = str(val).strip()

        if col_name and col_name in self.protected_values:
            if s_val in self.protected_values[col_name]:
                return False 

        if not s_val: return True
        if self.garbage_regex.match(s_val): return True
        return False
    
    # --- Cleaning Functions ---
    def _prune_empty(self, obj):
        """Recursively removes empty structures."""
        # keeps position by adding none
        if isinstance(obj, list):
            cleaned = [self._prune_empty(x) for x in obj]
            if all(x is None for x in cleaned):
                return None
            return cleaned

        # agressive: removes key, should work with pyarrow
        elif isinstance(obj, dict):
            cleaned = {k: self._prune_empty(v) for k, v in obj.items()}
            cleaned = {k: v for k, v in cleaned.items() if v is not None}
            return cleaned if cleaned else None
        # Base case
        return None if self._is_garbage(obj) else obj
    
    def _clean_complex(self, val, expected_type, col_name):
        if self._is_garbage(val, col_name): return None
        if isinstance(val, np.ndarray): val = val.tolist()
        if isinstance(val, str):
            try: val = ast.literal_eval(val)
            except: return None
        if isinstance(val, expected_type):
            return self._prune_empty(val)
        else:
            if expected_type is list and val is not None:
                print(f'UNEXPECTED DTYPE: {col_name}: {val}\nAttempting recovery through listing value.')
                new_val = [val]
                return self._prune_empty(new_val)
            else:
                print(f'UNEXPECTED DTYPE: {col_name}: {val}\nIrrecoverable: {expected_type} value set to None')
                return None
    
    def _clean_bool(self, val, col_name):
        if self._is_garbage(val, col_name): return None
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)):
            if val == 1: return True
            if val == 0: return False
            return None
        if isinstance(val, str):
            s = val.strip().lower()
            if s in self.true_vals: return True
            if s in self.false_vals: return False
        return None
    
    def _clean_scalar_str(self, val, col_name):
        # 1. Immediate Garbage Check
        if self._is_garbage(val, col_name): return None
        
        # 2. Handle PyArrow Scalars (Unbox to Python objects)
        if hasattr(val, 'as_py'):
            val = val.as_py()

        # 3. String Recovery (Parse "['text']" or "{'text'}")
        if isinstance(val, str):
            val = val.strip()
            if val.startswith(('[', '{')):
                try:
                    parsed = ast.literal_eval(val)
                    # Recursively clean the parsed object
                    return self._clean_scalar_str(parsed, col_name)
                except:
                    pass # Keep as string if parsing fails
            
            if self._is_garbage(val, col_name): return None
            return val

        # 4. Container Recovery (List, Tuple, Array, AND SET)
        # Added 'set' here to fix the volume issue: {9141} -> 9141
        if isinstance(val, (list, tuple, np.ndarray, set)):
            # Convert to list for consistent indexing
            if isinstance(val, (np.ndarray, set)): 
                val = list(val)
            
            # Filter out internal garbage
            valid_items = [v for v in val if not self._is_garbage(v, col_name)]
            
            if not valid_items:
                return None
            
            # RECOVERY STRATEGY: Take the first valid item
            # Recursively clean it (handles [['Text']])
            return self._clean_scalar_str(valid_items[0], col_name)

        # 5. Dict Recovery (Still unsafe to guess key, drop)
        if isinstance(val, dict):
            return None

        # 6. Numeric/Bool to String
        return str(val).strip()


    # --- Analysis Helper (UPDATED) ---
    def _scan_placeholders(self, col):
        """
        Vectorized scan to count AND identify regex matches.
        Returns: (count, list_of_unique_matches)
        """
        try:
            # 1. Get non-null values as strings
            s = self.df[col].dropna().astype(str).str.strip()
            if s.empty: return 0, []

            # 2. Exclude Protected Values
            if col in self.protected_values:
                mask_protected = s.isin(self.protected_values[col])
                s = s[~mask_protected]

            if s.empty: return 0, []

            # 3. Find Matches
            # Get boolean mask of garbage
            mask_garbage = s.str.match(self.garbage_regex)
            
            # Filter the series to just the garbage
            garbage_values = s[mask_garbage]
            
            count = garbage_values.shape[0]
            
            # Get unique examples (limit to top 5 to avoid massive lists)
            if count > 0:
                examples = garbage_values.unique().tolist()
            else:
                examples = []
                
            return count, examples
        except:
            return 0, []

    def apply(self, schema):
        print(f"--- Enforcing Schema on {len(schema)} columns ---")
        
        for col, dtype in schema.items():
            if col not in self.df.columns: continue
            
            initial_valid = self.df[col].notna().sum()
            
            # NEW: Get count AND specific values found
            n_placeholders, found_examples = self._scan_placeholders(col)

            # --- Transformation ---
            if dtype == 'list':
                self.df[col] = self.df[col].apply(lambda x: self._clean_complex(x, list, col))
            elif dtype == 'dict':
                self.df[col] = self.df[col].apply(lambda x: self._clean_complex(x, dict, col))
            elif dtype == 'bool':
                self.df[col] = self.df[col].map(lambda x: self._clean_bool(x, col))
            elif dtype == 'string':
                self.df[col] = self.df[col].map(lambda x: self._clean_scalar_str(x, col))
            elif dtype in ['int', 'float', 'number']:
                if col in self.protected_values:
                     self.df[col] = self.df[col].apply(lambda x: np.nan if self._is_garbage(x, col) else x)
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            elif dtype in ['date', 'datetime']:
                if col in self.protected_values:
                     self.df[col] = self.df[col].apply(lambda x: np.nan if self._is_garbage(x, col) else x)
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

            final_valid = self.df[col].notna().sum()
            
            # --- Stats Calculation ---
            total_cleaned = initial_valid - final_valid
            n_mismatch = max(0, total_cleaned - n_placeholders)
            
            # Consistency check
            if total_cleaned < n_placeholders:
                n_placeholders = total_cleaned

            if total_cleaned > 0:
                self.stats[col] = {
                    'total': total_cleaned,
                    'placeholders': n_placeholders,
                    'mismatch': n_mismatch,
                    'examples': found_examples # Store the specific values found
                }

        # --- Print Detailed Report (UPDATED) ---
        if self.stats:
            print("\n[Cleaning Report] Values converted to NaN:\n")
            # Adjusted widths to fit the new column
            print(f"{'Column':<20} | {'Total':<6} | {'Regex Match':<11} | {'Type Mismatch':<13} | {'Detected Garbage'}")
            print("-" * 100)
            
            for col, data in self.stats.items():
                # Format the examples list as a string, truncate if too long
                ex_str = str(data['examples'])
                if len(ex_str) > 40: 
                    ex_str = ex_str[:37] + "..."
                
                print(f"{col:<20} | {data['total']:<6} | {data['placeholders']:<11} | {data['mismatch']:<13} | {ex_str}")
        else:
            print("\n[Cleaning Report] No values were converted to NaN.\n")

        return self.df
    

class DataFrameCleaner:
    """
    A class to encapsulate a pandas DataFrame and apply a series of
    common cleaning operations using method chaining.
    """

    def __init__(self, data):
        # Create a copy to avoid setting warnings or modifying the original variable unexpectedly
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
            print('Dataframe cleaner initialized, working on copy')
        if isinstance(data, (Path, str)):
            try: path = Path(data)
            except: raise ValueError(f'couldnt convert string to path, invalid path')
            if not path.exists(): raise FileNotFoundError(f'file not found {path}')
            if path.suffix.lower() != '.parquet': raise ValueError(f'Input parquet file. got {path.suffix}')

            self.df = pd.read_parquet(path, engine='pyarrow',dtype_backend='pyarrow')
            print(f'Dataframe cleaner initialized.')



        pd.set_option('display.max_rows', None)
        print('Showing all rows, set back with pd.set_option(\'display.max_rows\', 10 idk)')


        # Regex patterns to identify "garbage" values that act as Nulls
        patterns = [
            r'(?i)^nan$',  # Case insensitive 'nan'
            r'(?i)^(?:n\/?a|null|none|<none>|not reported|unknown|undefined|missing)$',
            r'^(?:-+$|/+$)',  # Dashes or slashes only
            r'^\?+$',         # Question marks
            r'^(?:-99|-9999|999|9999)$', # Common numeric placeholders
        ]
        self.na_placeholders = '|'.join(patterns)

    # HELPERS
    def _get_placeholders(self, col):
        """Helper: Returns a list of unique values matching the garbage regex."""
        try:
            # Dropna first to avoid nan errors in unique
            unique_vals = self.df[col].dropna().unique()
            
            # Convert only the unique values to string for regex checking
            u_series = pd.Series(unique_vals).astype(str)
            
            # Check pattern match
            matches = u_series[u_series.str.match(self.na_placeholders, na=False)]
            
            if not matches.empty:
                return matches.tolist()
        except Exception:
            return None
        return None
    
    def _find_unhashable_columns(self):
        """
        Identifies columns that contain complex nested data (Lists, Dicts, Structs, Arrays).
        Supports both legacy 'object' columns and modern 'ArrowDtype' columns.
        """
        import pyarrow as pa
        
        unhashable_cols = []

        for col in self.df.columns:
            dtype = self.df[col].dtype

            # --- STRATEGY 1: Check PyArrow Dtypes (Metadata) ---
            if isinstance(dtype, pd.ArrowDtype):
                pa_type = dtype.pyarrow_dtype
                # Check for nested Arrow types
                if (pa.types.is_list(pa_type) or 
                    pa.types.is_large_list(pa_type) or 
                    pa.types.is_fixed_size_list(pa_type) or 
                    pa.types.is_struct(pa_type) or 
                    pa.types.is_map(pa_type)):
                    
                    unhashable_cols.append(col)
                    continue

            # --- STRATEGY 2: Check Object Dtypes (Value Inspection) ---
            if pd.api.types.is_object_dtype(dtype):
                valid = self.df[col].dropna()
                if not valid.empty:
                    if isinstance(valid.iloc[0], (list, dict, set, np.ndarray)):
                        unhashable_cols.append(col)

        return unhashable_cols
    
    def _analyze_arrow_data(self, array, indent=0):
            """
            Helper: Recursively walks through PyArrow DATA arrays.
            - If List: Flattens it and recurses.
            - If Struct: Extracts fields and recurses.
            - If Atomic: Calculates unique counts, gets sample, and formats output.
            """
            import pyarrow as pa
            import pyarrow.compute as pc
            
            prefix = " " * indent
            arrow_type = array.type
            
            # --- ALIGNMENT SETTINGS ---
            # Adjust these to change the visual width of columns
            KEY_WIDTH = 30    # Width for "  * fieldname:"
            TYPE_WIDTH = 15   # Width for "int64", "string", etc.
            COUNT_WIDTH = 15  # Width for "123 unique"
            SAMPLE_WIDTH = 40 # Width for sample text

            # CASE 1: LIST
            if pa.types.is_list(arrow_type):
                print(f"{prefix}- List of:")
                flattened = array.flatten()
                self._analyze_arrow_data(flattened, indent + 2)
                
            # CASE 2: STRUCT / DICT
            elif pa.types.is_struct(arrow_type):
                print(f"{prefix}- Object (Dict) with keys:")
                
                for field in arrow_type:
                    # Extract the field column
                    child_array = array.field(field.name)
                    
                    # ALIGNMENT FIX: Pad the key name so the next value starts aligned
                    # We construct the label "  * name:"
                    label = f"{prefix}  * {field.name}:"
                    
                    # If the child is complex, we print the label and newline
                    if pa.types.is_list(field.type) or pa.types.is_struct(field.type):
                        print(label) 
                        self._analyze_arrow_data(child_array, indent + 6)
                    # If the child is simple, we print the label PADDIED and stay on same line
                    else:
                        # Use ljust to ensure the Type starts at the same horizontal position
                        # We subtract the indent from KEY_WIDTH to keep alignment relative to nesting
                        print(f"{label.ljust(KEY_WIDTH + indent)}", end="")
                        self._analyze_arrow_data(child_array, indent=0)

            # CASE 3: ATOMIC -> STATS & SAMPLE
            else:
                total_count = len(array)
                if total_count == 0:
                    print(f"{arrow_type} (Empty)")
                    return

                # 1. Get Stats
                unique_vals = pc.unique(array)
                n_unique = len(unique_vals)
                
                ratio = n_unique / total_count
                if n_unique == 1: cat_label = "CONSTANT"
                elif ratio > 0.9: cat_label = "ID/TEXT"
                elif ratio < 0.1 or n_unique < 50: cat_label = "CATEGORY"
                else: cat_label = "DENSE"

                # 2. Get Sample (Safely)
                # Slice first 50 items to find a non-null without scanning everything
                sample_slice = array.slice(0, 50) 
                non_null_slice = sample_slice.drop_null()
                
                if len(non_null_slice) > 0:
                    val = non_null_slice[0].as_py() # Convert to Python object
                    val_str = str(val).replace('\n', ' ') # Remove newlines for clean print
                    if len(val_str) > SAMPLE_WIDTH - 3: 
                        val_str = val_str[:SAMPLE_WIDTH-3] + "..."
                else:
                    val_str = "NULL"

                # 3. Format Output (Columns)
                # If indent is 0, we are inline with a Key, so don't print prefix
                current_prefix = prefix if indent > 0 else ""
                
                # Formatted strings using f-string padding
                # < : Left Align, > : Right Align
                type_str = f"{str(arrow_type)}".ljust(TYPE_WIDTH)
                count_str = f"{n_unique} unique".rjust(10) # e.g. "   5 unique" #TODO where unique counts are below 10 or so, print all possible values
                sample_str = f"sample: {val_str}".ljust(SAMPLE_WIDTH + 8) # +8 for "sample: " length

                print(f"{current_prefix}{type_str} | {count_str} | {sample_str} | ({cat_label})")

    def _get_python_type_name(self, val):
        """Helper to get a clean string representation of a type."""
        return type(val).__name__



    # --- 1. CHECKING METHODS (Inspect without modifying) ---
    def summarize(self):
            """
            Prints a comprehensive summary table combining dtypes, missing values, 
            samples, and detected placeholder garbage values.
            """
            print(f"\n--- DataFrame Summary (Shape: {self.df.shape} (rows, cols)) ---")
            print(f"\nDUPLICATES")
            self.check_duplicates()
            print('-'*20)
            print(f"\nGetting dataframe overview...")
            summary = pd.DataFrame(index=self.df.columns)
            summary['missing#'] = self.df.isna().sum()
            summary['missing%'] = (self.df.isna().mean() * 100).round(2)
            summary['dtypes'] = self.df.dtypes.astype(str)

            unhashable_cols = self._find_unhashable_columns()
            unique_counts = pd.Series(0, index=self.df.columns, dtype='int64')
            hashable_cols = [c for c in self.df.columns if c not in unhashable_cols]
            if hashable_cols:
                unique_counts[hashable_cols] = self.df[hashable_cols].nunique(dropna=True)
                
            for col in unhashable_cols:
                unique_counts[col] = -1 # MAYBE CHECK PER ATOMIC VALUE, PLACEHOLDER, TODO

            summary['unique#'] = unique_counts.astype(int)
            # ----------------------------------

            samples = []
            found_placeholders = []
            
            for col in self.df.columns:
                valid_values = self.df[col].dropna()
                if not valid_values.empty:
                    val_str = str(valid_values.iloc[0])
                    samples.append(val_str[:40] + "..." if len(val_str) > 40 else val_str)
                else:
                    samples.append(np.nan)

                if col in hashable_cols:
                    matches = self._get_placeholders(col)
                    if matches:
                        found_placeholders.append(str(matches))
                    else:
                        found_placeholders.append("")
                else:
                    found_placeholders.append("")

            summary['sample_val'] = samples
            summary['placeholders'] = found_placeholders

            order = ['missing#', 'missing%', 'unique#', 'sample_val', 'placeholders', 'dtypes']
            summary = summary[order]
    
            # --- FORCE LEFT ALIGNMENT ---
            max_len = summary['dtypes'].map(len).max() if not summary.empty else 10
            
            # 2. Define a formatter
            formatters = {
                'dtypes': lambda x: f"{x:<{max_len}}"
            }

            # 3. Apply formatter in to_string
            print(summary.sort_values('dtypes', ascending=False).to_string(formatters=formatters))
            print('\n')
            print("-"*20)

            self.audit_mixed_types()
            print('-'*20)

            if unhashable_cols:
                print(f"\nChecking unhashable column structures...")
                self.analyze_structure_recursive()
            else:
                print('\nNo complex nested columns found.')
            return self
    
    
    def analyze_structure_recursive(self, sample_size=5000):
            """
            Converts unhashable columns to PyArrow arrays to infer the schema
            AND calculates unique value counts at every level of nesting.
            """
            import pyarrow as pa
            
            unhashable_cols = self._find_unhashable_columns()
            if not unhashable_cols:
                return self

            print(f"\n--- Deep Structure & Stats (Sample n={sample_size}) ---")

            for col in unhashable_cols:
                # 1. Get Sample Data
                valid_data = self.df[col].dropna()
                if valid_data.empty:
                    continue
                
                # Cap sample size
                if len(valid_data) > sample_size:
                    valid_data = valid_data.sample(n=sample_size, random_state=42)
                
                print(f"Column '{col}':")
                try:
                    # 2. Create PyArrow Array (Holds the actual data in memory)
                    arrow_array = pa.array(valid_data)
                    
                    # 3. Pass the DATA (not just the type) to the helper
                    self._analyze_arrow_data(arrow_array, indent=2)
                    
                except pa.ArrowInvalid:
                    print(f"  [!] Mixed types detected. Cannot infer strict schema.")
                except Exception as e:
                    print(f"  [!] Error: {e}")
                
                print("") 
                
            print("-----------------------------------------------------\n")
            return self

    def get_samples(self, columns=None, number=5):  # add func for str as well as list
        valid = [c for c in columns or [] if c in self.df.columns]
        if (invalid := set(columns or []) - set(valid)): print(f"Invalid: {invalid}")
        for col in valid:
            s = self.df[col].dropna()
            print(f"Column: {col}, {type(s.iloc[0])}", *s.sample(min(number, len(s))), "-" * 30, sep="\n") 
            
        return self

    def audit_mixed_types(self, verbose=True):
        """
        Analyzes object columns. Returns a dict containing detailed stats about mixed types.
        Structure: {col: {'majority_type': 'str', 'breakdown': {...}, 'outliers': [val1, val2]}}
        """
        report = {}
        
        # Only check object columns (where mixing happens)
        candidates = self.df.select_dtypes(include=['object']).columns
        
        for col in candidates:
            # Drop NAs to check actual values
            valid_series = self.df[col].dropna()
            if valid_series.empty: continue

            # Get type counts
            type_counts = valid_series.apply(self._get_python_type_name).value_counts()
            
            if len(type_counts) > 1:
                # Identify Majority and Minority
                majority_type = type_counts.idxmax()
                majority_count = type_counts.max()
                total_count = type_counts.sum()
                
                # Find samples of the minority types (the "Outliers")
                # We iterate to find indices where type != majority
                # Note: This can be slow on massive DFs, so we might want to sample if len > 1M
                types_series = valid_series.apply(self._get_python_type_name)
                outlier_mask = types_series != majority_type
                outlier_samples = valid_series[outlier_mask].head(5).tolist()
                
                report[col] = {
                    'majority_type': majority_type,
                    'majority_pct': (majority_count / total_count) * 100,
                    'breakdown': type_counts.to_dict(),
                    'outliers': outlier_samples
                }

        if verbose and report:
            print(f"\n⚠️  [Mixed Type Report] Found {len(report)} inconsistent columns:")
            print(f"{'Column':<20} | {'Majority':<8} | {'Breakdown':<30} | {'Outlier Samples'}")
            print("-" * 90)
            for col, data in report.items():
                breakdown_str = str(data['breakdown'])
                if len(breakdown_str) > 30: breakdown_str = breakdown_str[:27] + "..."
                print(f"{col:<20} | {data['majority_type']:<8} | {breakdown_str:<30} | {data['outliers']}")
        
        return report
    
    def check_missing_values(self):
        """Prints a report of missing values."""
        # Calculate isna once to save time
        na_counts = self.df.isna().sum()
        # Filter only columns with missing values
        missing_data = na_counts[na_counts > 0]
        
        if not missing_data.empty:
            print("\n--- Missing Values Report ---")
            # Vectorized calculation of percentage
            percentages = (missing_data / len(self.df)) * 100
            report = pd.DataFrame({
                'Missing Count': missing_data,
                'Percentage': percentages.round(2)
            })
            print(report)
        else:
            print("\nNo standard missing values (NaN) found.")
        return self

    def check_duplicates(self):
        """Checks for duplicate rows, skipping unhashable columns."""
        unhashable_cols = self._find_unhashable_columns()
        
        if unhashable_cols:
            print(f"\nWarning: Excluding unhashable columns from duplicate check: {unhashable_cols}")
            hashable_cols = [c for c in self.df.columns if c not in unhashable_cols]
            if not hashable_cols:
                print("No hashable columns to check.")
                return self
            dupes = self.df.duplicated(subset=hashable_cols).sum()
        else:
            dupes = self.df.duplicated().sum()

        if dupes > 0:
            print(f"\nFound {dupes} duplicate rows.")
        else:
            print("\nNo duplicates found.")
        return self

    
    def auto_infer_schema(self, sample_size=1000):
            import pyarrow as pa
            import numpy as np
            import ast

            inferred = {}
            # Optimization: Don't sample if df is small
            sample_size = min(len(self.df), sample_size)
            print(f"--- Auto-Inferring Schema (Sample n={sample_size}) ---")

            for col in self.df.columns:
                valid = self.df[col].dropna()
                if valid.empty: continue
                
                if len(valid) > sample_size: 
                    valid = valid.sample(n=sample_size, random_state=42)

                # --- PHASE 1: Native Checks (Fast) ---
                if pd.api.types.is_bool_dtype(self.df[col]): 
                    inferred[col] = 'bool'; continue
                if pd.api.types.is_integer_dtype(self.df[col]): 
                    inferred[col] = 'int'; continue
                if pd.api.types.is_float_dtype(self.df[col]): 
                    inferred[col] = 'float'; continue
                if pd.api.types.is_datetime64_any_dtype(self.df[col]): 
                    inferred[col] = 'datetime'; continue

                # --- PHASE 2: Mixed Type Guard (Crucial for Object Columns) ---
                # If a column contains multiple python types (e.g. str AND int), default to string
                # to avoid data loss.
                if pd.api.types.is_object_dtype(self.df[col]):
                    # We apply type() to get the class of each value
                    types_found = valid.apply(type).unique()
                    if len(types_found) > 1:
                        inferred[col] = 'string'
                        # Optional: Print warning if you want to know
                        print(f"  > MIXED TYPE Detected '{col}' as STRING (Mixed Types found)")
                        continue

                # --- PHASE 3: Numeric-as-Boolean Check ---
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    unique_nums = valid.unique()
                    # Check if it only contains 0 and 1
                    if set(unique_nums).issubset({0, 1, 0.0, 1.0}):
                        inferred[col] = 'bool'
                        print(f"  > Detected '{col}' as BOOL (Binary Numeric)")
                        continue
                    # Fallback to standard numeric
                    elif pd.api.types.is_integer_dtype(self.df[col]): inferred[col] = 'int'
                    else: inferred[col] = 'float'
                    continue

                # --- PHASE 4: Object Inspection ---
                try:
                    first_val = valid.iloc[0]
                    
                    # FIX: Handle PyArrow Scalars if data loaded with pyarrow backend
                    if hasattr(first_val, 'as_py'):
                        first_val = first_val.as_py()

                    is_complex_object = isinstance(first_val, (list, dict, set, np.ndarray))

                    # --- BOOLEAN DETECTION (String Scalar) ---
                    if not is_complex_object:
                        # Convert to string to safely check for 'yes'/'no'
                        unique_vals = valid.astype(str).str.lower().unique()
                        
                        if len(unique_vals) <= 10: 
                            u_series = pd.Series(unique_vals)
                            # Filter garbage using regex
                            mask_clean = ~u_series.str.match(self.na_placeholders)
                            clean_vals = u_series[mask_clean].tolist()
                            
                            if clean_vals:
                                if set(clean_vals).issubset(self.enforcer.true_vals | self.enforcer.false_vals):
                                    inferred[col] = 'bool'
                                    print(f"  > Detected '{col}' as BOOL (Semantic String)")
                                    continue

                    # --- PHASE 5: Deep Arrow Inference (Complex Types) ---
                    def normalize(x):
                        # FIX: Handle Arrow Scalars row-by-row
                        if hasattr(x, 'as_py'): x = x.as_py()
                        
                        if isinstance(x, np.ndarray): return x.tolist()
                        if isinstance(x, str) and x.strip().startswith(('[','{')):
                            try: return ast.literal_eval(x)
                            except: pass
                        return x
                    
                    sample_list = valid.apply(normalize).tolist()
                    arrow_type = pa.array(sample_list).type
                    
                    if pa.types.is_list(arrow_type): inferred[col] = 'list'; print(f"  > Detected '{col}' as LIST")
                    elif pa.types.is_struct(arrow_type) or pa.types.is_map(arrow_type): inferred[col] = 'dict'; print(f"  > Detected '{col}' as DICT")
                    else: inferred[col] = 'string'

                except Exception:
                    inferred[col] = 'string'
                    
            return inferred

    def show_duplicates(self, cols):
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            if col not in self.df.columns:
                continue
            counts = self.df[col].value_counts()
            duplicates = counts[counts>=2]

            print(f"\n--- '{col}' (Total Duplicates: {len(duplicates)}) ---")
            if not duplicates.empty:
                print(duplicates.to_string(header=False))
            else:
                print("No values with 2+ appearances.")
        



    # --- 2. CLEANING METHODS (Modify the DataFrame) ---
    def drop_short_strings(self, column, min_chars=10):
        """Sets values in a string column to NaN if they are too short."""
        if column in self.df.columns:
            # Calculate lengths (handling existing NaNs gracefully)
            lengths = self.df[column].astype(str).str.len()
            
            # Find indices where length is valid (not nan) but below threshold
            # We check 'notna' to distinguish between "short string" and "already empty"
            mask = (lengths < min_chars) & (self.df[column].notna())
            
            count = mask.sum()
            if count > 0:
                self.df.loc[mask, column] = np.nan
                print(f"Cleaned '{column}': Removed {count} values shorter than {min_chars} chars.")
        return self
    
    def reset_data(self, df_original):
        """
        Replaces the current (cleaned) dataframe with a fresh copy of the original.
        Useful for restarting the pipeline without re-initializing the whole class.
        """
        print("Resetting DataFrame to original state...")
        
        # Create a fresh copy so we don't mutate the external variable
        self.df = df_original.copy()
        
        # IMPORTANT: We must clear any previous analysis/enforcer state
        # Re-initialize the enforcer with the new data
        self.enforcer = SchemaEnforcer(self.df, self.na_placeholders)
        
        print(f"Reset complete. Shape: {self.df.shape}")
        return self

    def enforce_schema(self, schema_dict, protected_values=None):
        """
        The Master Cleaning Function. 
        Replaces: strip_whitespace, standardize_missing_values, and unify_na_values.
        """
        enforcer = SchemaEnforcer(self.df, self.na_placeholders, protected_values)
        self.df = enforcer.apply(schema_dict)
        return self

    def clean_column_names(self):
        self.df.columns = (self.df.columns.astype(str).str.strip().str.lower()
                           .str.replace(r'\s+', '_', regex=True)
                           .str.replace(r'[^a-z0-9_]', '', regex=True))
        return self
    
    def drop_constant_columns(self):
        """Drops columns where all values are the same."""
        # nunique(dropna=False) counts NaNs as a unique value if they exist
        # If nunique == 1, the column has only 1 value across all rows
        cols_to_drop = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped constant columns: {cols_to_drop}")
        return self
    
    def cap_outliers(self, columns, lower_quantile=0.05, upper_quantile=0.95):
        """Caps numerical values at specific quantiles to reduce outlier impact."""
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                lower = self.df[col].quantile(lower_quantile)
                upper = self.df[col].quantile(upper_quantile)
                
                # Clip values
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                print(f"Capped '{col}' between {lower:.2f} and {upper:.2f}")
        return self


    def drop_missing_cols(self, threshold=0.95, exclude=[]):
        """Drops columns that are missing more than `threshold` percent of data."""
        mask = self.df.isna().mean() > threshold
        cols_to_drop = [col for col in self.df.columns[mask] if col not in exclude]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped columns (> {threshold*100}% missing): {cols_to_drop}")
        return self

    def drop_duplicates(self):
        # Only check hashable columns to avoid crashing on lists
        unhashable = self._find_unhashable_columns()
        hashable_subset = [c for c in self.df.columns if c not in unhashable]
        if hashable_subset:
            initial = len(self.df)
            self.df.drop_duplicates(subset=hashable_subset,inplace=True)
            print(f"Dropped {initial - len(self.df)} duplicates.")
        return self

    def resolve_mixed_types(self, interactive=True):
        """
        Uses audit results to prompt the user for resolution strategies.
        Returns a schema dictionary with the user's choices.
        """
        report = self.audit_mixed_types(verbose=True)
        if not report:
            return {}

        schema_overrides = {}
        
        # Map Python type names to your SchemaEnforcer names
        type_map = {
            'str': 'string', 'int': 'int', 'float': 'float', 
            'bool': 'bool', 'list': 'list', 'dict': 'dict',
            'ndarray': 'list' # Convert numpy arrays to lists usually
        }

        if not interactive:
            print("\n[Non-Interactive Mode] Skipping manual resolution (defaulting to safe inference).")
            return {}

        print("\n--- Mixed Type Resolution ---")
        for col, data in report.items():
            maj_type = data['majority_type']
            maj_schema = type_map.get(maj_type, 'string')
            
            print(f"\nColumn: '{col}'")
            print(f" > Majority: {maj_type} ({data['majority_pct']:.1f}%)")
            print(f" > Minority Types: {[k for k in data['breakdown'] if k != maj_type]}")
            print(f" > Outlier Examples: {data['outliers']}")
            
            print(f"Options:")
            print(f"  [1] Coerce to Majority ('{maj_schema}') -> Outliers become NaN")
            print(f"  [2] Convert ALL to String -> Preserves data, loses type")
            print(f"  [3] Force specific type (Enter manually)")
            print(f"  [4] Skip (Let auto-inference decide)")
            
            choice = input(f"Action for '{col}': ").strip()
            
            if choice == '1':
                schema_overrides[col] = maj_schema
                print(f" -> Set schema for '{col}' to '{maj_schema}' (Strict)")
            elif choice == '2':
                schema_overrides[col] = 'string'
                print(f" -> Set schema for '{col}' to 'string' (Safe)")
            elif choice == '3':
                manual = input("Enter schema type (int, float, string, bool, list, dict): ").strip()
                if manual in type_map.values():
                    schema_overrides[col] = manual
                else:
                    print("Invalid type, skipping.")
            else:
                print(" -> Skipped.")
                
        return schema_overrides
    
    def convert_data_types_pandas(self):
        """Uses pandas convert_dtypes to infer best types."""
        self.df = self.df.convert_dtypes()
        print("Converted column types.")
        print(self.df.dtypes)
        return self
    
    def convert_data_types_arrow(self):
        """
        The Final Step: Convert everything to PyArrow backends.
        This locks in the schema and optimizes memory.
        
        """
        print("Step 1: Auto converting to PyArrow-backed types...")
        try:
            # automatic conversion
            self.df = self.df.convert_dtypes(dtype_backend="pyarrow")
            
            # Manual override for List/Structs if they remained objects
            # (Optional: PyArrow backend usually catches lists if they are clean)
            print(self.df.dtypes)
        except Exception as e:
            print(f"PyArrow conversion warning: {e}")

        print('Step 2: converting object types')
        for col in self.df.select_dtypes(include=['object']):
            try:
                arrow_array = pa.array(self.df[col].dropna())

                self.df[col] = self.df[col].astype(pd.ArrowDtype(arrow_array.type))
                print(f' > {col}: {arrow_array.type}')
            except Exception:
                pass
        print('Conversion complete.')
        return self
    
    def reset_index(self):
        self.df.reset_index(drop=True, inplace=True)
        return self

    # --- 3. PIPELINE ---
    #TODO 
        # check first for mixed types and adapt enforce schema
    def run_auto_pipeline(self, schema=None, protected_values=None, drop_empty_cols=False, interactive=True):
        # 1. Diagnostic & Remediation
        print("\n--- Pipeline Start: Pre-Flight Check ---")
        print('Looking for mixed types')
        # This will print the report AND ask you what to do
        # It returns a dict like {'authors': 'list', 'year': 'int'}
        mixed_type_fixes = self.resolve_mixed_types(interactive=interactive)
        # 2. Infer Base Schema
        detected_schema = self.auto_infer_schema()
        
        # 3. Merge Schemas
        # Priority: Manual Overrides (args) > Interactive Fixes > Auto Detect
        if mixed_type_fixes:
            print(f"Applying {len(mixed_type_fixes)} mixed-type resolutions...")
            detected_schema.update(mixed_type_fixes)
        else:
            print('No mixed types found.')
            
        if schema:
            print(f"Applying {len(schema)} manual overrides...")
            detected_schema.update(schema)
        
        # 4. Clean
        self.clean_column_names().enforce_schema(detected_schema, protected_values)
        if drop_empty_cols: self.drop_missing_cols()
        
        return self.drop_duplicates().convert_data_types_arrow().summarize()
    
    def save_parquet(self, path):
        if path:
            self.df.to_parquet(path, engine='pyarrow', index=False)
        else:
            print('No path given')


# ==============================================================================
#  requests
# ==============================================================================

def timed_request(
    url: str,
    session: Session | None = None,
    method: str ='GET',
    delay: float | None = None,
    timeout: int = 10,
    headers: dict | None = None,
    **kwargs
):
    """Makes a robust, timed HTTP request with error handling."""
    
    # 1. Delay
    if delay is None:
        delay = random.uniform(1.5, 4.5)
    time.sleep(delay)

    # 2. Headers
    request_headers = headers
    if headers is None and session is None:
        request_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
        }

    # 3. Request
    requester = session if session else requests
    try:
        response = requester.request(
            method=method, 
            url=url, 
            timeout=timeout, 
            headers=request_headers, 
            **kwargs
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        # Catching all Request exceptions (HTTPError, ConnectionError, etc)
        print(f"  [Error] Request failed for {url}: {e}")
        return None





