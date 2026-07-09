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
from collections import defaultdict
import re
import ast
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import csv as pa_csv

try:
    from rapidfuzz import process, fuzz
except ImportError:
    try:
        from fuzzywuzzy import process, fuzz
    except ImportError:
        process, fuzz = None, None
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


def sanitize_filename_old(filename: str, replacement: str = "_") -> str:
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


def sanitize_filename(filename: str, replacement: str = '_') -> str:
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
    sanitized = re.sub(invalid_chars, replacement, filename)

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
    save_to: str | None = None,
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
        if save_to:
            file_path = Path(save_to)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"[Success] HTTP response saved to: {save_to}")

        return response
    except requests.exceptions.RequestException as e:
        # Catching all Request exceptions (HTTPError, ConnectionError, etc)
        print(f"  [Error] Request failed for {url}: {e}")
        return None
    except IOError as e:
        print(f"  [Error] Could not save file to {save_to}: {e}")
        return None


from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

class PyarrowHelper:
    def __init__(self, source):
        """
        Initializes the helper, keeps the Parquet metadata reader open,
        and caches essential metadata elements.
        """
        self.source = Path(source)
        if not self.source.exists():
            raise FileNotFoundError(f"Parquet file not found at: {self.source}")
            
        # 1. Open metadata reader (very fast, does not read actual rows)
        self.pf = pq.ParquetFile(self.source)
        
        # 2. Cache metadata parameters to avoid redundant IO
        self.all_cols = self.pf.schema_arrow.names
        self.num_rows = self.pf.metadata.num_rows
        
        # Placeholder for lazy-loaded table cache
        self._table = None

    @property
    def table(self):
        """
        Lazily loads and caches the full PyArrow Table only if accessed.
        """
        if self._table is None:
            self._table = pq.read_table(self.source)
        return self._table

    @property
    def file_size(self) -> str:
        """
        Returns the file size on disk formatted in Megabytes.
        """
        bytes_size = self.source.stat().st_size
        mb_size = bytes_size / (1024 * 1024)
        return f"{mb_size:.2f} MB"

    def head(self, num=5) -> pd.DataFrame:
        """
        Returns a visual Pandas DataFrame preview of the first `num` rows 
        without loading the entire Parquet file.
        """
        table_slice = pq.read_table(self.source).slice(0, num)
        return table_slice.to_pandas()

    def info(self) -> pd.DataFrame:
        """
        Compiles structural profiling metrics (data types, null counts, 
        null percentages, nested status) for all columns.
        
        Returns a Pandas DataFrame for neat notebook rendering.
        """
        metrics = []
        for col_name in self.all_cols:
            col_idx = self.all_cols.index(col_name)
            
            # Extract Arrow Data Type and check nested status
            arrow_type = self.pf.schema_arrow.field(col_idx).type
            is_nested = isinstance(
                arrow_type, 
                (pa.ListType, pa.StructType, pa.MapType, pa.LargeListType, pa.FixedSizeListType)
            )
            
            # Fetch null counts
            null_count = self._get_null_count(col_name)
            null_pct = (null_count / self.num_rows * 100) if self.num_rows > 0 else 0.0
            
            metrics.append({
                "Column": col_name,
                "Type": str(arrow_type),
                "Null Count": null_count,
                "Null %": f"{null_pct:.2f}%",
                "Nested?": is_nested
            })
            
        return pd.DataFrame(metrics)

    def _get_null_count(self, col_name: str) -> int:
        """
        Attempts to read null counts directly from the Parquet metadata.
        Falls back to a memory-efficient compute scan if statistics are missing.
        """
        col_idx = self.all_cols.index(col_name)
        null_count = 0
        has_stats = True
        
        # Check metadata row groups for pre-computed statistics
        for rg_idx in range(self.pf.num_row_groups):
            rg = self.pf.metadata.row_group(rg_idx)
            stats = rg.column(col_idx).statistics
            if stats is not None and stats.has_null_count:
                null_count += stats.null_count
            else:
                has_stats = False
                break
                
        if has_stats:
            return null_count
            
        # Fallback if metadata lacks statistics
        single_col_table = pq.read_table(self.source, columns=[col_name])
        col_data = single_col_table[col_name]
        return pc.sum(pc.is_null(col_data)).as_py()

    def show_vals(self, col, num=10):
        """
        Loads only the specified column and prints a small list of sample values.
        """
        # Read only the target column to keep memory footprint low
        table = pq.read_table(self.source, columns=[col])
        sample_values = table[col].slice(0, num).to_pylist()
        print(f'{col}: {sample_values}')

    def show_value_counts(self, col=None, num=10, exclude=None):
        """
        Calculates value counts natively within PyArrow for the specified columns,
        sorts the frequencies in descending order, and displays the top results.
        
        If col=None, runs for all columns by default (excluding specified ones).
        """
        # 1. Normalize the 'exclude' list
        if exclude is None:
            exclude_list = []
        elif isinstance(exclude, str):
            exclude_list = [exclude]
        elif isinstance(exclude, list):
            exclude_list = exclude
        else:
            raise ValueError("The 'exclude' argument must be a string or a list of strings.")
            
        # 2. Determine which columns to check
        if col is None:
            cols_to_check = [c for c in self.all_cols if c not in exclude_list]
        else:
            if isinstance(col, str):
                temp_cols = [col]
            elif isinstance(col, list):
                temp_cols = col
            else:
                raise ValueError("The 'col' argument must be a string or a list of strings.")
            
            cols_to_check = [c for c in temp_cols if c not in exclude_list]

        if not cols_to_check:
            print("No columns left to check after applying exclusion filters.")
            return

        # 3. Process each column
        for col_name in cols_to_check:
            if col_name not in self.all_cols:
                print(f"Column '{col_name}' not found in source.")
                print("-" * 45)
                continue
                
            table = pq.read_table(self.source, columns=[col_name])
            col_data = table[col_name]
            
            try:
                counts = pc.value_counts(col_data)
                
                # Convert StructArray to Table to allow native sorting (handles ChunkedArrays)
                if isinstance(counts, pa.ChunkedArray):
                    counts_table = pa.Table.from_batches([
                        pa.RecordBatch.from_struct_array(chunk) 
                        for chunk in counts.chunks
                    ])
                else:
                    counts_table = pa.Table.from_batches([
                        pa.RecordBatch.from_struct_array(counts)
                    ])
                
                sorted_table = counts_table.sort_by([("counts", "descending")])
                top_results = sorted_table.slice(0, num)
                
                print(f"Top {num} Value Counts for '{col_name}':")
                results_list = top_results.to_pylist()
                if not results_list:
                    print("  - (No values found)")
                else:
                    for row in results_list:
                        print(f"  - {row['counts']}: {row['values']}")
                print("-" * 45)
                
            except Exception:
                print(f"Column '{col_name}':")
                print(f"  - [Skipped] Cannot compute value counts natively (complex/nested type: {col_data.type})")
                print("-" * 45)



    def show_duplicates(self, col=None, exclude=None):
        """
        Inspects duplication metrics for Parquet columns with optional exclusions.
        """
        # Normalize the 'exclude' list
        if exclude is None:
            exclude_list = []
        elif isinstance(exclude, str):
            exclude_list = [exclude]
        elif isinstance(exclude, list):
            exclude_list = exclude
        else:
            raise ValueError("The 'exclude' argument must be a string or a list of strings.")
            
        # Determine which columns to check
        if col is None:
            cols_to_check = [c for c in self.all_cols if c not in exclude_list]
            is_summary_mode = True
        else:
            is_summary_mode = False
            if isinstance(col, str):
                temp_cols = [col]
            elif isinstance(col, list):
                temp_cols = col
            else:
                raise ValueError("The 'col' argument must be a string or a list of strings.")
            
            cols_to_check = [c for c in temp_cols if c not in exclude_list]

        if not cols_to_check:
            print("No columns left to check after applying exclusion filters.")
            return

        # Process each column
        for col_name in cols_to_check:
            if col_name not in self.all_cols:
                print(f"Column '{col_name}' not found in source.")
                print("-" * 45)
                continue
                
            table = pq.read_table(self.source, columns=[col_name])
            col_data = table[col_name]
            total_rows = len(col_data)
            
            try:
                counts = pc.value_counts(col_data)
                dup_mask = pc.greater(counts.field('counts'), 1)
                dupes = pc.filter(counts.field('values'), dup_mask)
                
                if is_summary_mode:
                    num_unique = len(counts)
                    num_dup_rows = total_rows - num_unique
                    num_distinct_duped = len(dupes)
                    
                    print(f"Column '{col_name}':")
                    print(f"  - Total rows: {total_rows}")
                    print(f"  - Unique values: {num_unique}")
                    print(f"  - Duplicate entries (redundant rows): {num_dup_rows}")
                    print(f"  - Distinct values with duplicates: {num_distinct_duped}")
                    print("-" * 45)
                else:
                    print(f"Duplicate values in '{col_name}':")
                    print(dupes.to_pylist())
                    print("-" * 45)
                    
            except Exception:
                print(f"Column '{col_name}':")
                print(f"  - Total rows: {total_rows}")
                print(f"  - [Skipped] Cannot compute duplicates natively (complex/nested type: {col_data.type})")
                print("-" * 45)





#DATAFRAMECLEANER YOUPIE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RobustDataFrameCleaner")


class DataFrameCleaner:
    """
    A unified class that combines data cleaning, schema enforcement, structural auditing,
    and deduplication natively on PyArrow Tables, bypassing Pandas row-by-row iteration.
    """

    def __init__(self, data, regex_patterns=None, protected_values=None):
        self.stats = {}
        self.true_vals = {'y', 'yes', 't', 'true', '1', 'on'}
        self.false_vals = {'n', 'no', 'f', 'false', '0', 'off'}

        # Unified ingestion to PyArrow Table
        if isinstance(data, pa.Table):
            self.table = data
            self._log_info("DataFrame cleaner initialized from existing PyArrow Table.")
        elif isinstance(data, pd.DataFrame):
            self.table = pa.Table.from_pandas(data)
            self._log_info("DataFrame cleaner initialized from Pandas DataFrame and converted to PyArrow Table.")
        elif isinstance(data, (Path, str)):
            try:
                path = Path(data)
            except Exception as e:
                raise ValueError(f"Could not convert string to path: {e}")
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            suffix = path.suffix.lower()
            if suffix in ['.parquet', '.pq']:
                self.table = pq.read_table(path)
                self._log_info(f"DataFrame cleaner initialized from Parquet file: {path.name}")
            elif suffix == '.csv':
                self.table = pa_csv.read_csv(path)
                self._log_info(f"DataFrame cleaner initialized from CSV file: {path.name}")
            elif suffix == '.json':
                # Fast Pandas-to-Arrow bridge for JSON/JSONL formats
                df_temp = pd.read_json(path, lines=True)
                self.table = pa.Table.from_pandas(df_temp)
                self._log_info(f"DataFrame cleaner initialized from JSON file via Pandas bridge: {path.name}")
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        else:
            raise TypeError("Initialization parameter 'data' must be a pa.Table, pd.DataFrame, or a valid path pointing to a file.")

        # Determine the garbage replacement pattern
        if regex_patterns:
            # Vectorized Validation: Run a dry-run check against a dummy array 
            # to intercept any PCRE features (such as lookarounds) that RE2 does not support.
            try:
                dummy_arr = pa.array(["test_val"])
                pc.match_substring_regex(dummy_arr, pattern=regex_patterns, ignore_case=True)
            except Exception as e:
                raise ValueError(
                    f"The provided regex_patterns is incompatible with PyArrow's C++ RE2 engine: {e}"
                )
            
            self.garbage_regex = re.compile(regex_patterns, flags=re.IGNORECASE)
            self.na_placeholders = regex_patterns
        else:
            # Clean RE2-compliant patterns without inner anchors.
            # The outer combined_pattern wrapper handles start (^) and end ($) boundaries.
            patterns = [
                r"nan",
                r"(?:n/?a|null|none|<none>|not reported|unknown|undefined|missing)",
                r"(?:-+|\/+)",
                r"\?+",
                r"(?:-99|-9999|999|9999)",
            ]
            combined_pattern = r"^(?:" + "|".join(patterns) + r")$"
            
            # Quick validation check of the default pattern layout
            try:
                dummy_arr = pa.array(["test_val"])
                pc.match_substring_regex(dummy_arr, pattern=combined_pattern, ignore_case=True)
            except Exception as e:
                raise ValueError(f"Internal default pattern failed RE2 dry-run check: {e}")

            self.garbage_regex = re.compile(combined_pattern, flags=re.IGNORECASE)
            self.na_placeholders = combined_pattern

        self.protected_values = {k: set(map(str, v)) for k, v in (protected_values or {}).items()}

    # =========================================================================
    # UNIFORM PRINT & DISPLAY HELPERS
    # =========================================================================

    def _print_header(self, title: str):
        border = "=" * 80
        print(f"\n{border}\n{title.center(80)}\n{border}")

    def _log_info(self, msg: str):
        logger.info(msg)

    def _log_warning(self, msg: str):
        logger.warning(msg)

    # =========================================================================
    # PYARROW VECTORIZED CLEANING LOGIC
    # =========================================================================

    def _nullify_garbage_recursive(self, array: pa.Array, col_name: str=None) -> pa.Array:
        """
        Recursively traverses any nested PyArrow array schema (structs, lists) 
        and nullifies string-based garbage placeholders at C++ speed.
        """
        if pa.types.is_null(array.type) or len(array) == 0:
            return array

        # Leaf Case: String types
        if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
            trimmed = pc.utf8_trim_whitespace(array)
            is_garbage = pc.match_substring_regex(trimmed, pattern=self.na_placeholders, ignore_case=True)
            is_garbage = pc.fill_null(is_garbage, False)

            # Check for column-specific protected values
            if col_name and col_name in self.protected_values:
                protected_list = list(self.protected_values[col_name])
                if protected_list:
                    # Isolate elements that match protected values and exclude them from garbage nullification
                    is_protected = pc.is_in(trimmed, value_set=pa.array(protected_list))
                    is_protected = pc.fill_null(is_protected, False)
                    is_garbage = pc.and_not(is_garbage, is_protected)

            return pc.if_else(is_garbage, pa.scalar(None, type=array.type), array)

        # Recursion Case: Struct Arrays
        if pa.types.is_struct(array.type):
            cleaned_fields = []
            for i in range(array.type.num_fields):
                field = array.type.field(i)
                child_array = array.field(field.name)
                cleaned_child = self._nullify_garbage_recursive(child_array, col_name=col_name)
                cleaned_fields.append(cleaned_child)
            return pa.StructArray.from_arrays(
                cleaned_fields, 
                fields=array.type, 
                mask=array.is_null()
            )

        # Recursion Case: List/Large List Arrays
        if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
            values_array = array.values
            cleaned_values = self._nullify_garbage_recursive(values_array, col_name=col_name)
            return pa.ListArray.from_arrays(
                array.offsets, 
                cleaned_values, 
                mask=array.is_null()
            )

        return array


    def _clean_arrow_array(self, array: pa.Array, target_dtype: str, col_name:str = None) -> pa.Array:
        """
        Applies both structural tree cleaning and type coercion on a PyArrow Array.
        """
        if pa.types.is_null(array.type) or len(array) == 0:
            return array

        # Transition Case: Parse flat JSON strings to actual structures if required
        if (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)) and target_dtype in ['list', 'dict']:
            parsed_list = []
            for val in array.to_pylist():
                parsed = self._parse_string_structure(val)
                if target_dtype == 'list' and not isinstance(parsed, list):
                    parsed_list.append(None)
                elif target_dtype == 'dict' and not isinstance(parsed, dict):
                    parsed_list.append(None)
                else:
                    parsed_list.append(parsed)
            array = pa.array(parsed_list)

        # 1. Clean garbage values recursively across any leaf-string nodes
        array = self._nullify_garbage_recursive(array, col_name=col_name)

        # 2. Apply vectorized target casting
        if target_dtype == 'string':
            return pc.cast(array, pa.string(), safe=False)
        
        elif target_dtype == 'bool':
            if pa.types.is_boolean(array.type):
                return array
            try:
                str_arr = pc.cast(array, pa.string(), safe=False)
                str_arr = pc.utf8_lower(pc.utf8_trim_whitespace(str_arr))
                is_true = pc.is_in(str_arr, value_set=pa.array(list(self.true_vals)))
                is_false = pc.is_in(str_arr, value_set=pa.array(list(self.false_vals)))
                return pc.if_else(is_true, pa.scalar(True), pc.if_else(is_false, pa.scalar(False), pa.scalar(None, type=pa.bool_())))
            except Exception:
                return pc.cast(array, pa.bool_(), safe=False)

        elif target_dtype in ['int', 'float', 'number']:
            pa_type = pa.int64() if target_dtype == 'int' else pa.float64()
            
            if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
                # Step 1: Trim surrounding whitespace
                array = pc.utf8_trim_whitespace(array)
                
                # Step 2: Match against a robust, RE2-compatible floating-point regex.
                float_pattern = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
                
                is_valid = pc.match_substring_regex(array, pattern=float_pattern)
                is_valid = pc.fill_null(is_valid, False)
                
                # Step 3: Vectorized replacement of malformed elements to None
                array = pc.if_else(is_valid, array, pa.scalar(None, type=array.type))
                
                # Step 4: Cast strings to float64 safely
                array = pc.cast(array, pa.float64(), safe=False)
                
                # Step 5: For integer requests, cast float64 down to int64 (truncates decimal values)
                if target_dtype == 'int':
                    array = pc.cast(array, pa.int64(), safe=False)
            else:
                # Directly cast if the array is already numeric/boolean
                array = pc.cast(array, pa_type, safe=False)
            
            return array

        elif target_dtype in ['date', 'datetime']:
            pa_type = pa.date32() if target_dtype == 'date' else pa.timestamp('us')
            try:
                # Standardize parsing by casting to microsecond timestamp first
                timestamp_arr = pc.cast(array, pa.timestamp('us'), safe=False)
                return pc.cast(timestamp_arr, pa_type, safe=False)
            except Exception:
                # Fallback to Pandas for mixed/non-ISO dates, guaranteeing clean output types
                series = pd.to_datetime(pd.Series(array.to_pylist()), errors='coerce')
                arrow_arr = pa.array(series)
                return pc.cast(arrow_arr, pa_type, safe=False)

        return array


    def _parse_string_structure(self, val):
        if not isinstance(val, str):
            return val
        val_stripped = val.strip()
        if val_stripped.startswith(('[', '{')):
            try:
                import json
                return json.loads(val_stripped)
            except Exception:
                try:
                    return ast.literal_eval(val_stripped)
                except Exception:
                    pass
        return val

    def _scan_placeholders(self, col, is_complex=False):
        """
        Vectorized placeholder scanning using PyArrow regex compute functions [1].
        """
        if is_complex or col not in self.table.column_names:
            return 0, []
        try:
            arr = pc.drop_null(self.table.column(col).combine_chunks())
            if pa.types.is_null(arr.type):
                return 0, []

            str_arr = pc.cast(arr, pa.string())
            str_arr = pc.utf8_trim_whitespace(str_arr)
            is_garbage = pc.match_substring_regex(str_arr, pattern=self.na_placeholders, ignore_case=True)
            is_garbage = pc.fill_null(is_garbage, False)

            count = pc.sum(is_garbage).as_py()
            if count > 0:
                garbage_values = pc.filter(str_arr, is_garbage)
                examples = pc.unique(garbage_values).to_pylist()[:10]
            else:
                examples = []
            return count, examples
        except Exception:
            return 0, []

    # =========================================================================
    # AUDITING & DIAGNOSTICS (Inspect methods)
    # =========================================================================

    def _find_unhashable_columns(self):
        unhashable_cols = []
        for field in self.table.schema:
            t = field.type
            if (pa.types.is_list(t) or 
                pa.types.is_large_list(t) or 
                pa.types.is_fixed_size_list(t) or 
                pa.types.is_struct(t) or 
                pa.types.is_map(t) or 
                pa.types.is_dictionary(t)):
                unhashable_cols.append(field.name)
        return unhashable_cols
    
    def _get_hashable_subset(self, exclude_cols=None, subset=None) -> list:
        """
        Private helper to resolve which columns are safe to evaluate for deduplication,
        bypassing nested/unhashable schemas and logging warnings.
        """
        unhashable = self._find_unhashable_columns()
        
        if subset is not None:
            hashable = [c for c in subset if c in self.table.column_names and c not in unhashable]
            invalid_or_unhashable = set(subset) - set(hashable)
            if invalid_or_unhashable:
                self._log_warning(f"Excluding invalid or unhashable columns from specified subset: {list(invalid_or_unhashable)}")
        else:
            exclude_cols = exclude_cols or []
            hashable = [c for c in self.table.column_names if c not in unhashable and c not in exclude_cols]
            
            # Log auto-exclusions for clarity
            auto_excluded = set(unhashable) & set(self.table.column_names)
            if auto_excluded:
                self._log_warning(f"Excluding columns from precise duplicates calculation: {list(auto_excluded)}")
                
        return hashable

    def _analyze_arrow_data(self, array, indent=0):
        prefix = " " * indent
        arrow_type = array.type
        
        KEY_WIDTH = 30    
        TYPE_WIDTH = 15   
        SAMPLE_WIDTH = 40 

        if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
            print(f"{prefix}- List of:")
            self._analyze_arrow_data(array.flatten(), indent + 2)
            
        elif pa.types.is_struct(arrow_type):
            print(f"{prefix}- Object (Dict) with keys:")
            for field in arrow_type:
                child_array = array.field(field.name)
                label = f"{prefix}  * {field.name}:"
                
                is_list_like = (
                    pa.types.is_list(field.type) or 
                    pa.types.is_large_list(field.type) or 
                    pa.types.is_fixed_size_list(field.type)
                )
                if is_list_like or pa.types.is_struct(field.type):
                    print(label) 
                    self._analyze_arrow_data(child_array, indent + 6)
                else:
                    print(f"{label.ljust(KEY_WIDTH + indent)}", end="")
                    self._analyze_arrow_data(child_array, indent=0)
        else:
            total_count = len(array)
            if total_count == 0:
                print(f"{arrow_type} (Empty)")
                return

            unique_vals = pc.unique(array)
            n_unique = len(unique_vals)
            ratio = n_unique / total_count
            
            if n_unique == 1:
                cat_label = "CONSTANT"
            elif ratio > 0.9:
                cat_label = "ID/TEXT"
            elif ratio < 0.1 or n_unique < 50:
                cat_label = "CATEGORY"
            else:
                cat_label = "DENSE"

            sample_slice = array.slice(0, 50).drop_null()
            if len(sample_slice) > 0:
                val_str = str(sample_slice[0].as_py()).replace('\n', ' ')
                if len(val_str) > SAMPLE_WIDTH - 3: 
                    val_str = val_str[:SAMPLE_WIDTH-3] + "..."
            else:
                val_str = "NULL"

            current_prefix = prefix if indent > 0 else ""
            type_str = f"{str(arrow_type)}".ljust(TYPE_WIDTH)
            count_str = f"{n_unique} unique".rjust(12)
            sample_str = f"sample: {val_str}".ljust(SAMPLE_WIDTH + 8)

            print(f"{current_prefix}{type_str} | {count_str} | {sample_str} | ({cat_label})")

    def _get_python_type_name(self, val):
        return type(val).__name__

    def summarize(self, exclude_cols=None, subset=None):
        self._print_header(f"DataFrame Summary (Shape: {self.table.num_rows} rows x {self.table.num_columns} cols)")
        
        print("\n--- DUPLICATE COUNT AUDIT ---")
        self.check_duplicates(exclude_cols=exclude_cols, subset=subset)
        print("-" * 80)

        summary_records = []
        unhashable_cols = self._find_unhashable_columns()

        for col in self.table.column_names:
            col_data = self.table.column(col).combine_chunks()
            null_count = col_data.null_count
            null_pct = round((null_count / self.table.num_rows) * 100, 2)
            dtype_str = str(col_data.type)

            if col not in unhashable_cols:
                try:
                    unique_count = len(pc.unique(col_data))
                except Exception:
                    unique_count = -1
            else:
                unique_count = -1

            # Natively extract the first non-null sample
            first_valid_val = None
            valid_data = col_data.drop_null()
            if len(valid_data) > 0:
                first_valid_val = valid_data[0].as_py()
            
            if first_valid_val is not None:
                val_str = str(first_valid_val)
                sample_str = val_str[:40] + "..." if len(val_str) > 40 else val_str
            else:
                sample_str = "NaN"

            placeholders_str = ""
            if col not in unhashable_cols:
                try:
                    cnt, ex_list = self._scan_placeholders(col)
                    if cnt > 0:
                        placeholders_str = str(ex_list)
                except Exception:
                    pass

            summary_records.append({
                'column': col,
                'missing#': null_count,
                'missing%': null_pct,
                'unique#': unique_count,
                'sample_val': sample_str,
                'placeholders': placeholders_str,
                'dtypes': dtype_str
            })

        summary = pd.DataFrame(summary_records).set_index('column')
        max_len = summary['dtypes'].map(len).max() if not summary.empty else 10
        formatters = {'dtypes': lambda x: f"{x:<{max_len}}"}
        
        print("\n--- COLUMN ATTRIBUTES ---")
        print(summary.sort_values('dtypes', ascending=False).to_string(formatters=formatters))
        print("-" * 80)

        self.audit_mixed_types()
        print("-" * 80)

        if unhashable_cols:
            print("\n--- NESTED COMPLEX DATA STRUCTURES ---")
            self.analyze_structure_recursive()
        else:
            self._log_info("No complex nested columns found in structural layout.")
            
        return self

    def analyze_structure_recursive(self, sample_size=5000):
        unhashable_cols = self._find_unhashable_columns()
        if not unhashable_cols:
            return self

        print(f"\n--- Deep Structure & Stats (Sample size limit: {sample_size}) ---")
        for col in unhashable_cols:
            col_data = self.table.column(col).combine_chunks()
            valid_data = col_data.drop_null()
            if len(valid_data) == 0:
                continue
            
            if len(valid_data) > sample_size:
                valid_data = valid_data.slice(0, sample_size)
            
            print(f"Column '{col}':")
            try:
                self._analyze_arrow_data(valid_data, indent=2)
            except Exception as e:
                self._log_warning(f"  [!] Structural parser failure: {e}")
            print("")
        return self

    def get_samples(self, columns=None, number=5):
        valid = [c for c in columns or [] if c in self.table.column_names]
        invalid = set(columns or []) - set(valid)
        if invalid:
            self._log_warning(f"Columns not found in Table index: {invalid}")
            
        for col in valid:
            col_data = self.table.column(col).combine_chunks().drop_null()
            if len(col_data) == 0:
                print(f"Column: {col} is empty.")
                continue
            
            first_val = col_data[0].as_py()
            self._print_header(f"Samples for column '{col}' ({type(first_val).__name__})")
            
            sample_indices = np.random.choice(len(col_data), min(number, len(col_data)), replace=False)
            for idx in sample_indices:
                print(f" > {col_data[int(idx)].as_py()}")
            print("-" * 40)
        return self

    def audit_mixed_types(self, verbose=True):
        if verbose:
            print("\n--- MIXED DATA TYPE CONFLICTS ---")
            self._log_info("PyArrow database layer enforces strict type invariants. No mixed column conflicts exist.")
        return {}

    def check_missing_values(self):
        missing_report = []
        for col in self.table.column_names:
            null_count = self.table.column(col).null_count
            if null_count > 0:
                missing_report.append({
                    'Column': col,
                    'Missing Count': null_count,
                    'Percentage (%)': round((null_count / self.table.num_rows) * 100, 2)
                })
        
        if missing_report:
            self._print_header("Missing Value Verification")
            report_df = pd.DataFrame(missing_report).set_index('Column')
            print(report_df)
        else:
            self._log_info("No missing NaN values detected.")
        return self



    def _parse_string_structure(self, val: str):
        """
        Safely attempts to parse a string into a Python object (list or dict)
        using json.loads first, falling back to ast.literal_eval.
        All parser exceptions are caught silently to prevent console clutter.
        """
        # Try standard JSON parsing (fastest and most standard)
        try:
            return json.loads(val)
        except Exception:
            pass
        
        # Try Python literal parsing (handles single quotes and trailing commas)
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
            
        return None

    def _get_type_signature(self, obj) -> str:
        """
        Recursively determines the semantic type signature of a parsed Python object.
        """
        if obj is None:
            return "null"
        if isinstance(obj, bool):  # Check bool before int, as bool is a subclass of int in Python
            return "bool"
        if isinstance(obj, int):
            return "int"
        if isinstance(obj, float):
            return "float"
        if isinstance(obj, str):
            return "string"
        
        if isinstance(obj, list):
            if not obj:
                return "list"
            # Determine unique types of items inside the list
            item_sigs = {self._get_type_signature(item) for item in obj}
            if len(item_sigs) == 1:
                return f"list<{next(iter(item_sigs))}>"
            return "list<any>"
            
        if isinstance(obj, dict):
            if not obj:
                return "dict"
            # Build signatures for each key's value
            val_sigs = {str(k): self._get_type_signature(v) for k, v in obj.items()}
            unique_val_sigs = set(val_sigs.values())
            
            # If all values are homogeneous, describe it as a dynamic dictionary/map
            if len(unique_val_sigs) == 1:
                return f"dict<string: {next(iter(unique_val_sigs))}>"
            
            # If values have different types, describe it as a structured type (struct)
            # Limit fields to 10 to keep signatures legible and avoid giant console schemas
            keys_to_show = list(val_sigs.items())[:10]
            struct_fields = ", ".join(f"{k}: {sig}" for k, sig in keys_to_show)
            if len(val_sigs) > 10:
                struct_fields += ", ..."
            return f"struct<{struct_fields}>"
            
        return "unknown"

    def _detect_string_structure(self, val) -> str:
        """
        Detects if a string contains a valid dictionary or list structure
        and returns its recursive type signature.
        """
        if not isinstance(val, str):
            return None
        
        val_stripped = val.strip()
        if not val_stripped:
            return None

        # Structural fast-path check
        is_dict = val_stripped.startswith('{') and val_stripped.endswith('}')
        is_list = val_stripped.startswith('[') and val_stripped.endswith(']')
        
        if is_dict or is_list:
            parsed = self._parse_string_structure(val_stripped)
            if parsed is not None:
                return self._get_type_signature(parsed)

        return None


    def auto_infer_schema(self, sample_pct=0.05, min_sample=1000, max_sample=50000):
        if self.table.num_rows == 0:
            self._log_warning("Working Table is empty. Bypassing schema inference.")
            return {}

        total_rows = self.table.num_rows
        calculated_sample = int(total_rows * sample_pct)
        sample_size = max(min(calculated_sample, max_sample), min_sample)
        sample_size = min(sample_size, total_rows)

        inferred = {}
        self._log_info(
            f"Auto-inferring schema dynamically. Sample size configured to n={sample_size} "
            f"(calculated from {sample_pct * 100}% of {total_rows} rows)."
        )

        for col in self.table.column_names:
            col_data = self.table.column(col)
            valid = col_data.drop_null()
            if len(valid) == 0:
                continue
            
            if len(valid) > sample_size: 
                valid = valid.slice(0, sample_size)

            t = valid.type
            
            # 1. Unpack Dictionary / Categorical columns natively
            if pa.types.is_dictionary(t):
                value_type = t.value_type
                if pa.types.is_string(value_type) or pa.types.is_large_string(value_type):
                    t = value_type # Inspect categorical values semantically as strings
                elif pa.types.is_integer(value_type):
                    inferred[col] = 'int'
                    continue
                elif pa.types.is_floating(value_type):
                    inferred[col] = 'float'
                    continue
                else:
                    inferred[col] = 'string'
                    continue

            # 2. Check strict physical types
            if pa.types.is_boolean(t):
                inferred[col] = 'bool'
                continue
            if pa.types.is_integer(t):
                inferred[col] = 'int'
                continue
            if pa.types.is_floating(t):
                # Verify if every non-null float is mathematically a whole number (e.g., 2026.0 == 2026)
                is_int_equivalent = pc.equal(valid, pc.floor(valid)) #Change?
                is_int_equivalent = pc.fill_null(is_int_equivalent, False)
                if pc.sum(is_int_equivalent).as_py() == len(valid):
                    inferred[col] = 'int'
                else:
                    inferred[col] = 'float'
                continue
            if pa.types.is_timestamp(t) or pa.types.is_date(t):
                inferred[col] = 'datetime'
                continue
            if pa.types.is_list(t) or pa.types.is_large_list(t):
                inferred[col] = 'list'
                continue
            if pa.types.is_struct(t) or pa.types.is_map(t):
                inferred[col] = 'dict'
                continue
                
            #TODO handle string -> struct
            if pa.types.is_string(t) or pa.types.is_large_string(t):
                # Fast slice first to avoid large-scale computations
                sample_col = col_data.slice(0, sample_size).drop_null()
                if len(sample_col) == 0:
                    inferred[col] = 'string'
                    continue

                # Extract PyList of unique values to handle boolean/low-cardinality checks first
                # Doing unique checks first is cheap and prevents wasting regex work on flags
                unique_vals = [str(x).strip().lower() for x in pc.unique(sample_col).to_pylist() if x is not None and x != ""]
                
                if not unique_vals:
                    inferred[col] = 'string'
                    continue

                # A. Boolean Check (Run first because it's the fastest fail-fast check)
                if len(unique_vals) <= 10:
                    clean_vals = [x for x in unique_vals if not self.garbage_regex.match(x)]
                    if clean_vals and set(clean_vals).issubset(self.true_vals | self.false_vals):
                        inferred[col] = 'bool'
                        continue

                # B. Structural Container Check (Using the silent, recursive parser we rewrote)
                # Limit loop to 50 samples max for performance
                sample_vals = [x.as_py() for x in sample_col if x.is_valid and x.as_py() != ""][:50]
                if sample_vals:
                    signature_votes = {}
                    base_type_votes = {'dict': 0, 'list': 0}
                    for val_str in sample_vals:
                        detected_struct = self._detect_string_structure(val_str)
                        if detected_struct:
                            signature_votes[detected_struct] = signature_votes.get(detected_struct, 0) + 1
                            if detected_struct.startswith(('dict', 'struct')):
                                base_type_votes['dict'] += 1
                            elif detected_struct.startswith('list'):
                                base_type_votes['list'] += 1

                    threshold = int(len(sample_vals) * 0.80)
                    if signature_votes:
                        most_common_sig, sig_count = max(signature_votes.items(), key=lambda x: x[1])
                        if sig_count >= threshold:
                            inferred[col] = most_common_sig
                            continue
                        if base_type_votes['dict'] >= threshold:
                            inferred[col] = 'dict'
                            continue
                        if base_type_votes['list'] >= threshold:
                            inferred[col] = 'list'
                            continue

                # C. Regex Math (Whitespaces handled inside the pattern to avoid pc.utf8_trim_whitespace)
                # These run only if the column wasn't a container or boolean.
                total_valid_cnt = len(sample_vals) # Using Python sample count
                if total_valid_cnt > 0:
                    # Integer Pattern (with optional whitespace)
                    is_int = pc.match_substring_regex(sample_col, pattern=r"^\s*[+-]?[0-9]+\s*$")
                    is_int = pc.fill_null(is_int, False)
                    if pc.sum(is_int).as_py() == total_valid_cnt:
                        inferred[col] = 'int'
                        continue

                    # Float Pattern (with optional whitespace)
                    float_pattern = r"^\s*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\s*$"
                    is_float = pc.match_substring_regex(sample_col, pattern=float_pattern)
                    is_float = pc.fill_null(is_float, False)
                    if pc.sum(is_float).as_py() == total_valid_cnt:
                        inferred[col] = 'float'
                        continue

                    # Date Pattern (with optional whitespace)
                    date_pattern = r"^\s*(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\s*$"
                    is_date = pc.match_substring_regex(sample_col, pattern=date_pattern)
                    is_date = pc.fill_null(is_date, False)
                    if pc.sum(is_date).as_py() == total_valid_cnt:
                        inferred[col] = 'datetime'
                        continue

                inferred[col] = 'string'
        return inferred


    # =========================================================================
    # CORE CLEANING OPERATIONS (Modify methods)
    # =========================================================================

    def parse_and_flatten(self, col, mode='first'):
        """
        Extract structure from columns containing flat JSON-string representations of lists.
        """
        if col not in self.table.column_names:
            self._log_warning(f"Flatten failed. Column not found: '{col}'.")
            return self

        col_idx = self.table.schema.get_field_index(col)
        arr = self.table.column(col).combine_chunks()

        # Temporary Pandas mapping specifically for string parsing step
        parsed_series = pd.Series(arr.to_pylist(), dtype=object).apply(self._parse_string_structure)

        def _get_flattened_element(val):
            if isinstance(val, list):
                if not val:
                    return None
                return val[0] if mode == 'first' else "; ".join([str(i) for i in val])
            return val

        flattened_series = parsed_series.apply(_get_flattened_element)
        self.table = self.table.set_column(col_idx, col, pa.array(flattened_series))
        self._log_info(f"Extracted/flattened elements inside target column '{col}' using mode='{mode}'")
        return self

    def combine_columns(self, col1, col2, new_name, sep=' '):
        if col1 not in self.table.column_names or col2 not in self.table.column_names:
            self._log_warning(f"Action ignored. Missing columns: '{col1}' or '{col2}'.")
            return self

        c1_str = pc.cast(self.table.column(col1).combine_chunks(), pa.string())
        c2_str = pc.cast(self.table.column(col2).combine_chunks(), pa.string())

        c1_str = pc.utf8_trim_whitespace(pc.fill_null(c1_str, ""))
        c2_str = pc.utf8_trim_whitespace(pc.fill_null(c2_str, ""))

        # Vectorized string concatenation via C++ binary join
        combined = pc.binary_join_element_wise(c1_str, c2_str, sep)
        combined = pc.utf8_trim_whitespace(combined)

        if new_name in self.table.column_names:
            self.table = self.table.drop([new_name])
        self.table = self.table.append_column(new_name, combined)
        self._log_info(f"Merged columns '{col1}' and '{col2}' into unified target: '{new_name}'")
        return self

    def drop_short_strings(self, column, min_chars=10):
        if column in self.table.column_names:
            col_idx = self.table.schema.get_field_index(column)
            arr = self.table.column(column).combine_chunks()
            if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
                lengths = pc.utf8_length(arr)
                is_short = pc.less(lengths, min_chars)
                is_short = pc.fill_null(is_short, False)
                
                cleaned_arr = pc.if_else(is_short, pa.scalar(None, type=arr.type), arr)
                self.table = self.table.set_column(col_idx, column, cleaned_arr)
                self._log_info(f"Cleaned column '{column}': Nullified short values under {min_chars} characters.")
        return self
    
    def reset_data(self, original_data):
        self._log_info("Resetting working database Table to the original source state...")
        if isinstance(original_data, pa.Table):
            self.table = original_data
        elif isinstance(original_data, pd.DataFrame):
            self.table = pa.Table.from_pandas(original_data)
        else:
            raise TypeError("Original data state must be a pyarrow.Table or pandas.DataFrame.")
        self.stats = {}
        return self

    def enforce_schema(self, schema_dict, protected_values=None):
        self._print_header(f"Enforcing Schema Constraints Across {len(schema_dict)} Columns")
        
        if protected_values:
            self.protected_values = {k: set(map(str, v)) for k, v in protected_values.items()}

        for col, dtype in schema_dict.items():
            if col not in self.table.column_names:
                continue
            
            is_complex = dtype in ['list', 'dict']
            n_placeholders, found_examples = self._scan_placeholders(col, is_complex=is_complex)

            array = self.table.column(col).combine_chunks()
            initial_null_count = array.null_count

            cleaned_array = self._clean_arrow_array(array, dtype, col_name = col)
            final_null_count = cleaned_array.null_count

            col_idx = self.table.schema.get_field_index(col)
            self.table = self.table.set_column(col_idx, col, cleaned_array)

            total_cleaned = final_null_count - initial_null_count
            n_mismatch = max(0, total_cleaned - n_placeholders)
            if total_cleaned < n_placeholders:
                n_placeholders = total_cleaned

            if total_cleaned > 0:
                self.stats[col] = {
                    'total': total_cleaned,
                    'placeholders': n_placeholders,
                    'mismatch': n_mismatch,
                    'examples': found_examples
                }

        if self.stats:
            print(f"\n{'='*100}")
            print(f"| {'CLEANING REPORT: STRIPPED CONVERSIONS TO NaN':^96} |")
            print(f"{'='*100}")
            print(f"| {'Column':<20} | {'Total':<8} | {'Regex Match':<12} | {'Mismatch':<10} | {'Detected Sample Placeholders':<35} |")
            print(f"{'-'*100}")
            for col, data in self.stats.items():
                ex_str = str(data['examples'])
                if len(ex_str) > 35: 
                    ex_str = ex_str[:32] + "..."
                print(f"| {col:<20} | {data['total']:<8} | {data['placeholders']:<12} | {data['mismatch']:<10} | {ex_str:<35} |")
            print(f"{'='*100}\n")
        else:
            self._log_info("Schema validation complete: No garbage or type mismatches were found.")

        return self

    def clean_column_names(self):
        new_names = []
        for name in self.table.column_names:
            cleaned = name.strip().lower().replace(' ', '_')
            cleaned = re.sub(r'\s+', '_', cleaned)
            cleaned = re.sub(r'[^a-z0-9_]', '', cleaned)
            new_names.append(cleaned)
        self.table = self.table.rename_columns(new_names)
        self._log_info("Cleaned and normalized all Table column labels to standard snake_case.")
        return self
    
    #NOT IMPLEMENTED
    def drop_constant_columns(self):
        cols_to_drop = []
        for col in self.table.column_names:
            arr = self.table.column(col).combine_chunks()
            unique_vals = pc.unique(arr)
            if len(unique_vals) <= 1:
                cols_to_drop.append(col)
        if cols_to_drop:
            self.table = self.table.drop(cols_to_drop)
            self._log_info(f"Dropped non-informative constant columns: {cols_to_drop}")
        return self
    
    def cap_outliers(self, columns, lower_quantile=0.05, upper_quantile=0.95):
        for col in columns:
            if col in self.table.column_names:
                col_idx = self.table.schema.get_field_index(col)
                arr = self.table.column(col).combine_chunks()
                if pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type):
                    series = arr.to_pandas()
                    lower_val = series.quantile(lower_quantile)
                    upper_val = series.quantile(upper_quantile)
                    
                    clipped = pc.if_else(pc.less(arr, lower_val), pa.scalar(lower_val, type=arr.type), arr)
                    clipped = pc.if_else(pc.greater(clipped, upper_val), pa.scalar(upper_val, type=arr.type), clipped)
                    
                    self.table = self.table.set_column(col_idx, col, clipped)
                    self._log_info(f"Capped outlier values in numeric column '{col}' between values: ({lower_val:.2f}, {upper_val:.2f})")
        return self

    def drop_missing_cols(self, threshold=0.95, exclude=None):
        exclude = exclude or []
        cols_to_drop = []
        total_rows = self.table.num_rows
        if total_rows == 0:
            return self
            
        for col in self.table.column_names:
            null_count = self.table.column(col).null_count
            if (null_count / total_rows) > threshold:
                if col not in exclude:
                    cols_to_drop.append(col)
        if cols_to_drop:
            self.table = self.table.drop(cols_to_drop)
            self._log_info(f"Dropped high-density missing value columns (> {threshold*100}% empty): {cols_to_drop}")
        return self
    
    # =========================================================================
    # DEDUPLICATION PIPELINE
    # =========================================================================

    def check_duplicates(self, exclude_cols=None, subset=None):
        """
        Audits and logs duplicate counts matching on designated columns.
        """
        hashable_cols = self._get_hashable_subset(exclude_cols, subset)

        if not hashable_cols:
            print("No remaining primitive hashable columns left to check.")
            return self

        key_subset_df = self.table.select(hashable_cols).to_pandas()
        dupes = key_subset_df.duplicated().sum()

        if dupes > 0:
            print(f"Detected {dupes} duplicate rows matching on checked columns: {hashable_cols}")
        else:
            print(f"No duplicate rows encountered matching on checked columns: {hashable_cols}")
        return self

    def show_column_duplicates(self, cols):
        if isinstance(cols, str):
            cols = [cols]
            
        for col in cols:
            if col not in self.table.column_names:
                continue
            col_series = self.table.column(col).combine_chunks().to_pandas()
            counts = col_series.value_counts()
            duplicates = counts[counts >= 2]

            self._print_header(f"Value Duplicates Audit on Column: '{col}' (Total Repeated Values: {len(duplicates)})")
            if not duplicates.empty:
                print(duplicates.to_string(header=False))
            else:
                print("No values encountered with multiple duplicate references.")
        return self

    #TODO allow for unhashable duplicate checks
    def drop_exact_duplicates(self, exclude_cols=None, subset=None):
        """
        Drops duplicate rows from the PyArrow Table based on hashable columns.
        """
        hashable_subset = self._get_hashable_subset(exclude_cols, subset)
        
        if not hashable_subset:
            self._log_warning("Unable to execute exact duplicate drop. No hashable columns found.")
            return self

        initial_len = self.table.num_rows
        # Fixed: reset index to guarantee Pandas labels map perfectly to PyArrow positional indices
        key_subset_df = self.table.select(hashable_subset).to_pandas().reset_index(drop=True)
        clean_indices = key_subset_df.drop_duplicates().index
        
        self.table = self.table.take(pa.array(clean_indices))
        dropped = initial_len - self.table.num_rows
        
        if dropped > 0:
            self._log_info(f"Successfully dropped {dropped} duplicate rows based on columns: {hashable_subset}")
        else:
            self._log_info("Exact duplicate check completed. No rows require dropping.")
        return self

    #UNIMPLEMENTED
    def resolve_fuzzy_duplicates(self, title_col='title', author_col='authors', 
                                title_threshold=60, author_threshold=80):
        if process is None or fuzz is None:
            self._log_warning("Fuzzy logic resolution skipped. Please install 'rapidfuzz' or 'fuzzywuzzy' first.")
            return self

        if title_col not in self.table.column_names or author_col not in self.table.column_names:
            self._log_warning(f"Fuzzy resolution cancelled. Target columns do not exist in index: '{title_col}', '{author_col}'.")
            return self

        self._log_info(f"Initiating fuzzy deduplication loop (Title threshold: {title_threshold}, Author threshold: {author_threshold})")
        
        # Serialize ONLY the validation target columns to Pandas
        fuzzy_keys_df = self.table.select([title_col, author_col]).to_pandas()
        fuzzy_keys_df[title_col] = fuzzy_keys_df[title_col].astype(str).str.strip().str.lower()
        fuzzy_keys_df[author_col] = fuzzy_keys_df[author_col].astype(str).str.strip().str.lower()

        keep_idx = set()
        remove_idx = set()
        
        # Native score checking of row completeness directly on Column Chunks
        completeness = np.zeros(self.table.num_rows, dtype=int)
        for col in self.table.column_names:
            valid_mask = pc.is_valid(self.table.column(col).combine_chunks()).to_numpy()
            completeness += valid_mask.astype(int)

        title_map = fuzzy_keys_df.groupby(title_col).groups
        unique_titles = list(title_map.keys())
        candidate_pairs = []

        for t1 in unique_titles:
            matches = process.extract(t1, unique_titles, scorer=fuzz.token_sort_ratio, limit=5)
            for t2, score, _ in matches:
                if t1 == t2: 
                    continue
                if score >= title_threshold:
                    idxs_1 = title_map[t1]
                    idxs_2 = title_map[t2]
                    for i1 in idxs_1:
                        for i2 in idxs_2:
                            pair = tuple(sorted((i1, i2)))
                            candidate_pairs.append(pair)

        for title, indices in title_map.items():
            if len(indices) > 1:
                indices = list(indices)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        candidate_pairs.append((indices[i], indices[j]))

        candidate_pairs = list(set(candidate_pairs))
        self._log_info(f"  * Generated {len(candidate_pairs)} candidate duplicate pairs based on title alignment.")

        for idx1, idx2 in candidate_pairs:
            if idx1 in remove_idx or idx2 in remove_idx:
                continue

            auth1 = fuzzy_keys_df.loc[idx1, author_col]
            auth2 = fuzzy_keys_df.loc[idx2, author_col]

            if fuzz.token_sort_ratio(auth1, auth2) >= author_threshold:
                score1 = completeness[idx1]
                score2 = completeness[idx2]
                
                if score1 >= score2:
                    keep_idx.add(idx1)
                    remove_idx.add(idx2)
                else:
                    keep_idx.add(idx2)
                    remove_idx.add(idx1)

        if remove_idx:
            all_indices = set(range(self.table.num_rows))
            clean_indices = sorted(list(all_indices - remove_idx))
            self.table = self.table.take(pa.array(clean_indices))
            self._log_info(f"Fuzzy resolution removed {len(remove_idx)} duplicate rows, retaining higher-completeness items.")
        else:
            self._log_info("Fuzzy deduplication complete. No near-duplicate targets removed.")

        return self
    


    def resolve_mixed_types(self, interactive=False):
        self._log_info("Non-interactive mode: PyArrow column types are strict. No inline mixed resolution required.")
        return {}

    def remove_missing_values(self, how='any', subset=None):
        initial = self.table.num_rows
        subset = subset or self.table.column_names
        
        valid_masks = []
        for col in subset:
            if col in self.table.column_names:
                valid_masks.append(pc.is_valid(self.table.column(col).combine_chunks()))
        
        if not valid_masks:
            return self
            
        if how == 'any':
            final_mask = valid_masks[0]
            for m in valid_masks[1:]:
                final_mask = pc.and_(final_mask, m)
        elif how == 'all':
            final_mask = valid_masks[0]
            for m in valid_masks[1:]:
                final_mask = pc.or_(final_mask, m)
        else:
            raise ValueError("how parameter must be either 'any' or 'all'.")

        self.table = self.table.filter(final_mask)
        dropped = initial - self.table.num_rows
        self._log_info(f"Dropped {dropped} rows containing null attributes with criteria how='{how}'.")
        return self

    # =========================================================================
    # PIPELINE EXECUTION & EXPORT INTERFACE
    # =========================================================================

    def run_auto_pipeline(self, schema=None, protected_values=None, drop_empty_cols=False, interactive=False, dedupe_exclude=None, dedupe_subset=None):
        self._print_header("Initializing Automated Cleaning Pipeline Execution")

        self._log_info("Step 1: Ingesting database layouts...")
        
        self._log_info("Step 2: Performing dynamic auto-inference logic...")
        detected_schema = self.auto_infer_schema()
        
        if schema:
            self._log_info(f"Applying {len(schema)} static user schema instructions.")
            detected_schema.update(schema)
        
        self._log_info("Step 3: Enforcing integrated schema transformations...")
        self.enforce_schema(detected_schema, protected_values)
        
        if drop_empty_cols: 
            self.drop_missing_cols()
            
        self._log_info("Step 4: Executing deduplication filters and compiling database representations...")
        self.drop_exact_duplicates(exclude_cols=dedupe_exclude, subset=dedupe_subset)
        self.summarize(exclude_cols=dedupe_exclude, subset=dedupe_subset)
        
        self._log_info("Pipeline execution successfully completed.")
        return self

    def to_pandas(self, use_arrow_dtype=True) -> pd.DataFrame:
        """
        Converts the clean internal PyArrow Table back to a Pandas DataFrame.
        
        Parameters:
        -----------
        use_arrow_dtype : bool, default True
            If True, maps PyArrow's physical schemas directly to pd.ArrowDtype,
            preserving the unified null (<NA>) values and zero-copy performance [1, pyarrow].
            If False, falls back to traditional NumPy-backed Pandas types.
        """
        if use_arrow_dtype:
            return self.table.to_pandas(types_mapper=pd.ArrowDtype)
        else:
            return self.table.to_pandas()
    
    def save(self, path, file_format='csv', **kwargs):
        if not path:
            self._log_warning("Save command ignored. Missing valid path parameter.")
            return self

        target_path = Path(path)
        fmt = file_format.lower().strip()

        suffix = target_path.suffix.lower()
        if suffix == '.csv':
            fmt = 'csv'
        elif suffix in ['.parquet', '.pq']:
            fmt = 'parquet'

        if fmt == 'csv':
            # PyArrow multi-threaded C++ CSV serialization
            write_options = None
            if kwargs:
                valid_options = {k: v for k, v in kwargs.items() if k in ['include_header', 'batch_size']}
                if valid_options:
                    write_options = pa_csv.WriteOptions(**valid_options)
            
            pa_csv.write_csv(self.table, target_path, write_options=write_options)
            self._log_info(f"Successfully saved CSV format to disk via PyArrow: {target_path}")
        elif fmt in ['parquet', 'pq']:
            # PyArrow native write preserving lists/struct structures natively
            pq.write_table(self.table, target_path, **kwargs)
            self._log_info(f"Successfully saved Parquet format to disk via PyArrow: {target_path}")
        else:
            raise ValueError(f"Unsupported file format '{file_format}'. Supported values are 'csv' or 'parquet'.")

        return self