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
from rapidfuzz import process, fuzz
from collections import defaultdict
    
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



#  DATAFRAMECLEANER
    # TODO
        # checken voor type outliers
        # include exclude list voor placeholders + specific placeholders
        # per colom # unique waarden
        # 
import re
import ast
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

try:
    from rapidfuzz import process, fuzz
except ImportError:
    try:
        from fuzzywuzzy import process, fuzz
    except ImportError:
        process, fuzz = None, None


#TODO allow for csv or parquet output
#TODO check deduplication
#TODO check _prune
#TODO check other pruning like function
#TODO 

# dfcleaner unified with schemaenforcer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RobustDataFrameCleaner")


class DataFrameCleaner:
    """
    A unified class that combines data cleaning, schema enforcement, structural auditing,
    and deduplication on Pandas DataFrames (including PyArrow backends).
    """

    def __init__(self, data, regex_patterns=None, protected_values=None):
        self.stats = {}
        self.true_vals = {'y', 'yes', 't', 'true', '1', 'on'}
        self.false_vals = {'n', 'no', 'f', 'false', '0', 'off'}

        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self._log_info("DataFrame cleaner initialized, working on a copy.")
        elif isinstance(data, (Path, str)):
            try:
                path = Path(data)
            except Exception as e:
                raise ValueError(f"Could not convert string to path: {e}")
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() != '.parquet':
                raise ValueError(f"Input file must be a .parquet file, received: {path.suffix}")

            self.df = pd.read_parquet(path, engine='pyarrow', dtype_backend='pyarrow')
            self._log_info(f"DataFrame cleaner initialized from file: {path.name}")
        else:
            raise TypeError("Initialization parameter 'data' must be either a pandas DataFrame or a valid Path/str pointing to a Parquet file.")

        if regex_patterns:
            self.garbage_regex = re.compile(regex_patterns)
        else:
            patterns = [
                r'(?i)^nan$',
                r'(?i)^(?:n\/?a|null|none|<none>|not reported|unknown|undefined|missing)$',
                r'^(?:-+$|/+$)',
                r'^\?+$',
                r'^(?:-99|-9999|999|9999)$',
            ]
            combined_pattern = r'^(?:' + '|'.join(patterns) + r')$'
            self.garbage_regex = re.compile(combined_pattern, flags=re.IGNORECASE)

        self.na_placeholders = self.garbage_regex.pattern
        self.protected_values = {k: set(map(str, v)) for k, v in (protected_values or {}).items()}

        pd.set_option('display.max_rows', 100)

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
    # INTERNAL CLEANING & ENFORCING LOGIC (Merged from SchemaEnforcer)
    # =========================================================================

    def _is_garbage(self, val, col_name=None):
        if val is None or val is pd.NA:
            return True
        if isinstance(val, (float, int, np.float64, np.int64)) and np.isnan(val):
            return True

        s_val = str(val).strip()
        if not s_val:
            return True
        if col_name and col_name in self.protected_values:
            if s_val in self.protected_values[col_name]:
                return False
        if self.garbage_regex.match(s_val):
            return True
        return False

    def _prune_empty(self, obj):
        if isinstance(obj, list):
            cleaned = [self._prune_empty(x) for x in obj]
            if all(x is None for x in cleaned):
                return None
            return cleaned
        elif isinstance(obj, dict):
            cleaned = {k: self._prune_empty(v) for k, v in obj.items()}
            cleaned = {k: v for k, v in cleaned.items() if v is not None}
            return cleaned if cleaned else None
        return None if self._is_garbage(obj) else obj
    
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

    def _clean_complex(self, val, expected_type, col_name):
        if self._is_garbage(val, col_name): 
            return None
        if isinstance(val, np.ndarray): 
            val = val.tolist()
        
        if isinstance(val, str):
            val = self._parse_string_structure(val)
            if isinstance(val, str): # Parsing failed to resolve string into a container; discard element
                return None
                
        if isinstance(val, expected_type):
            return self._prune_empty(val)
        else:
            if expected_type is list and val is not None:
                self._log_warning(f"UNEXPECTED DTYPE: {col_name}: {val}\nAttempting recovery through listing value.")
                new_val = [val]
                return self._prune_empty(new_val)
            else:
                self._log_warning(f"UNEXPECTED DTYPE: {col_name}: {val}\nIrrecoverable: {expected_type} value set to None")
                return None
            

    def _clean_bool(self, val, col_name):
        if self._is_garbage(val, col_name):
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            if val == 1:
                return True
            if val == 0:
                return False
            return None
        if isinstance(val, str):
            s = val.strip().lower()
            if s in self.true_vals:
                return True
            if s in self.false_vals:
                return False
        return None

    def _clean_scalar_str(self, val, col_name):
        if self._is_garbage(val, col_name): 
            return None
        
        if hasattr(val, 'as_py'):
            val = val.as_py()

        if isinstance(val, str):
            parsed = self._parse_string_structure(val)
            if parsed is not val:  # Unified parser successfully parsed the string to container structure
                return self._clean_scalar_str(parsed, col_name)
            
            if self._is_garbage(val, col_name): 
                return None
            return val

        if isinstance(val, (list, tuple, np.ndarray, set)):
            if isinstance(val, (np.ndarray, set)): 
                val = list(val)
            
            valid_items = [v for v in val if not self._is_garbage(v, col_name)]
            
            if not valid_items:
                return None
            
            return self._clean_scalar_str(valid_items[0], col_name)

        if isinstance(val, dict):
            return None

        return str(val).strip()

    def _scan_placeholders(self, col):
        """
        Vectorized lookup scan to count and record unique occurrences of 
        pre-defined NA placeholder configurations in a given column.
        """
        try:
            s = self.df[col].dropna().astype(str).str.strip()
            if s.empty:
                return 0, []

            if col in self.protected_values:
                mask_protected = s.isin(self.protected_values[col])
                s = s[~mask_protected]

            if s.empty:
                return 0, []

            mask_garbage = s.str.match(self.garbage_regex)
            garbage_values = s[mask_garbage]
            count = garbage_values.shape[0]

            examples = garbage_values.unique().tolist() if count > 0 else []
            return count, examples
        except Exception:
            return 0, []

    # =========================================================================
    # AUDITING & DIAGNOSTICS (Inspect methods)
    # =========================================================================

    def _find_unhashable_columns(self):
        unhashable_cols = []
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if isinstance(dtype, pd.ArrowDtype):
                pa_type = dtype.pyarrow_dtype
                if (pa.types.is_list(pa_type) or 
                    pa.types.is_large_list(pa_type) or 
                    pa.types.is_fixed_size_list(pa_type) or 
                    pa.types.is_struct(pa_type) or 
                    pa.types.is_map(pa_type)):
                    unhashable_cols.append(col)
                    continue

            if pd.api.types.is_object_dtype(dtype):
                valid = self.df[col].dropna()
                if not valid.empty:
                    if isinstance(valid.iloc[0], (list, dict, set, np.ndarray)):
                        unhashable_cols.append(col)
        return unhashable_cols

    #recursive
    def _analyze_arrow_data(self, array, indent=0):
        prefix = " " * indent
        arrow_type = array.type
        
        KEY_WIDTH = 30    
        TYPE_WIDTH = 15   
        SAMPLE_WIDTH = 40 

        if pa.types.is_list(arrow_type):
            print(f"{prefix}- List of:")
            self._analyze_arrow_data(array.flatten(), indent + 2)
            
        elif pa.types.is_struct(arrow_type):
            print(f"{prefix}- Object (Dict) with keys:")
            for field in arrow_type:
                child_array = array.field(field.name)
                label = f"{prefix}  * {field.name}:"
                
                if pa.types.is_list(field.type) or pa.types.is_struct(field.type):
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
        """
        Prints a structured summary table of the working DataFrame, reflecting types,
        missing counts, percentage nulls, observed placeholders, and data samples.
        """
        self._print_header(f"DataFrame Summary (Shape: {self.df.shape[0]} rows x {self.df.shape[1]} cols)")
        
        print("\n--- DUPLICATE COUNT AUDIT ---")
        self.check_duplicates(exclude_cols=exclude_cols, subset=subset)
        print("-" * 80)

        # Build clean structural overview DataFrame
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
            unique_counts[col] = -1  # Placeholder marker for complex columns

        summary['unique#'] = unique_counts.astype(int)

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
                # Optimized logic retrieval
                try:
                    unique_vals = self.df[col].dropna().unique()
                    u_series = pd.Series(unique_vals).astype(str)
                    matches = u_series[u_series.str.match(self.na_placeholders, na=False)]
                    found_placeholders.append(str(matches.tolist()) if not matches.empty else "")
                except Exception:
                    found_placeholders.append("")
            else:
                found_placeholders.append("")

        summary['sample_val'] = samples
        summary['placeholders'] = found_placeholders

        order = ['missing#', 'missing%', 'unique#', 'sample_val', 'placeholders', 'dtypes']
        summary = summary[order]

        # Format printing column sizes explicitly for clean reading
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

    #TODO useless function just integrate into analyzearrow data???
    def analyze_structure_recursive(self, sample_size=5000):
        unhashable_cols = self._find_unhashable_columns()
        if not unhashable_cols:
            return self

        print(f"\n--- Deep Structure & Stats (Sample size limit: {sample_size}) ---")
        for col in unhashable_cols:
            valid_data = self.df[col].dropna()
            if valid_data.empty:
                continue
            
            if len(valid_data) > sample_size:
                valid_data = valid_data.sample(n=sample_size, random_state=42)
            
            print(f"Column '{col}':")
            try:
                arrow_array = pa.array(valid_data)
                self._analyze_arrow_data(arrow_array, indent=2)
            except pa.ArrowInvalid:
                self._log_warning(f"  [!] Mixed type configuration on column '{col}'. Cannot infer strict Arrow schema.")
            except Exception as e:
                self._log_warning(f"  [!] Structural parser failure: {e}")
            print("")
        return self

    def get_samples(self, columns=None, number=5):
        valid = [c for c in columns or [] if c in self.df.columns]
        invalid = set(columns or []) - set(valid)
        if invalid:
            self._log_warning(f"Columns not found in DataFrame index: {invalid}")
            
        for col in valid:
            s = self.df[col].dropna()
            if s.empty:
                print(f"Column: {col} is empty.")
                continue
            self._print_header(f"Samples for column '{col}' ({type(s.iloc[0]).__name__})")
            for item in s.sample(min(number, len(s))):
                print(f" > {item}")
            print("-" * 40)
        return self

    def audit_mixed_types(self, verbose=True):
        report = {}
        candidates = self.df.select_dtypes(include=['object']).columns
        
        for col in candidates:
            inferred_type = pd.api.types.infer_dtype(self.df[col], skipna=True)
            if not inferred_type.startswith("mixed"):
                continue

            valid_series = self.df[col].dropna()
            if valid_series.empty:
                continue

            type_series = valid_series.apply(self._get_python_type_name)
            type_counts = type_series.value_counts()
            
            if len(type_counts) > 1:
                majority_type = type_counts.idxmax()
                majority_count = type_counts.max()
                total_count = type_counts.sum()
                
                outlier_mask = type_series != majority_type
                outlier_samples = valid_series[outlier_mask].head(5).tolist()
                
                report[col] = {
                    'majority_type': majority_type,
                    'majority_pct': (majority_count / total_count) * 100,
                    'breakdown': type_counts.to_dict(),
                    'outliers': outlier_samples
                }

        if verbose and report:
            print("\n--- MIXED DATA TYPE CONFLICTS ---")
            for col, r_data in report.items():
                print(f"Column: {col}")
                print(f"  * Majority Representation: {r_data['majority_type']} ({r_data['majority_pct']:.2f}%)")
                print(f"  * Outliers Encountered: {r_data['outliers']}")
                print(f"  * Type Breakdown: {r_data['breakdown']}")
        return report

    def check_missing_values(self):
        na_counts = self.df.isna().sum()
        missing_data = na_counts[na_counts > 0]
        
        if not missing_data.empty:
            self._print_header("Missing Value Verification")
            percentages = (missing_data / len(self.df)) * 100
            report = pd.DataFrame({
                'Missing Count': missing_data,
                'Percentage (%)': percentages.round(2)
            })
            print(report)
        else:
            self._log_info("No missing NaN values detected.")
        return self


    def auto_infer_schema(self, sample_pct=0.05, min_sample=1000, max_sample=50000):
        if self.df.empty:
            self._log_warning("Working DataFrame is empty. Bypassing schema inference.")
            return {}

        total_rows = len(self.df)
        calculated_sample = int(total_rows * sample_pct)
        sample_size = max(min(calculated_sample, max_sample), min_sample)
        sample_size = min(sample_size, total_rows)

        inferred = {}
        self._log_info(
            f"Auto-inferring schema dynamically. Sample size configured to n={sample_size} "
            f"(calculated from {sample_pct * 100}% of {total_rows} rows)."
        )

        for col in self.df.columns:
            valid = self.df[col].dropna()
            if valid.empty:
                continue
            
            if len(valid) > sample_size: 
                valid = valid.sample(n=sample_size, random_state=42)

            if pd.api.types.is_bool_dtype(self.df[col]): 
                inferred[col] = 'bool'
                continue
            if pd.api.types.is_integer_dtype(self.df[col]): 
                inferred[col] = 'int'
                continue
            if pd.api.types.is_float_dtype(self.df[col]): 
                inferred[col] = 'float'
                continue
            if pd.api.types.is_datetime64_any_dtype(self.df[col]): 
                inferred[col] = 'datetime'
                continue

            if pd.api.types.is_object_dtype(self.df[col]):
                inferred_type = pd.api.types.infer_dtype(valid, skipna=True)
                if inferred_type in ['mixed-integer', 'mixed-integer-float']:
                    inferred[col] = 'string'
                    self._log_info(f"  > Mixed numeric profile for '{col}' ({inferred_type}). Assigned fallback target 'string'.")
                    continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                unique_nums = valid.unique()
                if set(unique_nums).issubset({0, 1, 0.0, 1.0}):
                    inferred[col] = 'bool'
                    self._log_info(f"  > Boolean binary structure detected in numeric column: '{col}' -> mapped 'bool'")
                    continue
                elif pd.api.types.is_integer_dtype(self.df[col]): 
                    inferred[col] = 'int'
                else: 
                    inferred[col] = 'float'
                continue

            try:
                first_val = valid.iloc[0]
                if hasattr(first_val, 'as_py'):
                    first_val = first_val.as_py()

                is_complex_object = isinstance(first_val, (list, dict, set, np.ndarray))

                if not is_complex_object:
                    raw_uniques = pd.Series(valid.unique()).dropna()
                    unique_vals = raw_uniques.astype(str).str.lower().unique()
                    
                    if len(unique_vals) <= 10:
                        u_series = pd.Series(unique_vals)
                        mask_clean = ~u_series.str.match(self.na_placeholders)
                        clean_vals = u_series[mask_clean].tolist()
                        
                        if clean_vals and set(clean_vals).issubset(self.true_vals | self.false_vals):
                            inferred[col] = 'bool'
                            self._log_info(f"  > Semantic Boolean flags inferred from text: '{col}' -> mapped 'bool'")
                            continue

                def normalize(x):
                    if hasattr(x, 'as_py'): 
                        x = x.as_py()
                    if isinstance(x, np.ndarray): 
                        return x.tolist()
                    if isinstance(x, str) and x.strip().startswith(('[', '{')):
                        try: 
                            return ast.literal_eval(x)
                        except Exception: 
                            pass
                    return x
                
                sample_list = valid.apply(normalize).tolist()
                arrow_type = pa.array(sample_list).type
                
                if pa.types.is_list(arrow_type): 
                    inferred[col] = 'list'
                elif pa.types.is_struct(arrow_type) or pa.types.is_map(arrow_type): 
                    inferred[col] = 'dict'
                else: 
                    inferred[col] = 'string'
            except Exception:
                inferred[col] = 'string'
                
        return inferred


    # =========================================================================
    # CORE CLEANING OPERATIONS (Modify methods)
    # =========================================================================

    def parse_and_flatten(self, col, mode='first'):
        """
        Parses complex structure columns and flattens nested elements.
        """
        return self.extract_first_element(col, mode=mode)

    def combine_columns(self, col1, col2, new_name, sep=' '):
        if col1 not in self.df.columns or col2 not in self.df.columns:
            self._log_warning(f"Action ignored. Missing columns: '{col1}' or '{col2}'.")
            return self

        s1 = self.df[col1].astype(str).replace('nan', '').str.strip()
        s2 = self.df[col2].astype(str).replace('nan', '').str.strip()

        self.df[new_name] = s1 + sep + s2
        self.df[new_name] = self.df[new_name].str.strip(sep)
        
        self._log_info(f"Merged columns '{col1}' and '{col2}' into unified target: '{new_name}'")
        return self

    def extract_first_element(self, column, mode='first'):
        if column not in self.df.columns:
            self._log_warning(f"Extraction failed. Column not found: '{column}'.")
            return self

        def _internal_parser(val):
            if isinstance(val, list):
                target_list = val
            elif isinstance(val, str) and val.strip().startswith('['):
                try:
                    target_list = ast.literal_eval(val.strip())
                except (ValueError, SyntaxError):
                    target_list = []
            else:
                return val

            if isinstance(target_list, list):
                if not target_list:
                    return None
                if mode == 'first':
                    return target_list[0]
                elif mode == 'join':
                    return "; ".join([str(i) for i in target_list])
            return val

        self.df[column] = self.df[column].apply(_internal_parser)
        self._log_info(f"Extracted/flattened elements inside target column '{column}' using mode='{mode}'")
        return self

    def drop_short_strings(self, column, min_chars=10):
        if column in self.df.columns:
            lengths = self.df[column].astype(str).str.len()
            mask = (lengths < min_chars) & (self.df[column].notna())
            count = mask.sum()
            if count > 0:
                self.df.loc[mask, column] = np.nan
                self._log_info(f"Cleaned column '{column}': Nullified {count} values shorter than {min_chars} characters.")
        return self
    
    def reset_data(self, df_original):
        self._log_info("Resetting working DataFrame back to the original source DataFrame...")
        self.df = df_original.copy()
        self.stats = {}
        return self


    def enforce_schema(self, schema_dict, protected_values=None):
        """
        Enforces type checking rules over structural columns. Modifies working state.
        """
        self._print_header(f"Enforcing Schema Constraints Across {len(schema_dict)} Columns")
        
        if protected_values:
            self.protected_values = {k: set(map(str, v)) for k, v in protected_values.items()}

        for col, dtype in schema_dict.items():
            if col not in self.df.columns:
                continue
            
            initial_valid = self.df[col].notna().sum()
            n_placeholders = 0
            found_examples = []

            # --- Fast Path Optimization ---
            if dtype == 'bool' and pd.api.types.is_bool_dtype(self.df[col]):
                if initial_valid == len(self.df): # No missing values to resolve
                    continue

            # --- Transformation & Verification Logic ---
            if dtype == 'list':
                if initial_valid > 0:
                    n_placeholders, found_examples = self._scan_placeholders(col)
                    self.df[col] = self.df[col].apply(lambda x: self._clean_complex(x, list, col))
            elif dtype == 'dict':
                if initial_valid > 0:
                    n_placeholders, found_examples = self._scan_placeholders(col)
                    self.df[col] = self.df[col].apply(lambda x: self._clean_complex(x, dict, col))
            elif dtype == 'bool':
                if initial_valid > 0:
                    n_placeholders, found_examples = self._scan_placeholders(col)
                    self.df[col] = self.df[col].map(lambda x: self._clean_bool(x, col))
            elif dtype == 'string':
                if initial_valid > 0:
                    n_placeholders, found_examples = self._scan_placeholders(col)
                    self.df[col] = self.df[col].map(lambda x: self._clean_scalar_str(x, col))
            elif dtype in ['int', 'float', 'number']:
                original_valid_mask = self.df[col].notna()
                coerced = pd.to_numeric(self.df[col], errors='coerce')
                failed_mask = original_valid_mask & coerced.isna()
                
                failed_vals = self.df.loc[failed_mask, col].dropna()
                if not failed_vals.empty:
                    failed_str = failed_vals.astype(str).str.strip()
                    is_garbage = failed_str.str.match(self.garbage_regex, na=False)
                    if col in self.protected_values:
                        is_protected = failed_str.isin(self.protected_values[col])
                        is_garbage = is_garbage & ~is_protected
                    
                    n_placeholders = is_garbage.sum()
                    found_examples = failed_str[is_garbage].unique().tolist()
                
                self.df[col] = coerced

            elif dtype in ['date', 'datetime']:
                original_valid_mask = self.df[col].notna()
                coerced = pd.to_datetime(self.df[col], errors='coerce')
                failed_mask = original_valid_mask & coerced.isna()
                
                failed_vals = self.df.loc[failed_mask, col].dropna()
                if not failed_vals.empty:
                    failed_str = failed_vals.astype(str).str.strip()
                    is_garbage = failed_str.str.match(self.garbage_regex, na=False)
                    if col in self.protected_values:
                        is_protected = failed_str.isin(self.protected_values[col])
                        is_garbage = is_garbage & ~is_protected
                    
                    n_placeholders = is_garbage.sum()
                    found_examples = failed_str[is_garbage].unique().tolist()
                
                self.df[col] = coerced

            final_valid = self.df[col].notna().sum()
            
            total_cleaned = initial_valid - final_valid
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
        self.df.columns = (self.df.columns.astype(str).str.strip().str.lower()
                           .str.replace(r'\s+', '_', regex=True)
                           .str.replace(r'[^a-z0-9_]', '', regex=True))
        self._log_info("Cleaned and normalized all DataFrame column labels to standard snake_case.")
        return self
    
    def drop_constant_columns(self):
        cols_to_drop = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self._log_info(f"Dropped non-informative constant columns: {cols_to_drop}")
        return self
    
    def cap_outliers(self, columns, lower_quantile=0.05, upper_quantile=0.95):
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                lower = self.df[col].quantile(lower_quantile)
                upper = self.df[col].quantile(upper_quantile)
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                self._log_info(f"Capped outlier values in numeric column '{col}' between quantiles: ({lower:.2f}, {upper:.2f})")
        return self

    #leave this to rot please
    def drop_missing_cols(self, threshold=0.95, exclude=None):
        exclude = exclude or []
        mask = self.df.isna().mean() > threshold
        cols_to_drop = [col for col in self.df.columns[mask] if col not in exclude]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self._log_info(f"Dropped high-density missing value columns (> {threshold*100}% empty): {cols_to_drop}")
        return self
    
    # =========================================================================
    # DEDUPLICATION PIPELINE
    # =========================================================================


    def check_duplicates(self, exclude_cols=None, subset=None):
        exclude_cols = exclude_cols or []
        unhashable_cols = self._find_unhashable_columns()
        
        if subset is not None:
            hashable_cols = [c for c in subset if c in self.df.columns and c not in unhashable_cols]
            invalid_or_unhashable = set(subset) - set(hashable_cols)
            if invalid_or_unhashable:
                self._log_warning(f"Excluding invalid or unhashable columns from specified subset: {list(invalid_or_unhashable)}")
        else:
            all_exclusions = set(unhashable_cols) | set(exclude_cols)
            if all_exclusions:
                self._log_warning(f"Excluding columns from precise duplicates calculation: {list(all_exclusions)}")
            hashable_cols = [c for c in self.df.columns if c not in all_exclusions]

        if not hashable_cols:
            print("No remaining primitive hashable columns left to check.")
            return self

        dupes = self.df.duplicated(subset=hashable_cols).sum()

        if dupes > 0:
            print(f"Detected {dupes} duplicate rows matching on checked columns: {hashable_cols}")
        else:
            print(f"No duplicate rows encountered matching on checked columns: {hashable_cols}")
        return self

    def show_column_duplicates(self, cols):
        if isinstance(cols, str):
            cols = [cols]
            
        for col in cols:
            if col not in self.df.columns:
                continue
            counts = self.df[col].value_counts()
            duplicates = counts[counts >= 2]

            self._print_header(f"Value Duplicates Audit on Column: '{col}' (Total Repeated Values: {len(duplicates)})")
            if not duplicates.empty:
                print(duplicates.to_string(header=False))
            else:
                print("No values encountered with multiple duplicate references.")
        return self

    def drop_exact_duplicates(self, exclude_cols=None, subset=None):
        unhashable = self._find_unhashable_columns()
        
        if subset is not None:
            hashable_subset = [c for c in subset if c in self.df.columns and c not in unhashable]
            invalid_or_unhashable = set(subset) - set(hashable_subset)
            if invalid_or_unhashable:
                self._log_warning(f"Excluding invalid or unhashable columns from specified subset: {list(invalid_or_unhashable)}")
        else:
            exclude_cols = exclude_cols or []
            hashable_subset = [c for c in self.df.columns if c not in unhashable and c not in exclude_cols]
        
        if not hashable_subset:
            self._log_warning("Unable to execute exact duplicate drop. No hashable columns found.")
            return self

        initial_len = len(self.df)
        self.df.drop_duplicates(subset=hashable_subset, inplace=True)
        dropped = initial_len - len(self.df)
        
        if dropped > 0:
            self._log_info(f"Successfully dropped {dropped} identical duplicate rows based on criteria columns: {hashable_subset}")
        else:
            self._log_info("Exact duplicate check completed. No rows require dropping.")
        return self

    def resolve_fuzzy_duplicates(self, title_col='title', author_col='authors', 
                                title_threshold=60, author_threshold=80):
        if process is None or fuzz is None:
            self._log_warning("Fuzzy logic resolution skipped. Please install 'rapidfuzz' or 'fuzzywuzzy' first.")
            return self

        if title_col not in self.df.columns or author_col not in self.df.columns:
            self._log_warning(f"Fuzzy resolution cancelled. Target columns do not exist in index: '{title_col}', '{author_col}'.")
            return self

        self._log_info(f"Initiating fuzzy deduplication loop (Title threshold: {title_threshold}, Author threshold: {author_threshold})")
        
        keep_idx = set()
        remove_idx = set()
        self.df['_completeness_score'] = self.df.notna().sum(axis=1)

        title_map = self.df.groupby(title_col).groups
        unique_titles = list(title_map.keys())
        candidate_pairs = []

        # Find potential approximate duplicate pairings
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

        # Catch exact text variations
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

            auth1 = str(self.df.loc[idx1, author_col])
            auth2 = str(self.df.loc[idx2, author_col])

            if fuzz.token_sort_ratio(auth1, auth2) >= author_threshold:
                score1 = self.df.loc[idx1, '_completeness_score']
                score2 = self.df.loc[idx2, '_completeness_score']
                
                if score1 >= score2:
                    keep_idx.add(idx1)
                    remove_idx.add(idx2)
                else:
                    keep_idx.add(idx2)
                    remove_idx.add(idx1)

        self.df.drop(columns=['_completeness_score'], inplace=True, errors='ignore')
        
        if remove_idx:
            self.df.drop(index=list(remove_idx), inplace=True)
            self._log_info(f"Fuzzy resolution removed {len(remove_idx)} duplicate rows, retaining higher-completeness items.")
        else:
            self._log_info("Fuzzy deduplication complete. No near-duplicate targets removed.")

        return self
    

    def drop_duplicates(self, check_unhashable=True, exclude_cols=None, subset=None):
        unhashable = self._find_unhashable_columns()
        
        if subset is not None:
            hashable_subset = [c for c in subset if c in self.df.columns and c not in unhashable]
            invalid_or_unhashable = set(subset) - set(hashable_subset)
            if invalid_or_unhashable:
                self._log_warning(f"Excluding invalid or unhashable columns from specified subset: {list(invalid_or_unhashable)}")
        else:
            exclude_cols = exclude_cols or []
            hashable_subset = [c for c in self.df.columns if c not in unhashable and c not in exclude_cols]
            
        if hashable_subset:
            initial = len(self.df)
            self.df.drop_duplicates(subset=hashable_subset, inplace=True)
            self._log_info(f"Checked for duplicates on columns:{hashable_subset}")
            self._log_info(f"Removed {initial - len(self.df)} duplicate rows based on columns: {hashable_subset}")
        return self
    



    def resolve_mixed_types(self, interactive=False):
        report = self.audit_mixed_types(verbose=True)
        if not report:
            return {}

        schema_overrides = {}
        type_map = {
            'str': 'string', 'int': 'int', 'float': 'float', 
            'bool': 'bool', 'list': 'list', 'dict': 'dict',
            'ndarray': 'list'
        }

        if not interactive:
            self._log_info("Non-interactive mode: Bypassing manual outlier resolution.")
            return {}

        self._print_header("Mixed Type Interaction Resolution Console")
        for col, data in report.items():
            maj_type = data['majority_type']
            maj_schema = type_map.get(maj_type, 'string')
            
            print(f"\nConflict in Column: '{col}'")
            print(f"  * Majority Representation: {maj_type} ({data['majority_pct']:.1f}%)")
            print(f"  * Conflict Types: {[k for k in data['breakdown'] if k != maj_type]}")
            print(f"  * Representative Outlier Values: {data['outliers']}")
            print("\nOptions:")
            print(f"  [1] Force values to Majority Representation type ('{maj_schema}') -> Invalid values coerced to NaN")
            print("  [2] Map complete column elements to string -> Retains value, drops specialized indexing")
            print("  [3] Manually type assign casting")
            print("  [4] Skip resolution and proceed with automated estimation rules")
            
            choice = input(f"Select action option [1-4] for '{col}': ").strip()
            if choice == '1':
                schema_overrides[col] = maj_schema
                self._log_info(f" -> Forced column '{col}' schema target to: '{maj_schema}'")
            elif choice == '2':
                schema_overrides[col] = 'string'
                self._log_info(f" -> Cast column '{col}' items completely to string representations.")
            elif choice == '3':
                manual = input("Enter schema choice keyword (int, float, string, bool, list, dict): ").strip()
                if manual in type_map.values():
                    schema_overrides[col] = manual
                    self._log_info(f" -> Applied manual type mapping for '{col}': '{manual}'")
                else:
                    self._log_warning("Invalid input. Proceeding with automated estimation rules.")
            else:
                self._log_info(f"Action bypassed on '{col}'.")
                
        return schema_overrides
    
    def convert_data_types_pandas(self):
        self.df = self.df.convert_dtypes()
        self._log_info("Mapped schema format to native Pandas nullable type system backend.")
        return self
    
    def convert_data_types_arrow(self):
        self._log_info("Converting columns to PyArrow-backed types for memory optimization...")
        try:
            self.df = self.df.convert_dtypes(dtype_backend="pyarrow")
        except Exception as e:
            self._log_warning(f"Automated PyArrow backend casting warning: {e}")

        for col in self.df.select_dtypes(include=['object']):
            try:
                arrow_array = pa.array(self.df[col].dropna())
                self.df[col] = self.df[col].astype(pd.ArrowDtype(arrow_array.type))
                self._log_info(f"  > Cast column '{col}' to arrow structure type: {arrow_array.type}")
            except Exception:
                pass
        self._log_info("Backend conversion to PyArrow formats completed.")
        return self
    
    def reset_index(self):
        self.df.reset_index(drop=True, inplace=True)
        return self

    def remove_missing_values(self, how='any', subset=None):
        """
        Drops missing rows completely using standard dropna behaviors.
        """
        initial = len(self.df)
        self.df.dropna(how=how, subset=subset, inplace=True)
        dropped = initial - len(self.df)
        self._log_info(f"Dropped {dropped} rows containing null attributes with criteria how='{how}'.")
        return self

    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================

    def run_auto_pipeline(self, schema=None, protected_values=None, drop_empty_cols=False, interactive=False, dedupe_exclude=None, dedupe_subset=None):
        self._print_header("Initializing Automated Cleaning Pipeline Execution")
        
        #self.clean_column_names()
        #TODO deduplicate edge cases where cols get same name
        #TODO clean schema col names

        self._log_info("Step 1: Auditing mixed datatype structures...")
        mixed_type_fixes = self.resolve_mixed_types(interactive=interactive)
        
        self._log_info("Step 2: Performing dynamic auto-inference logic...")
        detected_schema = self.auto_infer_schema()
        
        if mixed_type_fixes:
            self._log_info(f"Applying {len(mixed_type_fixes)} manual overrides resolved during audits.")
            detected_schema.update(mixed_type_fixes)
            
        if schema:
            self._log_info(f"Applying {len(schema)} static user schema instructions.")
            detected_schema.update(schema)
        
        self._log_info("Step 3: Enforcing integrated schema transformations...")
        self.enforce_schema(detected_schema, protected_values)
        
        if drop_empty_cols: 
            self.drop_missing_cols() #stay dead
            
        self._log_info("Step 4: Executing deduplication filters and locking database in Arrow representations...")
        self.convert_data_types_arrow()
        self.drop_duplicates(exclude_cols=dedupe_exclude, subset=dedupe_subset)
        self.summarize()
        
        self._log_info("Pipeline execution successfully completed.")
        return self.df
    
    
    def save(self, path, file_format='csv', **kwargs):
        """
        Saves the processed DataFrame to disk in either CSV or Parquet format.
        
        Parameters:
        -----------
        path : str or Path
            The file path where the DataFrame should be saved.
        file_format : str, default 'csv'
            The format to use. Supported options are 'csv' and 'parquet'.
            If the provided path extension is .csv or .parquet/.pq, it takes precedence.
        **kwargs : dict
            Optional keyword arguments passed directly to Pandas' `to_csv()` or `to_parquet()`.
        """
        if not path:
            self._log_warning("Save command ignored. Missing valid path parameter.")
            return self

        target_path = Path(path)
        fmt = file_format.lower().strip()

        # Deduce and override format dynamically based on path suffix if applicable
        suffix = target_path.suffix.lower()
        if suffix == '.csv':
            fmt = 'csv'
        elif suffix in ['.parquet', '.pq']:
            fmt = 'parquet'

        if fmt == 'csv':
            # Default options for CSV (can be overridden by kwargs)
            save_args = {'index': False}
            save_args.update(kwargs)
            self.df.to_csv(target_path, **save_args)
            self._log_info(f"Successfully saved CSV format to disk: {target_path}")
        elif fmt in ['parquet', 'pq']:
            # Default options for Parquet (can be overridden by kwargs)
            save_args = {'engine': 'pyarrow', 'index': False}
            save_args.update(kwargs)
            self.df.to_parquet(target_path, **save_args)
            self._log_info(f"Successfully saved Parquet format to disk: {target_path}")
        else:
            raise ValueError(f"Unsupported file format '{file_format}'. Supported values are 'csv' or 'parquet'.")

        return self

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

