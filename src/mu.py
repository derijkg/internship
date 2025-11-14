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

def basic_imports():
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
    import pandas as pd
    
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



#  DATAFRAMECLEANER
import pandas as pd
import numpy as np
import re

class DataFrameCleaner:
    """
    A class to encapsulate a pandas DataFrame and apply a series of
    common cleaning operations.

    Attributes:
        df (pd.DataFrame): The DataFrame being cleaned.
    """

    def __init__(self, dataframe):
        """
        Initializes the cleaner with a DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be cleaned.
                                      A copy is made to avoid side effects.
        """
        self.df = dataframe.copy()
        print("DataFrameCleaner initialized. A copy of the original DataFrame has been made.")
        
        print('Setting display.max_rows to None')
        pd.set_option('display.max_rows', None)

        patterns = [
                r'(?i)^nan$', #case insensitive nan
                r'(?i)^(?:n\/?a|null|none|<none>|not reported|unknown|undefined|missing)$',
                r'^(?:-+$|/+$)',
                r'^\?+$',
                r'^(?:-99|-9999|999|9999)$',
            ]
        self.na_placeholders = '|'.join(patterns)

        self.summarize()

    # --- 1. CHECKING METHODS (Inspect without modifying) ---

    def summarize(self):
        """Prints a summary of the DataFrame, including info and shape."""
        print("\n--- DataFrame Summary ---")
        print(f"Shape: {self.df.shape}")
        self.check_duplicates()
        self.check_missing_values()
        print("\nInfo:")
        self.df.info()
        print('\nDescribe:')
        print(self.df.describe())
        print('\nRandom row:')
        print(self.df.sample(n=1).T)
        self.check_string_placeholders()
        print("-------------------------\n")

        return self

    def check_missing_values(self):
        """Prints a report of missing values per column (count and percentage)."""
        missing_values = self.df.isna().sum()
        missing_percent = ((missing_values / len(self.df)) * 100).round(2)
        missing_df = pd.DataFrame({
            'missing_count': missing_values,
            'missing_percent': missing_percent
        })
        print("\n--- Missing Values Report ---")
        print(missing_df[missing_df['missing_count'] > 0])
        return self

    def check_duplicates(self):
        """Prints the number of duplicate rows found."""
        num_duplicates = self.df.duplicated().sum()
        print(f"\nFound {num_duplicates} duplicate rows.\n")
        return self
    
    def check_string_placeholders(self, na_placeholders=None):
        """
        Checks object/string columns for common string placeholders for NA values
        and prints a report of what it finds.
        
        Args:
            na_placeholders (list, optional): A custom list of strings to check for.
                                              Defaults to a comprehensive internal list.
        """
        print("\n--- Checking for String Placeholders ---")
        if na_placeholders is None:
            na_placeholders = self.na_placeholders

        found_placeholders = {}
        string_cols = self.df.select_dtypes(include=['object', 'string']).columns

        for col in string_cols:
            matching_mask = self.df[col].astype(str).str.match(na_placeholders, na=False)
            if matching_mask.any():
                # Get the value counts of only the matching placeholders
                counts = self.df[col][matching_mask].value_counts()
                if not counts.empty:
                    found_placeholders[col] = counts

        # Report the findings
        if not found_placeholders:
            print("No common string NA placeholders found in object/string columns.")
        else:
            print("Found the following string placeholders:")
            for col, counts in found_placeholders.items():
                print(f"\nColumn '{col}':")
                print(counts)
                
        self.found_placeholders = found_placeholders
        return self
        
    def head(self, n=5):
        """Shows the first n rows of the current DataFrame."""
        print(self.df.head(n))
        return self

    # --- 2. CLEANING METHODS (Modify the DataFrame) ---

    def standardize_missing_values(self, include_cols=None, exclude_cols=None, placeholders=None):
        if include_cols is not None and exclude_cols is not None:
            raise ValueError("Cannot specify both 'include_cols' and 'exclude_cols'. Please choose one.")

        if placeholders is None:
            placeholders = self.na_placeholders
        
        initial_nas = self.df.isna().sum().sum()
        
        target_cols = self.df.columns
        
        # Determine the target columns for the operation
        if include_cols is not None:
            # Validate that included columns actually exist
            target_cols = [col for col in include_cols if col in self.df.columns]
            print(f"Applying replacement to specified columns: {target_cols}")
            self.df[target_cols] = self.df[target_cols].replace(placeholders, pd.NA)

        elif exclude_cols is not None:
            target_cols = [col for col in self.df.columns if col not in exclude_cols]
            print(f"Applying replacement to all columns EXCEPT: {exclude_cols}")
            self.df[target_cols] = self.df[target_cols].replace(placeholders, pd.NA)
            
        else:
            # Default behavior: replace on the entire DataFrame
            print("Applying replacement to all columns.")
            self.df = self.df.replace(placeholders, pd.NA)
        
        new_nas = self.df.isna().sum().sum()
        print(f"Standardized missing values. Added {new_nas - initial_nas} new NA values.")
        return self

    def clean_column_names(self):
        """
        Standardizes all column names to snake_case (lowercase with underscores).
        """
        new_cols = []
        for col in self.df.columns:
            new_col = str(col).strip().lower()
            new_col = re.sub(r'\s+', '_', new_col) # Replace spaces with underscores
            new_col = re.sub(r'[^a-z0-9_]', '', new_col) # Remove special characters
            new_cols.append(new_col)
        self.df.columns = new_cols
        print("Cleaned column names.")
        return self


    def drop_duplicates(self):
        """Removes duplicate rows from the DataFrame."""
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        rows_dropped = initial_rows - len(self.df)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} duplicate rows.")
        else:
            print("No duplicate rows to drop.")
        return self

    def strip_whitespace(self, columns=None):
        """
        Strips leading/trailing whitespace from string columns.

        Args:
            columns (list, optional): Specific columns to strip.
                                      If None, applies to all object columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'string']).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip()
        print(f"Stripped whitespace from columns: {list(columns)}.")
        return self

    def convert_data_types(self):
        """
        Uses pandas' convert_dtypes() to infer and convert columns to the
        best possible types (e.g., string, Int64, boolean).
        """
        self.df = self.df.convert_dtypes()
        print("Attempted to convert columns to more efficient types.")
        return self
        
    def reset_dataframe_index(self):
        """Resets the DataFrame's index, useful after dropping rows."""
        self.df.reset_index(drop=True, inplace=True)
        print("Reset DataFrame index.")
        return self

    # --- 3. UTILITY METHODS ---

    def run_all(self):
        """
        Runs a standard sequence of cleaning operations.
        This is a great one-shot method for a quick clean.
        """
        print("\n--- Running Full Cleaning Pipeline ---")
        self.clean_column_names()
        self.standardize_missing_values()
        self.strip_whitespace()
        self.drop_duplicates()
        self.convert_data_types()
        self.reset_dataframe_index()
        print("\n--- Full Cleaning Pipeline Complete ---")
        self.summarize()
        return self

    def get_df(self):
        """Returns the cleaned DataFrame."""
        return self.df



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
    """Makes a robust, timed HTTP request with error handling.

    :param url: The URL to send the request to.
    :type url: str
    :param method: The HTTP method to use (e.g., 'GET', 'POST'). Defaults to 'GET'.
    :type method: str, optional
    :param delay: Time in seconds to wait before sending the request. If None,
                a random delay (1.5s-4.5s) is used. Defaults to None.
    :type delay: float, optional
    :param timeout: Seconds to wait for the server to respond before giving up.
    :type timeout: int, optional
    :param headers: A dictionary of HTTP headers. If None, default headers are used.
    :type headers: dict, optional
    :param **kwargs: Additional keyword arguments to pass to `requests.request`,
                    such as `json`, `data`, or `params`.
    :return: The `requests.Response` object on a successful request (status 2xx),
            otherwise None.
    :rtype: requests.Response or None
    """

    # delay
    if delay is None:
        delay = random.uniform(1.5, 4.5)
    time.sleep(delay)

    # headers
    request_headers = headers
    if headers is None and session is None:
        request_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }

    # request
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
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} for URL {url}. Full error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for URL {url}. Error: {e}")
        return None