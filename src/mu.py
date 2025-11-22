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
from collections import Counter
''' 
class DataFrameCleaner_old:
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
        self.check_duplicates()  # This is now simpler and more robust
        self.check_missing_values()
        print("\nInfo:")
        self.df.info(verbose=True, show_counts=True) # More detailed info, very long time, optim
        print('\nDescribe (numerical):')
        #print(self.df.describe())
        print('\nDescribe (categorical):')
        #print(self.df.describe(include=['object', 'category']))
        print('Describe turned off for now to spare time')
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


    def _find_unhashable_columns(self):
        """A helper method to identify columns with unhashable types."""
        unhashable_cols = []
        for col in self.df.columns:
            # We check the first non-NA value's type.
            # This is much safer than .iloc[0] as it handles empty/all-NA columns.
            non_na_series = self.df[col].dropna()
            if not non_na_series.empty:
                first_value = non_na_series.iloc[0]
                # Dictionaries and lists are the most common unhashable types in DataFrames.
                if isinstance(first_value, (dict, list)):
                    unhashable_cols.append(col)
        return unhashable_cols
    

    def check_duplicates(self):
        """
        Prints the number of duplicate rows.
        Automatically excludes unhashable columns (like those with dicts/lists)
        from the check and warns the user.
        """
        unhashable_cols = self._find_unhashable_columns()
        
        if unhashable_cols:
            print(f"\nWarning: Found unhashable columns: {unhashable_cols}.")
            print("Excluding them from the duplicate check.")
            # Check for duplicates only in the subset of hashable columns
            hashable_cols = [col for col in self.df.columns if col not in unhashable_cols]
            # It's possible all columns are unhashable
            if not hashable_cols:
                print("No hashable columns to check for duplicates.")
                return self
            num_duplicates = self.df.duplicated(subset=hashable_cols).sum()
        else:
            # If all columns are hashable, check the whole DataFrame
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

    def drop_missing_cols(self,threshold=0.95): # add specific cols func
        initial_cols = self.df.shape[1]
        missing_frac = self.df.isna().mean()
        cols_to_drop = missing_frac[missing_frac > threshold].index
        self.df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped {initial_cols - self.df.shape[1]} columns with >{threshold*100}% missing values.")
        print('Dropped cols: ', cols_to_drop.tolist())
        return self
    

    
    def map_yn_to_boolean(self, columns):
        """
        Converts columns with 'y'/'n' or 'Y'/'N' values to True/False.

        Args:
            columns (list): A list of column names to convert.
        """
        if not isinstance(columns, list):
            raise TypeError("The 'columns' argument must be a list of column names.")

        # Create the mapping dictionary
        # .str.lower() handles both 'Y' and 'y'
        yn_map = {'y': True, 'n': False}

        print(f"\nConverting 'y'/'n' to boolean for columns: {columns}")
        for col in columns:
            if col in self.df.columns:
                # First, store original non-NA count to see if we lose data
                original_non_na = self.df[col].notna().sum()
                
                # Apply the mapping
                self.df[col] = self.df[col].str.lower().map(yn_map)
                
                # See how many values were not in the map (became NA)
                unmapped_count = original_non_na - self.df[col].notna().sum()
                if unmapped_count > 0:
                    print(f"  - Note for column '{col}': {unmapped_count} values were not 'y' or 'n' and are now NA.")
            else:
                print(f"  - Warning: Column '{col}' not found in DataFrame.")
        
        # Use convert_dtypes to get a proper nullable BooleanDtype
        self.convert_data_types()
        return self
    
    def standardize_missing_values(self, include_cols=None, exclude_cols=None, placeholders=None):
        """
        Replaces string placeholders with a standard missing value indicator (NA/NaN).
        """
        if include_cols is not None and exclude_cols is not None:
            raise ValueError("Cannot specify both 'include_cols' and 'exclude_cols'.")

        if placeholders is None:
            placeholders = self.na_placeholders

        initial_nas = self.df.isna().sum().sum()

        if include_cols is not None:
            target_cols = [col for col in include_cols if col in self.df.columns]
            print(f"Applying replacement to specified columns: {target_cols}")
        elif exclude_cols is not None:
            target_cols = [col for col in self.df.columns if col not in exclude_cols]
            print(f"Applying replacement to all columns EXCEPT: {exclude_cols}")
        else:
            target_cols = self.df.columns.tolist() # Use .tolist() for clarity
            print("Applying replacement to all columns.")

        # We only act on the string columns within our target columns for efficiency
        string_cols_in_target = self.df[target_cols].select_dtypes(include=['object', 'string']).columns

        if not string_cols_in_target.empty:
            cleaned_slice = self.df[string_cols_in_target].replace(placeholders, pd.NA, regex=True)
            self.df.loc[:, string_cols_in_target] = cleaned_slice
        
        new_nas = self.df.isna().sum().sum()
        print(f"Standardized missing values. Added {new_nas - initial_nas} new NA/NaN values.")
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
        """
        Removes duplicate rows from the DataFrame.
        Automatically excludes unhashable columns from the check.
        """
        initial_rows = len(self.df)
        unhashable_cols = self._find_unhashable_columns()
        
        subset_to_check = None
        if unhashable_cols:
            print(f"Warning: Excluding unhashable columns from duplicate removal: {unhashable_cols}")
            subset_to_check = [col for col in self.df.columns if col not in unhashable_cols]
            if not subset_to_check:
                print("No hashable columns to check for duplicates. No rows dropped.")
                return self

        self.df.drop_duplicates(subset=subset_to_check, inplace=True)
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
'''

class DataFrameCleaner:
    """
    A class to encapsulate a pandas DataFrame and apply a series of
    common cleaning operations using method chaining.
    """

    def __init__(self, dataframe):
        # Create a copy to avoid setting warnings or modifying the original variable unexpectedly
        self.df = dataframe.copy()
        print("DataFrameCleaner initialized. (Working on a copy)")

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

    # --- 1. CHECKING METHODS (Inspect without modifying) ---
    def summarize(self):
            """
            Prints a comprehensive summary table combining dtypes, missing values, 
            samples, and detected placeholder garbage values.
            """
            print(f"\n--- DataFrame Summary (Shape: {self.df.shape} (rows, cols)) ---")
            print(f"\nChecking for duplicate rows...")
            self.check_duplicates()

            print(f"\nGetting dataframe overview...")
            summary = pd.DataFrame(self.df.dtypes, columns=['dtypes'])
            summary['missing#'] = self.df.isna().sum()
            summary['missing%'] = (self.df.isna().mean() * 100).round(2)
            


            unhashable_cols = self._find_unhashable_columns()
            unique_counts = pd.Series(index=self.df.columns, dtype='int64')
            hashable_cols = [c for c in self.df.columns if c not in unhashable_cols]
            if hashable_cols:
                unique_counts[hashable_cols] = self.df[hashable_cols].nunique(dropna=True)
                
            for col in unhashable_cols:
                try:
                    unique_counts[col] = self.df[col].astype(str).nunique(dropna=True)
                except Exception:
                    unique_counts[col] = 0 # Fallback if something goes really wrong

            summary['unique#'] = unique_counts
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

                if pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_string_dtype(self.df[col]):
                    matches = self._get_placeholders(col)
                    if matches:
                        found_placeholders.append(str(matches))
                    else:
                        found_placeholders.append("")
                else:
                    found_placeholders.append("")

            summary['sample_val'] = samples
            summary['placeholders'] = found_placeholders

            print(summary.sort_values('missing%', ascending=False).to_string())
            print("--------------------------------------------------\n")

            if unhashable_cols:
                print(f"\nChecking unhashable column structures...")
                self.analyze_structure_recursive()
            else:
                print('\nNo complex nested columns found.')
            return self
    
    ''' old nested analyze
    def inspect_nested_columns(self, sample_size=2000):
            """
            Analyzes unhashable columns (Lists/Dicts) to give structural insights.
            - Lists: Length stats, Total Unique items, Top occurring items.
            - Dicts: Key frequency, Key Cardinality (is it an ID or a Category?).
            """
            cols = self._find_unhashable_columns()
            if not cols:
                return self

            print(f"\n--- Nested Structure Analysis (Sample n={sample_size}) ---")

            for col in cols:
                # Get valid data only
                valid_series = self.df[col].dropna()
                if valid_series.empty:
                    continue
                
                # Check type of first item
                first_item = valid_series.iloc[0]

                # --- STRATEGY 1: LISTS ---
                if isinstance(first_item, list):
                    # 1. Length Statistics
                    lengths = valid_series.str.len()
                    
                    # 2. Content Analysis (Flatten the lists from the sample)
                    # We limit this to the sample_size to avoid exploding 1M rows
                    sample = valid_series.sample(n=min(len(valid_series), sample_size), random_state=42)
                    
                    # Flatten: list of lists -> single list
                    all_items = [item for sublist in sample for item in sublist]
                    
                    # If list contains unhashable items (like dicts inside lists), we can't count uniques easily
                    try:
                        unique_count = len(set(all_items))
                        top_items = Counter(all_items).most_common(3)
                        content_summary = f"Total Unique: ~{unique_count} (in sample)"
                        top_summary = f"Top items: {top_items}"
                    except TypeError:
                        # Fallback if list contains dicts
                        content_summary = "Content: Complex Objects (Dicts/Lists)"
                        top_summary = ""

                    print(f"Column '{col}' [List]:")
                    print(f"  - Lengths: Min={lengths.min()}, Max={lengths.max()}, Avg={lengths.mean():.2f}")
                    print(f"  - {content_summary}")
                    if top_summary:
                        print(f"  - {top_summary}")

                # --- STRATEGY 2: DICTIONARIES ---
                elif isinstance(first_item, dict):
                    actual_n = min(len(valid_series), sample_size)
                    sample = valid_series.sample(n=actual_n, random_state=42)
                    
                    all_keys = []
                    # Storage for values of the most common keys to check cardinality
                    key_values_map = {} 
                    
                    for d in sample:
                        keys = list(d.keys())
                        all_keys.extend(keys)
                        # Store values for analysis later
                        for k, v in d.items():
                            if k not in key_values_map:
                                key_values_map[k] = []
                            key_values_map[k].append(str(v)) # Convert to str to handle unhashables

                    # Key Frequency
                    key_counts = Counter(all_keys)
                    top_keys = [k for k, _ in key_counts.most_common(5)]
                    
                    print(f"Column '{col}' [Dict]:")
                    print(f"  - Top Keys found: {top_keys}")
                    
                    # Analyze Cardinality of Top Keys
                    print(f"  - Key Insights (Sample):")
                    for k in top_keys:
                        vals = key_values_map[k]
                        n_unique = len(set(vals))
                        example = vals[0] if vals else ""
                        if len(example) > 20: example = example[:20] + "..."
                        
                        # Heuristic: Low unique count = Categorical, High = ID/Text
                        val_type = "ID/Text" if n_unique > (actual_n * 0.8) else "Categorical"
                        if n_unique == 1: val_type = "Constant"
                        
                        print(f"    * '{k}': {n_unique} unique values ({val_type}). Ex: {example}")

                print("") # spacer
            
            print("--------------------------------------------------\n")
            return self
    '''

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

    def _analyze_arrow_data(self, array, indent=0):
        """
        Helper: Recursively walks through PyArrow DATA arrays.
        - If List: Flattens it and recurses.
        - If Struct: Extracts fields and recurses.
        - If Atomic: Calculates unique counts.
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        
        prefix = " " * indent
        arrow_type = array.type
        
        # CASE 1: LIST (e.g., [ [a,b], [c] ])
        if pa.types.is_list(arrow_type):
            # "List<string>"
            print(f"{prefix}- List of:")
            
            # .flatten() unwraps one level of nesting: [a, b, c]
            # This allows us to analyze the CONTENTS of the lists in aggregate
            flattened = array.flatten()
            self._analyze_arrow_data(flattened, indent + 2)
            
        # CASE 2: STRUCT / DICT (e.g., {x: 1, y: 2})
        elif pa.types.is_struct(arrow_type):
            print(f"{prefix}- Object (Dict) with keys:")
            
            for field in arrow_type:
                # .field(name) extracts the column for just this key
                child_array = array.field(field.name)
                
                print(f"{prefix}  * {field.name}: ", end="")
                
                # If child is complex, new line and recurse
                if pa.types.is_list(field.type) or pa.types.is_struct(field.type):
                    print("") 
                    self._analyze_arrow_data(child_array, indent + 6)
                # If child is simple, print stats on same line
                else:
                    self._analyze_arrow_data(child_array, indent=0)

        # CASE 3: ATOMIC (String, Int, Float, etc.) -> CALCULATE STATS
        else:
            # This is where the magic happens. We have the raw data array here.
            # We can count unique values instantly in C++.
            
            total_count = len(array)
            if total_count == 0:
                print(f"{arrow_type} (Empty)")
                return

            # pc.unique is very fast
            unique_vals = pc.unique(array)
            n_unique = len(unique_vals)
            
            # Heuristic for ID vs Category
            # If >90% of values are unique, it's likely an ID or free text
            # If <10% are unique, it's likely a category
            ratio = n_unique / total_count
            if n_unique == 1:
                cat_label = "CONSTANT"
            elif ratio > 0.9:
                cat_label = "ID/TEXT"
            elif ratio < 0.1 or n_unique < 50:
                cat_label = "CATEGORY"
            else:
                cat_label = "DENSE"

            # Print details
            # If indent is 0, we are inline with a Struct key, so don't print prefix
            current_prefix = prefix if indent > 0 else ""
            print(f"{current_prefix}{arrow_type} | {n_unique} unique ({cat_label})")


    def get_samples(self, columns=None, number=5):
        valid = [c for c in columns or [] if c in self.df.columns]
        if (invalid := set(columns or []) - set(valid)): print(f"Invalid: {invalid}")
        for col in valid:
            s = self.df[col].dropna()
            print(f"Column: {col}", *s.sample(min(number, len(s))), "-" * 30, sep="\n")
            
        return self

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

    def _find_unhashable_columns(self):
        """Helper: Find columns containing lists or dicts."""
        unhashable_cols = []
        # Only check object columns, numeric/bool are always hashable
        for col in self.df.select_dtypes(include=['object']):
            # Check first non-na value
            first_valid = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None
            if isinstance(first_valid, (dict, list, set)):
                unhashable_cols.append(col)
        return unhashable_cols

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
    
    def _get_placeholders(self, col):
        """Helper: Returns a list of unique values matching the garbage regex."""
        # Convert to string for regex check
        col_as_str = self.df[col].astype(str)
        
        # Check pattern match
        matches_mask = col_as_str.str.match(self.na_placeholders, na=False)
        
        # CRITICAL FIX: Exclude values that are ALREADY real NaN/None
        # We only want strings that look like garbage, not actual missing values
        is_real_nan = self.df[col].isna()
        final_mask = matches_mask & (~is_real_nan)

        if final_mask.any():
            return self.df.loc[final_mask, col].unique().tolist()
        return None

    # --- 2. CLEANING METHODS (Modify the DataFrame) ---
    def unify_na_values(self):
        """
        Reverts modern pd.NA (and None) back to np.nan for consistency 
        and compatibility with external libraries (like sklearn/plotting).
        """
        # Replace pd.NA with np.nan
        # Note: This might cast Int64 columns back to float64, which is usually expected in data science.
        self.df = self.df.replace({pd.NA: np.nan, None: np.nan})
        print("Unified missing values to np.nan.")
        return self

    def convert_to_datetime(self, columns):
        """
        Forces columns to datetime objects. Coerces errors to NaT.
        """
        for col in columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"Converted '{col}' to datetime.")
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

    def clean_column_names(self):
        """
        Vectorized cleanup of column names to snake_case.
        """
        # Use vector string methods for speed
        self.df.columns = (self.df.columns.astype(str)
                           .str.strip()
                           .str.lower()
                           .str.replace(r'\s+', '_', regex=True)  # spaces to underscore
                           .str.replace(r'[^a-z0-9_]', '', regex=True)) # remove special chars
        print("Cleaned column names to snake_case.")
        return self

    def drop_missing_cols(self, threshold=0.95, exclude=[]):
        """Drops columns that are missing more than `threshold` percent of data."""
        mask = self.df.isna().mean() > threshold
        cols_to_drop = [col for col in self.df.columns[mask] if col not in exclude]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped columns (> {threshold*100}% missing): {cols_to_drop}")
        return self

    def standardize_missing_values(self, include_cols=None, exclude_cols=None, placeholders=None):
        """Replaces placeholder strings with np.nan."""
        if placeholders is None:
            placeholders = self.na_placeholders

        # Determine target columns
        if include_cols:
            target_cols = include_cols
        elif exclude_cols:
            target_cols = [c for c in self.df.columns if c not in exclude_cols]
        else:
            target_cols = self.df.columns

        # Only apply to object/string columns within the target set
        # Regex replace on numeric columns is usually safe but inefficient
        cols_to_clean = self.df[target_cols].select_dtypes(include=['object', 'string']).columns

        if not cols_to_clean.empty:
            initial_nas = self.df[cols_to_clean].isna().sum().sum()
            
            # Apply regex replace
            self.df[cols_to_clean] = self.df[cols_to_clean].replace(
                to_replace=placeholders, 
                value=np.nan, 
                regex=True
            )
            
            new_nas = self.df[cols_to_clean].isna().sum().sum()
            diff = new_nas - initial_nas
            if diff > 0:
                print(f"Standardized {diff} values to NaN.")
        
        return self

    def map_yn_to_boolean(self, columns):
        """Maps 'yes/no' style columns to boolean True/False."""
        yn_map = {'y': True, 'n': False, 'yes': True, 'no': False, 't': True, 'f': False}
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().map(yn_map)
        
        print(f"Mapped {columns} to boolean.")
        return self

    def drop_duplicates(self):
        """Removes duplicate rows, ignoring unhashable columns."""
        unhashable = self._find_unhashable_columns()
        subset = [c for c in self.df.columns if c not in unhashable] if unhashable else None
        
        initial = len(self.df)
        self.df.drop_duplicates(subset=subset, inplace=True)
        dropped = initial - len(self.df)
        
        if dropped:
            print(f"Dropped {dropped} duplicate rows.")
        return self

    def strip_whitespace(self):
        """Strips whitespace from all string columns."""
        # Only select object columns
        str_cols = self.df.select_dtypes(include=['object', 'string']).columns
        # Vectorized strip
        self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.strip())
        print("Stripped whitespace from string columns.")
        return self

    def convert_data_types(self):
        """Uses pandas convert_dtypes to infer best types."""
        self.df = self.df.convert_dtypes()
        print("Converted column types.")
        print(self.df.dtypes)
        return self
    
    def reset_index(self):
        self.df.reset_index(drop=True, inplace=True)
        return self

    # --- 3. PIPELINE ---

    def run_all(self):
        """Runs the standard cleaning pipeline."""
        return (self
                .clean_column_names()
                .standardize_missing_values()
                .strip_whitespace()
                .drop_duplicates()
                .convert_data_types()
                .unify_na_values()
                .reset_index()
                .summarize())

    def get_df(self):
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