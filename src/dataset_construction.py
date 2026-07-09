import os
import json
import ast
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.json as pj
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, List
import hashlib
import uuid
from scrape import HBOScraper, ScriptiebankScraper
import requests
import argparse
from langdetect import detect, LangDetectException
import string
from tqdm import tqdm
import numpy as np

#targets
BASE_DIR = Path(__file__).resolve().parent.parent

SOURCES_CONFIG = {
    'UG': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'UG' / 'publications.json',
        'clean_file': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_cleaned.parquet',
        'selected_file': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_selected.parquet',
    },
    'HBO': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'HBO' / 'HBO_metadata.csv',
        'clean_file': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_cleaned.parquet',
        'selected_file': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_selected.parquet',
    },
    'SB': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'SB' / 'SB_metadata.csv',
        'clean_file': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_cleaned.parquet',
        'selected_file': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_selected.parquet',
    }
}


# -------------------------
# download

def download_raw_data(source_name: str, output_path: Path):
    """
    Downloads raw data from scrapers or remote endpoints.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if source_name == 'SB':
        print("[SB] Starting Scriptiebank scraper...")
        scraper = ScriptiebankScraper(base_folder=BASE_DIR / 'data')
        scraper.run(gather_metadata=True, gather_urls=True, download_files=False)
        
    elif source_name == 'HBO':
        print("[HBO] Starting HBO scraper...")
        scraper = HBOScraper(base_folder=BASE_DIR / 'data')
        scraper.run(gather_metadata=True, gather_urls=True, download_files=False)
        
    elif source_name == 'UG':
        datadump_url = 'https://biblio.ugent.be/exports/publications.json'
        print(f"[UG] Downloading UGent datadump from {datadump_url}...")
        response = requests.get(datadump_url)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        print("[UG] Download complete.")


# ------------------------
# cleaning
"""
Loads a csv, parquet or json file file
unifies all null values (NaN, None..) string sentinel values like 'nan', '-', '/'
detects semantically logical dtypes, detects nested python structures (list, dicts) in strings
saves to either csv or parquet
"""
#TODO clean individual abstr


# Helper to prevent semantic false positives on codes and identifiers
def is_identifier_name(name: str) -> bool:
    if not name:
        return False
    name_lower = name.lower()
    identifiers = {
        '_id', #'code', 'num', 'phone', 'zip', 'postal', 'serial', 'ssn', 
        #'ein', 'vabb', 'wos', 'arxiv', 'pubmed', 'esci', 'issn', 'isbn', 
        #'doi', 'ugent_id', 'biblio_id', 'handle', 'sha256'
    }
    return any(keyword in name_lower for keyword in identifiers)

def detect_logical_array(
    array: pa.Array,
    field_name: str = None, # Propagated down the recursion stack for semantic context
    infer_dates: bool = True,
    infer_booleans: bool = True,
    infer_numbers: bool = True,
    category_ratio_threshold: float = 0.15,
    category_max_unique: int = 5_000,
    max_category_string_len: int = 60
) -> pa.Array:
    """
    Recursively analyzes and infers the most logical semantic datatype of a PyArrow Array,
    natively cleaning NaNs and string placeholder sentinels (like "NaN", "null") on the fly.
    """
    arrow_type = array.type
    if len(array) == 0 or array.null_count == len(array):
        return array

    # Handle List Types
    if pa.types.is_list(arrow_type):
        flattened = array.flatten()
        optimized_values = detect_logical_array(
            flattened, field_name, infer_dates, infer_booleans, infer_numbers,
            category_ratio_threshold, category_max_unique, max_category_string_len
        )
        return pa.ListArray.from_arrays(offsets=array.offsets, values=optimized_values, mask=array.is_null())
        
    elif pa.types.is_large_list(arrow_type):
        flattened = array.flatten()
        optimized_values = detect_logical_array(
            flattened, field_name, infer_dates, infer_booleans, infer_numbers,
            category_ratio_threshold, category_max_unique, max_category_string_len
        )
        return pa.LargeListArray.from_arrays(offsets=array.offsets, values=optimized_values, mask=array.is_null())
        
    elif pa.types.is_fixed_size_list(arrow_type):
        flattened = array.flatten()
        optimized_values = detect_logical_array(
            flattened, field_name, infer_dates, infer_booleans, infer_numbers,
            category_ratio_threshold, category_max_unique, max_category_string_len
        )
        return pa.FixedSizeListArray.from_arrays(values=optimized_values, list_size=arrow_type.list_size, mask=array.is_null())

    # Handle Structs (Nested Dicts)
    elif pa.types.is_struct(arrow_type):
        new_arrays = []
        new_fields = []
        for i in range(arrow_type.num_fields):
            field = arrow_type.field(i)
            child_array = array.field(field.name)
            optimized_child = detect_logical_array(
                child_array, field.name, infer_dates, infer_booleans, infer_numbers,
                category_ratio_threshold, category_max_unique, max_category_string_len
            )
            new_arrays.append(optimized_child)
            new_fields.append(pa.field(field.name, optimized_child.type, nullable=field.nullable))
        return pa.StructArray.from_arrays(new_arrays, fields=new_fields, mask=array.is_null())

    # Standardize Dictionaries
    elif pa.types.is_dictionary(arrow_type):
        optimized_values = detect_logical_array(
            array.dictionary, field_name, infer_dates, infer_booleans, infer_numbers,
            category_ratio_threshold, category_max_unique, max_category_string_len
        )
        n_unique = len(optimized_values)
        index_type = pa.int16() if n_unique <= 32767 else pa.int32()
        return pa.DictionaryArray.from_arrays(pc.cast(array.indices, index_type), optimized_values)

    # -------------------------------------------------------------
    # SEMANTIC INFERENCE: INTEGER ARRAYS
    # -------------------------------------------------------------
    elif pa.types.is_integer(arrow_type):
        # A. Infer Unix Timestamps (Safeguard: Never run on ID or Code fields)
        if infer_dates and not is_identifier_name(field_name):
            try:
                min_max = pc.min_max(array).as_py()
                min_val, max_val = min_max['min'], min_max['max']
                if min_val is not None and max_val is not None:
                    if 315532800 <= min_val and max_val <= 2524608000:
                        return pc.cast(array, pa.timestamp('s'))
                    elif 315532800000 <= min_val and max_val <= 2524608000000:
                        return pc.cast(array, pa.timestamp('ms'))
            except Exception:
                pass

        # B. Infer Booleans flags masquerading as 0 and 1
        if infer_booleans:
            try:
                unique_vals = {v.as_py() for v in pc.unique(array) if v.is_valid}
                if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
                    return pc.cast(array, pa.bool_())
            except Exception:
                pass

        return pc.cast(array, pa.int64())

    # -------------------------------------------------------------
    # SEMANTIC INFERENCE: FLOATING ARRAYS (Now cleans NaNs recursively)
    # -------------------------------------------------------------
    elif pa.types.is_floating(arrow_type):
        try:
            # 1. Clean NaNs by turning them into actual nulls (integers cannot hold NaNs)
            is_nan_mask = pc.fill_null(pc.is_nan(array), False)
            if pc.any(is_nan_mask).as_py():
                null_scalar = pa.scalar(None, type=arrow_type)
                array = pc.if_else(is_nan_mask, null_scalar, array)
        except Exception:
            pass

        try:
            # 2. Check if all non-null numbers are equivalent to integers (e.g., 801001.0 == 801001)
            floor_array = pc.floor(array)
            is_integer_valued = pc.equal(array, floor_array)
            is_integer_valued = pc.fill_null(is_integer_valued, True)
            
            if pc.all(is_integer_valued).as_py():
                # 3. Cast to standard int64 and recursively let the integer block handle it
                int_array = pc.cast(array, pa.int64())
                return detect_logical_array(
                    int_array, field_name, infer_dates, infer_booleans, infer_numbers,
                    category_ratio_threshold, category_max_unique, max_category_string_len
                )
        except Exception:
            pass

        # Default to standard 64-bit floats
        return pc.cast(array, pa.float64())

    # -------------------------------------------------------------
    # SEMANTIC INFERENCE: STRING ARRAYS
    # -------------------------------------------------------------
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        try:
            sentinels = [
                "nan", "none", "null", "-", "--", "---", "_", "unknown", "missing", "niet beschikbaar"
                "<none>", "n/a", "na", "not reported", 
                "/",   # Safely matches single forward-slash placeholders
                "\\"   # Safely matches single backslash placeholders (escaped as double-backslash)
            ] #TODO can be expanded
            lower_array = pc.ascii_lower(array)
            is_sentinel_mask = pc.is_in(lower_array, value_set=pa.array(sentinels))
            
            if pc.any(is_sentinel_mask).as_py():
                null_scalar = pa.scalar(None, type=arrow_type)
                array = pc.if_else(is_sentinel_mask, null_scalar, array)
        except Exception:
            pass

        non_null_count = len(array) - array.null_count
        if non_null_count == 0:
            return array

        # A. Infer Booleans from string flags
        if infer_booleans:
            try:
                unique_vals = {str(v.as_py()).strip().lower() for v in pc.unique(array) if v.is_valid}
                if unique_vals.issubset({"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0","1.0","0.0"}) and len(unique_vals) > 0:
                    bool_list = []
                    for val in array:
                        py_val = val.as_py()
                        if py_val is None:
                            bool_list.append(None)
                        else:
                            clean_str = str(py_val).strip().lower()
                            bool_list.append(clean_str in ("true", "t", "yes", "y", "1"))
                    return pa.array(bool_list, type=pa.bool_())
            except Exception:
                pass

        # B. Infer Numeric Values (Digits & Decimals)
        if infer_numbers:
            try:
                is_digit_mask = pc.ascii_is_decimal(array)
                is_digit_mask = pc.fill_null(is_digit_mask, True)
                if pc.all(is_digit_mask).as_py():
                    # Protect postal codes, serials, and leading-zero IDs
                    has_leading_zero = pc.any(pc.and_(pc.starts_with(array, "0"), pc.not_equal(array, "0"))).as_py()
                    if not has_leading_zero:
                        int_array = pc.cast(array, pa.int64())
                        return detect_logical_array(
                            int_array, field_name, infer_dates, infer_booleans, infer_numbers,
                            category_ratio_threshold, category_max_unique, max_category_string_len
                        )
            except Exception:
                pass

            try:
                float_arr = pc.cast(array, pa.float64())
                return detect_logical_array(
                    float_arr, field_name, infer_dates, infer_booleans, infer_numbers,
                    category_ratio_threshold, category_max_unique, max_category_string_len
                )
            except Exception:
                pass

        # C. Infer Date & Time formats
        if infer_dates:
            try:
                has_date_separators = pc.any(pc.or_(pc.match_substring(array, "-"), pc.match_substring(array, "/"))).as_py()
                if has_date_separators:
                    try:
                        return pc.cast(array, pa.date32())
                    except Exception:
                        return pc.cast(array, pa.timestamp('us'))
            except Exception:
                pass

        # D. Infer Logical Categories (Dictionary Encoding)
        try:
            n_unique = pc.count_distinct(array).as_py()
            avg_len = pc.mean(pc.string_length(array)).as_py()
        except Exception:
            return array

        ratio = n_unique / non_null_count
        if ratio < category_ratio_threshold and n_unique < category_max_unique and avg_len <= max_category_string_len:
            index_type = pa.int16() if n_unique <= 32767 else pa.int32()
            try:
                return pc.cast(array.dictionary_encode(), pa.dictionary(index_type, pa.string()))
            except Exception:
                encoded = array.dictionary_encode()
                return pa.DictionaryArray.from_arrays(pc.cast(encoded.indices, index_type), encoded.dictionary)

    return array

def detect_logical_table(
    table: pa.Table, 
    infer_dates: bool = True,
    infer_booleans: bool = True,
    infer_numbers: bool = True,
    category_ratio_threshold: float = 0.15,
    category_max_unique: int = 5_000,
    max_category_string_len: int = 60
) -> pa.Table:
    optimized_columns = []
    optimized_fields = []
    for i in range(table.num_columns):
        col_name = table.column_names[i]
        col_array = table.column(i).combine_chunks()
        
        # Initiate the recursion passing the top-level column name
        optimized_array = detect_logical_array(
            col_array,
            field_name=col_name,
            infer_dates=infer_dates,
            infer_booleans=infer_booleans,
            infer_numbers=infer_numbers,
            category_ratio_threshold=category_ratio_threshold,
            category_max_unique=category_max_unique,
            max_category_string_len=max_category_string_len
        )
        optimized_columns.append(optimized_array)
        optimized_fields.append(pa.field(col_name, optimized_array.type, nullable=table.schema.field(i).nullable))
    return pa.Table.from_arrays(optimized_columns, schema=pa.schema(optimized_fields))


# =====================================================================
# 2. FILE DETECTOR & SERIALIZED PARSER
# =====================================================================
def detect_json_columns(table: pa.Table, sample_size: int = 100, success_threshold: float = 0.9) -> list[str]:
    """
    Scans string columns and flags those containing serialized JSON/Python literal dicts or lists.
    """
    detected_cols = []
    for i in range(table.num_columns):
        col_name = table.column_names[i]
        col_type = table.schema.field(i).type
        if not (pa.types.is_string(col_type) or pa.types.is_large_string(col_type)):
            continue
            
        sample_slice = table.column(i).slice(0, min(sample_size, len(table))).combine_chunks()
        sample_values = [v.as_py() for v in sample_slice if v.is_valid]
        if not sample_values:
            continue
            
        valid_json_count = 0
        for val in sample_values:
            if not isinstance(val, str):
                continue
            stripped = val.strip()
            if (stripped.startswith('{') and stripped.endswith('}')) or (stripped.startswith('[') and stripped.endswith(']')):
                parsed_as_complex = False
                try:
                    if isinstance(json.loads(stripped), (dict, list)):
                        parsed_as_complex = True
                except Exception:
                    try:
                        if isinstance(ast.literal_eval(stripped), (dict, list)):
                            parsed_as_complex = True
                    except Exception:
                        pass
                if parsed_as_complex:
                    valid_json_count += 1
                    
        if (valid_json_count / len(sample_values)) >= success_threshold:
            detected_cols.append(col_name)
    return detected_cols

def parse_serialized_columns(table: pa.Table, json_columns: list[str]) -> pa.Table:
    """
    Converts identified serialized string columns into native PyArrow Structs or Lists.
    """
    new_columns = []
    new_fields = []
    for i in range(table.num_columns):
        col_name = table.column_names[i]
        col_data = table.column(i)
        
        if col_name in json_columns:
            flat_array = col_data.combine_chunks()
            parsed_list = []
            for item in flat_array:
                val = item.as_py()
                if val is None or not isinstance(val, str) or val.strip() == "":
                    parsed_list.append(None)
                    continue
                try:
                    parsed_list.append(json.loads(val))
                except Exception:
                    try:
                        parsed_list.append(ast.literal_eval(val))
                    except Exception:
                        parsed_list.append(val)
            
            nested_array = pa.array(parsed_list)
            new_columns.append(nested_array)
            new_fields.append(pa.field(col_name, nested_array.type, nullable=table.schema.field(i).nullable))
        else:
            new_columns.append(col_data)
            new_fields.append(table.schema.field(i))
    return pa.Table.from_arrays(new_columns, schema=pa.schema(new_fields))


# =====================================================================
# 3. UNIFIED FILE RUNNER
# =====================================================================
def optimize_file_table(
    input: any,
    format: str = None, 
    infer_dates: bool = True,
    infer_booleans: bool = True,
    infer_numbers: bool = True,
    category_ratio_threshold: float = 0.15,
    category_max_unique: int = 5_000,
    max_category_string_len: int = 60,
    remove_duplicates: any = None,
    save: str = None
) -> pa.Table:
    """
    Accepts a filepath, an in-memory PyArrow Table, or a Pandas DataFrame,
    automatically detects types, applies deduplication, and optionally saves the output.
    """
    
    # -------------------------------------------------------------
    # Ingestion Routing
    # -------------------------------------------------------------
    
    # 1. Input is already a PyArrow Table
    if isinstance(input, pa.Table):
        table = input

    # 2. Input is a Pandas DataFrame
    elif isinstance(input, pd.DataFrame):
        # Convert to PyArrow Table. 
        table = pa.Table.from_pandas(input)

    # 3. Input is a File Path (string or Path object)
    else:
        input = str(input)
        if not os.path.exists(input):
            raise FileNotFoundError(f"The file path '{input}' does not exist.")

        if format is None:
            ext = os.path.splitext(input)[1].lower()
            if ext in ('.csv', '.txt'):
                format = 'csv'
            elif ext in ('.json', '.jsonl', '.ndjson'):
                format = 'json'
            elif ext in ('.parquet', '.pq'):
                format = 'parquet'
            else:
                raise ValueError(f"Could not auto-detect format for: '{input}'. Please specify 'format' explicitly.")

        # Read file into PyArrow Table
        if format == 'csv':
            table = pv.read_csv(input)
        elif format == 'json':
            table = pj.read_json(input)
        elif format == 'parquet':
            table = pq.read_table(input)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'csv', 'json', or 'parquet'.")
        
    # B. Auto-detect serialized string columns containing dicts/lists
    json_cols = detect_json_columns(table, sample_size=100)
    if json_cols:
        print(f"[{format.upper()}] Auto-detected serialized nested columns: {json_cols}")
        table = parse_serialized_columns(table, json_cols)

    # C. Run recursive logical/semantic type detection
    optimized_table = detect_logical_table(
        table,
        infer_dates=infer_dates,
        infer_booleans=infer_booleans,
        infer_numbers=infer_numbers,
        category_ratio_threshold=category_ratio_threshold,
        category_max_unique=category_max_unique,
        max_category_string_len=max_category_string_len
    )

    if remove_duplicates not in (False, None) and len(optimized_table) > 0:
        # Filter out nested columns (lists/structs) from the potential grouping keys
        flat_columns = [
            name for name, field in zip(optimized_table.column_names, optimized_table.schema)
            if not (pa.types.is_list(field.type) or 
                    pa.types.is_large_list(field.type) or 
                    pa.types.is_fixed_size_list(field.type) or 
                    pa.types.is_struct(field.type))
        ]

        dedup_keys = []

        # State 1: Deduplicate on ALL flat columns
        if remove_duplicates is True or remove_duplicates == "all":
            dedup_keys = flat_columns
            print(f"Deduplicating on all flat columns ({len(dedup_keys)} columns)...")

        # State 2: Deduplicate on all flat columns EXCEPT specified ones
        elif isinstance(remove_duplicates, dict) and "exclude" in remove_duplicates:
            exclude_list = remove_duplicates["exclude"]
            if isinstance(exclude_list, str):
                exclude_list = [exclude_list]
            dedup_keys = [col for col in flat_columns if col not in exclude_list]
            print(f"Deduplicating on all flat columns except {exclude_list} ({len(dedup_keys)} columns)...")

        # State 3: Deduplicate on a specific subset of columns
        elif isinstance(remove_duplicates, (list, tuple, set)):
            for key in remove_duplicates:
                if key not in optimized_table.column_names:
                    raise KeyError(f"Column '{key}' specified for deduplication does not exist.")
                if key not in flat_columns:
                    raise ValueError(
                        f"Cannot deduplicate on column '{key}' because it contains a nested structure (list/struct). "
                        "PyArrow hash-grouping only supports flat primitive columns."
                    )
            dedup_keys = list(remove_duplicates)
            print(f"Deduplicating on specified column subset: {dedup_keys}...")
        else:
            raise ValueError(
                "Invalid 'remove_duplicates' value. Expected True, 'all', a list of columns, "
                "or {'exclude': [...] }."
            )

        if dedup_keys:
            # Create monotonic row indices
            row_indices = pa.array(np.arange(len(optimized_table)), type=pa.int64())
            temp_table = optimized_table.append_column("__row_index__", row_indices)

            # Group and select minimum row index (first occurrence)
            grouped = temp_table.group_by(dedup_keys).aggregate([("__row_index__", "min")])
            unique_indices = grouped.column("__row_index___min")

            # Sort indices to preserve original table order
            sorted_positions = pc.sort_indices(unique_indices)
            ordered_unique_indices = pc.take(unique_indices, sorted_positions)

            # Filter table
            pre_count = len(optimized_table)
            optimized_table = optimized_table.take(ordered_unique_indices)
            print(f"Deduplication finished. Retained {len(optimized_table)} of {pre_count} rows.")

    if post_process is not None:
        if not callable(post_process):
            raise TypeError("post_process must be a callable fuction")
        optimized_table = post_process(optimized_table)

    # D. Handle optional saving
    if save is not None:
        save_path = str(save)
        save_ext = os.path.splitext(save_path)[1].lower()

        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(save_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if save_ext in ('.parquet', '.pq'):
            pq.write_table(optimized_table, save_path)
            print(f"Successfully saved optimized table to Parquet: {save_path}")
            
        elif save_ext == '.csv':
            df = optimized_table.to_pandas()
            df.to_csv(save_path, index=False)
            print(f"Successfully saved optimized table to CSV (via Pandas text-serialization): {save_path}")
            
        else:
            raise ValueError(
                f"Unsupported save format '{save_ext}'. Please specify a path ending with .csv, .parquet, or .pq."
            )

    return optimized_table

# -------------------------------
#Post processing cleaning: merge cols, clean abstracts
#post processing helpers
def extract_sb_homepage_text(val) -> Optional[str]:
    """
    Parses SB 'text_homepage' from either a JSON string, dictionary, or raw string,
    unifying list-based paragraph elements into a single cohesive block.
    """
    if pd.isna(val) or not val:
        return None
        
    # Attempt to decode string representation of dict
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('{') and val.endswith('}'):
            try:
                val = json.loads(val)
            except Exception:
                pass
                
    if isinstance(val, dict):
        all_paragraphs = []
        for k, v in val.items():
            if isinstance(v, list):
                all_paragraphs.extend([str(p).strip() for p in v if p])
            elif isinstance(v, str):
                all_paragraphs.append(v.strip())
        return " ".join(all_paragraphs) if all_paragraphs else None
        
    return str(val)

def _generate_robust_id(source: str, row: pd.Series) -> str:
    """
    Generates a unique and deterministic ID using MD5 hashing of 
    available content (abstract, title, or year) combined with a source prefix.
    """
    content_parts = []
    
    # Prioritize standardized abstract text
    for col in ['text_dut', 'abstract', 'abstract_full']:
        val = row.get(col)
        if val is not None and str(val).strip():
            content_parts.append(str(val))
            break
            
    # Include title for further uniqueness
    title_val = row.get('title')
    if title_val is not None and str(title_val).strip():
        content_parts.append(str(title_val))
        
    # Include year
    year_val = row.get('year')
    if year_val is not None:
        content_parts.append(str(year_val))
        
    combined_content = "|".join(content_parts)
    
    # Fallback to random identifier if identifying content is entirely missing
    if not combined_content.strip():
        combined_content = str(uuid.uuid4())
        
    content_hash = hashlib.md5(combined_content.encode('utf-8', errors='ignore')).hexdigest()
    return f"{source}_{content_hash[:16]}"

def _parse_semicolon_keywords(val) -> list:
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        return _parse_list_keywords(val)
    if pd.isna(val) or not isinstance(val, str):
        return []
    return [k.strip() for k in val.split(';') if k.strip()]


def _parse_list_keywords(val) -> list:
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        return [str(item).strip() for item in val if pd.notna(item) and str(item).strip()]
    if pd.isna(val):
        return []
    if isinstance(val, str):
        return [val.strip()]
    return []

#unified post processing functions
#TODO adapt from pandas to pyarrow or do as seperate step
def pp_ug():
    pass

def pp_hbo():
    pass

def pp_sb():
    pass
'''
    if source_name == 'UG':
        #subset for deduplicating
        subset = []
        excl = []

        protected_values = {
            'volume': [99, '99', 999, '999', 9999, '9999'],
            'issue': [99, '99', '999', 999, '9999', 9999]
        }

    # 2. Source override: Scriptiebank specific cleans (Float year to Int)
    if source_name == 'SB':
        #subset for deduplicating
        subset = []
        excl = []

        print("[SB] Casting float 'year' column to Integer type...")
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').round().astype('Int64')

    # 3. Source override: column harmonization & title merger (HBO only)
    if source_name == 'HBO':   
        #subset for deduplicating
        subset = []
        excl = []

        print("[HBO] Harmonizing columns: merging 'jaar' -> 'year' and 'partners' -> 'partner'...")
        # Merge 'jaar' into 'year'
        if 'jaar' in df.columns:
            if 'year' in df.columns:
                df['year'] = df['year'].fillna(df['jaar'])
            else:
                df['year'] = df['jaar']
            df = df.drop(columns=['jaar'])
            
        # Merge 'partners' into 'partner'
        if 'partners' in df.columns:
            if 'partner' in df.columns:
                df['partner'] = df['partner'].fillna(df['partners'])
            else:
                df['partner'] = df['partners']
            df = df.drop(columns=['partners'])
            
        # Combine 'title' and 'subtitle'
        if 'title' in df.columns and 'subtitle' in df.columns:
            print("[HBO] Merging 'title' and 'subtitle' columns...")
            df['title'] = df.apply(
                lambda r: f"{r['title']}: {r['subtitle']}" 
                if pd.notna(r['title']) and pd.notna(r['subtitle']) and str(r['subtitle']).strip()
                else r['title'], 
                axis=1
            )
            df = df.drop(columns=['subtitle'])
'''
# -------------------------------
#selection and adding cols: text_dut, sent_dut, _id, source #TODO check others
def clean_abstract(
    abstract: str,
    min_char_length: int = 100,
    tokenizer_lang: str = 'dutch',
    detect_lang_tag: str = 'nl',
    heading_words: Optional[List[str]] = None,
    logger = None
) -> Tuple[str, List[str]]:
    """
    Cleans and filters abstract text. Returns a tuple containing:
    (joined_clean_string, list_of_clean_sentences)
    """
    if heading_words is None:
        heading_words = [
            "achtergrond", "inleiding", "doelstelling", "methode", "methoden", 
            "resultaat", "resultaten", "conclusie", "conclusies", "discussie", 
            "aanbeveling", "aanbevelingen", "samenvatting", "abstract", 
            "trefwoorden", "kernwoorden"
        ]
        
    headings_list = []
    for h in heading_words:
        headings_list.extend([h.lower(), h.capitalize(), h.upper()])
    headings_pattern = '|'.join(set(headings_list))

    def _strip_layout_headers(sent: str) -> tuple[str, Optional[str]]:
        orig = sent
        sent_cleaned = re.sub(r'[*_]{1,2}', '', orig).strip()
        
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})([A-Z])', r'\1', sent_cleaned)
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})[\s]*[:.-]+[\s]*', '', sent_cleaned)
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})\s+([A-Z])', r'\1', sent_cleaned)
        
        if re.match(rf'^(?:{headings_pattern})$', sent_cleaned):
            sent_cleaned = ""

        if sent_cleaned != orig:
            if not sent_cleaned:
                removed = orig
            else:
                idx = orig.find(sent_cleaned)
                if idx != -1:
                    removed = orig[:idx]
                else:
                    removed = f"'{orig}' -> '{sent_cleaned}'"
            return sent_cleaned, removed
            
        return orig, None

    dutch_abstract = ""
    dutch_sentences = []
    
    if isinstance(abstract, str) and len(abstract) >= min_char_length and abstract.strip():
        abstract = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', abstract)
        
        raw_sentences = nltk.sent_tokenize(abstract, language=tokenizer_lang)
        cleaned_sentences = []
        
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            cleaned_sent, removed = _strip_layout_headers(sent)
            
            if removed and logger:
                logger.debug(f"Stripped layout header: {repr(removed)}")
            
            if not cleaned_sent:
                continue
            
            sent = cleaned_sent

            should_merge = False
            if cleaned_sentences:
                if not re.match(r'^[A-Z]', sent):
                    should_merge = True
                elif len(sent) >= 2 and sent[1] in string.punctuation:
                    should_merge = True
            
            if should_merge:
                cleaned_sentences[-1] = cleaned_sentences[-1] + ' ' + sent
            else:
                cleaned_sentences.append(sent)
        
        for sent in cleaned_sentences:
            try:
                if detect(sent) == detect_lang_tag:
                    dutch_sentences.append(sent)
            except LangDetectException:
                continue
                
        if dutch_sentences:
            dutch_abstract = ' '.join(dutch_sentences)
            
    return dutch_abstract, dutch_sentences


def select_and_clean_abstracts(
    source_name: str,
    input_path: Path,
    output_path: Path,
    min_year: int = 1980,
    max_year: int = 2022,
    min_char_length: int = 100,
    source_lang_tag: str = 'dut'
):
    """
    Extracts, tokenizes, filters, and standardizes abstracts based on language and dates.
    Tracks statistics for all excluded rows and generates the new text_dut and sent_dut columns.
    """
    print(f"[{source_name}] Parsing abstracts from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    #TODO maybe change to pa but rather not
    df = pd.read_parquet(input_path)
    filtered_rows = []

    removed_stats = {
        "missing_year": 0,
        "year_out_of_bounds": 0,
        "invalid_year_format": 0,
        "missing_text_content": 0,
        "text_too_short": 0,
        "no_dutch_sentences_detected": 0
    }

    # Convert the dataframe to records for fast row-by-row tqdm iteration
    records = df.to_dict(orient='records')

    for row in tqdm(records, desc=f"[{source_name}] Processing abstracts"):
        #YEAR CHECK
        year = row.get('year')

        # 1. Filter out missing years
        if pd.isna(year):
            removed_stats["missing_year"] += 1
            continue

        # 2. Check year validity
        try:
            year_int = int(year)
            if not (min_year <= year_int <= max_year):
                removed_stats["year_out_of_bounds"] += 1
                continue
        except (ValueError, TypeError):
            removed_stats["invalid_year_format"] += 1
            continue

        # ABSTRACT CHECK
        text_content = None
        if source_name == 'UG':
            abstract_full = row.get('abstract_full')
            if isinstance(abstract_full, (list, np.ndarray)):
                for item in abstract_full:
                    if isinstance(item, dict) and item.get('lang') == source_lang_tag:
                        text_content = item.get('text')
                        break
            # Fallback to general abstract if JSON list extraction is missing #TODO check this cuz its sus
            if not text_content and isinstance(row.get('abstract'), str):
                text_content = row.get('abstract')

        elif source_name == 'SB':
            # Check 'abstract' first, fallback to 'text_homepage' parsing
            if isinstance(row.get('abstract'), str) and row.get('abstract').strip():
                text_content = row.get('abstract')
            elif row.get('text_homepage') is not None:
                text_content = extract_sb_homepage_text(row.get('text_homepage'))

        else: # HBO #TODO
            if isinstance(row.get('abstract'), str):
                text_content = row.get('abstract')

        # 4. Filter out missing text
        if not text_content or not str(text_content).strip():
            removed_stats["missing_text_content"] += 1
            continue

        # 5. Filter out short text
        if len(text_content) < min_char_length:
            removed_stats["text_too_short"] += 1
            continue

        # 6 & 7. Clean and structure NLP sentences
        text_dut, sent_dut = clean_abstract(text_content, min_char_length=min_char_length) #TODO check validityof clean abstract but lgtm
        
        if not sent_dut:
            removed_stats["no_dutch_sentences_detected"] += 1
            continue

        # Append processing results
        row['text_dut'] = text_dut
        row['sent_dut'] = sent_dut
        filtered_rows.append(row)

    # Convert results back to DataFrame
    filtered_df = pd.DataFrame(filtered_rows)

    # Apply duplicate dropping based on standardized abstract text (keep first) #THis is superfluous but a good check lol
    if not filtered_df.empty:
        initial_length = len(filtered_df)
        filtered_df = filtered_df.drop_duplicates(subset=['text_dut'], keep='first')
        dropped_duplicates = initial_length - len(filtered_df)
    else:
        dropped_duplicates = 0

    print(f"\n[{source_name}] Process complete.")
    print(f"  Total records read:         {len(df)}")
    print(f"  Total records retained:     {len(filtered_df)}")
    print(f"  Duplicate entries dropped:  {dropped_duplicates}")
    print("  Exclusion counts by cause:")
    for cause, count in removed_stats.items():
        formatted_cause = cause.replace('_', ' ').capitalize()
        print(f"    - {formatted_cause}: {count}")
    print()

    # Save selection to silver parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not filtered_df.empty:
        filtered_df.to_parquet(output_path, index=False)
    else:
        # Save empty table structure if no records passed
        pd.DataFrame(columns=df.columns.tolist() + ['text_dut', 'sent_dut']).to_parquet(output_path, index=False)
        
    print(f"[{source_name}] Saved filtered table to: {output_path}")




#--------------------------------
#merging
#TODO csv saving since it was corrupt somehow
def merge(sources: list, output_format: str = 'csv', force: bool = False):
    """
    Merges the final {source}_selected files according to the source-specific 
    mapping schema, generates robust IDs where missing, and normalizes columns.
    """
    gold_dir = BASE_DIR / 'data' / 'gold'
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_output = gold_dir / 'merged_publications.parquet'
    csv_output = gold_dir / 'merged_publications.csv'
    
    outputs_exist = (
        (output_format in ['parquet', 'both'] and parquet_output.exists()) and
        (output_format in ['csv', 'both'] and csv_output.exists())
    )
    if outputs_exist and not force:
        print("[Merge] Merged outputs already exist. Skipping merge. Use --force or --force-merge to overwrite.")
        return

    merged_dfs = []

    for source in sources:
        config = SOURCES_CONFIG.get(source)
        if not config:
            continue
        
        selected_path = config['selected_file']
        if not selected_path.exists():
            print(f"[Merge] Selected file not found for {source} at: {selected_path}. Skipping.")
            continue

        print(f"[Merge] Standardizing and processing {source}...")
        df = pd.read_parquet(selected_path)
        
        if df.empty:
            print(f"[Merge] Selected data for {source} is empty. Skipping.")
            continue

        processed_df = pd.DataFrame()

        # --- Source-Specific Column Mapping Logic ---
        if source == 'UG':
            if '_id' in df.columns:
                processed_df['id'] = df['_id'].astype(str)
            else:
                processed_df['id'] = df.apply(lambda r: _generate_robust_id(source, r), axis=1)

            processed_df['source'] = 'UG'

            if 'keyword' in df.columns:
                processed_df['keywords'] = df['keyword'].apply(_parse_list_keywords)
            else:
                processed_df['keywords'] = [[] for _ in range(len(df))]

        elif source in ['HBO', 'SB']:
            processed_df['id'] = df.apply(lambda r: _generate_robust_id(source, r), axis=1)
            processed_df['source'] = source

            if 'keywords' in df.columns:
                processed_df['keywords'] = df['keywords'].apply(_parse_semicolon_keywords)
            else:
                processed_df['keywords'] = [[] for _ in range(len(df))]

        # --- Shared Column Standardization ---
        if 'year' in df.columns:
            processed_df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        else:
            processed_df['year'] = pd.Series([None] * len(df), dtype='Int64')

        # Simply grab processed NLP results directly from the Silver Selection files
        processed_df['abstract'] = df['text_dut'] if 'text_dut' in df.columns else None
        processed_df['abstract_sentence'] = df['sent_dut'] if 'sent_dut' in df.columns else None

        # Enforce target gold schema order
        target_cols = ['id', 'source', 'keywords', 'year', 'abstract', 'abstract_sentence']
        
        for col in target_cols:
            if col not in processed_df.columns:
                if col in ['keywords', 'abstract_sentence']:
                    processed_df[col] = [[] for _ in range(len(processed_df))]
                else:
                    processed_df[col] = None

        processed_df = processed_df[target_cols]
        merged_dfs.append(processed_df)

    if not merged_dfs:
        print("[Merge] No data sources were successfully compiled. Skipping merge output.")
        return

    final_df = pd.concat(merged_dfs, ignore_index=True)

    # --- Write Outputs ---
    if output_format in ['parquet', 'both']:
        print(f"[Merge] Saving Parquet merged data to: {parquet_output}")
        final_df.to_parquet(parquet_output, index=False)

    if output_format in ['csv', 'both']:
        print(f"[Merge] Saving CSV merged data to: {csv_output}")
        csv_df = final_df.copy()
        if 'keywords' in csv_df.columns:
            csv_df['keywords'] = csv_df['keywords'].apply(lambda x: ';'.join(x) if isinstance(x, list) else x)
        if 'abstract_sentence' in csv_df.columns:
            csv_df['abstract_sentence'] = csv_df['abstract_sentence'].apply(lambda x: ' | '.join(x) if isinstance(x, list) else x)
            
        csv_df.to_csv(csv_output, index=False)

    print("[Merge] Processing completed.")




# -------------------------------
#main
def main():
    parser = argparse.ArgumentParser(description="Multi-source NLP pipeline orchestrator")
    parser.add_argument(
        '--sources', 
        type=str, 
        nargs='+', 
        default=['UG', 'HBO', 'SB'], 
        choices=['UG', 'HBO', 'SB'], 
        help="Source dataset(s) to process. Default: all (UG, HBO, SB)"
    )
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        default=['download', 'clean', 'select', 'merge'],
        choices=['download', 'clean', 'select', 'merge'],
        help="Pipeline steps to execute. Default: all steps (download, clean, select, merge)"
    )
    parser.add_argument('--force', action='store_true', help="Force run all processes (ignores cache)")
    parser.add_argument('--force-download', action='store_true', help="Force run the download step")
    parser.add_argument('--force-clean', action='store_true', help="Force run the cleaning step")
    parser.add_argument('--force-select', action='store_true', help="Force run the selection/filtering step")
    parser.add_argument('--force-merge', action='store_true', help="Force run the merge step")
    parser.add_argument(
        '--output-format',
        type=str,
        default='parquet',
        choices=['parquet', 'csv', 'both'],
        help="Export file format for the merged step. Default: parquet. Options: parquet, csv, both"
    )
    
    args = parser.parse_args()

    sources = [s.upper() for s in args.sources]
    steps = [step.lower() for step in args.steps]

    for source in sources:
        print(f"\n--- Processing source: {source} ---")
        config = SOURCES_CONFIG[source]

        # 1. Download Step
        if 'download' in steps:
            raw_exists = config['raw_file'].exists()
            if not raw_exists or args.force or args.force_download:
                download_raw_data(source, config['raw_file'])
            else:
                print(f"[{source}] Raw dataset already exists. Skipping download.")

        #TODO add option for both csv and pq for cleaned and selected files
        # 2. Clean Step
        if 'clean' in steps:
            if not config['raw_file'].exists():
                print(f"[{source}] Missing raw file {config['raw_file']}. Skipping cleaning step.")
                continue

            clean_exists = config['clean_file'].exists()
            if not clean_exists or args.force or args.force_clean:
                if source == 'UG':
                    cleaned_table = optimize_file_table(config['raw_file'], post_process=pp_ug, save=config['clean_file'])
                elif source == 'SB':
                    cleaned_table = optimize_file_table(config['raw_file'], post_process=pp_sb, save=config['clean_file'])
                elif source == 'HBO':
                    cleaned_table = optimize_file_table(config['raw_file'], post_process=pp_hbo, save=config['clean_file'])
            
            else:
                print(f"[{source}] Cleaned dataset already exists. Skipping cleaning.")

        # 3. Select Step
        if 'select' in steps:
            if not config['clean_file'].exists():
                print(f"[{source}] Missing cleaned file {config['clean_file']}. Skipping selection step.")
                continue

            selected_exists = config['selected_file'].exists()
            if not selected_exists or args.force or args.force_select:
                select_and_clean_abstracts(
                    source_name=source,
                    input_path=config['clean_file'],
                    output_path=config['selected_file']
                )
            else:
                print(f"[{source}] Selected dataset already exists. Skipping selection.")

    # 4. Merge Step
    if 'merge' in steps:
        print("\n--- Running final merge step ---")
        merge(
            sources=sources,
            output_format=args.output_format,
            force=args.force or args.force_merge
        )

#-------------------------------


# =====================================================================
# 4. USER CONFIGURATION BLOCK
# =====================================================================
if __name__ == "__main__":
    main()




#TODO save cleaned as both csv and pq
#TODO change location of adding cols text_dut and sent_dut to before cleaning.


'''
logical structure

- download and scrape

- clean -> csv and pq
    - fix scraper mistakes

- select
    - select
    - make extra cols text_dut and sent_dut (temp?)

- clean again

- merge



'''