import os
import re
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
import nltk
import unicodedata

# Self-healing NLTK tokenizer download safeguards
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        nltk.download('punkt', quiet=True)

# Set base directories
BASE_DIR = Path(__file__).resolve().parent.parent

SOURCES_CONFIG = {
    'UG': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'UG' / 'publications.json',
        'clean_file_pq': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_cleaned.parquet',
        'clean_file_csv': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_cleaned.csv',
        'selected_file_pq': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_selected.parquet',
        'selected_file_csv': BASE_DIR / 'data' / 'silver' / 'UG' / 'ug_selected.csv',
    },
    'HBO': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'HBO' / 'HBO_metadata.csv',
        'clean_file_pq': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_cleaned.parquet',
        'clean_file_csv': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_cleaned.csv',
        'selected_file_pq': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_selected.parquet',
        'selected_file_csv': BASE_DIR / 'data' / 'silver' / 'HBO' / 'hbo_selected.csv',
    },
    'SB': {
        'raw_file': BASE_DIR / 'data' / 'bronze' / 'SB' / 'SB_metadata.csv',
        'clean_file_pq': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_cleaned.parquet',
        'clean_file_csv': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_cleaned.csv',
        'selected_file_pq': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_selected.parquet',
        'selected_file_csv': BASE_DIR / 'data' / 'silver' / 'SB' / 'sb_selected.csv',
    }
}


# =====================================================================
# DOWNLOAD STEP
# =====================================================================
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


# =====================================================================
# SEMANTIC OPTIMIZER HELPERS
# =====================================================================
def is_identifier_name(name: str) -> bool:
    if not name:
        return False
    name_lower = name.lower()
    identifiers = {
        '_id',# 'code', 'num', 'phone', 'zip', 'postal', 'serial', 'ssn', 
        #'ein', 'vabb', 'wos', 'arxiv', 'pubmed', 'esci', 'issn', 'isbn', 
        #'doi', 'ugent_id', 'biblio_id', 'handle', 'sha256'
    }
    return any(keyword in name_lower for keyword in identifiers)


def detect_logical_array(
    array: pa.Array,
    field_name: str = None,
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

    # Standardize Integers
    elif pa.types.is_integer(arrow_type):
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

        if infer_booleans:
            try:
                unique_vals = {v.as_py() for v in pc.unique(array) if v.is_valid}
                if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
                    return pc.cast(array, pa.bool_())
            except Exception:
                pass

        return pc.cast(array, pa.int64())

    elif pa.types.is_floating(arrow_type):
        try:
            is_nan_mask = pc.fill_null(pc.is_nan(array), False)
            if pc.any(is_nan_mask).as_py():
                null_scalar = pa.scalar(None, type=arrow_type)
                array = pc.if_else(is_nan_mask, null_scalar, array)
        except Exception:
            pass

        try:
            floor_array = pc.floor(array)
            is_integer_valued = pc.equal(array, floor_array)
            is_integer_valued = pc.fill_null(is_integer_valued, True)
            
            if pc.all(is_integer_valued).as_py():
                int_array = pc.cast(array, pa.int64())
                return detect_logical_array(
                    int_array, field_name, infer_dates, infer_booleans, infer_numbers,
                    category_ratio_threshold, category_max_unique, max_category_string_len
                )
        except Exception:
            pass

        return pc.cast(array, pa.float64())

    # Standardize Strings (Handles text placeholder sanitization and nested recursion)
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        try:
            sentinels = [
                "nan", "none", "null", "-", "--", "---", "_", "unknown", "missing", "niet beschikbaar",
                "<none>", "n/a", "na", "not reported", "/", "\\"
            ]
            trimmed_array = pc.utf8_trim_whitespace(array)
            lower_array = pc.ascii_lower(trimmed_array)
            is_sentinel_mask = pc.is_in(lower_array, value_set=pa.array(sentinels))
            
            if pc.any(is_sentinel_mask).as_py():
                null_scalar = pa.scalar(None, type=arrow_type)
                array = pc.if_else(is_sentinel_mask, null_scalar, trimmed_array)
            else:
                array = trimmed_array
        except Exception:
            pass

        non_null_count = len(array) - array.null_count
        if non_null_count == 0:
            return array

        if infer_booleans:
            try:
                unique_vals = {str(v.as_py()).strip().lower() for v in pc.unique(array) if v.is_valid}
                if unique_vals.issubset({"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0", "1.0", "0.0"}) and len(unique_vals) > 0:
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

        if infer_numbers:
            try:
                is_digit_mask = pc.ascii_is_decimal(array)
                is_digit_mask = pc.fill_null(is_digit_mask, True)
                if pc.all(is_digit_mask).as_py():
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

        if infer_dates:
            try:
                has_date_separators = pc.any(pc.or_(pc.match_substring(array, "-"), pc.match_substring(array, "/"))).as_py()
                if has_date_separators:
                    try:
                        return pc.cast(array, pa.date32())
                    except Exception:
                        return pc.cast(array, pa.timestamp('us'))
            except Exception:
                pass #TODO handle pass, maybe as string ?

        try:
            n_unique = pc.count_distinct(array).as_py()
            avg_len = pc.mean(pc.utf8_length(array)).as_py()
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
# RECURSIVE SCHEMA LOADER & PARSER
# =====================================================================
def optimize_file_table(
    input_source: any,
    format: str = None, 
    infer_dates: bool = True,
    infer_booleans: bool = True,
    infer_numbers: bool = True,
    category_ratio_threshold: float = 0.15,
    category_max_unique: int = 5_000,
    max_category_string_len: int = 60,
    remove_duplicates: any = None,
    save: any = None, # Can be string, Path, or a list of multiple paths
    post_process: callable = None
) -> pa.Table:
    """
    Ingests and optimizes raw files (CSV, JSON Lines, Parquet) or in-memory arrays.
    Optionally executes customized post-processing callables and saves results to multiple paths.
    """
    if isinstance(input_source, pa.Table):
        table = input_source
    elif isinstance(input_source, pd.DataFrame):
        table = pa.Table.from_pandas(input_source)
    else:
        input_source = str(input_source)
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Raw dataset file path not found: {input_source}")

        if format is None:
            ext = os.path.splitext(input_source)[1].lower()
            if ext in ('.csv', '.txt'):
                format = 'csv'
            elif ext in ('.json', '.jsonl', '.ndjson'):
                format = 'json'
            elif ext in ('.parquet', '.pq'):
                format = 'parquet'
            else:
                raise ValueError(f"Could not auto-detect format for: '{input_source}'. Please specify 'format' explicitly.")

        if format == 'csv':
            table = pv.read_csv(input_source)
        elif format == 'json':
            try:
                table = pj.read_json(input_source)
            except:
                with open(input_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    table = pa.Table.from_pandas(pd.DataFrame(data))
                else:
                    raise
        elif format == 'parquet':
            table = pq.read_table(input_source)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'csv', 'json', or 'parquet'.")
        
    json_cols = detect_json_columns(table, sample_size=100)
    if json_cols:
        print(f"[{format.upper()}] Auto-detected serialized nested columns: {json_cols}")
        table = parse_serialized_columns(table, json_cols)

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
        flat_columns = [
            name for name, field in zip(optimized_table.column_names, optimized_table.schema)
            if not (pa.types.is_list(field.type) or 
                    pa.types.is_large_list(field.type) or 
                    pa.types.is_fixed_size_list(field.type) or 
                    pa.types.is_struct(field.type))
        ]

        dedup_keys = []
        if remove_duplicates is True or remove_duplicates == "all":
            dedup_keys = flat_columns
            print(f"Deduplicating on all flat columns ({len(dedup_keys)} columns)...")
        elif isinstance(remove_duplicates, dict) and "exclude" in remove_duplicates:
            exclude_list = remove_duplicates["exclude"]
            if isinstance(exclude_list, str):
                exclude_list = [exclude_list]
            dedup_keys = [col for col in flat_columns if col not in exclude_list]
            print(f"Deduplicating on all flat columns except {exclude_list} ({len(dedup_keys)} columns)...")
        elif isinstance(remove_duplicates, (list, tuple, set)):
            for key in remove_duplicates:
                if key not in optimized_table.column_names:
                    raise KeyError(f"Deduplication column '{key}' does not exist.")
                if key not in flat_columns:
                    raise ValueError(
                        f"Cannot deduplicate on column '{key}' because it is nested."
                    )
            dedup_keys = list(remove_duplicates)
            print(f"Deduplicating on specified column subset: {dedup_keys}...")

        if dedup_keys:
            row_indices = pa.array(np.arange(len(optimized_table)), type=pa.int64())
            temp_table = optimized_table.append_column("__row_index__", row_indices)
            grouped = temp_table.group_by(dedup_keys).aggregate([("__row_index__", "min")])
            unique_indices = grouped.column("__row_index___min")
            sorted_positions = pc.sort_indices(unique_indices)
            ordered_unique_indices = pc.take(unique_indices, sorted_positions)

            pre_count = len(optimized_table)
            optimized_table = optimized_table.take(ordered_unique_indices)
            print(f"Deduplication finished. Retained {len(optimized_table)} of {pre_count} rows.")

    # Call custom post-processing callbacks prior to saving
    if post_process is not None:
        if not callable(post_process):
            raise TypeError("post_process argument must be a callable function.")
        optimized_table = post_process(optimized_table)

    # Handle multiple save targets (e.g. Parquet and CSV concurrently)
    if save is not None:
        save_paths = [save] if isinstance(save, (str, Path)) else list(save)
        
        for p in save_paths:
            save_path = str(p)
            save_ext = os.path.splitext(save_path)[1].lower()

            parent_dir = os.path.dirname(save_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            if save_ext in ('.parquet', '.pq'):
                pq.write_table(optimized_table, save_path)
                print(f"Saved cleaned table to Parquet: {save_path}")
            elif save_ext == '.csv':
                df = optimized_table.to_pandas()
                df.to_csv(save_path, index=False)
                print(f"Saved cleaned table to CSV: {save_path}")
            else:
                raise ValueError(f"Unsupported save format '{save_ext}'. Must be .csv or .parquet")

    return optimized_table


# =====================================================================
# SOURCE-SPECIFIC POST-PROCESSING FUNCTIONS
# =====================================================================
#NOT USED, NOT NEEDED
def pp_ug(table: pa.Table, **kwargs) -> pa.Table:
    """
    UG (University of Ghent) Specific Cleans:
    Converts academic sentinel placeholder values (99, 999, 9999) 
    in volume and issue fields into true null values.
    """
    print("[UG] Applying UG-specific cleans (nullifying sentinel volume/issue values)...")
    
    protected_values = {
        'volume': [99, '99', 999, '999', 9999, '9999'],
        'issue': [99, '99', '999', 999, '9999', 9999]
    }
    
    for col_name, sentinels in protected_values.items():
        if col_name in table.column_names:
            idx = table.schema.get_field_index(col_name)
            col_array = table.column(idx).combine_chunks()
            
            typed_sentinels = []
            for s in sentinels:
                try:
                    typed_sentinels.append(pa.scalar(s).cast(col_array.type))
                except Exception:
                    pass
            
            if typed_sentinels:
                value_set = pa.array(typed_sentinels, type=col_array.type)
                is_sentinel = pc.fill_null(pc.is_in(col_array, value_set=value_set), False)
                
                null_scalar = pa.scalar(None, type=col_array.type)
                cleaned_array = pc.if_else(is_sentinel, null_scalar, col_array)
                table = table.set_column(idx, col_name, cleaned_array)
                
    return table


def pp_sb(table: pa.Table, **kwargs) -> pa.Table:
    """
    Scriptiebank (SB) Specific Cleans:
    Safely converts floating-point or stringified float 'year' values 
    to integer representations (e.g., "2023.0" -> 2023).
    """
    print("[SB] Casting 'year' column to Integer type...")
    
    if 'year' in table.column_names:
        idx = table.schema.get_field_index('year')
        col_array = table.column(idx).combine_chunks()
        
        if pa.types.is_floating(col_array.type):
            try:
                is_nan = pc.fill_null(pc.is_nan(col_array), False)
                if pc.any(is_nan).as_py():
                    null_scalar = pa.scalar(None, type=col_array.type)
                    col_array = pc.if_else(is_nan, null_scalar, col_array)
                
                rounded = pc.round(col_array)
                integer_years = pc.cast(rounded, pa.int64())
                table = table.set_column(idx, 'year', integer_years)
            except Exception as e:
                print(f"[SB] Failed to cast float year to Integer: {e}")
                
        elif pa.types.is_string(col_array.type) or pa.types.is_large_string(col_array.type):
            try:
                floats = pc.cast(col_array, pa.float64())
                rounded = pc.round(floats)
                integer_years = pc.cast(rounded, pa.int64())
                table = table.set_column(idx, 'year', integer_years)
            except Exception as e:
                print(f"[SB] Failed to cast string year to Integer: {e}")
                
    return table


def pp_hbo(table: pa.Table, **kwargs) -> pa.Table:
    """
    HBO Specific Cleans:
    1. Jaar -> Year: Merges 'jaar' into 'year' (via coalesce) and drops 'jaar'.
    2. Partners -> Partner: Merges 'partners' into 'partner' and drops 'partners'.
    3. Title & Subtitle merger: Combines "Title" and "Subtitle" element-wise.
    """
    # 1. Merge 'jaar' into 'year'
    if 'jaar' in table.column_names:
        print("[HBO] Merging 'jaar' -> 'year'...")
        if 'year' in table.column_names:
            year_col = table.column('year')
            jaar_col = table.column('jaar')
            if year_col.type != jaar_col.type:
                try:
                    jaar_col = pc.cast(jaar_col, year_col.type)
                except Exception as e:
                    print(f"[HBO] Warning: Failed to cast 'jaar' to {year_col.type}: {e}")
            
            # FIXED: Pass columns directly instead of inside a list
            merged_year = pc.coalesce(year_col, jaar_col)
            table = table.set_column(table.schema.get_field_index('year'), 'year', merged_year)
        else:
            table = table.append_column('year', table.column('jaar'))
        table = table.remove_column(table.schema.get_field_index('jaar'))

    # 2. Merge 'partners' into 'partner'
    if 'partners' in table.column_names:
        print("[HBO] Merging 'partners' -> 'partner'...")
        if 'partner' in table.column_names:
            partner_col = table.column('partner')
            partners_col = table.column('partners')
            if partner_col.type != partners_col.type:
                try:
                    partners_col = pc.cast(partners_col, partner_col.type)
                except Exception as e:
                    print(f"[HBO] Warning: Failed to cast 'partners' to {partner_col.type}: {e}")
            
            # FIXED: Pass columns directly instead of inside a list
            merged_partner = pc.coalesce(partner_col, partners_col)
            table = table.set_column(table.schema.get_field_index('partner'), 'partner', merged_partner)
        else:
            table = table.append_column('partner', table.column('partners'))
        table = table.remove_column(table.schema.get_field_index('partners'))

    # 3. Combine 'title' and 'subtitle' element-wise (C++ parallel string processing)
    if 'title' in table.column_names and 'subtitle' in table.column_names:
        print("[HBO] Merging 'title' and 'subtitle' columns...")
        title_col = table.column('title')
        sub_col = table.column('subtitle')
        
        # FIXED: Changed pc.string_length -> pc.utf8_length
        sub_len = pc.fill_null(pc.utf8_length(sub_col), 0)
        has_subtitle = pc.and_(pc.is_valid(sub_col), pc.greater(sub_len, 0))
        
        joined_title = pc.binary_join_element_wise(title_col, sub_col, ": ")
        merged_title = pc.if_else(has_subtitle, joined_title, title_col)
        
        table = table.set_column(table.schema.get_field_index('title'), 'title', merged_title)
        table = table.remove_column(table.schema.get_field_index('subtitle'))

    return table

# =====================================================================
# SELECTION AND NLP CLEANING HELPERS
# =====================================================================
def extract_sb_homepage_text(val) -> Optional[str]:
    """
    Parses SB 'text_homepage' from either a JSON string, dictionary, or raw string,
    unifying list-based paragraph elements into a single cohesive block.
    """
    if pd.isna(val) or not val:
        return None
        
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
    
    for col in ['text_dut', 'abstract', 'abstract_full']:
        val = row.get(col)
        if val is not None and str(val).strip():
            content_parts.append(str(val))
            break
            
    title_val = row.get('title')
    if title_val is not None and str(title_val).strip():
        content_parts.append(str(title_val))
        
    year_val = row.get('year')
    if year_val is not None:
        content_parts.append(str(year_val))
        
    combined_content = "|".join(content_parts)
    
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

#TODO check
def safe_join_list(val, separator: str) -> str:
    """
    Safely converts lists, numpy arrays, or other iterables into a unified string.
    Prevents corrupt brackets and layout formatting in CSV serialization.
    """
    if val is None:
        return ""
    # Check for iterable types first to avoid Pandas ambiguity checks
    if isinstance(val, (list, np.ndarray, tuple, set)):
        return separator.join([str(item).strip() for item in val if pd.notna(item)])
    if pd.isna(val):
        return ""
    if isinstance(val, str):
        return val
    if hasattr(val, '__iter__') and not isinstance(val, (bytes, str)):
        return separator.join([str(item).strip() for item in val if pd.notna(item)])
    return str(val)

# =====================================================================
# ABSTRACT FILTERING AND NLP PARSING
# =====================================================================
def normalize_text(text: str) -> str:
    """
    Applies NFKC normalization, standardizes smart punctuation (quotes/dashes),
    and collapses duplicate whitespaces or hidden layout breaks.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. NFKC normalization (standardizes ligatures, accents, and symbol variants)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Convert curly/smart punctuation and non-standard dashes to standard ASCII
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    
    # 3. Collapse multiple whitespaces and convert carriage returns/tabs into standard spaces
    text = " ".join(text.split())
    
    return text

def clean_abstract(
    abstract: str,
    min_char_length: int = 100,
    min_sentences: int = 4,
    tokenizer_lang: str = 'dutch',
    detect_lang_tag: str = 'nl',
    heading_words: Optional[List[str]] = None,
    logger = None
) -> Tuple[str, List[str]]:
    """
    Cleans, tokenizes, and filters abstracts. Resolves formatting errors,
    strips layout headers, protects emails/URLs from splitting, and extracts Dutch sentences.
    Filters out non-Dutch abstracts entirely and applies minimum sentence cutoffs.
    """
    # 1. Normalize unicode and clean whitespace variants before running any NLP matches
    abstract = normalize_text(abstract)

    # Reject early if abstract doesn't meet minimum length requirements
    if not isinstance(abstract, str) or len(abstract) < min_char_length or not abstract.strip():
        return "", []

    # --- 2. FULL ABSTRACT LANGUAGE CHECK ---
    try:
        detected_lang = detect(abstract)
        if detected_lang != detect_lang_tag:
            if logger:
                logger.info(
                    f"Skipping abstract: Detected language '{detected_lang}' "
                    f"does not match target '{detect_lang_tag}'."
                )
            return "", []
    except LangDetectException:
        # Fall back to proceeding with execution if language detection fails on raw text
        pass

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
        # Strip bold/italic markdown characters
        sent_cleaned = re.sub(r'[*_]{1,2}', '', orig).strip()
        
        # 1. Strip compressed OCR headers (e.g., "INLEIDINGDit is")
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})([A-Z])', r'\1', sent_cleaned)
        # 2. Strip standard headers followed by separators (e.g., "Inleiding: Dit is")
        sent_cleaned = re.sub(rf'^(?:{headings_pattern})[\s]*[:.-]+[\s]*', '', sent_cleaned)
        
        # Clean up isolated matching heading sentences (e.g. a sentence that is just "Inleiding")
        if re.match(rf'^(?:{headings_pattern})$', sent_cleaned):
            sent_cleaned = ""

        if sent_cleaned != orig:
            if not sent_cleaned:
                removed = orig
            else:
                idx = orig.find(sent_cleaned)
                removed = orig[:idx] if idx != -1 else f"'{orig}' -> '{sent_cleaned}'"
            return sent_cleaned, removed
            
        return orig, None

    # --- 3. EMAIL & URL PROTECTION MASKING ---
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    url_pattern = r'\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    
    raw_emails = re.findall(email_pattern, abstract)
    raw_urls = re.findall(url_pattern, abstract)
    
    # Post-process: Strip trailing sentence-ending punctuation from URLs
    cleaned_urls = [url.rstrip('.,!?")];*_') for url in raw_urls]
    
    # Deduplicate and sort by length descending to prevent substring-replacement issues
    emails = sorted(list(set(raw_emails)), key=len, reverse=True)
    urls = sorted(list(set(cleaned_urls)), key=len, reverse=True)
    
    # Temporarily replace emails/URLs with safe placeholders
    protected_abstract = abstract
    for idx, email in enumerate(emails):
        protected_abstract = protected_abstract.replace(email, f"__EMAIL_PLACEHOLDER_{idx}__")
    for idx, url in enumerate(urls):
        protected_abstract = protected_abstract.replace(url, f"__URL_PLACEHOLDER_{idx}__")

    # Fix missing spaces after punctuation (safe from breaking masked emails/URLs)
    protected_abstract = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', protected_abstract)
    
    raw_sentences = nltk.sent_tokenize(protected_abstract, language=tokenizer_lang)
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
            # Merge if the current fragment does not start with a capital letter (split abbreviation)
            if not re.match(r'^[A-Z]', sent):
                should_merge = True
            # Merge if character index 1 is punctuation (like 's-Gravenhage) BUT NOT a list item (like "1." or "A.")
            elif len(sent) >= 2 and sent[1] in string.punctuation:
                if not sent[0].isalnum(): # Only merge if the prefix is not a list indicator
                    should_merge = True
        
        if should_merge:
            cleaned_sentences[-1] = cleaned_sentences[-1] + ' ' + sent
        else:
            cleaned_sentences.append(sent)
    
    # Helper to restore masked values
    def restore_placeholders(text: str) -> str:
        for idx, email in enumerate(emails):
            text = text.replace(f"__EMAIL_PLACEHOLDER_{idx}__", email)
        for idx, url in enumerate(urls):
            text = text.replace(f"__URL_PLACEHOLDER_{idx}__", url)
        return text

    # Restore placeholders before running sentence language detection
    restored_sentences = [restore_placeholders(s) for s in cleaned_sentences]

    # --- 4. SEMANTIC RUN-LENGTH LANGUAGE FILTER ---
    sentence_flags = []
    for sent in restored_sentences:
        try:
            is_dutch = (detect(sent) == detect_lang_tag)
        except LangDetectException:
            is_dutch = True
        sentence_flags.append((sent, is_dutch))
        
    dutch_sentences = []
    current_non_dutch_run = []
    max_non_dutch_sequence = 3  # Drop run of non-Dutch text if length >= 3
    
    for sent, is_dutch in sentence_flags:
        if is_dutch:
            if len(current_non_dutch_run) < max_non_dutch_sequence and len(dutch_sentences) > 0:
                dutch_sentences.extend(current_non_dutch_run)
            else:
                if logger:
                    logger.debug(
                        f"Dropped non-Dutch abstract block of length {len(current_non_dutch_run)}"
                    )
            current_non_dutch_run = []
            dutch_sentences.append(sent)
        else:
            current_non_dutch_run.append(sent)
            
    if len(current_non_dutch_run) < max_non_dutch_sequence:
        dutch_sentences.extend(current_non_dutch_run)
        
    # --- 5. MINIMUM SENTENCE CUTOFF FILTER ---
    if len(dutch_sentences) < min_sentences:
        if logger:
            logger.info(
                f"Skipping abstract: Number of remaining valid sentences ({len(dutch_sentences)}) "
                f"is fewer than the minimum cutoff ({min_sentences})."
            )
        return "", []

    dutch_abstract = ""
    if dutch_sentences:
        dutch_abstract = ' '.join(dutch_sentences)
        
    return dutch_abstract, dutch_sentences


#TODO curr always works from pq file, maybe add csv option but could also always make parquet file
def select_and_clean_abstracts(
    source_name: str,
    input_path: Path,
    output_path_pq: Path,
    output_path_csv: Path,
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
    
    if str(input_path).endswith('.csv'):
        df = pd.read_csv(input_path)
    elif str(input_path).endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        raise TypeError('couldnt read filetype, use pq or csv')
    filtered_rows = []

    removed_stats = {
        "missing_year": 0,
        "year_out_of_bounds": 0,
        "invalid_year_format": 0,
        "missing_text_content": 0,
        "text_too_short": 0,
        "no_dutch_sentences_detected": 0
    }

    records = df.to_dict(orient='records')

    for row in tqdm(records, desc=f"[{source_name}] Processing abstracts"):
        # 1. Year Validation
        year = row.get('year')
        if pd.isna(year):
            removed_stats["missing_year"] += 1
            continue

        try:
            year_int = int(float(year))
            if not (min_year <= year_int <= max_year):
                removed_stats["year_out_of_bounds"] += 1
                continue
        except (ValueError, TypeError):
            removed_stats["invalid_year_format"] += 1
            continue

        # 2. Abstract Content Extraction
        text_content = None
        if source_name == 'UG':
            abstract_full = row.get('abstract_full')
            if isinstance(abstract_full, (list, np.ndarray)):
                for item in abstract_full:
                    if isinstance(item, dict) and item.get('lang') == source_lang_tag:
                        text_content = item.get('text')
                        break
            if not text_content and isinstance(row.get('abstract'), str): #TODO turbo sus fallback
                text_content = row.get('abstract')

        elif source_name == 'SB':
            if isinstance(row.get('abstract'), str) and row.get('abstract').strip():
                text_content = row.get('abstract')
            elif row.get('text_homepage') is not None:
                text_content = extract_sb_homepage_text(row.get('text_homepage'))

        else: # HBO
            if isinstance(row.get('abstract'), str):
                text_content = row.get('abstract')

        # 3. Content Filters
        if not text_content or not str(text_content).strip():
            removed_stats["missing_text_content"] += 1
            continue

        if len(text_content) < min_char_length:
            removed_stats["text_too_short"] += 1
            continue

        # 4. Clean and tokenise Dutch NLP components
        text_dut, sent_dut = clean_abstract(text_content, min_char_length=min_char_length)
        
        if not sent_dut:
            removed_stats["no_dutch_sentences_detected"] += 1
            continue

        # Append calculated structures to row
        row['text_dut'] = text_dut
        row['sent_dut'] = sent_dut
        filtered_rows.append(row)

    # Rebuild DataFrame and apply deduplication on the standardized abstract content
    filtered_df = pd.DataFrame(filtered_rows)

    if not filtered_df.empty:
        initial_length = len(filtered_df)
        filtered_df = filtered_df.drop_duplicates(subset=['text_dut'], keep='first') #should be superfluous but good check #TODO keep most populated entry
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

    # Save selection silver files to Parquet and CSV
    if output_path_pq is not None:
        output_path_pq.parent.mkdir(parents=True, exist_ok=True)
        if not filtered_df.empty:
            filtered_df.to_parquet(output_path_pq, index=False)
        else:
            empty_df = pd.DataFrame(columns=df.columns.tolist() + ['text_dut', 'sent_dut'])
            empty_df.to_parquet(output_path_pq, index=False)
            
    if output_path_csv is not None:
        output_path_csv.parent.mkdir(parents=True, exist_ok=True)
        if not filtered_df.empty:
            filtered_df.to_csv(output_path_csv, index=False)
        else:
            empty_df = pd.DataFrame(columns=df.columns.tolist() + ['text_dut', 'sent_dut'])
            empty_df.to_csv(output_path_csv, index=False)
        
    print(f"[{source_name}] Saved filtered table to requested formats.")


# =====================================================================
# GOLD MERGE STEP
# =====================================================================
def merge(sources: list, output_format: str = 'csv', force: bool = False):
    """
    Merges the final {source}_selected files according to the source-specific 
    mapping schema, generates robust deterministic IDs, and normalizes columns.
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
        
        # DYNAMIC PATH RESOLVER: Falls back to CSV if Parquet Silver files are missing
        selected_path = config['selected_file_pq'] if config['selected_file_pq'].exists() else config['selected_file_csv']
        if not selected_path.exists():
            print(f"[Merge] Selected file not found for {source} at: {selected_path}. Skipping.")
            continue
        
        # DYNAMIC LOADER: Read whichever format is available
        if str(selected_path).endswith('.csv'):
            df = pd.read_csv(selected_path)
        else:
            df = pd.read_parquet(selected_path)

        print(f"[Merge] Standardizing and processing {source}...")
        
        if df.empty:
            print(f"[Merge] Selected data for {source} is empty. Skipping.")
            continue

        processed_df = pd.DataFrame()

        # --- Source-Specific Column Mapping ---
        if source == 'UG':
            if '_id' in df.columns:
                processed_df['_id'] = df['_id'].astype(str)  # Changed from 'id' to '_id'
            else:
                processed_df['_id'] = df.apply(lambda r: _generate_robust_id(source, r), axis=1)  # Changed from 'id' to '_id'

            processed_df['source'] = 'UG'

            if 'keyword' in df.columns:
                processed_df['keywords'] = df['keyword'].apply(_parse_list_keywords)
            else:
                processed_df['keywords'] = [[] for _ in range(len(df))]

        elif source in ['HBO', 'SB']:
            processed_df['_id'] = df.apply(lambda r: _generate_robust_id(source, r), axis=1)  # Changed from 'id' to '_id'
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

        # Map processed NLP results directly from the selection layer
        processed_df['abstract'] = df['text_dut'] if 'text_dut' in df.columns else None
        processed_df['abstract_sentence'] = df['sent_dut'] if 'sent_dut' in df.columns else None

        # Enforce target Gold schema order
        target_cols = ['_id', 'source', 'keywords', 'year', 'abstract', 'abstract_sentence']  # Changed 'id' to '_id'
        
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
        
        # Safe string joining protects against NumPy structural cell overflows in CSV layout
        if 'keywords' in csv_df.columns:
            csv_df['keywords'] = csv_df['keywords'].apply(lambda x: safe_join_list(x, ';'))
        if 'abstract_sentence' in csv_df.columns:
            csv_df['abstract_sentence'] = csv_df['abstract_sentence'].apply(lambda x: safe_join_list(x, ' | '))
            
        csv_df.to_csv(csv_output, index=False)

    print("[Merge] Processing completed successfully.")

# =====================================================================
# PIPELINE ORCHESTRATOR
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-source NLP pipeline orchestrator")
    parser.add_argument(
        '--sources', 
        type=str, 
        nargs='+', 
        default=['UG', 'HBO'], 
        choices=['UG', 'HBO', 'SB'], 
        help="Source dataset(s) to process. Default: only (UG, HBO)"
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
        default='both',
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
            print(f'Starting download: {source}')
            raw_exists = config['raw_file'].exists()
            if not raw_exists or args.force or args.force_download:
                download_raw_data(source, config['raw_file'])
            else:
                print(f"[{source}] Raw dataset already exists. Skipping download.")

        # 2. Clean Step (Silver Cleaned)
        if 'clean' in steps:
            print(f'Starting cleaning: {source}')
            if not config['raw_file'].exists():
                print(f"[{source}] Missing raw file {config['raw_file']}. Skipping cleaning step.")
                continue

            # Determine which targets to check and generate based on the --output-format option
            saves = []
            checks = []
            if args.output_format in ['parquet', 'both']:
                saves.append(config['clean_file_pq'])
                checks.append(config['clean_file_pq'].exists())
            if args.output_format in ['csv', 'both']:
                saves.append(config['clean_file_csv'])
                checks.append(config['clean_file_csv'].exists())

            # Build only if any requested format is missing, or if forced
            clean_exists = all(checks) if checks else False
            if not clean_exists or args.force or args.force_clean:
            #TODO check post processing functions
            #TODO check deduplication cols
                if source == 'UG':
                    opt_table = optimize_file_table(config['raw_file'], save=saves) #doesnt need pp
                elif source == 'SB':
                    opt_table = optimize_file_table(config['raw_file'], post_process=pp_sb, save=saves)
                elif source == 'HBO':
                    opt_table = optimize_file_table(config['raw_file'], post_process=pp_hbo, save=saves, remove_duplicates=['abstract'])
                print(opt_table.schema)
            else:
                print(f"[{source}] Cleaned dataset already exists for target format(s). Skipping cleaning.")

        # 3. Select Step (Silver Selected)
        if 'select' in steps:
            print(f'Starting row selection: {source}')
            # Safely fallback to reading CSV as the input source for NLP if PQ doesn't exist
            input_file = config['clean_file_pq'] if config['clean_file_pq'].exists() else config['clean_file_csv']
            if not input_file.exists():
                print(f"[{source}] Missing cleaned file. Skipping selection step.")
                continue

            checks = []
            output_pq = None
            output_csv = None

            if args.output_format in ['parquet', 'both']:
                output_pq = config['selected_file_pq']
                checks.append(config['selected_file_pq'].exists())
            if args.output_format in ['csv', 'both']:
                output_csv = config['selected_file_csv']
                checks.append(config['selected_file_csv'].exists())

            selected_exists = all(checks) if checks else False
            if not selected_exists or args.force or args.force_select:
                select_and_clean_abstracts(
                    source_name=source,
                    input_path=input_file,
                    output_path_pq=output_pq,
                    output_path_csv=output_csv
                )
            else:
                print(f"[{source}] Selected dataset already exists for target format(s). Skipping selection.")

    # 4. Merge Step (Gold Merged)
    if 'merge' in steps:
        print("\n--- Running final merge step ---")
        merge(
            sources=sources,
            output_format=args.output_format,
            force=args.force or args.force_merge
        )


if __name__ == "__main__":
    main()