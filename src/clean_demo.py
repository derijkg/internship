import os
import json
import ast
import argparse
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.json as pj
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np

# =====================================================================
# SEMANTIC OPTIMIZER HELPERS
# =====================================================================
def is_identifier_name(name: str) -> bool:
    if not name:
        return False
    name_lower = name.lower()
    identifiers = {'_id'}
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
                pass

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
# INGESTION & COMPLEMENTARY COMPARER
# =====================================================================
def load_table(file_path: str, format: str = None) -> pa.Table:
    """Ingests a file based on format extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if format is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.csv', '.txt'):
            format = 'csv'
        elif ext in ('.json', '.jsonl', '.ndjson'):
            format = 'json'
        elif ext in ('.parquet', '.pq'):
            format = 'parquet'
        else:
            raise ValueError(f"Could not auto-detect format for: '{file_path}'. Specify 'format' explicitly.")

    if format == 'csv':
        return pv.read_csv(file_path)
    elif format == 'json':
        try:
            return pj.read_json(file_path)
        except Exception:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return pa.Table.from_pandas(pd.DataFrame(data))
            else:
                raise
    elif format == 'parquet':
        return pq.read_table(file_path)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'csv', 'json', or 'parquet'.")


def show_schema_difference(original: pa.Table, optimized: pa.Table):
    """Prints a structured diff between the original and optimized PyArrow schemas."""
    orig_fields = {f.name: f.type for f in original.schema}
    opt_fields = {f.name: f.type for f in optimized.schema}

    all_keys = sorted(list(set(orig_fields.keys()) | set(opt_fields.keys())))

    header = f"{'Column Name':<35} | {'Original Type':<35} | {'Optimized Type':<35} | Status"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for col in all_keys:
        orig_t = orig_fields.get(col)
        opt_t = opt_fields.get(col)

        orig_str = str(orig_t) if orig_t else "N/A"
        opt_str = str(opt_t) if opt_t else "N/A"

        if orig_t is None:
            status = "ADDED"
        elif opt_t is None:
            status = "REMOVED"
        elif orig_t != opt_t:
            status = "CHANGED"
        else:
            status = "UNFILTERED"

        print(f"{col:<35} | {orig_str:<35} | {opt_str:<35} | {status}")
    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clean raw data tables and display schema changes.")
    parser.add_argument('--file', type=str, required=True, help="Path to your raw CSV, JSON, or Parquet file.")
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'parquet'], default=None, help="Force format option.")
    args = parser.parse_args()

    print(f"Loading dataset: {args.file}")
    original_table = load_table(args.file, format=args.format)

    # 1. Detect JSON columns
    json_cols = detect_json_columns(original_table, sample_size=100)
    if json_cols:
        print(f"Detected nested serialized columns: {json_cols}")
        table_prepped = parse_serialized_columns(original_table, json_cols)
    else:
        table_prepped = original_table

    # 2. Run schema type coercion
    print("Running logical semantic data-type optimizer...")
    optimized_table = detect_logical_table(table_prepped)

    # 3. Present Results
    show_schema_difference(original_table, optimized_table)


if __name__ == "__main__":
    main()