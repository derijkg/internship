# inspect_parquet.py
import argparse
import sys
import numpy as np
import pandas as pd


def inspect_parquet(file_path: str, sample_size: int = 3, output_txt: str = None):
    print(f"Loading parquet file: {file_path} ...\n")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        sys.exit(1)

    lines = []
    lines.append("=" * 80)
    lines.append("                       PARQUET FILE INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"File Path    : {file_path}")
    lines.append(f"Total Rows   : {len(df):,}")
    lines.append(f"Total Columns: {len(df.columns)}")
    lines.append(f"Memory Usage : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    lines.append("=" * 80 + "\n")

    lines.append("1. COLUMN OVERVIEW TABLE")
    lines.append("-" * 80)
    lines.append(f"{'Column Name':<28} | {'Pandas dtype':<12} | {'Python Type':<20} | {'Nulls (%)':<12}")
    lines.append("-" * 80)

    for col in df.columns:
        non_nulls = df[col].dropna()
        first_val = non_nulls.iloc[0] if len(non_nulls) > 0 else None
        
        py_type = type(first_val).__name__ if first_val is not None else "All Null"
        if isinstance(first_val, (list, tuple, np.ndarray, set)):
            if len(first_val) > 0:
                inner_type = type(first_val[0]).__name__
                py_type = f"{py_type}[{inner_type}]"
            else:
                py_type = f"{py_type}[empty]"

        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        lines.append(f"{col:<28} | {str(df[col].dtype):<12} | {py_type:<20} | {null_pct:.1f}%")

    lines.append("\n" + "=" * 80)
    lines.append("2. DETAILED COLUMN BREAKDOWN & SAMPLE VALUES")
    lines.append("=" * 80 + "\n")

    for col in df.columns:
        non_nulls = df[col].dropna()
        total_rows = len(df)
        null_count = df[col].isna().sum()
        null_pct = (null_count / total_rows) * 100
        
        first_val = non_nulls.iloc[0] if len(non_nulls) > 0 else None
        py_type = type(first_val).__name__ if first_val is not None else "All Null"
        
        is_collection = isinstance(first_val, (list, tuple, np.ndarray, set))
        
        lines.append(f"COLUMN: '{col}'")
        lines.append(f"  ├─ Pandas Dtype : {df[col].dtype}")
        lines.append(f"  ├─ Python Type  : {py_type}")
        lines.append(f"  ├─ Null Count   : {null_count:,} / {total_rows:,} ({null_pct:.1f}%)")
        
        if is_collection:
            lens = [len(x) for x in non_nulls.iloc[:200] if isinstance(x, (list, tuple, np.ndarray, set))]
            avg_len = np.mean(lens) if lens else 0
            min_len = min(lens) if lens else 0
            max_len = max(lens) if lens else 0
            lines.append(f"  ├─ Collection   : Min Len={min_len}, Max Len={max_len}, Avg Len={avg_len:.1f}")
            if len(first_val) > 0:
                lines.append(f"  ├─ Element Type : {type(first_val[0]).__name__}")
            lines.append(f"  ├─ Unique Values: N/A (Collection)")
        else:
            try:
                lines.append(f"  ├─ Unique Values: {df[col].nunique():,}")
            except Exception:
                lines.append(f"  ├─ Unique Values: N/A")

        lines.append(f"  └─ Non-Null Samples ({min(sample_size, len(non_nulls))} examples):")
        
        for i in range(min(sample_size, len(non_nulls))):
            val = non_nulls.iloc[i]
            val_str = repr(val)
            # Truncate extremely long strings for clean text output
            if len(val_str) > 160:
                val_str = val_str[:157] + "..."
            lines.append(f"       [{i+1}] {val_str}")
        lines.append("")

    lines.append("=" * 80)
    report_text = "\n".join(lines)

    # Print to stdout
    print(report_text)

    # Optionally save to file
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n-> Full inspection report successfully saved to: '{output_txt}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Parquet file and output detailed column schema & samples.")
    parser.add_argument("data_path", type=str, help="Path to the Parquet file")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample values to print per column")
    parser.add_argument("--output", type=str, default="parquet_summary.txt", help="Path to save output TXT file")

    args = parser.parse_args()
    inspect_parquet(args.data_path, sample_size=args.samples, output_txt=args.output)