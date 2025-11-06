#!/bin/bash

# --- Configuration ---
# Define input and output paths as variables for easy modification.
INPUT_PDF="../data/test/test.pdf"
OUTPUT_DIR="../data/test/marker_output"
# The output filename will be the same as the input, but with a .md extension.
OUTPUT_MD_PATH="$OUTPUT_DIR/test.md"

# --- Script Logic ---
echo "Starting Marker PDF conversion..."
echo "Input file: $INPUT_PDF"
echo "Output file: $OUTPUT_MD_PATH"

# Best Practice: Create the output directory if it doesn't exist.
# The "-p" flag ensures it doesn't complain if the directory is already there.
mkdir -p "$OUTPUT_DIR"

# The main command.
# - The environment variable is placed on the same line to apply it only to this command.
# - We provide both the input PDF path and the full output Markdown file path.
# - Variables are quoted to handle spaces or special characters in paths gracefully.
CUDA_VISIBLE_DEVICES=0 marker_single "$INPUT_PDF" --output_dir "$OUTPUT_MD_PATH"

echo "Conversion complete. Output saved to $OUTPUT_MD_PATH"