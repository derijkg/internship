import os
import sys
import subprocess
import zipfile
import tempfile
import argparse
from pathlib import Path
from tqdm import tqdm

def detect_gpus() -> int:
    """Detects the number of available NVIDIA GPUs using torch."""
    print("INFO: Detecting available GPUs...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"SUCCESS: Found {gpu_count} NVIDIA GPU(s).")
            return gpu_count
        else:
            print("INFO: No NVIDIA GPUs found. Marker will run on CPU.")
            return 0
    except ImportError:
        print("WARNING: 'torch' is not installed. Cannot detect GPUs. Assuming CPU execution.", file=sys.stderr)
        return 0

def unzip_to_tempdir(zip_path: Path) -> tempfile.TemporaryDirectory:
    """Unzips an archive to a temporary directory and returns the directory object."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"The specified zip file does not exist: {zip_path}")
    
    temp_dir = tempfile.TemporaryDirectory()
    print(f"INFO: Unzipping '{zip_path.name}' to temporary directory...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir.name)
    print("SUCCESS: Unzipping complete.")
    return temp_dir

# --- Processing Functions ---

def process_pdfs_batch(pdf_dir: Path, output_dir: Path, gpu_count: int):
    """
    Processes all PDF files in a directory using marker's batch command.
    Configures workers and GPU devices based on hardware detection.
    """
    print("\n--- Starting PDF Processing ---")
    command = ["marker", str(pdf_dir), str(output_dir)]

    if gpu_count > 0:
        # Per marker documentation/best practice, set workers relative to GPU count
        num_workers = 4 * gpu_count
        gpu_devices = ",".join(map(str, range(gpu_count)))
        
        command.extend(["--num_workers", str(num_workers)])
        command.extend(["--gpu_devices", gpu_devices])
        
        print(f"INFO: Configuring marker for {gpu_count} GPU(s) with {num_workers} workers.")
    else:
        print("INFO: Configuring marker for CPU execution.")

    try:
        # Marker has its own TQDM progress bar, so we just run it directly.
        subprocess.run(command, check=True)
        print("SUCCESS: PDF processing complete.")
    except FileNotFoundError:
        print("Error: 'marker' command not found.", file=sys.stderr)
        print("Please ensure 'marker-pdf' is installed in your environment.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Marker failed with exit code {e.returncode}.", file=sys.stderr)
        print("Please review the output from the tool above.", file=sys.stderr)
        sys.exit(1)

def process_docx_files(docx_files: list, output_dir: Path):
    """Processes a list of DOCX files using pandoc."""
    print("\n--- Starting DOCX Processing ---")
    if not docx_files:
        print("INFO: No DOCX files found to process.")
        return

    check_system_dependency("pandoc")

    for docx_path in tqdm(docx_files, desc="Converting DOCX"):
        output_filename = docx_path.stem + ".md"
        output_path = output_dir / output_filename
        
        command = [
            "pandoc",
            str(docx_path),
            "-f", "docx",
            "-t", "markdown",
            "-o", str(output_path)
        ]
        subprocess.run(command, capture_output=True, check=True)
    print("SUCCESS: DOCX processing complete.")


# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(
        description="Batch convert PDF and DOCX files from a ZIP archive to Markdown.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("zip_path", type=Path, help="Path to the input .zip file.")
    parser.add_argument("output_dir", type=Path, help="Path to the directory where markdown files will be saved.")
    args = parser.parse_args()

    # --- Setup ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_count = detect_gpus()
    
    # Unzip the source files to a managed temporary directory
    try:
        temp_source_dir = unzip_to_tempdir(args.zip_path)
        source_path = Path(temp_source_dir.name)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- File Discovery ---
    # Using rglob to find files in subdirectories of the zip
    all_files = list(source_path.rglob("*"))
    pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
    docx_files = [f for f in all_files if f.suffix.lower() == ".docx"]

    # --- Processing ---
    # The `marker` batch command needs a single input directory.
    # We can just point it to our main unzipped directory, and it will
    # automatically find and process only the PDF files.
    if pdf_files:
        process_pdfs_batch(source_path, args.output_dir, gpu_count)
    else:
        print("\nINFO: No PDF files found to process.")

    # Pandoc needs to be run on each file individually.
    process_docx_files(docx_files, args.output_dir)

    # --- Cleanup ---
    temp_source_dir.cleanup()
    print(f"\n✅ All tasks complete. Markdown files are saved in '{args.output_dir}'.")


if __name__ == "__main__":
    main()