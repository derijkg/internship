import subprocess
import re
import os
import tempfile
import sys
from tqdm import tqdm
import time

#!sudo apt-get update && sudo apt-get install pandoc
#!pip install --upgrade marker-pdf torch torchvision torchaudio tqdm

def run_marker_command(command: list):
    """
    Runs a subprocess command and streams its output directly to the console.
    This allows us to see the real-time progress from the command itself.

    Args:
        command (list): The command and its arguments to execute.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.
    """
    print(f"\n--- Running Marker Command ---\n$ {' '.join(command)}\n")
    
    # By not capturing stdout/stderr, the subprocess will print directly to our console
    process = subprocess.run(command, check=True) # check=True will raise an exception on failure
    
    print("\n--- Marker execution completed successfully ---\n")
    # Setup and run the indeterminate progress bar
    with tqdm(desc="Converting PDF...", unit=" ticks") as pbar:
        while process.poll() is None:  # poll() returns None while the process is running
            pbar.update(1) # Animate the bar
            time.sleep(0.1) # Prevents the loop from using 100% CPU

    # The process has finished. Now we check the result.
    return_code = process.wait()
    if return_code != 0:
        stdout, stderr = process.communicate()
        print("--- Marker execution failed ---", file=sys.stderr)
        print(f"Return Code: {return_code}", file=sys.stderr)
        print("\n--- STDOUT ---", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print("\n--- STDERR ---", file=sys.stderr)
        print(stderr, file=sys.stderr)
        raise RuntimeError("Marker CLI failed. See the output above for details.")
    
    print("\nMarker conversion completed successfully.")

def convert_and_save_pdf(pdf_path: str, output_dir: str):
    """
    Converts a PDF to Markdown using the Marker CLI and saves the structured output.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory to save the final Markdown files.
        
    Raises:
        FileNotFoundError: If the input PDF file does not exist.
        RuntimeError: If the Marker CLI tool fails or is not found.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The input file was not found at: {pdf_path}. Please check the path.")

    # Use a temporary directory to handle Marker's output files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for conversion: {temp_dir}")
        
        # 1. Run the Marker command-line tool via subprocess
        try:
            marker_command = ["marker_single", pdf_path, '--output_dir', temp_dir]
            run_marker_command(marker_command)

        except FileNotFoundError:
            raise RuntimeError(
                "The 'marker_single' command was not found. "
                "Please ensure marker-pdf is installed and the correct "
                "virtual environment is activated."
            )

        # 2. Read the full markdown from the temporary output file
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_filename = f"{base_filename}.md"
        temp_markdown_path = os.path.join(temp_dir, markdown_filename)

        if not os.path.exists(temp_markdown_path):
            raise FileNotFoundError(f"Marker did not produce the expected output file: {temp_markdown_path}")
            
        with open(temp_markdown_path, 'r', encoding='utf-8') as f:
            full_markdown = f.read()

    # 3. Save the full markdown output
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, "marker_test_output.md")
    with open(full_output_path, 'w', encoding='utf-8') as f:
        f.write(full_markdown)
    print(f"\nFull markdown output saved to: '{full_output_path}'")

    # 4. Split by chapter and save individual files
    print("Splitting by chapter and saving individual files...")
    # This regex splits the text by lines that start with '# ' (a Level 1 heading)
    split_content = re.split(r'(?m)^#\s(.+)', full_markdown)
    
    # Handle content before the first chapter (e.g., preface, title page)
    preface_content = split_content[0].strip()
    if preface_content:
        preface_path = os.path.join(output_dir, "Preface.md")
        with open(preface_path, 'w', encoding='utf-8') as f:
            f.write(preface_content)
        print(f"  - Saved 'Preface.md'")

    # Handle main chapters
    for i in range(1, len(split_content), 2):
        title = split_content[i].strip()
        # Reconstruct the chapter content with its title
        content = f"# {title}\n\n{split_content[i+1].strip()}"
        
        # Sanitize the chapter title to create a valid filename
        filename = re.sub(r'[\\/*?:"<>|]', "", title).replace(" ", "_") + ".md"
        chapter_path = os.path.join(output_dir, filename)
        
        with open(chapter_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  - Saved '{chapter_path}'")

### --- Main Execution Block ---
if __name__ == "__main__":
    PDF_PATH = "data/test/test_nl.pdf"

    # Define the directory where the output files will be saved
    TARGET_DIR = os.path.join("data", "test")
    
    # Ensure the target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)

    try:
        # Run the main conversion and saving logic
        convert_and_save_pdf(pdf_path=PDF_PATH, output_dir=TARGET_DIR)
        
        print("\n" + "="*50)
        print("      PROCESS COMPLETED SUCCESSFULLY")
        print("="*50 + "\n")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nAN ERROR OCCURRED: {e}", file=sys.stderr)