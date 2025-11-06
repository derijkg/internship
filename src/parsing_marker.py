#mamba install -n pdf_parsing marker-pdf
import subprocess
import re
import os
import tempfile
import shutil

def convert_pdf_to_structured_markdown(pdf_path: str) -> tuple[str, dict[str, str]]:
    """
    Converts a PDF file to Markdown using Marker and structures the output.

    This function runs the marker_single command-line tool to convert the PDF,
    then reads the output and splits it into chapters based on Level 1 Markdown
    headings (# Heading).

    Args:
        pdf_path (str): The absolute or relative path to the input PDF file.

    Returns:
        A tuple containing:
        - full_markdown (str): The complete Markdown content of the entire document.
        - chapters (dict): A dictionary where keys are the chapter titles 
          (e.g., "Chapter 1: Introduction") and values are the Markdown
          content of each chapter.
          
    Raises:
        FileNotFoundError: If the input PDF file does not exist.
        RuntimeError: If the Marker CLI tool fails to execute.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file was not found at: {pdf_path}")

    # Create a temporary directory to store Marker's output
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # 1. Run the Marker command-line tool
        try:
            print(f"Running Marker on '{pdf_path}'...")
            # Using subprocess.run to execute the command
            # The output of the command (stdout/stderr) will be printed to the console
            subprocess.run(
                ["marker_single", pdf_path, temp_dir],
                check=True, # This will raise a CalledProcessError if Marker fails
                capture_output=True, # Captures stdout and stderr
                text=True # Decodes stdout/stderr as text
            )
            print("Marker conversion completed successfully.")
        except FileNotFoundError:
            raise RuntimeError(
                "The 'marker_single' command was not found. "
                "Please ensure that marker-pdf is installed and your "
                "virtual environment is activated."
            )
        except subprocess.CalledProcessError as e:
            # If Marker fails, print its error output for easier debugging
            print("--- Marker STDOUT ---")
            print(e.stdout)
            print("--- Marker STDERR ---")
            print(e.stderr)
            raise RuntimeError(f"Marker CLI failed with exit code {e.returncode}.")

        # 2. Find the output Markdown file
        output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
        markdown_path = os.path.join(temp_dir, output_filename)

        if not os.path.exists(markdown_path):
            raise FileNotFoundError(f"Marker did not produce the expected output file: {markdown_path}")
            
        # 3. Read the full Markdown content
        print(f"Reading generated Markdown file: {markdown_path}")
        with open(markdown_path, 'r', encoding='utf-8') as f:
            full_markdown = f.read()

    # 4. Split the content by chapter (Level 1 headings)
    chapters = {}
    # This regex splits the text by lines that start with '# ' (a Level 1 heading)
    # The `(?m)` flag enables multi-line mode
    split_content = re.split(r'(?m)^\#\s(.+)', full_markdown)
    
    # The first element is any content before the first chapter title
    preface_content = split_content[0].strip()
    if preface_content:
        chapters["Preface"] = preface_content

    # The rest of the list is alternating [title, content, title, content, ...]
    for i in range(1, len(split_content), 2):
        title = split_content[i].strip()
        content = split_content[i+1].strip()
        chapters[title] = content
        
    print("Markdown has been split into chapters.")
    
    return full_markdown, chapters

### --- Example Usage ---
if __name__ == "__main__":
    try:

        DUMMY_PDF_PATH = "data/test/test_nl.pdf"
        try:
            full_text, chapters_dict = convert_pdf_to_structured_markdown(DUMMY_PDF_PATH)
            
            print("\n" + "="*50)
            print("      CONVERSION RESULTS")
            print("="*50 + "\n")
            
            print(f"Successfully processed the PDF into {len(chapters_dict)} sections.")
            print("Chapter titles found:", list(chapters_dict.keys()))
            
            print("\n--- Content of 'Chapter 1: The Beginning' ---")
            if "Chapter 1: The Beginning" in chapters_dict:
                # Print the first 200 characters of the chapter
                print(chapters_dict["Chapter 1: The Beginning"][:200] + "...")
            else:
                print("Chapter 1 not found.")
                
            print("\n--- Full Markdown Text (first 400 characters) ---")
            print(full_text[:400] + "...")

        except (FileNotFoundError, RuntimeError) as e:
            print(f"\nAn error occurred: {e}")
        finally:
            # Clean up the dummy PDF
            if os.path.exists(DUMMY_PDF_PATH):
                os.remove(DUMMY_PDF_PATH)
                print(f"\nCleaned up dummy PDF: '{DUMMY_PDF_PATH}'")

    except ImportError:
        print("Please provide your own PDF and call the function directly.")
        print("Example: full_text, chapters = convert_pdf_to_structured_markdown('path/to/your/file.pdf')")