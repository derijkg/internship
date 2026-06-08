import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import random
random.seed(42)
import nltk
from ollama import Client

# Download the sentence tokenizer model for Dutch
nltk.download('punkt_tab')
nltk.download('punkt')

# ==========================================
# 1. CONFIGURE OLLAMA & PARAMETERS
# ==========================================
# Connect using the official Ollama Python Client
client = Client(host='http://localhost:11434')

# Define the Ollama model and instruction you want to use
MODEL_CONFIG = {
    "model_id": "llama3.1", # Ensure you have run `ollama pull llama3.1` in your terminal
    "instruction": "Herschrijf deze zin zodat het simpeler is voor een leek (B1 niveau). Maak de zin niet langer dan de originele zin."
}

# The list of percentages you want to run sequentially. 
# 0.10 = 10% of sentences rewritten, 0.50 = 50%, etc.
PERCENTAGES_TO_RUN = [0.10, 0.25, 0.50]

# ==========================================
# 2. LLM REWRITE FUNCTION
# ==========================================
def rewrite_sentence(sentence: str) -> str:
    """Sends a single sentence to Ollama for rewriting."""
    system_prompt = (
        "Je bent een professionele Nederlandse tekstverwerker. "
        f"Jouw taak is: {MODEL_CONFIG['instruction']} "
        "Geef UITSLUITEND de herschreven zin terug. Geef geen introductie, geen uitleg en geen aanhalingstekens."
    )

    try:
        # Using the official Ollama chat implementation
        response = client.chat(
            model=MODEL_CONFIG["model_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sentence}
            ],
            options={
                "temperature": 0.3, # Low temperature to prevent hallucinations
                "num_predict": 150  # Ollama's equivalent of max_tokens
            }
        )
        
        # Extract the text content from the Ollama response dictionary
        rewritten = response['message']['content'].strip()
        
        # Clean up stray quotes if the LLM adds them
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
            
        return rewritten
    
    except Exception as e:
        print(f"Error calling Ollama ({MODEL_CONFIG['model_id']}): {e}")
        return sentence # Fallback to original sentence if the API fails

# ==========================================
# 3. CORE PROCESSING LOGIC (PER PERCENTAGE)
# ==========================================
def process_abstract(abstract: str, percentage: float) -> str:
    """Splits abstract, selects X% of sentences at random, rewrites them, and rebuilds it."""
    if not isinstance(abstract, str) or not abstract.strip():
        return abstract

    # Split into Dutch sentences robustly
    sentences = nltk.sent_tokenize(abstract, language='dutch')
    total_sents = len(sentences)
    
    if total_sents == 0:
        return abstract

    # Calculate exactly how many sentences to rewrite
    count_to_rewrite = int(round(total_sents * percentage))
    
    # If the percentage is so low it rounds to 0, just return the original
    if count_to_rewrite == 0:
        return abstract

    # Pick random sentence indices to rewrite
    indices_to_rewrite = set(random.sample(range(total_sents), count_to_rewrite))

    # Process the sentences
    new_sentences = []
    for i, sentence in enumerate(sentences):
        if i in indices_to_rewrite:
            new_sent = rewrite_sentence(sentence)
            new_sentences.append(new_sent)
            print(f"[REWRITTEN] {new_sent}")
        else:
            new_sentences.append(sentence)

    # Reconstruct the abstract
    return " ".join(new_sentences)

# ==========================================
# 4. PYARROW / PARQUET HANDLING
# ==========================================
def run_percentage_pipeline(input_path: str, base_output_path: str, column_name: str):
    """Loads the parquet file and runs the rewrite logic for each percentage in the list."""
    
    print(f"Loading {input_path} via PyArrow...")
    table = pq.read_table(input_path)
    original_df = table.to_pandas()
    
    if column_name not in original_df.columns:
        raise ValueError(f"Column '{column_name}' not found in the parquet file.")

    print(f"Total rows to process: {len(original_df)}")

    # Loop through the list of percentages
    for pct in PERCENTAGES_TO_RUN:
        pct_label = int(pct * 100)
        print(f"\n==================================================")
        print(f"🚀 STARTING RUN FOR {pct_label}% REWRITE")
        print(f"==================================================")
        
        # Create a fresh copy of the dataframe for this run
        df_current_run = original_df.copy()
        processed_abstracts = []

        # Process each row
        for idx, row in df_current_run.iterrows():
            print(f"--- Row {idx+1}/{len(df_current_run)} ({pct_label}% run) ---")
            new_text = process_abstract(row[column_name], pct)
            processed_abstracts.append(new_text)
            
        # Update the column with the new data
        df_current_run[column_name] = processed_abstracts
        
        # Format the output filename (e.g., data_10pct.parquet)
        output_name = base_output_path.replace(".parquet", f"_{pct_label}pct.parquet")
        
        # Convert back to PyArrow Table and save
        print(f"\n💾 Saving {pct_label}% dataset to {output_name}...")
        updated_table = pa.Table.from_pandas(df_current_run)
        pq.write_table(updated_table, output_name)
        print(f"✅ Finished saving {output_name}!\n")

# ==========================================
# 5. EXECUTE
# ==========================================
if __name__ == "__main__":
    
    # Replace these with your actual filenames
    INPUT_PARQUET = "dutch_dataset.parquet"
    OUTPUT_PARQUET_BASE = "dutch_dataset_rewritten.parquet" 
    COLUMN_TO_REWRITE = "abstract"
    
    # Uncomment the line below to run!
    # run_percentage_pipeline(INPUT_PARQUET, OUTPUT_PARQUET_BASE, COLUMN_TO_REWRITE)
    pass