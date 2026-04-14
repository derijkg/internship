import ollama
import pandas as pd
from tqdm import tqdm
import time

# --- Configuration ---
OLLAMA_MODEL = 'gemma:2b'  # Use 'gemma:7b' for higher quality, but slower generation
INPUT_FILE = 'human_data.parquet'
OUTPUT_FILE = 'training_data_full.parquet'

# --- Step 1: Create a dummy input file for the example ---
human_texts = [
    "Masterproef ingediend met het oog op het behalen van de graad van Master in de Geneeskunde.",
    "Geweld tegen artsen is een actueel, maar zeker geen nieuw probleem. Met dit onderzoek werd geprobeerd om voor België actuele informatie te bekomen over agressie en geweld tegen artsen binnen de arts-patiëntrelatie.",
    "Voor Patrik Roelandt leek het een normale werkdag te worden. Het was op 1 december 2015 dat de 64-jarige huisarts, gevestigd met zijn solopraktijk te Izegem, aan zijn namiddagronde van de huisbezoeken begon."
]
human_df = pd.DataFrame({
    'text': human_texts,
    'label': 0,
    'source_id': [f"thesis_nl_seg_{i}" for i in range(len(human_texts))]
})
human_df.to_parquet(INPUT_FILE, index=False)
print(f"Created dummy input file: {INPUT_FILE}")


# --- Step 2: Define the rewriting function ---
def rewrite_with_gemma(text_to_rewrite: str, model: str = OLLAMA_MODEL) -> str | None:
    system_prompt = (
        "You are an expert text rephrasing assistant. Your task is to rewrite the "
        "given text while preserving its original meaning, tone, and key information. "
        "Do not add any commentary, preamble, or explanation. "
        "Provide only the rewritten text as your response."
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text_to_rewrite},
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"\nError while processing text: {text_to_rewrite[:50]}...")
        print(f"Error: {e}")
        return None

# --- Step 3: Load data and process in a loop ---
try:
    human_df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(human_df)} human-written text segments from {INPUT_FILE}")
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found. Please create it first.")
    exit()

llm_generated_data = []

for index, row in tqdm(human_df.iterrows(), total=human_df.shape[0], desc="Rewriting texts with Gemma"):
    rewritten_text = rewrite_with_gemma(row['text'])
    if rewritten_text:
        llm_generated_data.append({
            'text': rewritten_text,
            'label': 1,
            'source_id': row['source_id']
        })
    else:
        print(f"Skipping failed rewrite for source_id: {row['source_id']}")

# --- Step 4: Assemble and save the final dataset ---
if llm_generated_data:
    llm_df = pd.DataFrame(llm_generated_data)
    final_df = pd.concat([human_df, llm_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n--- Final Dataset ---")
    print(final_df.head())
    print("\n--- Label Distribution ---")
    print(final_df['label'].value_counts())
    print(f"\n✅ Complete dataset saved to '{OUTPUT_FILE}'")
else:
    print("\nNo LLM data was generated. The final file was not created.")