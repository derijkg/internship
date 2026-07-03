# Dutch Abstract + LLM rewritten Database

This project is part of my internship at the CLiPS research lab at UAntwerpen. It compiles a dataset consisting of Dutch abstracts along with LLM rewrites of individual sentences, full abstracts and different percentages of randomly selected sentences within abstracts done by four models. Three levels of data are created; bronze (raw data), silver (cleaned and selected), and gold (LLM data added). Main orchestration occurs inside run.py.

## Features

- Scraping and downloading:
    scrape.py consists of a base scraper class along with two instances geared toward scraping data from scriptiebank and HBO Kennisbank. Downloading the jsonl datadump from UGent occurs inside run.py.

- Cleaning:
    Inside the main run.py file clean_df will use the DataframeCleaner class inside mu.py to replace placeholder values, duplicates and enforce a strict pyarrow data scheme to remove aberrant values.

- Filtering:
    Cleaning and selection of individual abstracts occurs inside of the select_and_clean_abstracts_ug() function which is also found in run.py. Currently only Dutch abstracts from between the years 1980 and 2022 are kept for generation.

- LLM generation: 
    From the abstracts three types of tasks are generated using a local Ollama server (port: 11435) running gemma4:e4b, gemma4:26b, qwen3.5:4b and qwen3.6:27b. The results are appended to the checkpoint_rewrites.jsonl file in the silver tier of data. When generation is completed all valid results are added to the dataframe using the following column names:
        - model_single: contains a list of all individually rewritten sentences.
        - model_pct: pct has three values: 25, 50 and 75, meaning every model generates three columns. This contains a rewrite of the specified percentage of randomly selected sentences in context of the abstract.
        - model_full: full rewrite of the abstract for the specified model.

## Getting Started

Simply running the run.py file should work.

### Prerequisites

    * Ollama version 0.24.0 was used for compatibility with hardware.
    * Packages used for this project were downloaded using conda


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/derijkg/internship