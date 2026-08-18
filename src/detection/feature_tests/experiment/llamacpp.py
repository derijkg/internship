#TODO change auc to also check low. show all significant features.

#!/usr/bin/env python3
"""
Experiment Scheduler for LLM Sentence & Abstract Trajectory Analysis
Features: High-Performance GGUF Extraction via llama-cpp-python, Multi-GPU Offloading (tensor_split),
          Automated CUDA Preloading, Two-Tier Data Cache Architecture, Base Model Trajectory Evaluation,
          Model Catalog Integration, Per-Model Generation Tracking ({model}_LLM), and Master Summary Report.
"""
from joblib import Parallel, delayed
import os
import sys
import glob
import ctypes
import ast
import zlib  # Deterministic document hashing across process restarts

# =====================================================================
# 0. AUTOMATIC CUDA LIBRARY PRELOADER (Runs before importing llama_cpp)
# =====================================================================
def setup_cuda_libraries():
    """
    1. Preloads Conda's modern libstdc++.so.6 globally to resolve GLIBCXX version 
       conflicts (e.g. pyarrow / scikit-learn).
    2. Preloads all PyPI NVIDIA CUDA shared libraries (.so) into process memory.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    conda_libstdcxx = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if os.path.exists(conda_libstdcxx):
        try:
            ctypes.CDLL(conda_libstdcxx, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

    site_packages = [p for p in sys.path if "site-packages" in p]
    for sp in site_packages:
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_dir):
            for pkg in os.listdir(nvidia_dir):
                lib_dir = os.path.join(nvidia_dir, pkg, "lib")
                if os.path.isdir(lib_dir):
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                    for libfile in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                        try:
                            ctypes.CDLL(libfile, mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass

setup_cuda_libraries()

# Now import llama_cpp safely
from llama_cpp import Llama

import argparse
import gc
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score
import nltk
from datasets import load_dataset
from huggingface_hub import hf_hub_download


# =====================================================================
# 1. MODEL CATALOG & RESOLUTION MANAGER
# =====================================================================
MODEL_CATALOG = {
    "qwen2.5-0.5b-base": { 
        "repo_id": "QuantFactory/Qwen2.5-0.5B-GGUF",
        "filename": "Qwen2.5-0.5B.gguf",
        "description": "Qwen 2.5 0.5B Base (Q8_0)",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": [0.5,0.5],
            "n_ctx": 4096,
            "n_batch": 4096,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "qwen2.5-14b-base": {
        "repo_id": "Qwen/Qwen2.5-14B-GGUF",
        "filename": "qwen2.5-14b-q5_k_m.gguf",
        "description": "Qwen 2.5 14B Base (Q5_K_M) - Top choice for Dutch/multilingual trajectory evaluation (~10.5 GB file).",
        "llama_kwargs": {
            "n_gpu_layers": 60,         
            "tensor_split": [0.5, 0.5], 
            "n_ctx": 4096,
            "n_batch": 4096,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "qwen2.5-32b-baseQ4": {
        "repo_id": "mradermacher/Qwen2.5-32B-GGUF",
        "filename": "Qwen2.5-32B.Q4_K_M.gguf",
        "description": "Qwen 2.5 32B Base (Q4_K_M) - Top-tier 32B foundation model for Dutch/multilingual tasks (~19.8 GB file).",
        "llama_kwargs": {
            "n_gpu_layers": 62,         
            "tensor_split": [0.38, 0.62], 
            "n_ctx": 2048,              
            "n_batch": 512,
            "logits_all": True,         
            "flash_attn": False,        
            "verbose": False
        }
    },
    "qwen2.5-7b-base": {
        "repo_id": "Qwen/Qwen2.5-7B-GGUF",
        "filename": "qwen2.5-7b-q8_0.gguf",
        "description": "Qwen 2.5 7B Base (Q8_0) - High-precision 8-bit baseline (~7.7 GB file).",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": [0.5, 0.5],
            "n_ctx": 4096,
            "n_batch": 4096,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "geitje-7b-ultra": {
        "repo_id": "BramVanroy/GEITje-7B-ultra-GGUF",
        "filename": "geitje-7b-ultra-q5_k_m.gguf",
        "description": "GEITje 7B Ultra (Q5_K_M) - Specialized Dutch language model (~5.1 GB file).",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": [0.5, 0.5],
            "n_ctx": 4096,
            "n_batch": 4096,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "llama-3.1-8b-base": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-GGUF",
        "filename": "Meta-Llama-3.1-8B-Q5_K_M.gguf",
        "description": "Llama 3.1 8B Base (Q5_K_M) - Strong general baseline (~5.7 GB file).",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": [0.5, 0.5],
            "n_ctx": 4096,
            "n_batch": 4096,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "eurollm-9b-base-q6": {
        "repo_id": "QuantFactory/EuroLLM-9B-GGUF",
        "filename": "EuroLLM-9B.Q6_K.gguf",
        "description": "EuroLLM 9B Base Q6_K - Multilingual evaluation model.",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": [0.5, 0.5],
            "n_ctx": 4096,
            "n_batch": 4096,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "eurollm-22b-q6_k": {
        "repo_id": "mradermacher/EuroLLM-22B-2512-GGUF",
        "filename": "EuroLLM-22B-2512.Q6_K.gguf",
        "description": "EuroLLM 22B 2512 (Q6_K) - Near-lossless precision for Dutch & EU multilingual logit analysis (~18.8 GB file).",
        "llama_kwargs": {
            "n_gpu_layers": 52,         
            "tensor_split": [0.48, 0.52], 
            "n_ctx": 2048,              
            "n_batch": 512,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    }
}


def resolve_and_download_model(model_key, models_dir="llama_cpp_models"):
    if model_key not in MODEL_CATALOG:
        valid_keys = list(MODEL_CATALOG.keys())
        raise ValueError(f"Unknown model key '{model_key}'. Valid options are: {valid_keys}")

    spec = MODEL_CATALOG[model_key]
    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, spec["filename"])

    if os.path.exists(local_path):
        print(f"\n[MODEL FOUND] Local GGUF model ready: '{local_path}'")
    else:
        print(f"\n[MODEL DOWNLOAD] Downloading '{spec['filename']}' from '{spec['repo_id']}'...")
        local_path = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=models_dir,
        )
        print(f"[MODEL DOWNLOAD COMPLETE] Saved to '{local_path}'")

    return local_path, spec["llama_kwargs"]


# =====================================================================
# 2. GLOBAL DATA CACHE MANAGER (TIER 1: STUDY-AGNOSTIC)
# =====================================================================
def get_split_cache_paths(config, cache_root="data_cache"):
    model_clean = config["model_name"].replace("/", "_").replace(":", "_")
    
    llm_col_val = config.get("llm_col", "default")
    if isinstance(llm_col_val, (list, tuple)):
        llm_col_clean = "multi_" + "_".join(str(c) for c in llm_col_val)
    else:
        llm_col_clean = str(llm_col_val)
        
    for invalid_char in [":", "/", "[", "]", "'", '"', " ", ","]:
        llm_col_clean = llm_col_clean.replace(invalid_char, "_")
    while "__" in llm_col_clean:
        llm_col_clean = llm_col_clean.replace("__", "_")

    ds_name = config.get("dataset", "dataset")
    eval_unit = config.get("eval_unit", "sentence")
    
    n_samples_cfg = config.get("n_samples")
    is_full = (n_samples_cfg is None) or (n_samples_cfg <= 0)
    sample_tag = "FULL" if is_full else f"SAMPLE_{n_samples_cfg}"
    
    model_dir = os.path.join(cache_root, model_clean)
    os.makedirs(model_dir, exist_ok=True)
    
    human_tok_path = os.path.join(model_dir, f"{ds_name}_HUMAN_sentence_{sample_tag}_tokens.csv")
    llm_tok_path = os.path.join(model_dir, f"{ds_name}_{llm_col_clean}_LLM_{eval_unit}_{sample_tag}_tokens.csv")
    combined_feat_path = os.path.join(model_dir, f"{ds_name}_{llm_col_clean}_{eval_unit}_{sample_tag}_features.csv")
    
    return human_tok_path, llm_tok_path, combined_feat_path


def append_chunk_to_csv(records, csv_path):
    if not records:
        return
    df_chunk = pd.DataFrame(records)
    file_exists = os.path.exists(csv_path)
    df_chunk.to_csv(csv_path, mode='a', index=False, header=not file_exists)


# =====================================================================
# 3. UTILITY & DATA LOADING FUNCTIONS
# =====================================================================

def get_model_n_ctx(llm, default=4096):
    if llm is None:
        return default
    try:
        n_ctx_attr = getattr(llm, "n_ctx", default)
        if callable(n_ctx_attr):
            return int(n_ctx_attr())
        return int(n_ctx_attr)
    except Exception:
        return default


def setup_nltk():
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


def is_valid_sentence(s, min_words):
    if s is None or pd.isna(s):
        return False
    s_str = str(s).strip()
    invalid_flags = {'FAILED_GENERATION', 'FAILED_VALIDATION', 'NONE', 'NAN', 'NA', 'NULL', '<NA>', ''}
    if s_str.upper() in invalid_flags:
        return False
    if len(s_str.split()) < min_words:
        return False
    return True


def parse_sentence_list(val):
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [val_str]
    return [val] if val is not None else []


def resolve_llm_column_name(col_name, is_full_mode, row_keys):
    """
    Dynamically maps base column names to '_single' or '_full' counterparts based on eval_unit.
    """
    col_str = str(col_name)
    if not is_full_mode:
        # Sentence Mode: Target _single
        if col_str.endswith("_full") and col_str.replace("_full", "_single") in row_keys:
            return col_str.replace("_full", "_single")
        return col_str
    else:
        # Full Mode: Target _full
        if col_str.endswith("_single") and col_str.replace("_single", "_full") in row_keys:
            return col_str.replace("_single", "_full")
        elif f"{col_str}_full" in row_keys:
            return f"{col_str}_full"
        elif col_str in row_keys:
            return col_str
        return col_str


def load_data(language, dataset_type, n_samples, min_words, parquet_path=None, llm_col=None, eval_unit="sentence"):
    setup_nltk()
    from nltk.tokenize import sent_tokenize

    human_texts = []  # Tuples: (doc_id, text, model_source)
    llm_texts = []    # Tuples: (doc_id, text, model_source)

    is_full_mode = str(eval_unit).lower() in ["full", "abstract", "document"]

    if dataset_type == "abstracts":
        print(f"Loading Custom Parquet Dataset: {parquet_path} (eval_unit: '{eval_unit}')")
        if not llm_col:
            raise ValueError("Must specify llm_col when dataset='abstracts'.")

        df_parquet = pd.read_parquet(parquet_path)

        if isinstance(llm_col, str):
            target_cols = [llm_col]
        elif isinstance(llm_col, (list, tuple)):
            target_cols = list(llm_col)
        else:
            target_cols = [str(llm_col)]

        for idx, row in df_parquet.iterrows():
            doc_id = row['_id'] if '_id' in row else (row['id'] if 'id' in row else f'doc_{idx}')
            row_keys = set(row.index)
            
            # 1. Parse Human text (Always from abstract_sentence)
            h_sents = parse_sentence_list(row.get("abstract_sentence"))
            valid_h = [str(s).strip() for s in h_sents if is_valid_sentence(s, min_words)]

            if not valid_h:
                continue

            # 2. Select LLM column deterministically per doc_id
            seed = zlib.crc32(str(doc_id).encode("utf-8"))
            rng = np.random.RandomState(seed)
            selected_base_col = rng.choice(target_cols)

            def extract_llm_sents_from_col(base_col):
                actual_col = resolve_llm_column_name(base_col, is_full_mode, row_keys)
                col_val = row.get(actual_col)

                if is_full_mode:
                    # Full Mode: Sentence tokenize raw string from {model}_full
                    if isinstance(col_val, (list, np.ndarray)):
                        raw_text = " ".join(parse_sentence_list(col_val))
                    else:
                        raw_text = str(col_val) if col_val is not None else ""

                    if not raw_text or raw_text.upper() in {'FAILED_GENERATION', 'FAILED_VALIDATION', 'NONE', 'NAN', 'NA', 'NULL', '<NA>'}:
                        return [], actual_col

                    sents = sent_tokenize(raw_text)
                else:
                    # Sentence Mode: Parse list of sentences directly from {model}_single
                    sents = parse_sentence_list(col_val)

                valid_s = [str(s).strip() for s in sents if is_valid_sentence(s, min_words)]
                return valid_s, actual_col

            valid_l, final_col_used = extract_llm_sents_from_col(selected_base_col)

            # Fallback to alternative model columns if chosen column yielded 0 valid sentences
            if not valid_l:
                available_cols = [c for c in target_cols if c != selected_base_col]
                for fallback_col in available_cols:
                    valid_l, final_col_used = extract_llm_sents_from_col(fallback_col)
                    if valid_l:
                        break

            if not valid_l:
                continue

            # 3. Append matched 1-to-1 sentence samples
            min_sents = min(len(valid_h), len(valid_l))
            for s_i in range(min_sents):
                human_texts.append((doc_id, valid_h[s_i], "Human"))
                llm_texts.append((doc_id, valid_l[s_i], final_col_used))

    elif dataset_type == "multitude":
        print(f"Loading MULTITuDE Dataset for Language: [{language.upper()}] (eval_unit: '{eval_unit}')")

        # 1. Load MULTITuDE dataset (from parquet_path if specified, or via Hugging Face)
        df_mult = None
        if parquet_path and os.path.exists(parquet_path):
            print(f"Loading local MULTITuDE file: {parquet_path}")
            if parquet_path.endswith(".csv"):
                df_mult = pd.read_csv(parquet_path)
            else:
                df_mult = pd.read_parquet(parquet_path)
        else:
            print("Loading MULTITuDE dataset from Hugging Face...")
            try:
                ds = load_dataset("kinit/multitude", split="train")
                df_mult = pd.DataFrame(ds)
            except Exception as e1:
                try:
                    ds = load_dataset("multitude", split="train")
                    df_mult = pd.DataFrame(ds)
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load MULTITuDE from Hugging Face ({e1}; {e2}). "
                        f"If using a local file, please set a valid 'parquet_path' in your study config."
                    )

        # 2. Filter by Language if language column exists
        lang_col = next((c for c in ["language", "lang", "locale"] if c in df_mult.columns), None)
        if lang_col:
            lang_str = str(language).lower().strip()
            dutch_aliases = {"dutch", "nl", "nl_nl", "nl-nl"}
            if lang_str in dutch_aliases:
                df_mult = df_mult[df_mult[lang_col].astype(str).str.lower().isin(dutch_aliases)].copy()
            else:
                df_mult = df_mult[df_mult[lang_col].astype(str).str.lower() == lang_str].copy()
            print(f"Filtered MULTITuDE to language '{language}': {len(df_mult)} records remaining.")

        if df_mult.empty:
            raise ValueError(f"Extracted 0 records from MULTITuDE for language '{language}'.")

        row_keys = set(df_mult.columns)

        # 3. Check dataset structure (Paired columns vs Unrolled rows)
        has_paired_cols = ("human_text" in row_keys or "human" in row_keys or "abstract_sentence" in row_keys) and \
                          (llm_col or "machine_text" in row_keys or "generated_text" in row_keys or "llm_text" in row_keys)

        if has_paired_cols:
            # PAIRED COLUMNS STRUCTURE (e.g. human text and LLM text on the same row)
            target_h_col = "abstract_sentence" if "abstract_sentence" in row_keys else ("human_text" if "human_text" in row_keys else "human")
            
            if llm_col:
                if isinstance(llm_col, str):
                    target_m_cols = [llm_col]
                elif isinstance(llm_col, (list, tuple)):
                    target_m_cols = list(llm_col)
                else:
                    target_m_cols = [str(llm_col)]
            else:
                target_m_cols = [c for c in ["machine_text", "generated_text", "llm_text", "gpt-4", "chatgpt"] if c in row_keys]

            for idx, row in df_mult.iterrows():
                doc_id = str(row.get('_id', row.get('id', f'mult_{idx}')))

                h_val = row.get(target_h_col)
                h_sents = parse_sentence_list(h_val) if isinstance(h_val, (list, np.ndarray)) or (isinstance(h_val, str) and h_val.startswith('[')) else (sent_tokenize(str(h_val)) if h_val and not pd.isna(h_val) else [])
                valid_h = [str(s).strip() for s in h_sents if is_valid_sentence(s, min_words)]

                if not valid_h:
                    continue

                seed = zlib.crc32(str(doc_id).encode("utf-8"))
                rng = np.random.RandomState(seed)
                selected_base_col = rng.choice(target_m_cols) if target_m_cols else list(row_keys)[0]

                def extract_llm_sents_from_col_mult(base_col):
                    actual_col = resolve_llm_column_name(base_col, is_full_mode, row_keys)
                    col_val = row.get(actual_col)
                    if col_val is None or pd.isna(col_val):
                        return [], actual_col
                    if is_full_mode:
                        raw_text = " ".join(parse_sentence_list(col_val)) if isinstance(col_val, (list, np.ndarray)) else str(col_val)
                        sents = sent_tokenize(raw_text)
                    else:
                        sents = parse_sentence_list(col_val) if isinstance(col_val, (list, np.ndarray)) or (isinstance(col_val, str) and str(col_val).startswith('[')) else sent_tokenize(str(col_val))
                    valid_s = [str(s).strip() for s in sents if is_valid_sentence(s, min_words)]
                    return valid_s, actual_col

                valid_l, final_col_used = extract_llm_sents_from_col_mult(selected_base_col)
                if not valid_l:
                    for fallback_col in [c for c in target_m_cols if c != selected_base_col]:
                        valid_l, final_col_used = extract_llm_sents_from_col_mult(fallback_col)
                        if valid_l:
                            break

                if not valid_l:
                    continue

                min_sents = min(len(valid_h), len(valid_l))
                for s_i in range(min_sents):
                    human_texts.append((doc_id, valid_h[s_i], "Human"))
                    llm_texts.append((doc_id, valid_l[s_i], final_col_used))

        else:
            # UNROLLED MULTITUDE STRUCTURE (Standard: 1 text per row with model/label columns)
            text_col = next((c for c in ["text", "content", "document", "abstract"] if c in row_keys), "text")
            model_col = next((c for c in ["model", "generator", "src", "generated_by"] if c in row_keys), None)
            label_col = next((c for c in ["label", "is_machine", "is_human", "generated"] if c in row_keys), None)
            id_col = next((c for c in ["id", "doc_id", "_id", "item_id"] if c in row_keys), None)

            target_llm_models = None
            if llm_col:
                if isinstance(llm_col, str):
                    target_llm_models = [llm_col.lower()]
                elif isinstance(llm_col, (list, tuple)):
                    target_llm_models = [str(c).lower() for c in llm_col]

            doc_groups = defaultdict(lambda: {"human": [], "llm": []})

            for idx, row in df_mult.iterrows():
                raw_id = row.get(id_col) if id_col else idx
                doc_id = str(raw_id).split("_")[0] if raw_id is not None else f"mult_{idx}"
                text_val = str(row.get(text_col, "")).strip()

                if not text_val or pd.isna(text_val):
                    continue

                sents = sent_tokenize(text_val)
                valid_sents = [s.strip() for s in sents if is_valid_sentence(s, min_words)]
                if not valid_sents:
                    continue

                is_human = False
                model_name = "LLM"

                if model_col and row.get(model_col):
                    m_val = str(row.get(model_col)).strip()
                    m_val_lower = m_val.lower()
                    if m_val_lower in ["human", "massivesumm", "original", "source", "0"]:
                        is_human = True
                        model_name = "Human"
                    else:
                        model_name = m_val
                elif label_col is not None:
                    l_val = row.get(label_col)
                    if str(l_val).lower() in ["human", "0", "false", "original"]:
                        is_human = True
                        model_name = "Human"
                    else:
                        model_name = str(l_val)
                else:
                    is_human = (idx % 2 == 0)

                if not is_human and target_llm_models:
                    if not any(tm in model_name.lower() for tm in target_llm_models):
                        continue

                if is_human:
                    doc_groups[doc_id]["human"].extend([(s, "Human") for s in valid_sents])
                else:
                    doc_groups[doc_id]["llm"].extend([(s, model_name) for s in valid_sents])

            for d_id, group in doc_groups.items():
                h_list = group["human"]
                l_list = group["llm"]

                if h_list and l_list:
                    min_len = min(len(h_list), len(l_list))
                    for i in range(min_len):
                        human_texts.append((d_id, h_list[i][0], h_list[i][1]))
                        llm_texts.append((d_id, l_list[i][0], l_list[i][1]))
                else:
                    for s_text, m_src in h_list:
                        human_texts.append((d_id, s_text, m_src))
                    for s_text, m_src in l_list:
                        llm_texts.append((d_id, s_text, m_src))


    elif dataset_type == "clin33":
        print(f"Loading CLIN33 LLM Texts + Abstracts Human Baseline (eval_unit: '{eval_unit}')")

        clin33_csv_path = parquet_path
        if not clin33_csv_path or not os.path.exists(clin33_csv_path):
            raise ValueError(
                f"CLIN33 CSV file not found at 'parquet_path': {clin33_csv_path}"
            )

        # Default to your gold abstracts parquet if abstracts_path is not specified
        human_parquet_path =  "/home/gderijck/internship/data/gold/llm_added.parquet"

        # -------------------------------------------------------------
        # 1. EXTRACT LLM TEXTS ONLY FROM CLIN33 CSV
        # -------------------------------------------------------------
        print(f"Reading CLIN33 LLM texts from: {clin33_csv_path}")
        try:
            df_clin = pd.read_csv(clin33_csv_path)
        except Exception:
            df_clin = pd.read_csv(clin33_csv_path, engine="python", on_bad_lines="skip")

        # Strip unnamed index column (leading comma)
        df_clin = df_clin.loc[:, ~df_clin.columns.str.contains('^Unnamed|^$')].copy()
        row_keys = set(df_clin.columns)

        gen_col_name = next((c for c in ["generated_text", "machine_text", "llm_text", "text", "content"] if c in row_keys), df_clin.columns[0])
        genre_col_name = next((c for c in ["genre", "model", "generator", "category"] if c in row_keys), None)

        target_llm_models = None
        if llm_col:
            if isinstance(llm_col, str):
                target_llm_models = [llm_col.lower()]
            elif isinstance(llm_col, (list, tuple)):
                target_llm_models = [str(c).lower() for c in llm_col]

        for idx, row in df_clin.iterrows():
            doc_id = f"clin_llm_{idx}"
            raw_text = str(row.get(gen_col_name, "")).strip()
            genre_val = str(row.get(genre_col_name, "LLM")).strip() if genre_col_name else "LLM"

            if not raw_text or pd.isna(raw_text) or raw_text.lower() in ["nan", "none", "null"]:
                continue

            # Optional filter if llm_col is specified in config (e.g. "News")
            if target_llm_models:
                if not any(tm in genre_val.lower() for tm in target_llm_models):
                    continue

            sents = sent_tokenize(raw_text)
            valid_s = [s.strip() for s in sents if is_valid_sentence(s, min_words)]

            for s in valid_s:
                llm_texts.append((doc_id, s, f"CLIN33_{genre_val}"))

        print(f"-> Extracted {len(llm_texts)} LLM sentences from CLIN33.")

        # -------------------------------------------------------------
        # 2. EXTRACT HUMAN TEXTS ONLY FROM CUSTOM ABSTRACTS PARQUET
        # -------------------------------------------------------------
        if not os.path.exists(human_parquet_path):
            raise FileNotFoundError(
                f"Could not find Human baseline dataset at 'abstracts_path': {human_parquet_path}"
            )

        print(f"Reading Human baseline text from Abstracts dataset: {human_parquet_path}")
        df_abs = pd.read_parquet(human_parquet_path)

        for idx, row in df_abs.iterrows():
            doc_id = row['_id'] if '_id' in row else (row['id'] if 'id' in row else f'abs_hum_{idx}')
            h_sents = parse_sentence_list(row.get("abstract_sentence"))
            valid_h = [str(s).strip() for s in h_sents if is_valid_sentence(s, min_words)]

            for s in valid_h:
                human_texts.append((doc_id, s, "Human"))

        print(f"-> Extracted {len(human_texts)} Human sentences from Abstracts dataset.")

    else:
        print(f"Loading Standard Dataset for Language: [{language.upper()}]")
        try:
            ds = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")
        except Exception:
            ds = load_dataset("Hello-SimpleAI/HC3", revision="refs/convert/parquet", split="train")

        for entry_idx, entry in enumerate(ds):
            doc_id = entry.get("_id", entry.get("id", f"hc3_{entry_idx}"))
            for ans in entry["human_answers"]:
                for s in sent_tokenize(ans):
                    if is_valid_sentence(s, min_words):
                        human_texts.append((doc_id, s.strip(), "Human"))
            for ans in entry["chatgpt_answers"]:
                for s in sent_tokenize(ans):
                    if is_valid_sentence(s, min_words):
                        llm_texts.append((doc_id, s.strip(), "chatgpt"))

    if len(human_texts) == 0 or len(llm_texts) == 0:
        raise ValueError(f"Extracted 0 valid texts! (Human: {len(human_texts)}, LLM: {len(llm_texts)})")

    np.random.seed(42)
    if n_samples is not None and n_samples > 0:
        sample_h_size = min(n_samples, len(human_texts))
        sample_l_size = min(n_samples, len(llm_texts))
        
        idx_h = np.random.choice(len(human_texts), sample_h_size, replace=False)
        idx_l = np.random.choice(len(llm_texts), sample_l_size, replace=False)
        
        human_sample = [human_texts[i] for i in idx_h]
        llm_sample = [llm_texts[i] for i in idx_l]
        print(f"Sampled {len(human_sample)} Human and {len(llm_sample)} LLM [{eval_unit.upper()}] sentence samples.")
    else:
        print(f"Using ALL available data ({len(human_texts)} Human, {len(llm_texts)} LLM [{eval_unit.upper()}] sentence samples).")
        human_sample = human_texts
        llm_sample = llm_texts

    return human_sample, llm_sample


# =====================================================================
# 4. LLAMA.CPP TRAJECTORY EXTRACTION FUNCTION
# =====================================================================
def compute_vectorized_gini(probs):
    """
    Computes standard Gini coefficient for probability distributions.
    Supports both 1D and 2D arrays.
    """
    probs = np.atleast_2d(probs)
    M, V = probs.shape
    sorted_probs = np.sort(probs, axis=-1)
    index = np.arange(1, V + 1, dtype=np.float32)
    weights = (V - index + 0.5) / V
    gini = 1.0 - 2.0 * np.sum(sorted_probs * weights, axis=-1)
    return gini.squeeze() if M == 1 else gini


def compute_zipf_exponent(v_logits, top_k=20):
    """
    Computes Zipf's exponent alpha by regressing top-k logits (log-probabilities
    up to an additive constant) against log-ranks.
    """
    v_logits = np.atleast_2d(v_logits)
    M, V = v_logits.shape
    actual_k = min(top_k, V)
    
    # Select top-k logits per token
    topk_logits = np.partition(v_logits, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_logits, axis=-1)[:, ::-1]  # z_1 >= z_2 >= ... >= z_k
    
    log_ranks = np.log(np.arange(1, actual_k + 1, dtype=np.float32))
    
    mean_x = np.mean(log_ranks)
    var_x = np.var(log_ranks)
    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)
    
    # Linear regression slope: z_r = C - alpha * ln(r)
    cov_xy = np.mean((log_ranks - mean_x) * (sorted_topk - mean_y), axis=-1)
    zipf_alpha = -cov_xy / (var_x + 1e-8)
    return zipf_alpha


def extract_trajectory_llama_cpp(
    text, 
    doc_id, 
    label_prefix, 
    sentence_id, 
    llm, 
    model_source="LLM", 
    max_tokens=2048,
    unigram_log_probs=None  # Added missing parameter to avoid NameError
):
    text_clean = text.strip()
    tokens = llm.tokenize(text_clean.encode("utf-8"))
    
    bos_id = llm.token_bos()
    eos_id = llm.token_eos()
    start_id = bos_id if (bos_id is not None and bos_id != -1) else eos_id

    if start_id is not None and start_id != -1:
        if len(tokens) == 0 or tokens[0] != start_id:
            tokens = [start_id] + tokens

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    if len(tokens) < 3:
        return [], sentence_id + 1

    llm.reset()
    llm.eval(tokens)

    logits = np.array(llm.eval_logits, dtype=np.float32)
    shift_logits = logits[:-1, :]
    shift_labels = np.array(tokens[1:], dtype=np.int64)

    special_ids = {tid for tid in (bos_id, eos_id) if tid is not None and tid != -1}
    n_vocab = llm.n_vocab()

    valid_mask = np.array([
        (tok not in special_ids) and (tok < n_vocab)
        for tok in shift_labels
    ], dtype=bool)

    valid_positions = np.where(valid_mask)[0]
    total_valid_tokens = len(valid_positions)

    if total_valid_tokens < 2:
        return [], sentence_id + 1

    v_logits = shift_logits[valid_positions]
    v_labels = shift_labels[valid_positions]

    lse = scipy.special.logsumexp(v_logits, axis=-1, keepdims=True)
    log_probs = v_logits - lse
    probs = np.exp(log_probs)

    raw_log_probs = log_probs[np.arange(total_valid_tokens), v_labels]
    surprisal = -raw_log_probs
    entropies = -np.sum(probs * log_probs, axis=-1)

    entropy_norm_score = entropies - surprisal
    surprisal_entropy_gap = surprisal - entropies

    # Top-1 and Top-2 probabilities with safe partition
    top_k_partition = min(2, n_vocab)
    top2_logits = np.partition(v_logits, -top_k_partition, axis=-1)[:, -top_k_partition:]
    p_top1 = np.exp(top2_logits[:, -1] - lse.squeeze(-1))
    p_top2 = np.exp(top2_logits[:, -2] - lse.squeeze(-1)) if top_k_partition >= 2 else np.zeros_like(p_top1)
    
    min_entropies = -np.log(p_top1 + 1e-12)
    renyi2_entropies = -np.log(np.sum(probs ** 2, axis=-1) + 1e-12)

    target_logits = v_logits[np.arange(total_valid_tokens), v_labels, None]
    ranks = np.sum(v_logits > target_logits, axis=-1) + 1
    cdf_mass = np.sum(np.where(v_logits >= target_logits, probs, 0.0), axis=-1)

    eff_vocab = np.exp(entropies)
    rank_eff_ratio = ranks / (eff_vocab + 1e-8)
    logit_std = np.std(v_logits, axis=-1)
    top1_top2_margins = p_top1 - p_top2

    gini_coefs = compute_vectorized_gini(probs)
    zipf_alphas = compute_zipf_exponent(v_logits, top_k=20)
    
    # Safe top-50 partitioning for variable vocabulary size
    k50 = min(50, n_vocab)
    top50_p = np.partition(probs, -k50, axis=-1)[:, -k50:]
    sorted_top50_p = np.sort(top50_p, axis=-1)
    
    k5 = min(5, n_vocab)
    k10 = min(10, n_vocab)
    top5_mass = np.sum(sorted_top50_p[:, -k5:], axis=-1)
    top10_mass = np.sum(sorted_top50_p[:, -k10:], axis=-1)
    top50_mass = np.sum(sorted_top50_p, axis=-1)
    concentration_gradient = top5_mass / (top50_mass + 1e-8)

    logit_skewness = scipy.stats.skew(v_logits, axis=-1)
    logit_kurtosis = scipy.stats.kurtosis(v_logits, axis=-1)

    # Unigram Prior Surprisal & IGR
    if unigram_log_probs is not None:
        unigram_prior_surprisal = -unigram_log_probs[v_labels]
    else:
        # Fallback approximation: Uniform maximum entropy prior
        unigram_prior_surprisal = np.full_like(surprisal, np.log(n_vocab))
    
    unigram_igr = (unigram_prior_surprisal - surprisal) / (unigram_prior_surprisal + 1e-8)
    bci = gini_coefs * (1.0 - top1_top2_margins)

    # Zipf Anomaly including LogSumExp normalization constant Z(alpha)
    k_ranks = np.arange(1, n_vocab + 1, dtype=np.float64)
    log_k = np.log(k_ranks)
    log_z = scipy.special.logsumexp(-zipf_alphas[:, None] * log_k[None, :], axis=-1)
    predicted_zipf_surprisal = zipf_alphas * np.log(ranks) + log_z
    zipf_anomaly = np.abs(surprisal - predicted_zipf_surprisal)

    # Concentration-Diversity Gap
    max_entropy = np.log(n_vocab)
    norm_entropy = entropies / max_entropy
    gini_entropy_gap = gini_coefs - norm_entropy

    if total_valid_tokens >= 3:
        acc_vals = np.diff(surprisal, n=2)
        surprisal_acc = np.pad(acc_vals, (2, 0), mode='edge')
    else:
        surprisal_acc = np.zeros(total_valid_tokens, dtype=np.float32)

    sid = f"{label_prefix}_{sentence_id}"
    label_name = "Human" if str(model_source).upper() == "HUMAN" else (f"{model_source}_LLM" if not str(model_source).endswith("_LLM") else str(model_source))

    records = [
        {
            "token_pos": idx + 1,
            "norm_pos": (idx + 1) / total_valid_tokens,
            "raw_log_prob": float(raw_log_probs[idx]),
            "surprisal": float(surprisal[idx]),
            "entropy": float(entropies[idx]),
            "entropy_norm_score": float(entropy_norm_score[idx]),
            "surprisal_entropy_gap": float(surprisal_entropy_gap[idx]),
            "min_entropy": float(min_entropies[idx]),
            "renyi2_entropy": float(renyi2_entropies[idx]),
            "cdf_mass": float(cdf_mass[idx]),
            "rank_eff_ratio": float(rank_eff_ratio[idx]),
            "logit_std": float(logit_std[idx]),
            "gini_coef": float(gini_coefs[idx]),
            "zipf_alpha": float(zipf_alphas[idx]),
            "top5_mass": float(top5_mass[idx]),
            "top10_mass": float(top10_mass[idx]),
            "top50_mass": float(top50_mass[idx]),
            "logit_skewness": float(logit_skewness[idx]),
            "logit_kurtosis": float(logit_kurtosis[idx]),
            "unigram_igr": float(unigram_igr[idx]),
            "rank": int(ranks[idx]),
            "log_rank": float(np.log(ranks[idx])),
            "top1_top2_margin": float(top1_top2_margins[idx]),
            "surprisal_acc": float(surprisal_acc[idx]),
            "bci": float(bci[idx]),
            "concentration_gradient": float(concentration_gradient[idx]),
            "zipf_anomaly": float(zipf_anomaly[idx]),
            "gini_entropy_gap": float(gini_entropy_gap[idx]),
            "sentence_id": sid,
            "_id": doc_id,
            "generator_model": model_source,
            "label": label_name
        }
        for idx in range(total_valid_tokens)
    ]

    return records, sentence_id + 1




def calc_slope(x, y):
    if len(x) >= 2 and np.std(x) > 1e-8:
        return float(scipy.stats.linregress(x, y).slope)
    return 0.0


def extract_positional_and_spectral_features(norm_pos, raw_log_prob, entropy, num_bins=10):
    features = {}
    target_bins = np.linspace(0.1, 1.0, num_bins)
    
    lp_interpolated = np.interp(target_bins, norm_pos, raw_log_prob)
    ent_interpolated = np.interp(target_bins, norm_pos, entropy)
    
    for i in range(num_bins):
        features[f"lp_step_{i+1:02d}"] = float(lp_interpolated[i])
        features[f"ent_step_{i+1:02d}"] = float(ent_interpolated[i])
        
    if len(raw_log_prob) >= 4:
        fft_raw = np.fft.rfft(raw_log_prob - np.mean(raw_log_prob))[1:] 
        fft_vals = np.abs(fft_raw)
        
        # Compute Energy using Power Spectrum (|X[k]|^2)
        power_spectrum = (fft_vals ** 2)
        
        # FIX: Dynamically partition frequency spectrum into low and high bands
        mid = max(1, len(power_spectrum) // 2)
        
        features["fft_low_freq_energy"] = float(np.sum(power_spectrum[:mid]))
        features["fft_high_freq_energy"] = float(np.sum(power_spectrum[mid:]))
        features["fft_spectral_ratio"] = float(features["fft_high_freq_energy"] / (features["fft_low_freq_energy"] + 1e-8))
        
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        nonzero_p = power_norm[power_norm > 0]
        features["fft_spectral_entropy"] = float(-np.sum(nonzero_p * np.log(nonzero_p)))
    else:
        features["fft_low_freq_energy"] = 0.0
        features["fft_high_freq_energy"] = 0.0
        features["fft_spectral_ratio"] = 0.0
        features["fft_spectral_entropy"] = 0.0
        
    return features


def _process_single_sentence_group(sid, group, text_map=None, log_base=np.e):
    label = group["label"].iloc[0]
    generator_model = group["generator_model"].iloc[0] if "generator_model" in group.columns else ("Human" if label == "Human" else "LLM")
    is_llm = 0 if (label == "Human" or str(generator_model).upper() == "HUMAN") else 1

    doc_id = group['_id'].iloc[0]
    group = group.sort_values("token_pos")
    
    norm_pos = group["norm_pos"].values
    log_rank = group["log_rank"].values
    
    if "rank" in group.columns:
        ranks = group["rank"].values
    else:
        ranks = log_base ** log_rank if log_base != np.e else np.exp(log_rank)

    raw_log_prob = group["raw_log_prob"].values
    surprisal = group["surprisal"].values
    entropy = group["entropy"].values
    length = len(group)

    # Standardized sample standard deviation (ddof=1)
    mean_e = float(np.mean(entropy))
    std_e = float(np.std(entropy, ddof=1)) if length > 1 else 0.0

    entropy_surprisal_diff = group["entropy_norm_score"].values if "entropy_norm_score" in group.columns else (entropy - surprisal)
    mean_entropy_surprisal_diff = float(np.mean(entropy_surprisal_diff))
    std_entropy_surprisal_diff = float(np.std(entropy_surprisal_diff, ddof=1)) if length > 1 else 0.0
    p25_entropy_surprisal_diff = float(np.percentile(entropy_surprisal_diff, 25))
    p75_entropy_surprisal_diff = float(np.percentile(entropy_surprisal_diff, 75))
    iqr_entropy_surprisal_diff = float(p75_entropy_surprisal_diff - p25_entropy_surprisal_diff)

    diff_entropy = np.diff(entropy) if length > 1 else np.array([0.0])
    mean_abs_diff_entropy = float(np.mean(np.abs(diff_entropy))) if length > 1 else 0.0

    if length >= 3:
        std_diff_entropy = float(np.std(diff_entropy, ddof=1))
        volatility_log_rank = float(np.var(np.diff(log_rank), ddof=1))
        volatility_log_prob = float(np.var(np.diff(raw_log_prob), ddof=1))
    else:
        std_diff_entropy = 0.0
        volatility_log_rank = 0.0
        volatility_log_prob = 0.0

    centered_entropy = entropy - mean_e
    zero_crossings = np.where(np.diff(centered_entropy >= 0))[0] if length > 1 else np.array([])
    entropy_mean_crossing_rate = float(len(zero_crossings) / (length - 1)) if length > 1 else 0.0

    # Leave-One-Out (LOO) rolling shocks & absolute magnitude max
    if length >= 5:
        surp_series = pd.Series(surprisal)
        w = 5
        roll_sum = surp_series.rolling(window=w, min_periods=w, center=True).sum()
        roll_sq_sum = (surp_series**2).rolling(window=w, min_periods=w, center=True).sum()
        
        # Calculate leave-one-out context mean and std (excluding token i)
        loo_mean = (roll_sum - surp_series) / (w - 1)
        loo_var = (roll_sq_sum - surp_series**2 - (w - 1) * (loo_mean**2)) / (w - 2)
        loo_std = np.sqrt(np.maximum(loo_var, 0.0))
        
        local_shocks = ((surp_series - loo_mean) / (loo_std + 1e-8)).dropna().values
        
        if len(local_shocks) > 0:
            max_local_surprisal_shock = float(np.max(np.abs(local_shocks)))
            mean_local_surprisal_shock = float(np.mean(np.abs(local_shocks)))
        else:
            max_local_surprisal_shock, mean_local_surprisal_shock = 0.0, 0.0
    else:
        max_local_surprisal_shock, mean_local_surprisal_shock = 0.0, 0.0

    mean_bci = float(np.mean(group["bci"].values)) if "bci" in group.columns else 0.0
    mean_concentration_gradient = float(np.mean(group["concentration_gradient"].values)) if "concentration_gradient" in group.columns else 0.0
    mean_zipf_anomaly = float(np.mean(group["zipf_anomaly"].values)) if "zipf_anomaly" in group.columns else 0.0
    mean_gini_entropy_div = float(np.mean(group["gini_entropy_div"].values)) if "gini_entropy_div" in group.columns else 0.0

    if length >= 4 and std_e > 1e-8:
        local_stds = [np.std(entropy[i:i+3], ddof=1) for i in range(length - 2)]
        mean_local_entropy_std = float(np.mean(local_stds))
        local_global_std_ratio = float(mean_local_entropy_std / (std_e + 1e-8))
    else:
        mean_local_entropy_std = 0.0
        local_global_std_ratio = 1.0 if length < 4 else 0.0

    # Sample Covariance and Correlation (ddof=1)
    if length >= 3 and std_e > 1e-8 and np.std(surprisal, ddof=1) > 1e-8:
        cov_matrix = np.cov(surprisal, entropy, ddof=1)
        surprisal_entropy_cov = float(cov_matrix[0, 1])
        surprisal_entropy_corr = float(np.nan_to_num(np.corrcoef(surprisal, entropy)[0, 1], nan=0.0))
    else:
        surprisal_entropy_cov = 0.0
        surprisal_entropy_corr = 0.0

    # Markov Regime Entropy Rate
    if length >= 4:
        states = np.digitize(entropy, bins=[1.0, 3.0])
        trans_matrix = np.zeros((3, 3), dtype=np.float64)
        
        for i_st, j_st in zip(states[:-1], states[1:]):
            trans_matrix[i_st, j_st] += 1.0
            
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        total_transitions = trans_matrix.sum()
        
        if total_transitions > 0:
            pi = (row_sums / total_transitions).squeeze(-1)
            trans_probs = np.zeros_like(trans_matrix)
            np.divide(trans_matrix, row_sums, out=trans_probs, where=row_sums > 0)
            
            row_entropies = np.zeros(3, dtype=np.float64)
            for i in range(3):
                p_row = trans_probs[i]
                nonzero_p = p_row[p_row > 0]
                if len(nonzero_p) > 0:
                    row_entropies[i] = -np.sum(nonzero_p * np.log2(nonzero_p))
            
            markov_regime_entropy = float(np.sum(pi * row_entropies))
        else:
            markov_regime_entropy = 0.0
    else:
        markov_regime_entropy = 0.0

    p25_entropy = float(np.percentile(entropy, 25))
    p75_entropy = float(np.percentile(entropy, 75))
    iqr_entropy = p75_entropy - p25_entropy
    median_e = float(np.median(entropy))
    iqr_entropy_ratio = float(iqr_entropy / (median_e + 1e-8))

    p25_log_prob = float(np.percentile(raw_log_prob, 25))
    p75_log_prob = float(np.percentile(raw_log_prob, 75))
    
    surprisal_skew = float(np.nan_to_num(scipy.stats.skew(surprisal), nan=0.0)) if length >= 3 else 0.0
    surprisal_kurtosis = float(np.nan_to_num(scipy.stats.kurtosis(surprisal), nan=0.0)) if length >= 4 else 0.0
    entropy_skew = float(np.nan_to_num(scipy.stats.skew(entropy), nan=0.0)) if length >= 3 else 0.0

    surprisal_var = float(np.var(surprisal, ddof=1)) if length > 1 else 0.0
    surprisal_mean = float(np.mean(surprisal))
    fano_factor = float(np.nan_to_num(surprisal_var / (surprisal_mean + 1e-8), nan=0.0))

    cdf_vals = group["cdf_mass"].values if "cdf_mass" in group.columns else np.zeros(length)
    mean_cdf = float(np.mean(cdf_vals))
    tail_breach_90 = float(np.mean(cdf_vals > 0.90))
    tail_breach_95 = float(np.mean(cdf_vals > 0.95))

    min_e = group["min_entropy"].values if "min_entropy" in group.columns else np.zeros(length)
    renyi2_e = group["renyi2_entropy"].values if "renyi2_entropy" in group.columns else np.zeros(length)
    min_shannon_ratio = float(np.nan_to_num(np.mean(min_e / (entropy + 1e-8)), nan=0.0)) if "min_entropy" in group.columns else 0.0
    renyi2_shannon_ratio = float(np.nan_to_num(np.mean(renyi2_e / (entropy + 1e-8)), nan=0.0)) if "renyi2_entropy" in group.columns else 0.0

    entropy_spike_ratio = float(np.mean(entropy > (mean_e + 1.5 * std_e))) if std_e > 1e-8 else 0.0
    
    if std_e > 1e-8 and length >= 4:
        ac = float(np.corrcoef(entropy[:-1], entropy[1:])[0, 1])
        entropy_autocorr = 0.0 if np.isnan(ac) else ac
    else:
        entropy_autocorr = 0.0

    if length >= 3 and std_e > 1e-8:
        h_centered = entropy - mean_e
        phi1 = np.sum(h_centered[1:] * h_centered[:-1]) / (np.sum(h_centered[:-1] ** 2) + 1e-8)
        ar1_residuals = h_centered[1:] - phi1 * h_centered[:-1]
        ar1_residual_var = float(np.var(ar1_residuals, ddof=1)) if len(ar1_residuals) > 1 else 0.0
    else:
        ar1_residual_var = 0.0

    terminal_mask = norm_pos >= 0.70
    terminal_entropy_slope = calc_slope(norm_pos[terminal_mask], entropy[terminal_mask]) if np.sum(terminal_mask) >= 2 else 0.0

    margins = group["top1_top2_margin"].values if "top1_top2_margin" in group.columns else np.zeros(length)
    p90_margin = float(np.percentile(margins, 90)) if length > 0 else 0.0
    max_rank = float(np.max(ranks)) if length > 0 else 1.0
    bimodal_extreme_index = float((max_rank * p90_margin) / (mean_e + 1e-8))

    entropy_texture_index = float((np.abs(surprisal_kurtosis) * entropy_autocorr) / (std_e + 1e-8))
    surprisal_jitter_index = float(np.mean(np.abs(np.diff(surprisal)))) if length > 1 else 0.0

    mean_gini_coef = float(np.mean(group["gini_coef"].values)) if "gini_coef" in group.columns else 0.0
    mean_zipf_alpha = float(np.mean(group["zipf_alpha"].values)) if "zipf_alpha" in group.columns else 0.0
    mean_top5_mass = float(np.mean(group["top5_mass"].values)) if "top5_mass" in group.columns else 0.0
    mean_top10_mass = float(np.mean(group["top10_mass"].values)) if "top10_mass" in group.columns else 0.0
    mean_top50_mass = float(np.mean(group["top50_mass"].values)) if "top50_mass" in group.columns else 0.0
    mean_logit_skewness = float(np.mean(group["logit_skewness"].values)) if "logit_skewness" in group.columns else 0.0
    mean_logit_kurtosis = float(np.mean(group["logit_kurtosis"].values)) if "logit_kurtosis" in group.columns else 0.0
    mean_unigram_igr = float(np.mean(group["unigram_igr"].values)) if "unigram_igr" in group.columns else 0.0

    head_mask, tail_mask = norm_pos <= 0.25, norm_pos > 0.75
    head_lp = float(np.mean(raw_log_prob[head_mask])) if np.any(head_mask) else float(np.mean(raw_log_prob))
    tail_lp = float(np.mean(raw_log_prob[tail_mask])) if np.any(tail_mask) else float(np.mean(raw_log_prob))

    traj_features = extract_positional_and_spectral_features(norm_pos, raw_log_prob, entropy)

    # FIX: Safely compute optional column metrics to prevent KeyError crashes
    rank_eff_ratio = float(np.mean(group["rank_eff_ratio"].values)) if "rank_eff_ratio" in group.columns else 0.0
    mean_logit_std = float(np.mean(group["logit_std"].values)) if "logit_std" in group.columns else 0.0
    
    mean_margin = float(np.mean(margins)) if length > 0 else 0.0
    std_margin = float(np.std(margins, ddof=1)) if length > 1 else 0.0
    
    surp_acc_vals = group["surprisal_acc"].values if "surprisal_acc" in group.columns else np.zeros(length)
    mean_surp_acc = float(np.mean(surp_acc_vals)) if length > 0 else 0.0
    std_surp_acc = float(np.std(surp_acc_vals, ddof=1)) if length > 1 else 0.0

    return {
        "sentence_id": sid,
        '_id': doc_id,
        "label": label,                      
        "generator_model": generator_model,  
        "is_llm": is_llm,                    
        "token_length": length,
        
        "mean_log_rank": np.mean(log_rank),
        "std_log_rank": np.std(log_rank, ddof=1) if length > 1 else 0.0,
        "slope_log_rank": calc_slope(norm_pos, log_rank),
        "volatility_log_rank": volatility_log_rank,

        "mean_log_prob": np.mean(raw_log_prob),
        "std_log_prob": np.std(raw_log_prob, ddof=1) if length > 1 else 0.0,
        "slope_log_prob": calc_slope(norm_pos, raw_log_prob),
        "volatility_log_prob": volatility_log_prob,
        "p25_log_prob": p25_log_prob,
        "p75_log_prob": p75_log_prob,
        "iqr_log_prob": p75_log_prob - p25_log_prob,
        "surprisal_skew": surprisal_skew,
        "surprisal_kurtosis": surprisal_kurtosis,
        "fano_factor_burstiness": fano_factor,

        "mean_entropy": mean_e,
        "std_entropy": std_e,
        "slope_entropy": calc_slope(norm_pos, entropy),
        "entropy_skew": entropy_skew,
        "entropy_spike_ratio": entropy_spike_ratio,
        "entropy_autocorr": entropy_autocorr,
        "iqr_entropy_ratio": iqr_entropy_ratio,

        "mean_gini_coef": mean_gini_coef,
        "mean_zipf_alpha": mean_zipf_alpha,
        "mean_top5_mass": mean_top5_mass,
        "mean_top10_mass": mean_top10_mass,
        "mean_top50_mass": mean_top50_mass,
        "mean_logit_skewness": mean_logit_skewness,
        "mean_logit_kurtosis": mean_logit_kurtosis,
        "mean_unigram_igr": mean_unigram_igr,

        "ar1_residual_var": ar1_residual_var,
        "terminal_entropy_slope": terminal_entropy_slope,
        "bimodal_extreme_index": bimodal_extreme_index,
        "entropy_texture_index": entropy_texture_index,
        "surprisal_jitter_index": surprisal_jitter_index,

        "mean_entropy_surprisal_diff": mean_entropy_surprisal_diff,
        "std_entropy_surprisal_diff": std_entropy_surprisal_diff,
        "iqr_entropy_surprisal_diff": iqr_entropy_surprisal_diff,

        "std_diff_entropy": std_diff_entropy,
        "mean_abs_diff_entropy": mean_abs_diff_entropy,
        "entropy_mean_crossing_rate": entropy_mean_crossing_rate,
        "local_global_std_ratio": local_global_std_ratio,
        "surprisal_entropy_cov": surprisal_entropy_cov,
        "surprisal_entropy_corr": surprisal_entropy_corr,
        "markov_regime_entropy": markov_regime_entropy,

        "min_shannon_ratio": min_shannon_ratio,
        "renyi2_shannon_ratio": renyi2_shannon_ratio,
        "mean_cdf_mass": mean_cdf,
        "tail_breach_ratio_90": tail_breach_90,
        "tail_breach_ratio_95": tail_breach_95,
        "rank_eff_ratio": rank_eff_ratio,
        "mean_logit_std": mean_logit_std,

        "diff_head_tail_log_prob": head_lp - tail_lp,
        
        "mean_top1_top2_margin": mean_margin,
        "std_top1_top2_margin": std_margin,
        "mean_surprisal_acc": mean_surp_acc,
        "std_surprisal_acc": std_surp_acc,
        "mean_bci": mean_bci,
        "mean_concentration_gradient": mean_concentration_gradient,
        "mean_zipf_anomaly": mean_zipf_anomaly,
        "mean_gini_entropy_div": mean_gini_entropy_div,
        "max_local_surprisal_shock": max_local_surprisal_shock,
        "mean_local_surprisal_shock": mean_local_surprisal_shock,
        **traj_features
    }



def aggregate_sentence_features(token_df, text_map=None, n_jobs=-1):
    if token_df is None or token_df.empty or "sentence_id" not in token_df.columns:
        print("[WARNING] token_df is empty or missing 'sentence_id'. Returning empty DataFrame.")
        return pd.DataFrame()

    groups = [group for _, group in token_df.groupby("sentence_id")]
    
    records = Parallel(n_jobs=n_jobs, batch_size=100)(
        delayed(_process_single_sentence_group)(
            group["sentence_id"].iloc[0], group, text_map
        ) 
        for group in groups
    )
        
    return pd.DataFrame(records)


def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if np.isnan(s_pooled) or s_pooled == 0:
        return 0.0
    return float((np.mean(group2) - np.mean(group1)) / s_pooled)


def format_p_value(p):
    if p < 1e-4:
        return f"{p:.2e} ***"
    elif p < 0.001:
        return f"{p:.2e} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (ns)"


def calculate_significance(sent_df):
    if sent_df is None or sent_df.empty:
        return pd.DataFrame()

    human_df = sent_df[sent_df["is_llm"] == 0]
    llm_df = sent_df[sent_df["is_llm"] == 1]
    
    ignore_cols = ["sentence_id", "_id", "doc_id", "label", "generator_model", "is_llm", "text", "token_length"]
    feature_cols = [col for col in sent_df.columns if col not in ignore_cols]
    results = []
    
    for feat in feature_cols:
        h_vals = pd.to_numeric(human_df[feat], errors='coerce').dropna().values
        l_vals = pd.to_numeric(llm_df[feat], errors='coerce').dropna().values

        h_vals = np.nan_to_num(h_vals, nan=0.0, posinf=0.0, neginf=0.0)
        l_vals = np.nan_to_num(l_vals, nan=0.0, posinf=0.0, neginf=0.0)

        if len(h_vals) == 0 or len(l_vals) == 0:
            continue
        
        try:
            u_stat, p_mw = stats.mannwhitneyu(h_vals, l_vals, alternative='two-sided')
        except Exception:
            p_mw = 1.0

        try:
            lev_stat, p_lev = stats.levene(h_vals, l_vals)
        except Exception:
            p_lev = 1.0

        d_val = compute_cohens_d(h_vals, l_vals)
        
        try:
            if len(np.unique(sent_df["is_llm"])) > 1:
                clean_y_true = sent_df["is_llm"].values
                clean_y_scores = np.nan_to_num(sent_df[feat].values, nan=0.0, posinf=0.0, neginf=0.0)
                auc_val = roc_auc_score(clean_y_true, clean_y_scores)
                directional_auc = max(auc_val, 1.0 - auc_val)
            else:
                directional_auc = 0.5
        except Exception:
            directional_auc = 0.5
        
        results.append({
            "feature": feat,
            "human_mean_std": f"{np.mean(h_vals):.3f} ± {np.std(h_vals):.3f}",
            "llm_mean_std": f"{np.mean(l_vals):.3f} ± {np.std(l_vals):.3f}",
            "p_location (MW-U)": format_p_value(p_mw),
            "p_variance (Levene)": format_p_value(p_lev),
            "cohens_d": round(d_val, 3),
            "roc_auc": round(directional_auc, 3),
            "_raw_auc": directional_auc,
            "_raw_p_mw": p_mw,
            "_raw_p_lev": p_lev
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="_raw_auc", ascending=False).reset_index(drop=True)
    return res_df


def generate_visualizations(token_df, sent_df, sig_df, output_png_path, exp_title):
    if sent_df is None or sent_df.empty:
        print("[WARNING] Skipping visualization generation due to empty input DataFrame.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    unique_labels = list(sent_df["label"].unique())
    palette = {}
    color_palette = sns.color_palette("tab10", len(unique_labels))
    for i, lbl in enumerate(unique_labels):
        if lbl == "Human":
            palette[lbl] = "#2b5c8f"
        else:
            palette[lbl] = color_palette[i]

    # 1. Token Trajectory
    if token_df is not None and not token_df.empty:
        token_df_copy = token_df.copy()
        token_df_copy["pos_bin"] = pd.cut(
            token_df_copy["norm_pos"], 
            bins=np.linspace(0, 1, 11), 
            labels=np.round(np.linspace(0.1, 1.0, 10), 2),
            include_lowest=True
        ).astype(float)

        sns.lineplot(data=token_df_copy, x="pos_bin", y="log_rank", hue="label", palette=palette, ax=axes[0, 0], marker="o", err_style="band")
        axes[0, 0].set_title("1. Token Log-Rank Trajectory Across Depth", fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel("Normalized Depth (0.0 = Start, 1.0 = End)")
        axes[0, 0].set_ylabel("Log Rank")
        axes[0, 0].legend(title="Source", fontsize='small')
    else:
        axes[0, 0].text(0.5, 0.5, "Token trajectories cached / unavailable", ha='center', va='center')

    # 2. Slope KDE
    human_slopes = sent_df[sent_df["is_llm"] == 0]["slope_log_rank"]
    llm_slopes = sent_df[sent_df["is_llm"] == 1]["slope_log_rank"]
    if len(human_slopes) > 0 and len(llm_slopes) > 0:
        _, p_val = stats.mannwhitneyu(human_slopes, llm_slopes, alternative='two-sided')
    else:
        p_val = 1.0

    sns.kdeplot(data=sent_df, x="slope_log_rank", hue="label", fill=True, common_norm=False, palette=palette, ax=axes[0, 1], alpha=0.3)
    axes[0, 1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title(f"2. Trajectory Slope Distribution (Overall MW-U p={p_val:.2e})", fontsize=11, fontweight='bold')
    if axes[0, 1].get_legend() is not None:
        sns.move_legend(axes[0, 1], "upper right", title="Source", fontsize='small')

    # 3. Dynamic ROC Curve
    top_feat = sig_df.iloc[0]["feature"] if (sig_df is not None and not sig_df.empty) else "mean_log_rank"
    y_true = sent_df["is_llm"].values
    y_scores = np.nan_to_num(sent_df[top_feat].values, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        if roc_auc_score(y_true, y_scores) < 0.5:
            y_scores = -y_scores
            
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc_val = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc_val = [0, 1], [0, 1], 0.5
    except Exception:
        fpr, tpr, roc_auc_val = [0, 1], [0, 1], 0.5

    axes[1, 0].plot(fpr, tpr, color='#d95f02', lw=2.5, label=f'AUC ({top_feat}) = {roc_auc_val:.3f}')
    axes[1, 0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    axes[1, 0].set_title(f'3. ROC Curve: Top Feature ({top_feat})', fontsize=11, fontweight='bold')
    axes[1, 0].legend(loc="lower right")

    # 4. Mean Log-Rank vs Token Length
    sns.scatterplot(data=sent_df, x="token_length", y="mean_log_rank", hue="label", palette=palette, alpha=0.5, ax=axes[1, 1], s=30)
    for lbl in unique_labels:
        sub_df = sent_df[sent_df['label'] == lbl]
        if len(sub_df) > 1:
            sns.regplot(data=sub_df, x='token_length', y='mean_log_rank', ax=axes[1, 1], scatter=False, color=palette[lbl])
    axes[1, 1].set_title("4. Mean Log-Rank vs. Token Length", fontsize=11, fontweight='bold')
    axes[1, 1].legend(title="Source", fontsize='small')

    plt.suptitle(exp_title, fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close('all')


# =====================================================================
# 6. EXPERIMENT SCHEDULER CLASS
# =====================================================================
class ExperimentScheduler:
    def __init__(self, configs, model_key="eurollm-9b-base-q6", models_dir="llama_cpp_models", root_dir="experiments_output", cache_dir="data_cache", reset_studies=None, reset_all=False):
        self.configs = configs
        self.model_key = model_key
        self.models_dir = models_dir
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.reset_studies = set(reset_studies) if reset_studies else set()
        self.reset_all = reset_all
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def is_study_completed(self, config):
        study_name = config["study_name"]
        if self.reset_all or study_name in self.reset_studies:
            return False
        
        exp_dir = os.path.join(self.root_dir, study_name)
        required_files = [
            "trajectory_tokens.csv", 
            "aggregated_features.csv", 
            "feature_significance.csv", 
            "visualization_dashboard.png",
            "study_config.json"
        ]
        for f in required_files:
            if not os.path.exists(os.path.join(exp_dir, f)):
                return False
        return True

    def run_all(self):
        pending_studies = []
        for config in self.configs:
            if self.is_study_completed(config):
                print(f"[SKIP] Study '{config['study_name']}' is already completed.")
            else:
                pending_studies.append(config)

        if not pending_studies:
            print("\nAll scheduled studies are already completed! Generating Master Summary Report...")
            self.generate_master_summary()
            return

        print(f"\n==================================================================")
        print(f"  SCHEDULING {len(pending_studies)} STUDIES FOR EXECUTION")
        print(f"==================================================================")

        grouped_studies = defaultdict(list)
        for config in pending_studies:
            grouped_studies[config["model_name"]].append(config)

        for model_name, study_group in grouped_studies.items():
            print(f"\n" + "="*80)
            print(f"EVALUATOR MODEL GROUP: [{model_name}] ({len(study_group)} STUDIES)")
            print("="*80)

            llm = None

            try:
                model_path, llama_kwargs = resolve_and_download_model(model_name, self.models_dir)

                needs_gpu = False
                for config in study_group:
                    human_tok_path, llm_tok_path, global_feat_path = get_split_cache_paths(config, self.cache_dir)
                    cache_hit = (
                        os.path.exists(global_feat_path) and 
                        not self.reset_all and 
                        config["study_name"] not in self.reset_studies
                    )
                    if not cache_hit:
                        if not (os.path.exists(human_tok_path) and os.path.exists(llm_tok_path)):
                            needs_gpu = True
                            break

                if needs_gpu and llm is None:
                    print(f"Loading GGUF Model via llama-cpp-python: [{model_path}]...")
                    llm = Llama(
                        model_path=model_path,
                        **llama_kwargs
                    )

                for config in study_group:
                    self._execute_single_study(config, llm)

            except Exception as e:
                print(f"[CRITICAL ERROR] Execution failed for model '{model_name}': {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

            finally:
                if llm is not None:
                    print(f"\n[Memory Manager] Unloading GGUF model from GPU...")
                    del llm
                    gc.collect()

        self.generate_master_summary()

    def _execute_single_study(self, config, llm):
        study_name = config["study_name"]
        exp_dir = os.path.join(self.root_dir, study_name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n>>> EXECUTING STUDY: [{study_name}] <<<")
        print(f"Dataset: {config['dataset']} | LLM Col: {config.get('llm_col')} | Unit: {config.get('eval_unit', 'sentence')}")

        human_tok_path, llm_tok_path, global_feat_path = get_split_cache_paths(config, self.cache_dir)
        
        cache_hit = (
            os.path.exists(global_feat_path) and 
            not self.reset_all and 
            study_name not in self.reset_studies
        )

        if cache_hit:
            print(f"[GLOBAL CACHE HIT] Reusing pre-computed features from:\n  -> {global_feat_path}")
            sent_df = pd.read_csv(global_feat_path)
            
            human_tok_df = pd.read_csv(human_tok_path) if os.path.exists(human_tok_path) else pd.DataFrame()
            llm_tok_df = pd.read_csv(llm_tok_path) if os.path.exists(llm_tok_path) else pd.DataFrame()
            token_df = pd.concat([human_tok_df, llm_tok_df], ignore_index=True) if not (human_tok_df.empty and llm_tok_df.empty) else pd.DataFrame()
        else:
            print(f"[CACHE CHECK] Verifying Human & LLM trajectory caches...")
            human_sample, llm_sample = load_data(
                language=config.get("language", "english"),
                dataset_type=config.get("dataset", "default"),
                n_samples=config.get("n_samples"),
                min_words=config.get("min_words", 8),
                parquet_path=config.get("parquet_path"),
                llm_col=config.get("llm_col"),
                eval_unit=config.get("eval_unit", "sentence")
            )

            text_map = {}
            for idx, item in enumerate(human_sample):
                text_map[f"H_{idx}"] = item[1]
            for idx, item in enumerate(llm_sample):
                text_map[f"L_{idx}"] = item[1]

            FLUSH_THRESHOLD = 50_000
            max_ctx_val = get_model_n_ctx(llm, default=2048)
            # 1. HUMAN DATA EVALUATION
            human_cache_valid = os.path.exists(human_tok_path) and not self.reset_all
            if human_cache_valid:
                print(f"[HUMAN CACHE HIT] Reusing pre-computed Human trajectories from:\n  -> {human_tok_path}")
                human_tok_df = pd.read_csv(human_tok_path)
            else:
                if llm is None:
                    raise RuntimeError(f"GPU inference required for Human texts in '{study_name}', but LLM is not loaded!")
                print(f"[HUMAN CACHE MISS] Extracting Human trajectories via llama.cpp on GPUs...")
                if os.path.exists(human_tok_path):
                    os.remove(human_tok_path)
                
                chunk_records = []
                sentence_id = 0
                pbar = tqdm(total=len(human_sample), desc="Evaluating Human Texts (GGUF)")
                for doc_id, text, model_src in human_sample:
                    records, sentence_id = extract_trajectory_llama_cpp(
                        text, doc_id, "H", sentence_id, llm, model_source=model_src, max_tokens=max_ctx_val
                    )
                    chunk_records.extend(records)
                    pbar.update(1)

                    if len(chunk_records) >= FLUSH_THRESHOLD:
                        append_chunk_to_csv(chunk_records, human_tok_path)
                        chunk_records.clear()
                pbar.close()

                if chunk_records:
                    append_chunk_to_csv(chunk_records, human_tok_path)
                    chunk_records.clear()

                human_tok_df = pd.read_csv(human_tok_path) if os.path.exists(human_tok_path) else pd.DataFrame()

            # 2. LLM DATA EVALUATION
            llm_cache_valid = os.path.exists(llm_tok_path) and not self.reset_all and study_name not in self.reset_studies
            if llm_cache_valid:
                print(f"[LLM CACHE HIT] Reusing pre-computed LLM trajectories from:\n  -> {llm_tok_path}")
                llm_tok_df = pd.read_csv(llm_tok_path)
            else:
                if llm is None:
                    raise RuntimeError(f"GPU inference required for LLM texts in '{study_name}', but LLM is not loaded!")
                print(f"[LLM CACHE MISS] Extracting LLM trajectories [{config.get('llm_col')}] via llama.cpp on GPUs...")
                if os.path.exists(llm_tok_path):
                    os.remove(llm_tok_path)

                chunk_records = []
                sentence_id = 0
                pbar = tqdm(total=len(llm_sample), desc=f"Evaluating LLM [{config.get('llm_col')}] (GGUF)")
                for doc_id, text, model_src in llm_sample:
                    records, sentence_id = extract_trajectory_llama_cpp(
                        text, doc_id, "L", sentence_id, llm, model_source=model_src, max_tokens=max_ctx_val
                    )
                    chunk_records.extend(records)
                    pbar.update(1)

                    if len(chunk_records) >= FLUSH_THRESHOLD:
                        append_chunk_to_csv(chunk_records, llm_tok_path)
                        chunk_records.clear()
                pbar.close()

                if chunk_records:
                    append_chunk_to_csv(chunk_records, llm_tok_path)
                    chunk_records.clear()

                llm_tok_df = pd.read_csv(llm_tok_path) if os.path.exists(llm_tok_path) else pd.DataFrame()

            token_df = pd.concat([human_tok_df, llm_tok_df], ignore_index=True)

            sent_df = aggregate_sentence_features(token_df, text_map=text_map)
            if not sent_df.empty:
                sent_df.to_csv(global_feat_path, index=False)
                print(f"[SAVED TO GLOBAL CACHE]:\n  -> {global_feat_path}")

        if not sent_df.empty:
            std_feat_path = os.path.join(exp_dir, "aggregated_features.csv")
            std_tok_path = os.path.join(exp_dir, "trajectory_tokens.csv")
            sent_df.to_csv(std_feat_path, index=False)
            
            if not token_df.empty:
                token_df.to_csv(std_tok_path, index=False)

            sig_df = calculate_significance(sent_df)
            if not sig_df.empty:
                report_cols = ["feature", "human_mean_std", "llm_mean_std", "p_location (MW-U)", "p_variance (Levene)", "cohens_d", "roc_auc"]
                available_report_cols = [c for c in report_cols if c in sig_df.columns]
                export_df = sig_df[available_report_cols]

                print("\n" + "="*80)
                print(f"  STATISTICAL SIGNIFICANCE REPORT: [{study_name.upper()}]")
                print("="*80)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                print(export_df.to_string(index=False))
                print("="*80 + "\n")

                sig_csv_path = os.path.join(exp_dir, "feature_significance.csv")
                export_df.to_csv(sig_csv_path, index=False)

                exp_title = f"Study: {study_name}\nModel: {config.get('model_name', 'GGUF')} | Unit: {config.get('eval_unit', 'sentence')}"
                generate_visualizations(token_df, sent_df, sig_df, os.path.join(exp_dir, "visualization_dashboard.png"), exp_title)

                top_feature = sig_df.iloc[0]
                auc_val = top_feature.get('roc_auc', 0.0)
                feat_name = top_feature.get('feature', 'N/A')
                print(f"[FINISHED] [{study_name}] Top Feature: '{feat_name}' (AUC = {auc_val:.3f})")
        else:
            print(f"[WARNING] Study '{study_name}' produced an empty sent_df. No features or reports were generated.")

        with open(os.path.join(exp_dir, "study_config.json"), "w") as f:
            json.dump(config, f, indent=4)


    def generate_master_summary(self):
        summary_rows = []
        for config in self.configs:
            study_name = config["study_name"]
            sig_csv = os.path.join(self.root_dir, study_name, "feature_significance.csv")
            
            if os.path.exists(sig_csv):
                sig_df = pd.read_csv(sig_csv)
                if not sig_df.empty:
                    top_row = sig_df.iloc[0]
                    summary_rows.append({
                        "study_name": study_name,
                        "evaluator_model": config["model_name"],
                        "dataset": config["dataset"],
                        "llm_col": str(config.get("llm_col", "default")),
                        "eval_unit": config.get("eval_unit", "sentence"),
                        "top_feature": top_row["feature"],
                        "top_roc_auc": top_row["roc_auc"],
                        "cohens_d": top_row["cohens_d"],
                        "p_location (MW-U)": top_row["p_location (MW-U)"],
                        "p_variance (Levene)": top_row["p_variance (Levene)"]
                    })

        if summary_rows:
            master_df = pd.DataFrame(summary_rows).sort_values(by="top_roc_auc", ascending=False)
            master_path = os.path.join(self.root_dir, "MASTER_SUMMARY_REPORT.csv")
            master_df.to_csv(master_path, index=False)
            print("\n" + "="*85)
            print("  MASTER EXPERIMENTS COMPARISON REPORT")
            print("="*85)
            print(master_df.to_string(index=False))
            print("="*85)
            print(f"Master summary report saved to: {master_path}\n")


# =====================================================================
# 7. CLI INTERFACE & MAIN EXECUTION
# =====================================================================
def apply_cli_overrides(configs):
    parser = argparse.ArgumentParser(description="LLM Trajectory Detection Scheduler")

    parser.add_argument("--n_samples", type=int, default=argparse.SUPPRESS, help="Override n_samples for all studies")
    parser.add_argument("--min_words", type=int, default=argparse.SUPPRESS, help="Override min_words for all studies")
    parser.add_argument("--eval_unit", type=str, choices=["sentence", "document", "full", "abstract"], default=argparse.SUPPRESS, help="Override eval_unit")
    parser.add_argument("--language", type=str, default=argparse.SUPPRESS, help="Override language")
    parser.add_argument("--parquet_path", type=str, default=argparse.SUPPRESS, help="Override parquet_path")
    parser.add_argument("--llm_col", type=str, default=argparse.SUPPRESS, help="Override LLM column name")

    parser.add_argument("--reset_all", action="store_true", default=argparse.SUPPRESS, help="Force reset all studies")
    parser.add_argument("--reset_studies", nargs="+", default=argparse.SUPPRESS, help="List of study names to reset")
    parser.add_argument("--model_key", type=str, default=argparse.SUPPRESS, help="Override evaluator model_key")

    args, _ = parser.parse_known_args()
    cli_kwargs = vars(args)

    scheduler_keys = {"reset_all", "reset_studies", "model_key"}
    scheduler_overrides = {k: v for k, v in cli_kwargs.items() if k in scheduler_keys}
    config_overrides = {k: v for k, v in cli_kwargs.items() if k not in scheduler_keys}

    if config_overrides:
        print(f"\n[CLI OVERRIDES APPLIED]: {config_overrides}")
        for config in configs:
            for key, val in config_overrides.items():
                config[key] = val

    return configs, scheduler_overrides


if __name__ == "__main__":
    STUDY_CONFIGS = [
        {
            "study_name": "exp6_qwen36b_mixed_model", #DONE
            "model_name": "qwen2.5-32b-baseQ4",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": [
                "qwen3.6:27b_single",
                "gemma4:e4b_single",
                "qwen3.5:4b_single",
                "gemma4:26b_single"
            ],
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "sentence",
            "n_samples": 1000,
            "min_words": 10,
        },
        {
            "study_name": "exp10_euro22b_mixed_model", #euro big,all model, sentence
            "model_name": "eurollm-22b-q6_k",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": [
                "qwen3.6:27b_single",
                "gemma4:e4b_single",
                "qwen3.5:4b_single",
                "gemma4:26b_single"
            ],
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "sentence",
            "n_samples": 1000,
            "min_words": 10,
        },
        {
            "study_name": "exp7_qwen36b_qwen27b", #big qwen qwen only sentence
            "model_name": "qwen2.5-32b-baseQ4",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": "qwen3.6:27b_single",
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "sentence",
            "n_samples": 1000,
            "min_words": 10,
        },
        {
            "study_name": "exp8_qwen36b_mixed_abstract", #big qwen, all models, abstract
            "model_name": "qwen2.5-32b-baseQ4",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": [
                "qwen3.6:27b_single",
                "gemma4:e4b_single",
                "qwen3.5:4b_single",
                "gemma4:26b_single"
            ],
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "abstract",
            "n_samples": 1000,
            "min_words": 10,
        },
        {
            "study_name": "exp9_qwen36b_gemma26b_abstract", #qwen qwen abstract
            "model_name": "qwen2.5-32b-baseQ4",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": "qwen3.6:27b_single",
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "abstract",
            "n_samples": 1000,
            "min_words": 10,
        },
        {
            "study_name": "exp11_euro22_qwen27b", #euro qwen only sentence
            "model_name": "eurollm-22b-q6_k",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": "qwen3.6:27b_single",
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "sentence",
            "n_samples": 1000,
            "min_words": 10,
        },
        {
            "study_name": "exp12_euro22b_qwen27bonly", #euro qwen only abstract
            "model_name": "eurollm-22b-q6_k",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": "qwen3.6:27b_single",
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "abstract",
            "n_samples": 1000,
            "min_words": 10,
        },
        {
            "study_name": "exp13_euro_mixed_abstract", #big qwen, all models, abstract
            "model_name": "eurollm-22b-q6_k",
            "language": "dutch",
            "dataset": "abstracts",
            "llm_col": [
                "qwen3.6:27b_single",
                "gemma4:e4b_single",
                "qwen3.5:4b_single",
                "gemma4:26b_single"
            ],
            "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
            "eval_unit": "abstract",
            "n_samples": 1000,
            "min_words": 10,
        },
#        {
#            "study_name": "exp14_multitude_dutch_sentence",
#            "model_name": "qwen2.5-32b-baseQ4",
#            "language": "dutch",
#            "dataset": "multitude",
#            # "parquet_path": "/path/to/local_multitude.parquet",  # Optional: only if using local file
#            "eval_unit": "sentence",
#            "n_samples": 1000,
#            "min_words": 10,
#        },
        {
            "study_name": "exp14_clin_dutch_sentence",
            "model_name": "qwen2.5-32b-baseQ4",
            "language": "dutch",
            "dataset": "clin33",
            "parquet_path": "/home/gderijck/internship/src/detection/feature_tests/data/clin33_shared_task_generated_dutch.csv",  # Optional: only if using local file
            "eval_unit": "sentence",
            "n_samples": 1000,
            "min_words": 10,
        },

    ]

    active_configs, scheduler_args = apply_cli_overrides(STUDY_CONFIGS)

    reset_all = scheduler_args.get("reset_all", False)
    reset_studies = scheduler_args.get("reset_studies", None)

    if reset_all or reset_studies:
        if reset_all:
            target_names = [cfg["study_name"] for cfg in active_configs]
            reset_msg = "ALL ACTIVE STUDIES:\n  - " + "\n  - ".join(target_names)
        else:
            reset_msg = "SPECIFIED STUDIES:\n  - " + "\n  - ".join(reset_studies)

        print("\n" + "!" * 80)
        print("  WARNING: SAFETY RESET CONFIRMATION REQUIRED")
        print("!" * 80)
        print(f"You are about to RESET / OVERWRITE the following study results:\n")
        print(reset_msg)
        print("\n" + "-" * 80)
        
        confirm = input("Are you sure you want to proceed and overwrite existing data? [y/N]: ")
        if confirm.strip().lower() not in ['y', 'yes']:
            print("\n[ABORTED] Reset cancelled by user. Exiting safely.\n")
            sys.exit(0)

    scheduler = ExperimentScheduler(
        configs=active_configs,
        model_key=scheduler_args.get("model_key", "eurollm-9b-base-q6"),
        models_dir=scheduler_args.get("models_dir", "llama_cpp_models"),
        root_dir=scheduler_args.get("root_dir", "experiments_output"),
        cache_dir=scheduler_args.get("cache_dir", "data_cache"),
        reset_studies=reset_studies,
        reset_all=reset_all
    )

    scheduler.run_all()