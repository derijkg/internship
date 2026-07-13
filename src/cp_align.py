import json
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
import pyarrow as pa

#useful?
def normalize_text(text: str) -> str:
    """Normalizes text by applying Unicode NFKC normalization and collapsing whitespace."""
    if not text:
        return ""
    # Strip unicode control characters / unusual spaces
    text = unicodedata.normalize("NFKC", text)
    # Replace all whitespace runs (tabs, newlines, non-breaking spaces) with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_similarity_ratio(s1: str, s2: str) -> float:
    """Returns SequenceMatcher similarity ratio (0.0 to 1.0)."""
    return SequenceMatcher(None, s1, s2).ratio()


def generate_alignment_report(
    dataset_path: str,
    checkpoint_path: str,
    fuzzy_threshold: float = 0.85
):
    dataset_file = Path(dataset_path)
    checkpoint_file = Path(checkpoint_path)

    # ---------------------------------------------------------
    # 1. Load Dataset & Construct Mapping
    # ---------------------------------------------------------
    print(f"Loading dataset from {dataset_file}...")
    if dataset_file.suffix == ".parquet":
        table = pq.read_table(dataset_file)
    else:
        table = pcsv.read_csv(dataset_file)

    rows = table.to_pylist()

    # Determine ID column name
    if '_id' in table.column_names:
        id_key = '_id'
    elif 'id' in table.column_names:
        id_key = 'id'
    elif 'page_link' in table.column_names:
        id_key = 'page_link'
    else:
        id_key = 'synthetic_id'
        for idx, r in enumerate(rows):
            r[id_key] = str(idx)

    # We map ALL rows to track presence and sources
    all_dataset_ids = {}  # ID -> source
    ug_data_map = {}      # ID -> preprocessed sentence structures (UG-only)

    for row in rows:
        r_id = str(row.get(id_key))
        source = str(row.get("source", ""))
        all_dataset_ids[r_id] = source

        if source == "UG":
            abstract = row.get("abstract") or ""
            sentences = row.get("abstract_sentence") or []
            if not isinstance(sentences, list):
                sentences = []

            ug_data_map[r_id] = {
                "abstract": abstract,
                "norm_abstract": normalize_text(abstract),
                "sentences": sentences,
                "norm_sentences": [normalize_text(s) for s in sentences]
            }

    print(f"Loaded {len(all_dataset_ids)} unique records from dataset.")
    print(f"Identified {len(ug_data_map)} records with source == 'UG'.")

    # ---------------------------------------------------------
    # 2. Analyze Checkpoint Alignment
    # ---------------------------------------------------------
    stats = {
        "total_checkpoint_records": 0,
        "corrupted_lines": 0,
        
        # ID Discrepancies
        "missing_id_dropped": 0,       # Completely missing from the new dataset
        "missing_id_non_ug": 0,        # Exists in new dataset, but source is not 'UG'
        
        # Sentence level alignment breakdown
        "sentence_exact_match": 0,
        "sentence_norm_match": 0,
        "sentence_index_drift": 0,
        "sentence_fuzzy_match": 0,
        "sentence_containment_match": 0,
        "sentence_unmatched": 0,

        # Full/Percentage abstract breakdown
        "full_exact_match": 0,
        "full_fuzzy_match": 0,
        "full_unmatched": 0,

        # Distributions
        "type_counts": {},
        "model_counts": {},
    }

    fuzzy_scores = []
    index_shifts = []

    print(f"Analyzing checkpoint alignment against {checkpoint_file}...")

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                stats["corrupted_lines"] += 1
                continue

            stats["total_checkpoint_records"] += 1

            row_id = str(record.get("id"))
            t_type = record.get("type", "sentence")
            model = record.get("model", "unknown")
            orig_text = record.get("text", "") or ""
            sent_idx = record.get("sent_idx")

            # Update categorical distributions
            stats["type_counts"][t_type] = stats["type_counts"].get(t_type, 0) + 1
            stats["model_counts"][model] = stats["model_counts"].get(model, 0) + 1

            # Check ID state against overall dataset mapping
            if row_id not in all_dataset_ids:
                stats["missing_id_dropped"] += 1
                continue
            
            if all_dataset_ids[row_id] != "UG":
                stats["missing_id_non_ug"] += 1
                continue

            doc_data = ug_data_map[row_id]
            norm_orig = normalize_text(orig_text)

            # --- ALIGNMENT FOR SENTENCE TASKS ---
            if t_type == "sentence":
                sentences = doc_data["sentences"]
                norm_sentences = doc_data["norm_sentences"]

                if not norm_sentences:
                    stats["sentence_unmatched"] += 1
                    continue

                matched = False

                # Tier 1 & 2: Exact or Normalized match at exact index
                if sent_idx is not None and 0 <= sent_idx < len(sentences):
                    if sentences[sent_idx] == orig_text:
                        stats["sentence_exact_match"] += 1
                        matched = True
                    elif norm_sentences[sent_idx] == norm_orig:
                        stats["sentence_norm_match"] += 1
                        matched = True

                # Tier 3: Index Drift (Exact text match at another index)
                if not matched:
                    for idx, norm_s in enumerate(norm_sentences):
                        if norm_s == norm_orig:
                            stats["sentence_index_drift"] += 1
                            if sent_idx is not None:
                                index_shifts.append(idx - sent_idx)
                            matched = True
                            break

                # Tier 4: Fuzzy Match (SequenceMatcher similarity)
                if not matched:
                    best_score = 0.0
                    best_idx = -1
                    for idx, norm_s in enumerate(norm_sentences):
                        score = get_similarity_ratio(norm_orig, norm_s)
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_score >= fuzzy_threshold:
                        stats["sentence_fuzzy_match"] += 1
                        fuzzy_scores.append(best_score)
                        if sent_idx is not None:
                            index_shifts.append(best_idx - sent_idx)
                        matched = True

                # Tier 5: Substring / Containment Match (Min length restriction)
                if not matched:
                    for idx, norm_s in enumerate(norm_sentences):
                        # Ensure we don't accidentally match tiny fragments (like stopwords)
                        if (norm_orig in norm_s or norm_s in norm_orig) and len(norm_orig) > 15:
                            stats["sentence_containment_match"] += 1
                            matched = True
                            break

                if not matched:
                    stats["sentence_unmatched"] += 1

            # --- ALIGNMENT FOR FULL ABSTRACT / PERCENTAGE TASKS ---
            else:
                norm_abstract = doc_data["norm_abstract"]
                if norm_orig == norm_abstract:
                    stats["full_exact_match"] += 1
                elif norm_orig and norm_abstract:
                    score = get_similarity_ratio(norm_orig, norm_abstract)
                    if score >= fuzzy_threshold:
                        stats["full_fuzzy_match"] += 1
                        fuzzy_scores.append(score)
                    else:
                        stats["full_unmatched"] += 1
                else:
                    stats["full_unmatched"] += 1

    # ---------------------------------------------------------
    # 3. Print Comprehensive Report
    # ---------------------------------------------------------
    total = stats["total_checkpoint_records"]
    aligned_sentence = (
        stats["sentence_exact_match"]
        + stats["sentence_norm_match"]
        + stats["sentence_index_drift"]
        + stats["sentence_fuzzy_match"]
        + stats["sentence_containment_match"]
    )
    aligned_full = stats["full_exact_match"] + stats["full_fuzzy_match"]
    total_aligned = aligned_sentence + aligned_full

    retention_pct = (total_aligned / total * 100) if total > 0 else 0

    print("\n" + "=" * 65)
    print("      CHECKPOINT vs NEW DATASET ALIGNMENT REPORT")
    print("=" * 65)
    print(f" Total Loaded Checkpoint Records   : {total}")
    print(f" Corrupted / Invalid JSON Lines    : {stats['corrupted_lines']}")
    print("-" * 65)
    print(f" UNMATCHED BY ID (DROPPED/FILTERED):")
    print(f"  • Completely Dropped from Dataset: {stats['missing_id_dropped']}")
    print(f"  • Non-UG Source Row (Filtered Out): {stats['missing_id_non_ug']}")
    print("-" * 65)
    print(f" TOTAL ALIGNED / RECOVERABLE DATA  : {total_aligned} / {total} ({retention_pct:.2f}%)")
    print(f" TOTAL UNMATCHED / DISCARDED DATA  : {total - total_aligned - stats['missing_id_dropped'] - stats['missing_id_non_ug']}")
    print("=" * 65)

    print("\n--- SENTENCE TASKS ALIGNMENT DETAILS ---")
    print(f"  [Tier 1] Exact Matches (Same Index & Text) : {stats['sentence_exact_match']}")
    print(f"  [Tier 2] Normalized Matches (Whitespace)   : {stats['sentence_norm_match']}")
    print(f"  [Tier 3] Index Drift (Shifted Sentence)    : {stats['sentence_index_drift']}")
    print(f"  [Tier 4] Fuzzy Matches (Score >= {fuzzy_threshold})      : {stats['sentence_fuzzy_match']}")
    print(f"  [Tier 5] Containment Matches (Split/Joined): {stats['sentence_containment_match']}")
    print(f"  [Failed] Unmatched Sentences               : {stats['sentence_unmatched']}")

    print("\n--- FULL / PERCENTAGE TASKS ALIGNMENT DETAILS ---")
    print(f"  • Full Exact / Normalized Matches          : {stats['full_exact_match']}")
    print(f"  • Full Fuzzy Matches                       : {stats['full_fuzzy_match']}")
    print(f"  • Full Unmatched                           : {stats['full_unmatched']}")

    print("\n--- TASK TYPE DISTRIBUTION ---")
    for task_type, count in stats["type_counts"].items():
        print(f"  • {task_type}: {count}")

    print("\n--- MODEL DISTRIBUTION ---")
    for model_name, count in stats["model_counts"].items():
        print(f"  • {model_name}: {count}")

    if fuzzy_scores:
        avg_score = sum(fuzzy_scores) / len(fuzzy_scores)
        print(f"\n--- METRICS ---")
        print(f" Average Fuzzy Match Score: {avg_score:.4f}")
    if index_shifts:
        avg_shift = sum(index_shifts) / len(index_shifts)
        print(f" Average Sentence Index Shift: {avg_shift:+.2f} positions")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    generate_alignment_report(
        dataset_path=Path('/home/gderijck/internship/data/gold/merged_publications.parquet'), 
        checkpoint_path=Path('/home/gderijck/internship/data/gold/checkpoint_rewrites.jsonl'),
        fuzzy_threshold=0.85
    )