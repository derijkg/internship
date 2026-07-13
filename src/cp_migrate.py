import json
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
from pathlib import Path

def migrate_checkpoint(
    dataset_path: str,
    old_checkpoint_path: str,
    new_checkpoint_path: str
):
    dataset_file = Path(dataset_path)
    old_checkpoint = Path(old_checkpoint_path)
    new_checkpoint = Path(new_checkpoint_path)

    # ---------------------------------------------------------
    # 1. Load Reference Dataset (Filter source == 'UG')
    # ---------------------------------------------------------
    print(f"Loading reference dataset from {dataset_file}...")
    if dataset_file.suffix == ".parquet":
        table = pq.read_table(dataset_file)
    else:
        table = pcsv.read_csv(dataset_file)

    rows = table.to_pylist()

    # Determine ID column name using prepare_tasks key resolution
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

    # Map only rows where source == 'UG'
    new_ug_map = {}
    for idx, row in enumerate(rows):
        row_id = str(row.get(id_key, idx))
        source = str(row.get("source", ""))

        if source == "UG":
            abstract = row.get("abstract") or ""
            sentences = row.get("abstract_sentence")
            
            # Handle potential CSV stringified lists
            if isinstance(sentences, str):
                try:
                    sentences = json.loads(sentences)
                except json.JSONDecodeError:
                    sentences = [sentences]
            if not isinstance(sentences, list):
                sentences = []

            new_ug_map[row_id] = {
                "abstract": abstract,
                "sentences": sentences
            }

    print(f"Loaded reference map with {len(new_ug_map)} UG documents.")

    # ---------------------------------------------------------
    # 2. Process and Migrate Old Checkpoint
    # ---------------------------------------------------------
    stats = {
        "processed": 0,
        "corrupted": 0,
        "discarded_missing_id": 0,
        "migrated_exact_index": 0,
        "migrated_index_shift": 0,
        "migrated_full_abstract": 0,
        "discarded_unmatched": 0
    }

    print(f"Migrating records from {old_checkpoint} -> {new_checkpoint}...")

    with open(old_checkpoint, "r", encoding="utf-8") as f_in, \
         open(new_checkpoint, "w", encoding="utf-8") as f_out:
        
        for line_idx, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                stats["corrupted"] += 1
                continue

            stats["processed"] += 1

            # Extract fields with backward-compatibility safety
            if "type" in record:
                row_id = str(record.get("id"))
                t_type = record.get("type", "sentence")
                model = record.get("model")
                sent_idx = record.get("sent_idx")
                pct = record.get("percentage")
                orig_text = record.get("text", "") or ""
                rewritten_text = record.get("rewritten")
            else:
                row_id = str(record.get("id"))
                t_type = "sentence"
                sent_idx = record.get("sent_idx")
                pct = None
                orig_text = record.get("text", "") or ""
                
                metadata_keys = {"id", "_id", "sent_idx", "text"}
                model_keys = [k for k in record.keys() if k not in metadata_keys] 
                if not model_keys or sent_idx is None:
                    stats["discarded_unmatched"] += 1
                    continue
                model = model_keys[0]
                rewritten_text = record[model]

            if rewritten_text is None:
                stats["discarded_unmatched"] += 1
                continue

            # Strip LLM channel wraps if present
            if isinstance(rewritten_text, str) and '<channel|>' in rewritten_text:
                rewritten_text = rewritten_text.split('<channel|>')[1].strip()

            # Ensure ID exists in the active UG reference set
            if row_id not in new_ug_map:
                stats["discarded_missing_id"] += 1
                continue

            ref_doc = new_ug_map[row_id]
            clean_orig = orig_text.strip()

            # --- SENTENCE TYPE ALIGNMENT ---
            if t_type == "sentence":
                sentences = ref_doc["sentences"]
                matched_idx = -1
                matched_text = ""

                # Tier 1: Try exact matching at the expected index
                if sent_idx is not None:
                    try:
                        sent_idx_int = int(sent_idx)
                        if 0 <= sent_idx_int < len(sentences):
                            if sentences[sent_idx_int].strip() == clean_orig:
                                matched_idx = sent_idx_int
                                matched_text = sentences[sent_idx_int]
                                stats["migrated_exact_index"] += 1
                    except (ValueError, TypeError):
                        pass

                # Tier 2: Try index drift matching (any exact match in the document)
                if matched_idx == -1:
                    for idx, sent in enumerate(sentences):
                        if sent.strip() == clean_orig:
                            matched_idx = idx
                            matched_text = sent
                            stats["migrated_index_shift"] += 1
                            break

                # If found, save standardized output with updated metadata
                if matched_idx != -1:
                    migrated_record = {
                        "id": row_id,
                        "type": "sentence",
                        "model": model,
                        "sent_idx": matched_idx,
                        "percentage": pct,
                        "text": matched_text,
                        "rewritten": rewritten_text
                    }
                    f_out.write(json.dumps(migrated_record, ensure_ascii=False) + "\n")
                else:
                    stats["discarded_unmatched"] += 1

            # --- FULL / PERCENTAGE TYPE ALIGNMENT ---
            else:
                ref_abstract = ref_doc["abstract"]
                if clean_orig == ref_abstract.strip():
                    migrated_record = {
                        "id": row_id,
                        "type": t_type,
                        "model": model,
                        "sent_idx": sent_idx,
                        "percentage": pct,
                        "text": ref_abstract,
                        "rewritten": rewritten_text
                    }
                    f_out.write(json.dumps(migrated_record, ensure_ascii=False) + "\n")
                    stats["migrated_full_abstract"] += 1
                else:
                    stats["discarded_unmatched"] += 1

    # ---------------------------------------------------------
    # 3. Print Summary Report
    # ---------------------------------------------------------
    total_saved = (
        stats["migrated_exact_index"]
        + stats["migrated_index_shift"]
        + stats["migrated_full_abstract"]
    )
    
    print("\n" + "=" * 60)
    print("             MIGRATION PROCESS COMPLETED")
    print("=" * 60)
    print(f" Total Checked Old Records         : {stats['processed']}")
    print(f" Corrupted Old JSON Lines          : {stats['corrupted']}")
    print(f" Discarded (Missing/Dropped IDs)   : {stats['discarded_missing_id']}")
    print(f" Discarded (Unmatched text/index)  : {stats['discarded_unmatched']}")
    print("-" * 60)
    print(f" TOTAL SUCCESSFULLY MIGRATED       : {total_saved} records")
    print(f"  • Sentences on Exact Index       : {stats['migrated_exact_index']}")
    print(f"  • Sentences with Shifted Indices : {stats['migrated_index_shift']}")
    print(f"  • Full/Percentage Abstracts      : {stats['migrated_full_abstract']}")
    print("=" * 60 + "\n")


# Execution Wrapper
if __name__ == "__main__":
    migrate_checkpoint(
        dataset_path=Path('/home/gderijck/internship/data/gold/merged_publications.parquet'), # or .csv
        old_checkpoint_path=Path('/home/gderijck/internship/data/gold/checkpoint_rewrites.jsonl'),
        new_checkpoint_path=Path('/home/gderijck/internship/data/gold/checkpoint_rewrites_NEW.jsonl')
    )