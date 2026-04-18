import argparse
import json
import sys
from pathlib import Path

try:
    from src.preprocessing.io_utils import load_jsonl as load_jsonl_with_row_id, resolve_patient_id
except ImportError:
    from ..preprocessing.io_utils import load_jsonl as load_jsonl_with_row_id, resolve_patient_id


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_pids(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data)


def val_pids_from_jsonl(path: str) -> set:
    """Validation PIDs from JSONL using the same resolve_patient_id as the split pipeline."""
    records = load_jsonl_with_row_id(path)
    return {resolve_patient_id(r) for r in records}

def check_kfold_leakage(val_pids: set, candidates: list[str], fold_idx: int, n_splits: int = 5, seed: int = 42):
    try:
        from src.training_validation.split import make_kfold_splits
    except ImportError:
        print("Error: Could not import make_kfold_splits from src.training_validation.split", file=sys.stderr)
        sys.exit(1)

    print(f"Checking k-fold leakage for {len(val_pids)} validation PIDs...")
    print(f"Target fold: {fold_idx} (0-indexed) out of {n_splits} splits (seed={seed})\n")

    has_leakage = False
    
    print(f"{'Candidate File':<40} | {'Train Size':<10} | {'Overlap':<10} | {'Leakage %':<10} | {'Status'}")
    print("-" * 85)

    for cand in candidates:
        try:
            records = load_jsonl(cand)
        except Exception as e:
            print(f"{cand:<40} | ERROR: {str(e)}")
            continue

        splits = make_kfold_splits(records, n_splits=n_splits, seed=seed)
        
        if fold_idx >= len(splits):
            print(f"{cand:<40} | ERROR: fold_idx {fold_idx} out of bounds")
            continue

        train_recs, _ = splits[fold_idx]
        train_pids = {resolve_patient_id(r) for r in train_recs}
        
        overlap = train_pids.intersection(val_pids)
        overlap_count = len(overlap)
        leakage_pct = (overlap_count / len(val_pids)) * 100 if val_pids else 0
        
        status = "LEAKED" if overlap_count > 0 else "CLEAN"
        if overlap_count > 0:
            has_leakage = True
            
        print(f"{Path(cand).name:<40} | {len(train_pids):<10} | {overlap_count:<10} | {leakage_pct:>5.1f}%     | {status}")

    return has_leakage

def check_global_leakage(val_pids: set, train_file: str):
    print(f"Checking global disjointness for {len(val_pids)} validation PIDs vs {train_file}...\n")
    
    try:
        records = load_jsonl(train_file)
        train_pids = {resolve_patient_id(r) for r in records}
    except Exception as e:
        print(f"ERROR reading {train_file}: {str(e)}")
        return True

    overlap = train_pids.intersection(val_pids)
    overlap_count = len(overlap)
    
    print(f"Train PIDs: {len(train_pids)}")
    print(f"Overlap:    {overlap_count} PIDs")
    
    if overlap_count > 0:
        print(f"\nStatus: LEAKED")
        return True
    else:
        print(f"\nStatus: CLEAN")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify data leakage between train and val splits.")
    parser.add_argument("--mode", choices=["kfold", "global"], default="kfold", help="Checking mode")
    parser.add_argument(
        "--val-pids",
        default=None,
        help="JSON file with a list of validation PIDs (use this or --val-jsonl).",
    )
    parser.add_argument(
        "--val-jsonl",
        default=None,
        help="Validation split JSONL; PIDs via resolve_patient_id (alternative to --val-pids).",
    )
    
    # kfold args
    parser.add_argument("--fold-idx", type=int, default=4, help="Fold index (0-indexed) used for training the model")
    parser.add_argument("--candidates", nargs="+", default=["data/processed/training_set.jsonl", "data/processed/cleaned.jsonl"], 
                        help="Candidate data files that might have been used for training")
    
    # global args
    parser.add_argument("--train-file", default="data/processed/training_set.jsonl", help="Train file to check against in global mode")

    args = parser.parse_args()

    if bool(args.val_pids) == bool(args.val_jsonl):
        parser.error("Provide exactly one of --val-pids or --val-jsonl.")

    try:
        if args.val_jsonl:
            val_pids = val_pids_from_jsonl(args.val_jsonl)
            print(f"Loaded {len(val_pids)} validation PIDs from {args.val_jsonl}")
        else:
            val_pids = load_pids(args.val_pids)
    except Exception as e:
        print(f"Error loading validation PIDs: {e}", file=sys.stderr)
        sys.exit(1)

    has_leakage = False
    if args.mode == "kfold":
        has_leakage = check_kfold_leakage(val_pids, args.candidates, args.fold_idx)
    else:
        has_leakage = check_global_leakage(val_pids, args.train_file)

    if has_leakage:
        print("\nWARNING: Data leakage detected!", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nSUCCESS: No leakage detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
