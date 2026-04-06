import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from evaluator import score_document, micro_f1

def load_gold(gold_jsonl_path: str) -> Dict[int, List[List[str]]]:
    gold_data = {}
    with open(gold_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            gold_data[record['patient_id']] = record['document_level_annotations']
    return gold_data

def evaluate_thresholds(
    scores: np.ndarray,
    patient_ids: List[int],
    gold_data: Dict[int, List[List[str]]],
    thresholds: np.ndarray,
    label_names: List[str]
) -> float:
    """
    Evaluates the given thresholds and returns the micro-F1 score.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # Binarize scores
    preds_bin = scores >= thresholds
    
    for i, pid in enumerate(patient_ids):
        if pid not in gold_data:
            continue
        gold_groups = gold_data[pid]
        
        # Get predicted codes for this document
        pred_indices = np.where(preds_bin[i])[0]
        pred_codes = [label_names[idx] for idx in pred_indices]
        
        tp, fp, fn = score_document(gold_groups, pred_codes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    _, _, f1 = micro_f1(total_tp, total_fp, total_fn)
    return f1

def tune_thresholds(
    scores: np.ndarray, 
    patient_ids: List[int], 
    gold_data: Dict[int, List[List[str]]], 
    label_names: List[str]
) -> np.ndarray:
    """
    Tunes thresholds to maximize micro-F1.
    Step 1: Global sweep
    Step 2: Per-class greedy sweep
    """
    num_classes = scores.shape[1]
    
    # Step 1: Global search
    print("Starting global threshold search...")
    best_global_t = 0.5
    best_global_f1 = 0.0
    
    for t in np.arange(0.05, 0.96, 0.01):
        global_thresh = np.full(num_classes, t)
        f1 = evaluate_thresholds(scores, patient_ids, gold_data, global_thresh, label_names)
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_t = t
            
    print(f"Best global threshold: {best_global_t:.2f} (F1: {best_global_f1:.4f})")
    
    # Step 2: Per-class greedy search
    print("Starting per-class greedy search...")
    best_thresholds = np.full(num_classes, best_global_t)
    current_best_f1 = best_global_f1
    
    for c in range(num_classes):
        best_c_t = best_thresholds[c]
        best_c_f1 = current_best_f1
        
        for t in np.arange(0.05, 0.96, 0.01):
            test_thresholds = best_thresholds.copy()
            test_thresholds[c] = t
            
            f1 = evaluate_thresholds(scores, patient_ids, gold_data, test_thresholds, label_names)
            if f1 > best_c_f1:
                best_c_f1 = f1
                best_c_t = t
                
        if best_c_f1 > current_best_f1:
            best_thresholds[c] = best_c_t
            current_best_f1 = best_c_f1
            
    print(f"Final tuned F1: {current_best_f1:.4f}")
    return best_thresholds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune thresholds for MLC outputs.")
    parser.add_argument("--scores", required=True, help="Path to .npy file containing sigmoid scores (N_docs, 115)")
    parser.add_argument("--pids", required=True, help="Path to JSON file containing list of patient_ids corresponding to rows in scores")
    parser.add_argument("--labels", required=True, help="Path to JSON file containing list of 115 label names")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL file")
    parser.add_argument("--out", required=True, help="Output JSON file for best thresholds")
    args = parser.parse_args()
    
    # Load data
    scores = np.load(args.scores)
    with open(args.pids, 'r', encoding='utf-8') as f:
        patient_ids = json.load(f)
    with open(args.labels, 'r', encoding='utf-8') as f:
        label_names = json.load(f)
        
    gold_data = load_gold(args.gold)
    
    # Tune
    best_thresholds = tune_thresholds(scores, patient_ids, gold_data, label_names)
    
    # Save
    out_dict = {label: float(thresh) for label, thresh in zip(label_names, best_thresholds)}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2)
        
    print(f"Thresholds saved to {args.out}")
