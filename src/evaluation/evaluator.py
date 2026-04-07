import json
import argparse
from typing import List, Set, Tuple, Dict

def score_document(gold_groups: List[List[str]], pred_codes: List[str]) -> Tuple[int, int, int]:
    """
    Scores a single document based on the list-of-lists synonym group logic.
    
    Args:
        gold_groups: List of lists, where each inner list contains synonymous ICD-10 codes.
        pred_codes: Flat list of predicted ICD-10 codes for the document.
        
    Returns:
        (tp, fp, fn) tuple.
    """
    pred_set = set(pred_codes)
    
    tp = 0
    # A prediction is a true positive if it hits at least one code in a gold group
    for group in gold_groups:
        group_set = set(group)
        if pred_set.intersection(group_set):
            tp += 1
            
    fn = len(gold_groups) - tp
    
    # FP = number of predicted codes not in *any* gold group
    all_gold_codes = set(code for group in gold_groups for code in group)
    fp = len([code for code in pred_set if code not in all_gold_codes])
    
    return tp, fp, fn

def micro_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculates micro-averaged Precision, Recall, and F1 score.
    """
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def evaluate_file(gold_jsonl_path: str, pred_jsonl_path: str) -> Dict:
    """
    Evaluates a prediction JSONL file against a gold JSONL file.
    """
    gold_data = {}
    with open(gold_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            gold_data[record['patient_id']] = record['document_level_annotations']
            
    pred_data = {}
    with open(pred_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # Flatten predictions to a single set of codes
            flat_preds = []
            for group in record.get('document_level_annotations', []):
                flat_preds.extend(group)
            pred_data[record['patient_id']] = list(set(flat_preds))
            
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for patient_id, gold_groups in gold_data.items():
        pred_codes = pred_data.get(patient_id, [])
        tp, fp, fn = score_document(gold_groups, pred_codes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    p, r, f1 = micro_f1(total_tp, total_fp, total_fn)
    
    return {
        'micro_f1': f1,
        'precision': p,
        'recall': r,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'docs_evaluated': len(gold_data)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ELCardioCC predictions.")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL file")
    parser.add_argument("--pred", required=True, help="Path to prediction JSONL file")
    args = parser.parse_args()
    
    metrics = evaluate_file(args.gold, args.pred)
    print(f"Evaluated {metrics['docs_evaluated']} documents.")
    print(f"Micro-F1:  {metrics['micro_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"TP: {metrics['total_tp']} | FP: {metrics['total_fp']} | FN: {metrics['total_fn']}")
