import os
import glob
import argparse
from evaluator import evaluate_file

def main():
    parser = argparse.ArgumentParser(description="Generate leaderboard for prediction files.")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL file")
    parser.add_argument("--pred-dir", required=True, help="Directory containing prediction JSONL files")
    args = parser.parse_args()
    
    pred_files = glob.glob(os.path.join(args.pred_dir, "*.jsonl"))
    if not pred_files:
        print(f"No .jsonl files found in {args.pred_dir}")
        return
        
    results = []
    
    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        try:
            metrics = evaluate_file(args.gold, pred_file)
            
            # Calculate average predictions per doc
            # We can estimate it by summing TP and FP, but wait, FP is calculated differently.
            # Let's just read the file again to count preds per doc
            total_preds = 0
            docs = 0
            import json
            with open(pred_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    flat_preds = []
                    for group in record.get('document_level_annotations', []):
                        flat_preds.extend(group)
                    total_preds += len(set(flat_preds))
                    docs += 1
            
            avg_preds = total_preds / docs if docs > 0 else 0.0
            
            results.append({
                'system': filename,
                'f1': metrics['micro_f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'avg_preds': avg_preds
            })
        except Exception as e:
            print(f"Error evaluating {filename}: {e}")
            
    # Sort by F1 descending
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    # Print table
    print(f"{'System':<30} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Avg Preds/Doc'}")
    print("-" * 75)
    for res in results:
        print(f"{res['system']:<30} | {res['f1']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['avg_preds']:.2f}")

if __name__ == "__main__":
    main()
