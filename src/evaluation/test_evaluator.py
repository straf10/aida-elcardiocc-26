import json
from evaluator import score_document

def test_scoring_logic():
    # Example 1: Perfect match with exact codes
    ground_truth = [["R55"], ["I10", "I11"], ["I44"], ["Z95"]]
    pred = ["R55", "I10", "I44", "Z95"]
    tp, fp, fn = score_document(ground_truth, pred)
    assert (tp, fp, fn) == (4, 0, 0), f"Ex 1 failed: {tp}, {fp}, {fn}"
    
    # Example 2: Perfect match with alternate synonym
    ground_truth = [["R55"], ["I10", "I11"], ["I44"], ["Z95"]]
    pred = ["R55", "I11", "I44", "Z95"]
    tp, fp, fn = score_document(ground_truth, pred)
    assert (tp, fp, fn) == (4, 0, 0), f"Ex 2 failed: {tp}, {fp}, {fn}"

    # Example 3: Multiple synonyms predicted for the same group (neutral FP)
    ground_truth = [["R55"], ["I10", "I11"], ["I44"], ["Z95"]]
    pred = ["R55", "I10", "I11", "I44", "Z95"]
    tp, fp, fn = score_document(ground_truth, pred)
    assert (tp, fp, fn) == (4, 0, 0), f"Ex 3 failed: {tp}, {fp}, {fn}"

    # Example 4: Missing one group, one wrong code
    ground_truth = [["R55"], ["I10", "I11"], ["I44"], ["Z95"]]
    pred = ["R55", "I10", "I44", "E11"]
    tp, fp, fn = score_document(ground_truth, pred)
    assert (tp, fp, fn) == (3, 1, 1), f"Ex 4 failed: {tp}, {fp}, {fn}"
    
    # Example 5: Empty prediction
    ground_truth = [["R55"], ["I10", "I11"], ["I44"], ["Z95"]]
    pred = []
    tp, fp, fn = score_document(ground_truth, pred)
    assert (tp, fp, fn) == (0, 0, 4), f"Ex 5 failed: {tp}, {fp}, {fn}"

    print("All 5 hand-verified examples passed successfully!")

if __name__ == "__main__":
    test_scoring_logic()
