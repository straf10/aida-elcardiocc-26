from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.evaluator import per_class_report, score_document
from src.evaluation.io_utils import load_ground_truth


def test_scoring_logic() -> None:
    ground_truth = [["R55"], ["I10", "I11"], ["I44"], ["Z95"]]

    pred = ["R55", "I10", "I44", "Z95"]
    assert score_document(ground_truth, pred) == (4, 0, 0)

    pred = ["R55", "I11", "I44", "Z95"]
    assert score_document(ground_truth, pred) == (4, 0, 0)

    pred = ["R55", "I10", "I11", "I44", "Z95"]
    assert score_document(ground_truth, pred) == (4, 0, 0)

    pred = ["R55", "I10", "I44", "E11"]
    assert score_document(ground_truth, pred) == (3, 1, 1)

    pred = []
    assert score_document(ground_truth, pred) == (0, 0, 4)


def test_per_class_report_basic() -> None:
    ground_truth_data = {1: [["I10", "I11"], ["R55"]]}
    pred_data = {1: ["I10", "E11"]}
    labels = ["I10", "I11", "R55", "E11"]
    report = per_class_report(ground_truth_data, pred_data, labels)
    lookup = {row["code"]: row for row in report}

    assert lookup["I10"]["support"] == 1
    assert lookup["I10"]["groups_hit"] == 1
    assert lookup["I10"]["fp_count"] == 0
    assert lookup["R55"]["groups_hit"] == 0
    assert lookup["E11"]["fp_count"] == 1


def test_known_patient_from_train_set_2026() -> None:
    train_path = Path("data/Train_Set_2026/train_dataset.jsonl")
    records = load_ground_truth(str(train_path))
    expected = [["R55"], ["I10", "I11"], ["I44"], ["Z95"]]
    assert records[2] == expected


if __name__ == "__main__":
    test_scoring_logic()
    test_per_class_report_basic()
    test_known_patient_from_train_set_2026()
    print("All evaluator tests passed.")
