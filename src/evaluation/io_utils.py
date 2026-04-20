import json
from pathlib import Path
from typing import Dict, Iterable, List

from preprocessing.io_utils import load_jsonl, save_jsonl


def flatten_annotation_groups(annotation_groups: List[List[str]]) -> List[str]:
    seen = set()
    flattened: List[str] = []
    for group in annotation_groups:
        for code in group:
            if code not in seen:
                seen.add(code)
                flattened.append(code)
    return flattened


def load_ground_truth(path: str) -> Dict[int, List[List[str]]]:
    ground_truth_data: Dict[int, List[List[str]]] = {}
    for record in load_jsonl(path):
        patient_id = int(record["patient_id"])
        groups = record.get("document_level_annotations", [])
        if groups is None:
            groups = []
        ground_truth_data[patient_id] = groups
    return ground_truth_data


def load_predictions(path: str) -> Dict[int, List[str]]:
    pred_data: Dict[int, List[str]] = {}
    for record in load_jsonl(path):
        patient_id = int(record["patient_id"])
        groups = record.get("document_level_annotations", [])
        if groups is None:
            groups = []
        pred_data[patient_id] = flatten_annotation_groups(groups)
    return pred_data


def save_predictions_jsonl(pred_data: Dict[int, List[str]], path: str | Path) -> None:
    """
    Write flat per-code predictions in the same JSONL shape expected by ``load_predictions``.

    Each line: ``{"patient_id": int, "document_level_annotations": [[code], ...]}`` with one
    inner list per predicted code (singleton groups).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for patient_id in sorted(pred_data.keys()):
            codes = sorted(set(pred_data[patient_id]))
            ann = [[c] for c in codes]
            handle.write(
                json.dumps(
                    {"patient_id": patient_id, "document_level_annotations": ann},
                    ensure_ascii=False,
                )
                + "\n"
            )


def average_pred_codes_per_doc(pred_data: Dict[int, List[str]]) -> float:
    if not pred_data:
        return 0.0
    total = sum(len(set(codes)) for codes in pred_data.values())
    return total / len(pred_data)
