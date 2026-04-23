"""Code description lookup and JSONL submission export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

from preprocessing.io_utils import resolve_patient_id

from .matcher import predict_codes_for_text

if TYPE_CHECKING:
    from .config import DictionaryConfig


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_code_description_csv(csv_path: str) -> dict:
    """Load code→greek_description mapping from CSV."""
    code_desc: dict[str, str] = {}
    if not Path(csv_path).exists():
        print(f"WARNING: {csv_path} not found — skipping code lookup")
        return code_desc
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"].strip()
            desc = row["greek_description"].strip()
            if code and desc:
                code_desc[code] = desc
    return code_desc


def export_code_lookup(code_desc_map: dict, output_path: str, *, output_dir: Path | None = None) -> None:
    """Export code→description mapping as JSON for easy use by other workstreams."""
    od = output_dir
    if od is None:
        from preprocessing.io_utils import PROJECT_ROOT

        od = PROJECT_ROOT / "outputs" / "models" / "dictionary_baseline"
    _ensure_output_dir(od)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(code_desc_map, f, ensure_ascii=False, indent=2)
    print(f"Code lookup saved: {output_path}")


def export_predictions_jsonl(
    records: list,
    matcher,
    output_path: str,
    *,
    config: "DictionaryConfig | None" = None,
    labelset: list[str] | None = None,
    code_desc_map: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Export predictions to JSONL submission format."""
    od = output_dir
    if od is None:
        from preprocessing.io_utils import PROJECT_ROOT

        od = PROJECT_ROOT / "outputs" / "models" / "dictionary_baseline"
    _ensure_output_dir(od)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            pred = predict_codes_for_text(
                rec.get("text", ""),
                matcher,
                config=config,
                labelset=labelset,
                code_desc_map=code_desc_map,
            )
            doc_annotations = [[code] for code in sorted(pred)]
            pid = resolve_patient_id(rec)
            line = {
                "patient_id": pid,
                "document_level_annotations": doc_annotations,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Predictions saved: {output_path}")
