"""
Evaluate IR pipelines with ``src.evaluation.evaluator.evaluate_data``.

Supports mention-expanded corpus (fit on train mentions only), relative score filtering,
and dictionary hybrid. Optional small grid search on a validation set.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Literal

try:
    from src.dictionary.dictionary import (
        LABELSET_PATH,
        TRAIN_PATH,
        load_jsonl,
        load_labelset,
        load_term_code_csv,
        TERM_CODE_CSV,
    )
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from ..dictionary.dictionary import (
        LABELSET_PATH,
        TRAIN_PATH,
        load_jsonl,
        load_labelset,
        load_term_code_csv,
        TERM_CODE_CSV,
    )
    from ..evaluation.evaluator import evaluate_data

from .corpus import build_code_documents, build_code_documents_with_mention_expansion
from .prediction import IRPredictionParams, predict_codes_from_retriever
from .term_retrieval import BM25CodeRetriever, TfidfCodeRetriever

RetrieverKind = Literal["bm25", "tfidf"]


def _patient_id(rec: dict) -> int:
    raw = rec.get("patient_id") or rec.get("id") or rec.get("doc_id") or rec.get("_row_id")
    return int(raw)


def build_ground_truth_map(records: list[dict]) -> dict[int, list[list[str]]]:
    return { _patient_id(r): (r.get("document_level_annotations") or []) for r in records }


def evaluate_ir_on_records(
    records: list[dict],
    labelset: list[str],
    retriever,
    *,
    params: IRPredictionParams | None = None,
    term_code_map: dict | None = None,
) -> dict[str, Any]:
    """Run IR (+ optional dictionary) on ``records`` and return evaluator metrics."""
    gt = build_ground_truth_map(records)
    pred: dict[int, list[str]] = {}
    p = params or IRPredictionParams()
    for rec in records:
        pid = _patient_id(rec)
        text = rec.get("text", "")
        pred[pid] = predict_codes_from_retriever(
            text, retriever, p, term_code_map=term_code_map
        )
    metrics = evaluate_data(gt, pred, label_space=labelset)
    return metrics


def fit_retriever(
    kind: RetrieverKind,
    documents: list,
) -> BM25CodeRetriever | TfidfCodeRetriever:
    if kind == "bm25":
        return BM25CodeRetriever().fit(documents)
    if kind == "tfidf":
        return TfidfCodeRetriever().fit(documents)
    raise ValueError(f"Unknown retriever kind: {kind!r}")


def tune_ir_hyperparams(
    val_records: list[dict],
    labelset: list[str],
    documents: list,
    *,
    kind: RetrieverKind = "bm25",
    term_code_map: dict | None = None,
    grid: dict[str, list] | None = None,
) -> tuple[IRPredictionParams, dict[str, Any]]:
    """
    Small default grid over ``fraction_of_top_score``, ``max_codes``, ``include_dictionary``.

    Returns best params (by micro-F1 on ``val_records``) and that metric dict.
    """
    retriever = fit_retriever(kind, documents)
    grid = grid or {
        "fraction_of_top_score": [0.15, 0.22, 0.30, 0.40],
        "max_codes": [8, 12, 16],
        "include_dictionary": [True, False],
    }

    keys = list(grid.keys())
    best_metrics: dict[str, Any] | None = None
    best_params: IRPredictionParams | None = None

    for values in product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, values))
        params = IRPredictionParams(
            search_top_k=80,
            fraction_of_top_score=kw["fraction_of_top_score"],
            max_codes=kw["max_codes"],
            include_dictionary=kw["include_dictionary"],
        )
        metrics = evaluate_ir_on_records(
            val_records, labelset, retriever, params=params, term_code_map=term_code_map
        )
        f1 = metrics["micro_f1"]
        if best_metrics is None or f1 > best_metrics["micro_f1"]:
            best_metrics = metrics
            best_params = params

    assert best_params is not None and best_metrics is not None
    return best_params, best_metrics


def main() -> None:
    """Train-only mention expansion, optional val tune, eval on full train (sanity)."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    train_path = Path(TRAIN_PATH)
    if not train_path.exists():
        print("Train file not found:", train_path)
        return

    records = load_jsonl(str(train_path))
    labelset = load_labelset(str(LABELSET_PATH))
    term_map = load_term_code_csv(str(TERM_CODE_CSV))

    # 80/20 chronological split by row order as a simple holdout (same as many baselines).
    n = len(records)
    split = int(n * 0.8)
    train_recs = records[:split]
    val_recs = records[split:]

    docs_plain = build_code_documents(codes=labelset)
    docs_exp_train_split = build_code_documents_with_mention_expansion(train_recs, codes=labelset)
    docs_exp_full = build_code_documents_with_mention_expansion(records, codes=labelset)

    gt = build_ground_truth_map(records)

    print("=== Baseline: plain corpus, BM25 fixed top-25 (no dictionary) ===")
    r0 = BM25CodeRetriever().fit(docs_plain)
    pred_top25: dict[int, list[str]] = {}
    for rec in records:
        pred_top25[_patient_id(rec)] = [h.code for h in r0.search(rec.get("text", ""), 25)]
    m_old = evaluate_data(gt, pred_top25, label_space=labelset)
    print("micro_f1", round(m_old["micro_f1"], 4), "precision", round(m_old["precision"], 4), "recall", round(m_old["recall"], 4))

    print("\n=== Expanded corpus (all-train mentions), relative cut + hybrid (defaults) ===")
    r1 = BM25CodeRetriever().fit(docs_exp_full)
    p_default = IRPredictionParams()
    m_new = evaluate_ir_on_records(
        records, labelset, r1, params=p_default, term_code_map=term_map
    )
    print("micro_f1", round(m_new["micro_f1"], 4), "precision", round(m_new["precision"], 4), "recall", round(m_new["recall"], 4))

    print("\n=== Tune on val (%d docs); corpus mined from first 80%% train only ===" % len(val_recs))
    best_p, best_val = tune_ir_hyperparams(
        val_recs, labelset, docs_exp_train_split, kind="bm25", term_code_map=term_map
    )
    print("Best val micro_f1", round(best_val["micro_f1"], 4), "params", best_p)

    print("\n=== Refit BM25 on full-train expanded corpus; same params; eval on full train ===")
    r2 = BM25CodeRetriever().fit(docs_exp_full)
    m_tuned = evaluate_ir_on_records(
        records, labelset, r2, params=best_p, term_code_map=term_map
    )
    print("micro_f1", round(m_tuned["micro_f1"], 4), "precision", round(m_tuned["precision"], 4), "recall", round(m_tuned["recall"], 4))

    out_dir = project_root / "outputs" / "information_retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_top25_plain_corpus": {k: m_old[k] for k in ("micro_f1", "precision", "recall")},
        "default_expanded_hybrid": {k: m_new[k] for k in ("micro_f1", "precision", "recall")},
        "tuned_params": {
            "fraction_of_top_score": best_p.fraction_of_top_score,
            "max_codes": best_p.max_codes,
            "include_dictionary": best_p.include_dictionary,
        },
        "tuned_full_train": {k: m_tuned[k] for k in ("micro_f1", "precision", "recall")},
    }
    with open(out_dir / "ir_tune_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nWrote", out_dir / "ir_tune_summary.json")


if __name__ == "__main__":
    main()
