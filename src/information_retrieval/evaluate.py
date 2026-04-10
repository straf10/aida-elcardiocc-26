"""
Evaluate IR pipelines with ``src.evaluation.evaluator.evaluate_data``.

Supports mention-expanded corpus, relative score filtering, dictionary hybrid, tuning,
**BM25 / TF-IDF / embeddings / hybrid (RRF)** (``--retriever hybrid`` fuses BM25 + dense), and raw or processed splits.
"""

from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "training_set.jsonl"
PROCESSED_VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation_set.jsonl"

RetrieverKind = Literal["bm25", "tfidf", "embedding", "hybrid"]


def _patient_id(rec: dict) -> int:
    raw = rec.get("patient_id") or rec.get("id") or rec.get("doc_id") or rec.get("_row_id")
    return int(raw)


def build_ground_truth_map(records: list[dict]) -> dict[int, list[list[str]]]:
    return {_patient_id(r): (r.get("document_level_annotations") or []) for r in records}


def raw_records_by_patient_id(raw_path: str) -> dict[int, dict]:
    """Map ``patient_id`` → raw JSONL row (includes ``mention_level_annotations``)."""
    return {_patient_id(row): row for row in load_jsonl(raw_path)}


def raw_rows_for_processed_split(
    processed_records: list[dict],
    raw_by_pid: dict[int, dict],
) -> list[dict]:
    out: list[dict] = []
    for r in processed_records:
        raw = raw_by_pid.get(_patient_id(r))
        if raw is not None:
            out.append(raw)
    return out


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
    return evaluate_data(gt, pred, label_space=labelset)


def fit_retriever(
    kind: RetrieverKind,
    documents: list,
    *,
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    Fit lexical or dense retriever on the code corpus.

    ``embedding`` and ``hybrid`` require ``sentence-transformers`` (and a PyTorch backend).
    """
    if kind == "bm25":
        return BM25CodeRetriever().fit(documents)
    if kind == "tfidf":
        return TfidfCodeRetriever().fit(documents)
    if kind == "embedding":
        from .embedding_retrieval import EmbeddingCodeRetriever

        return EmbeddingCodeRetriever(model_name=embedding_model).fit(documents)
    if kind == "hybrid":
        from .embedding_retrieval import EmbeddingCodeRetriever
        from .hybrid_retrieval import HybridRrfRetriever

        bm25 = BM25CodeRetriever().fit(documents)
        dense = EmbeddingCodeRetriever(model_name=embedding_model).fit(documents)
        return HybridRrfRetriever(bm25, dense)
    raise ValueError(f"Unknown retriever kind: {kind!r}")


def _retriever_label(kind: RetrieverKind) -> str:
    return {
        "bm25": "BM25",
        "tfidf": "TF-IDF",
        "embedding": "Embedding (sentence-transformers)",
        "hybrid": "Hybrid RRF (BM25 + embeddings)",
    }[kind]


def tune_ir_hyperparams(
    val_records: list[dict],
    labelset: list[str],
    documents: list,
    *,
    kind: RetrieverKind = "bm25",
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    term_code_map: dict | None = None,
    grid: dict[str, list] | None = None,
) -> tuple[IRPredictionParams, dict[str, Any]]:
    """
    Small default grid over ``fraction_of_top_score``, ``max_codes``, ``include_dictionary``.

    Returns best params (by micro-F1 on ``val_records``) and that metric dict.
    """
    retriever = fit_retriever(kind, documents, embedding_model=embedding_model)
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
    parser = argparse.ArgumentParser(description="IR evaluation / tuning (ELCardioCC)")
    parser.add_argument(
        "--source",
        choices=("raw", "processed"),
        default="raw",
        help="raw: one train JSONL + 80/20 row split. processed: stratified splits under data/processed/.",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "tfidf", "embedding", "hybrid"),
        default="bm25",
        help="Lexical (bm25, tfidf), dense (embedding), or hybrid RRF of BM25+embedding.",
    )
    parser.add_argument(
        "--embedding-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for --retriever embedding or hybrid.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip validation grid search; use default IRPredictionParams for the tuned-metrics block (faster, esp. with embedding).",
    )
    args = parser.parse_args()

    train_path = Path(TRAIN_PATH)
    if not train_path.exists():
        print("Train file not found:", train_path)
        return

    labelset = load_labelset(str(LABELSET_PATH))
    term_map = load_term_code_csv(str(TERM_CODE_CSV))
    kind: RetrieverKind = args.retriever

    if args.source == "raw":
        records = load_jsonl(str(train_path))
        split = int(len(records) * 0.8)
        train_recs = records[:split]
        val_recs = records[split:]
        docs_exp_train_split = build_code_documents_with_mention_expansion(train_recs, codes=labelset)
        docs_exp_full = build_code_documents_with_mention_expansion(records, codes=labelset)
        data_source = "raw_train_jsonl_80_20_rows"
    else:
        if not PROCESSED_TRAIN_PATH.exists() or not PROCESSED_VAL_PATH.exists():
            print("Processed splits not found. Run src/data/cleaning.py first.")
            print(" ", PROCESSED_TRAIN_PATH)
            print(" ", PROCESSED_VAL_PATH)
            return
        train_proc = load_jsonl(str(PROCESSED_TRAIN_PATH))
        val_proc = load_jsonl(str(PROCESSED_VAL_PATH))
        records = train_proc + val_proc
        train_recs = train_proc
        val_recs = val_proc
        raw_by_pid = raw_records_by_patient_id(str(train_path))
        mining_train = raw_rows_for_processed_split(train_proc, raw_by_pid)
        mining_all = raw_rows_for_processed_split(records, raw_by_pid)
        docs_exp_train_split = build_code_documents_with_mention_expansion(mining_train, codes=labelset)
        docs_exp_full = build_code_documents_with_mention_expansion(mining_all, codes=labelset)
        data_source = "processed_train_val_stratified"

    docs_plain = build_code_documents(codes=labelset)
    gt = build_ground_truth_map(records)

    print(
        "Data source:", data_source,
        "| retriever:", kind,
        "| eval:", len(records), "docs | train split:", len(train_recs), "| val:", len(val_recs),
    )
    if kind in ("embedding", "hybrid"):
        print("Embedding model:", args.embedding_model)

    rlabel = _retriever_label(kind)
    print(f"\n=== Baseline: plain corpus, {rlabel} top-25 (no dictionary) ===")
    r0 = fit_retriever(kind, docs_plain, embedding_model=args.embedding_model)
    pred_top25: dict[int, list[str]] = {}
    for rec in records:
        pred_top25[_patient_id(rec)] = [h.code for h in r0.search(rec.get("text", ""), 25)]
    m_old = evaluate_data(gt, pred_top25, label_space=labelset)
    print("micro_f1", round(m_old["micro_f1"], 4), "precision", round(m_old["precision"], 4), "recall", round(m_old["recall"], 4))

    print(f"\n=== Expanded corpus + relative cut + hybrid (defaults), {rlabel} ===")
    r1 = fit_retriever(kind, docs_exp_full, embedding_model=args.embedding_model)
    p_default = IRPredictionParams()
    m_new = evaluate_ir_on_records(
        records, labelset, r1, params=p_default, term_code_map=term_map
    )
    print("micro_f1", round(m_new["micro_f1"], 4), "precision", round(m_new["precision"], 4), "recall", round(m_new["recall"], 4))

    if args.no_tune:
        print(f"\n=== Skipping val grid (--no-tune); using default IRPredictionParams ===")
        best_p = IRPredictionParams()
        best_val = evaluate_ir_on_records(
            val_recs, labelset, r1, params=best_p, term_code_map=term_map
        )
        print("Val micro_f1 (defaults)", round(best_val["micro_f1"], 4), "params", best_p)
    else:
        print(f"\n=== Tune on val ({len(val_recs)} docs); corpus from train-split mentions only ===")
        best_p, best_val = tune_ir_hyperparams(
            val_recs,
            labelset,
            docs_exp_train_split,
            kind=kind,
            embedding_model=args.embedding_model,
            term_code_map=term_map,
        )
        print("Best val micro_f1", round(best_val["micro_f1"], 4), "params", best_p)

    print(f"\n=== Refit {rlabel} on full expanded corpus; tuned params; all eval records ===")
    if args.no_tune:
        r_final = r1
    else:
        r_final = fit_retriever(kind, docs_exp_full, embedding_model=args.embedding_model)
    m_tuned = evaluate_ir_on_records(
        records, labelset, r_final, params=best_p, term_code_map=term_map
    )
    print("micro_f1", round(m_tuned["micro_f1"], 4), "precision", round(m_tuned["precision"], 4), "recall", round(m_tuned["recall"], 4))

    out_dir = PROJECT_ROOT / "outputs" / "information_retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_source": data_source,
        "retriever": kind,
        "embedding_model": args.embedding_model if kind in ("embedding", "hybrid") else None,
        "no_tune": args.no_tune,
        "baseline_top25_plain_corpus": {k: m_old[k] for k in ("micro_f1", "precision", "recall")},
        "default_expanded_hybrid": {k: m_new[k] for k in ("micro_f1", "precision", "recall")},
        "tuned_params": {
            "fraction_of_top_score": best_p.fraction_of_top_score,
            "max_codes": best_p.max_codes,
            "include_dictionary": best_p.include_dictionary,
        },
        "tuned_full_train": {k: m_tuned[k] for k in ("micro_f1", "precision", "recall")},
    }
    out_name = {
        "bm25": "ir_tune_summary_bm25.json",
        "tfidf": "ir_tune_summary_tfidf.json",
        "embedding": "ir_tune_summary_embedding.json",
        "hybrid": "ir_tune_summary_hybrid.json",
    }[kind]
    out_path = out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
