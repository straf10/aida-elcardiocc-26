"""
Evaluate IR pipelines with ``src.evaluation.evaluator.evaluate_data``.

Supports mention-expanded corpus, relative score filtering, dictionary-aware prediction
strategies, tuning, and **BM25 / TF-IDF / embeddings / hybrid (RRF)** retrieval.

Default data source is ``processed`` splits under ``data/processed/``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Literal

try:
    from src.dictionary.dictionary import (
        load_term_code_csv,
        predict_codes_for_text,
    )
    from src.preprocessing.io_utils import (
        LABELSET_PATH,
        TRAIN_PATH,
        TERM_CODE_CSV,
        load_jsonl,
        load_labelset,
    )
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from ..dictionary.dictionary import (
        load_term_code_csv,
        predict_codes_for_text,
    )
    from ..preprocessing.io_utils import (
        LABELSET_PATH,
        TRAIN_PATH,
        TERM_CODE_CSV,
        load_jsonl,
        load_labelset,
    )
    from ..evaluation.evaluator import evaluate_data

from .corpus import build_code_documents, build_code_documents_with_mention_expansion
from .prediction import IRPredictionParams, predict_codes_from_retriever
from .types import RetrievalHit
from .term_retrieval import BM25CodeRetriever, TfidfCodeRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "training_set_raw.jsonl"
PROCESSED_VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation_set_raw.jsonl"

RetrieverKind = Literal["bm25", "tfidf", "embedding", "hybrid"]
PredictionStrategy = Literal["standard", "dict-rerank"]


def _patient_id(rec: dict) -> int:
    raw = rec.get("patient_id") or rec.get("id") or rec.get("doc_id") or rec.get("_row_id")
    return int(raw)


def build_ground_truth_map(records: list[dict]) -> dict[int, list[list[str]]]:
    return {_patient_id(r): (r.get("document_level_annotations") or []) for r in records}


def evaluate_ir_on_records(
    records: list[dict],
    labelset: list[str],
    retriever,
    *,
    params: IRPredictionParams | None = None,
    term_code_map: dict | None = None,
    strategy: PredictionStrategy = "standard",
    fallback_to_standard_if_no_dictionary: bool = True,
) -> dict[str, Any]:
    """Run IR (+ optional dictionary) on ``records`` and return evaluator metrics."""
    gt = build_ground_truth_map(records)
    pred: dict[int, list[str]] = {}
    p = params or IRPredictionParams()
    for rec in records:
        pid = _patient_id(rec)
        text = rec.get("text", "")
        if strategy == "dict-rerank":
            pred[pid] = predict_codes_with_dictionary_rerank(
                text,
                retriever,
                p,
                term_code_map=term_code_map,
                fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
            )
        else:
            pred[pid] = predict_codes_from_retriever(
                text, retriever, p, term_code_map=term_code_map
            )
    return evaluate_data(gt, pred, label_space=labelset)


def predict_codes_with_dictionary_rerank(
    text: str,
    retriever,
    params: IRPredictionParams,
    *,
    term_code_map: dict | None = None,
    fallback_to_standard_if_no_dictionary: bool = True,
) -> list[str]:
    """
    Candidate-first prediction: dictionary candidates are reranked by IR scores.

    If dictionary yields no candidates, optionally fall back to standard IR flow.
    """
    dict_candidates = (
        predict_codes_for_text(text, term_code_map) if term_code_map is not None else set()
    )
    if not dict_candidates:
        if fallback_to_standard_if_no_dictionary:
            return predict_codes_from_retriever(text, retriever, params, term_code_map=term_code_map)
        return []

    hits: list[RetrievalHit] = retriever.search(text, top_k=params.search_top_k)
    rank_by_code = {h.code: rank for rank, h in enumerate(hits, start=1)}
    score_by_code = {h.code: h.score for h in hits}
    best_score = hits[0].score if hits else 0.0

    ranked: list[tuple[str, float, int]] = []
    for code in dict_candidates:
        raw_score = score_by_code.get(code, 0.0)
        rel_score = (raw_score / best_score) if best_score > 0 else 0.0
        rank = rank_by_code.get(code, 10**9)
        ranked.append((code, rel_score, rank))

    ranked.sort(key=lambda x: (-x[1], x[2], x[0]))
    threshold = max(0.0, params.fraction_of_top_score)
    kept = [code for code, rel, _ in ranked if rel >= threshold][: params.max_codes]
    if not kept and ranked:
        kept = [ranked[0][0]]
    return sorted(kept)


def fit_retriever(
    kind: RetrieverKind,
    documents: list,
    *,
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    hybrid_rrf_k: int = 60,
    hybrid_bm25_weight: float = 1.0,
    hybrid_dense_weight: float = 1.0,
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
        return HybridRrfRetriever(
            bm25,
            dense,
            rrf_k=hybrid_rrf_k,
            bm25_weight=hybrid_bm25_weight,
            dense_weight=hybrid_dense_weight,
        )
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
    strategy: PredictionStrategy = "standard",
    fallback_to_standard_if_no_dictionary: bool = True,
) -> tuple[IRPredictionParams, dict[str, Any]]:
    """
    Small default grid over ``fraction_of_top_score``, ``max_codes``, ``include_dictionary``.

    Returns best params (by micro-F1 on ``val_records``) and that metric dict.
    """
    hybrid_components = None
    if kind == "hybrid":
        from .embedding_retrieval import EmbeddingCodeRetriever
        from .hybrid_retrieval import HybridRrfRetriever

        bm25 = BM25CodeRetriever().fit(documents)
        dense = EmbeddingCodeRetriever(model_name=embedding_model).fit(documents)
        hybrid_components = (bm25, dense, HybridRrfRetriever)

    if grid is None:
        if kind == "hybrid":
            # Keep hybrid grid compact; each point runs full retrieval over val.
            grid = {
                "fraction_of_top_score": [0.22, 0.40],
                "max_codes": [8, 12],
                "include_dictionary": [True],
                "hybrid_rrf_k": [20, 60],
                "hybrid_bm25_weight": [0.8, 1.2],
                "hybrid_dense_weight": [0.8, 1.2],
            }
        else:
            if strategy == "dict-rerank":
                # Finer grid around lower thresholds where dict-rerank performs best.
                grid = {
                    "fraction_of_top_score": [0.08, 0.10, 0.12, 0.15, 0.18, 0.22],
                    "max_codes": [8, 10, 12, 14, 16],
                    "include_dictionary": [True],
                }
            else:
                grid = {
                    "fraction_of_top_score": [0.15, 0.22, 0.30, 0.40],
                    "max_codes": [8, 12, 16],
                    "include_dictionary": [True, False],
                }

    keys = list(grid.keys())
    best_metrics: dict[str, Any] | None = None
    best_params: IRPredictionParams | None = None
    best_hybrid_kwargs = {
        "hybrid_rrf_k": 60,
        "hybrid_bm25_weight": 1.0,
        "hybrid_dense_weight": 1.0,
    }

    for values in product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, values))
        if kind == "hybrid" and hybrid_components is not None:
            bm25, dense, HybridRrfRetriever = hybrid_components
            retriever = HybridRrfRetriever(
                bm25,
                dense,
                rrf_k=int(kw.get("hybrid_rrf_k", 60)),
                bm25_weight=float(kw.get("hybrid_bm25_weight", 1.0)),
                dense_weight=float(kw.get("hybrid_dense_weight", 1.0)),
            )
        else:
            retriever = fit_retriever(
                kind,
                documents,
                embedding_model=embedding_model,
                hybrid_rrf_k=int(kw.get("hybrid_rrf_k", 60)),
                hybrid_bm25_weight=float(kw.get("hybrid_bm25_weight", 1.0)),
                hybrid_dense_weight=float(kw.get("hybrid_dense_weight", 1.0)),
            )
        params = IRPredictionParams(
            search_top_k=80,
            fraction_of_top_score=kw["fraction_of_top_score"],
            max_codes=kw["max_codes"],
            include_dictionary=kw["include_dictionary"],
        )
        metrics = evaluate_ir_on_records(
            val_records,
            labelset,
            retriever,
            params=params,
            term_code_map=term_code_map,
            strategy=strategy,
            fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
        )
        f1 = metrics["micro_f1"]
        if best_metrics is None or f1 > best_metrics["micro_f1"]:
            best_metrics = metrics
            best_params = params
            best_hybrid_kwargs = {
                "hybrid_rrf_k": int(kw.get("hybrid_rrf_k", 60)),
                "hybrid_bm25_weight": float(kw.get("hybrid_bm25_weight", 1.0)),
                "hybrid_dense_weight": float(kw.get("hybrid_dense_weight", 1.0)),
            }

    assert best_params is not None and best_metrics is not None
    if kind == "hybrid":
        best_metrics["_best_hybrid_kwargs"] = best_hybrid_kwargs
    return best_params, best_metrics


def _flatten_gold_codes(rec: dict) -> set[str]:
    return {code for group in (rec.get("document_level_annotations") or []) for code in group}


def learn_per_code_fraction_thresholds(
    records: list[dict],
    retriever,
    *,
    search_top_k: int = 80,
    candidate_thresholds: list[float] | None = None,
    min_pos_hits: int = 5,
    min_total_hits: int = 12,
) -> dict[str, float]:
    """
    Learn per-code relative-score cutoffs from validation retrieval hits.

    For each code, choose the threshold that maximizes binary F1 on retrieved hits where
    positive means "code belongs to this document's gold set".
    """
    candidate_thresholds = candidate_thresholds or [0.10, 0.15, 0.22, 0.30, 0.40, 0.50]
    obs: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for rec in records:
        text = rec.get("text", "")
        hits: list[RetrievalHit] = retriever.search(text, top_k=search_top_k)
        if not hits or hits[0].score <= 0:
            continue
        best = hits[0].score
        gold_codes = _flatten_gold_codes(rec)
        for h in hits:
            rel = float(h.score / best)
            obs[h.code].append((rel, 1 if h.code in gold_codes else 0))

    out: dict[str, float] = {}
    for code, values in obs.items():
        pos = sum(y for _, y in values)
        if pos < min_pos_hits or len(values) < min_total_hits:
            continue

        best_thr = candidate_thresholds[0]
        best_f1 = -1.0
        for thr in candidate_thresholds:
            tp = fp = fn = 0
            for rel, y in values:
                pred = rel >= thr
                if pred and y == 1:
                    tp += 1
                elif pred and y == 0:
                    fp += 1
                elif (not pred) and y == 1:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        out[code] = best_thr
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="IR evaluation / tuning (ELCardioCC)")
    parser.add_argument(
        "--source",
        choices=("raw", "processed"),
        default="processed",
        help="processed (default): stratified splits under data/processed/. raw: one train JSONL + 80/20 row split.",
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
    parser.add_argument(
        "--enable-per-code-thresholds",
        action="store_true",
        help="Learn per-code relative-score thresholds on val and apply at inference.",
    )
    parser.add_argument(
        "--hybrid-rrf-k",
        type=int,
        default=60,
        help="RRF k for hybrid retrieval.",
    )
    parser.add_argument(
        "--hybrid-bm25-weight",
        type=float,
        default=1.0,
        help="BM25 channel weight for hybrid RRF.",
    )
    parser.add_argument(
        "--hybrid-dense-weight",
        type=float,
        default=1.0,
        help="Dense channel weight for hybrid RRF.",
    )
    parser.add_argument(
        "--prediction-strategy",
        choices=("standard", "dict-rerank"),
        default="standard",
        help="standard: existing IR+optional dictionary union; dict-rerank: dictionary candidates reranked by IR.",
    )
    parser.add_argument(
        "--no-fallback-standard-when-no-dict",
        action="store_true",
        help="With dict-rerank, do not fall back to standard prediction when dictionary finds no candidates.",
    )
    args = parser.parse_args()

    labelset = load_labelset(str(LABELSET_PATH))
    term_map = load_term_code_csv(str(TERM_CODE_CSV))
    kind: RetrieverKind = args.retriever
    strategy: PredictionStrategy = args.prediction_strategy
    fallback_to_standard_if_no_dictionary = not args.no_fallback_standard_when_no_dict
    selected_hybrid_rrf_k = args.hybrid_rrf_k
    selected_hybrid_bm25_weight = args.hybrid_bm25_weight
    selected_hybrid_dense_weight = args.hybrid_dense_weight

    if args.source == "raw":
        train_path = Path(TRAIN_PATH)
        if not train_path.exists():
            print("Train file not found:", train_path)
            return
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
        docs_exp_train_split = build_code_documents_with_mention_expansion(train_recs, codes=labelset)
        docs_exp_full = build_code_documents_with_mention_expansion(records, codes=labelset)
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
    r0 = fit_retriever(
        kind,
        docs_plain,
        embedding_model=args.embedding_model,
        hybrid_rrf_k=selected_hybrid_rrf_k,
        hybrid_bm25_weight=selected_hybrid_bm25_weight,
        hybrid_dense_weight=selected_hybrid_dense_weight,
    )
    pred_top25: dict[int, list[str]] = {}
    for rec in records:
        pred_top25[_patient_id(rec)] = [h.code for h in r0.search(rec.get("text", ""), 25)]
    m_old = evaluate_data(gt, pred_top25, label_space=labelset)
    print("micro_f1", round(m_old["micro_f1"], 4), "precision", round(m_old["precision"], 4), "recall", round(m_old["recall"], 4))

    print(f"\n=== Expanded corpus + relative cut + hybrid (defaults), {rlabel} ===")
    r1 = fit_retriever(
        kind,
        docs_exp_full,
        embedding_model=args.embedding_model,
        hybrid_rrf_k=selected_hybrid_rrf_k,
        hybrid_bm25_weight=selected_hybrid_bm25_weight,
        hybrid_dense_weight=selected_hybrid_dense_weight,
    )
    p_default = IRPredictionParams()
    m_new = evaluate_ir_on_records(
        records,
        labelset,
        r1,
        params=p_default,
        term_code_map=term_map,
        strategy=strategy,
        fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
    )
    print("micro_f1", round(m_new["micro_f1"], 4), "precision", round(m_new["precision"], 4), "recall", round(m_new["recall"], 4))

    if args.no_tune:
        print(f"\n=== Skipping val grid (--no-tune); using default IRPredictionParams ===")
        best_p = IRPredictionParams()
        best_val = evaluate_ir_on_records(
            val_recs,
            labelset,
            r1,
            params=best_p,
            term_code_map=term_map,
            strategy=strategy,
            fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
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
            strategy=strategy,
            fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
        )
        print("Best val micro_f1", round(best_val["micro_f1"], 4), "params", best_p)
        if kind == "hybrid":
            tuned = best_val.get("_best_hybrid_kwargs") or {}
            selected_hybrid_rrf_k = int(tuned.get("hybrid_rrf_k", selected_hybrid_rrf_k))
            selected_hybrid_bm25_weight = float(
                tuned.get("hybrid_bm25_weight", selected_hybrid_bm25_weight)
            )
            selected_hybrid_dense_weight = float(
                tuned.get("hybrid_dense_weight", selected_hybrid_dense_weight)
            )
            print(
                "Best hybrid fusion:",
                {
                    "rrf_k": selected_hybrid_rrf_k,
                    "bm25_weight": selected_hybrid_bm25_weight,
                    "dense_weight": selected_hybrid_dense_weight,
                },
            )

    if args.enable_per_code_thresholds:
        print("\n=== Learn per-code relative-score thresholds on val ===")
        thr_retriever = fit_retriever(
            kind,
            docs_exp_train_split,
            embedding_model=args.embedding_model,
            hybrid_rrf_k=selected_hybrid_rrf_k,
            hybrid_bm25_weight=selected_hybrid_bm25_weight,
            hybrid_dense_weight=selected_hybrid_dense_weight,
        )
        code_thr = learn_per_code_fraction_thresholds(
            val_recs,
            thr_retriever,
            search_top_k=best_p.search_top_k,
        )
        best_p = IRPredictionParams(
            search_top_k=best_p.search_top_k,
            fraction_of_top_score=best_p.fraction_of_top_score,
            max_codes=best_p.max_codes,
            min_ir_codes=best_p.min_ir_codes,
            include_dictionary=best_p.include_dictionary,
            code_fraction_thresholds=code_thr,
        )
        best_val = evaluate_ir_on_records(
            val_recs,
            labelset,
            thr_retriever,
            params=best_p,
            term_code_map=term_map,
            strategy=strategy,
            fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
        )
        print("Codes with learned thresholds:", len(code_thr))
        print("Val micro_f1 (with per-code thresholds)", round(best_val["micro_f1"], 4))

    print(f"\n=== Refit {rlabel} on full expanded corpus; tuned params; all eval records ===")
    if args.no_tune:
        r_final = r1
    else:
        r_final = fit_retriever(
            kind,
            docs_exp_full,
            embedding_model=args.embedding_model,
            hybrid_rrf_k=selected_hybrid_rrf_k,
            hybrid_bm25_weight=selected_hybrid_bm25_weight,
            hybrid_dense_weight=selected_hybrid_dense_weight,
        )
    m_tuned = evaluate_ir_on_records(
        records,
        labelset,
        r_final,
        params=best_p,
        term_code_map=term_map,
        strategy=strategy,
        fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
    )
    print("micro_f1", round(m_tuned["micro_f1"], 4), "precision", round(m_tuned["precision"], 4), "recall", round(m_tuned["recall"], 4))

    out_dir = PROJECT_ROOT / "outputs" / "information_retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_source": data_source,
        "retriever": kind,
        "prediction_strategy": strategy,
        "fallback_to_standard_if_no_dictionary": fallback_to_standard_if_no_dictionary,
        "embedding_model": args.embedding_model if kind in ("embedding", "hybrid") else None,
        "enable_per_code_thresholds": args.enable_per_code_thresholds,
        "hybrid_rrf_k": selected_hybrid_rrf_k if kind == "hybrid" else None,
        "hybrid_bm25_weight": selected_hybrid_bm25_weight if kind == "hybrid" else None,
        "hybrid_dense_weight": selected_hybrid_dense_weight if kind == "hybrid" else None,
        "no_tune": args.no_tune,
        "baseline_top25_plain_corpus": {k: m_old[k] for k in ("micro_f1", "precision", "recall")},
        "default_expanded_hybrid": {k: m_new[k] for k in ("micro_f1", "precision", "recall")},
        "tuned_params": {
            "fraction_of_top_score": best_p.fraction_of_top_score,
            "max_codes": best_p.max_codes,
            "include_dictionary": best_p.include_dictionary,
            "num_per_code_thresholds": len(best_p.code_fraction_thresholds or {}),
        },
        "tuned_full_train": {k: m_tuned[k] for k in ("micro_f1", "precision", "recall")},
    }
    base_name = {
        "bm25": "ir_tune_summary_bm25.json",
        "tfidf": "ir_tune_summary_tfidf.json",
        "embedding": "ir_tune_summary_embedding.json",
        "hybrid": "ir_tune_summary_hybrid.json",
    }[kind]
    strategy_suffix = "" if strategy == "standard" else f"_{strategy}"
    fallback_suffix = "" if fallback_to_standard_if_no_dictionary else "_no_fallback"
    out_name = base_name.replace(".json", f"{strategy_suffix}{fallback_suffix}.json")
    out_path = out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
