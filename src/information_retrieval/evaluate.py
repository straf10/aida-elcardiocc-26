"""
Evaluate IR pipelines with ``evaluation.evaluator.evaluate_data``.

Supports mention-expanded corpus, relative score filtering, dictionary-aware prediction
strategies, tuning, and **BM25 / TF-IDF / embeddings / hybrid (RRF)** retrieval.

With ``--source processed``, evaluation uses raw train/val JSONL under ``data/raw/``
(IR mention expansion); ensure ``python -m preprocessing`` has been run for cleaned data elsewhere.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Literal

import ahocorasick

from dictionary.dictionary import (
    build_automaton,
    load_term_code_csv,
    predict_codes_for_text,
)
from preprocessing.io_utils import (
    LABELSET_PATH,
    RAW_TRAIN_PATH,
    RAW_VAL_PATH,
    TRAIN_PATH,
    TERM_CODE_CSV,
    load_jsonl,
    load_labelset,
    resolve_patient_id,
)
from evaluation.evaluator import evaluate_data
from evaluation.io_utils import save_predictions_jsonl

from .corpus import build_code_documents, build_code_documents_with_mention_expansion
from .prediction import IRPredictionParams, predict_codes_from_retriever, filter_hits_by_relative_score
from .types import RetrievalHit
from .term_retrieval import BM25CodeRetriever, TfidfCodeRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IR_RAW_TRAIN_PATH = Path(RAW_TRAIN_PATH)
IR_RAW_VAL_PATH = Path(RAW_VAL_PATH)

RetrieverKind = Literal["bm25", "tfidf", "embedding", "hybrid"]
PredictionStrategy = Literal["standard", "dict-rerank"]


def build_ground_truth_map(records: list[dict]) -> dict[int, list[list[str]]]:
    return {resolve_patient_id(r): (r.get("document_level_annotations") or []) for r in records}


def predict_ir_codes_for_records(
    records: list[dict],
    retriever,
    *,
    params: IRPredictionParams | None = None,
    term_code_map: dict[str, set[str]] | ahocorasick.Automaton | None = None,
    strategy: PredictionStrategy = "standard",
    fallback_to_standard_if_no_dictionary: bool = True,
) -> dict[int, list[str]]:
    """Run IR (+ optional dictionary) on ``records``; return ``patient_id -> sorted code list``."""
    pred: dict[int, list[str]] = {}
    p = params or IRPredictionParams()
    for rec in records:
        pid = resolve_patient_id(rec)
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
    return pred


def evaluate_ir_on_records(
    records: list[dict],
    labelset: list[str],
    retriever,
    *,
    params: IRPredictionParams | None = None,
    term_code_map: dict[str, set[str]] | ahocorasick.Automaton | None = None,
    strategy: PredictionStrategy = "standard",
    fallback_to_standard_if_no_dictionary: bool = True,
) -> dict[str, Any]:
    """Run IR (+ optional dictionary) on ``records`` and return evaluator metrics."""
    gt = build_ground_truth_map(records)
    pred = predict_ir_codes_for_records(
        records,
        retriever,
        params=params,
        term_code_map=term_code_map,
        strategy=strategy,
        fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
    )
    return evaluate_data(gt, pred, label_space=labelset)


def predict_codes_with_dictionary_rerank(
    text: str,
    retriever,
    params: IRPredictionParams,
    *,
    term_code_map: dict[str, set[str]] | ahocorasick.Automaton | None = None,
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


def _e5_prefixes(model_name: str) -> dict:
    if "e5" in model_name.lower():
        return {"query_prefix": "query: ", "doc_prefix": "passage: "}
    return {}


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

        return EmbeddingCodeRetriever(model_name=embedding_model, **_e5_prefixes(embedding_model)).fit(documents)
    if kind == "hybrid":
        from .embedding_retrieval import EmbeddingCodeRetriever
        from .hybrid_retrieval import HybridRrfRetriever

        bm25 = BM25CodeRetriever().fit(documents)
        dense = EmbeddingCodeRetriever(model_name=embedding_model, **_e5_prefixes(embedding_model)).fit(documents)
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
    term_code_map: dict[str, set[str]] | ahocorasick.Automaton | None = None,
    grid: dict[str, list] | None = None,
    strategy: PredictionStrategy = "standard",
    fallback_to_standard_if_no_dictionary: bool = True,
) -> tuple[IRPredictionParams, dict[str, Any]]:
    """
    Grid search over prediction + fusion hyperparams; returns best params by micro-F1.

    For ``hybrid``, channel hits are pre-computed once and re-fused cheaply per combo
    so model inference runs only once per val record regardless of grid size.
    """
    if grid is None:
        if kind == "hybrid":
            grid = {
                "fraction_of_top_score": [0.04, 0.06, 0.08, 0.10, 0.14],
                "max_codes": [2, 3, 4],
                "include_dictionary": [True, False],
                "hybrid_rrf_k": [10, 20, 30],
                "hybrid_bm25_weight": [1.0],
                "hybrid_dense_weight": [0.2, 0.3, 0.4],
            }
        elif strategy == "dict-rerank":
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

    best_metrics: dict[str, Any] | None = None
    best_params: IRPredictionParams | None = None
    best_hybrid_kwargs = {"hybrid_rrf_k": 60, "hybrid_bm25_weight": 1.0, "hybrid_dense_weight": 1.0}

    if kind == "hybrid":
        from .embedding_retrieval import EmbeddingCodeRetriever

        bm25_r = BM25CodeRetriever().fit(documents)
        dense_r = EmbeddingCodeRetriever(model_name=embedding_model, **_e5_prefixes(embedding_model)).fit(documents)

        # Pre-compute channel hits and dictionary candidates once per val record.
        pool = 200
        n_codes = len(bm25_r._codes)
        pool = min(n_codes, pool)
        print(f"  Pre-computing channel hits for {len(val_records)} val records (pool={pool})…")
        bm25_cache = [bm25_r.search(r.get("text", ""), top_k=pool) for r in val_records]
        dense_cache = [dense_r.search(r.get("text", ""), top_k=pool) for r in val_records]
        dict_cache = [
            (predict_codes_for_text(r.get("text", ""), term_code_map) if term_code_map else set())
            for r in val_records
        ]
        pids = [resolve_patient_id(r) for r in val_records]
        gt_val = build_ground_truth_map(val_records)

        fusion_keys = ["hybrid_rrf_k", "hybrid_bm25_weight", "hybrid_dense_weight"]
        pred_keys = ["fraction_of_top_score", "max_codes", "include_dictionary"]
        fk = [k for k in fusion_keys if k in grid]
        pk = [k for k in pred_keys if k in grid]

        for fusion_vals in product(*[grid[k] for k in fk]):
            fkw = dict(zip(fk, fusion_vals))
            rrf_k = int(fkw.get("hybrid_rrf_k", 60))
            bm25_w = float(fkw.get("hybrid_bm25_weight", 1.0))
            dense_w = float(fkw.get("hybrid_dense_weight", 1.0))

            # Fuse cached rankings — pure Python, no model calls.
            fused_hits: list[list[RetrievalHit]] = []
            for bm25_hits, dense_hits in zip(bm25_cache, dense_cache):
                scores: dict[str, float] = {}
                for rank, h in enumerate(bm25_hits, start=1):
                    scores[h.code] = scores.get(h.code, 0.0) + bm25_w / (rrf_k + rank)
                for rank, h in enumerate(dense_hits, start=1):
                    scores[h.code] = scores.get(h.code, 0.0) + dense_w / (rrf_k + rank)
                ordered = sorted(scores.items(), key=lambda x: -x[1])[:80]
                fused_hits.append([RetrievalHit(code=c, score=s, document_text="") for c, s in ordered])

            for pred_vals in product(*[grid[k] for k in pk]):
                pkw = dict(zip(pk, pred_vals))
                fraction = float(pkw["fraction_of_top_score"])
                max_c = int(pkw["max_codes"])
                use_dict = bool(pkw["include_dictionary"])

                preds: dict[int, list[str]] = {}
                for pid, hits, dict_codes in zip(pids, fused_hits, dict_cache):
                    ir_codes = filter_hits_by_relative_score(hits, fraction_of_top=fraction, max_codes=max_c)
                    out: set[str] = set(ir_codes)
                    if use_dict:
                        out |= dict_codes
                    preds[pid] = sorted(out)

                metrics = evaluate_data(gt_val, preds, label_space=labelset)
                f1 = metrics["micro_f1"]
                if best_metrics is None or f1 > best_metrics["micro_f1"]:
                    best_metrics = metrics
                    best_params = IRPredictionParams(
                        search_top_k=80,
                        fraction_of_top_score=fraction,
                        max_codes=max_c,
                        include_dictionary=use_dict,
                    )
                    best_hybrid_kwargs = {
                        "hybrid_rrf_k": rrf_k,
                        "hybrid_bm25_weight": bm25_w,
                        "hybrid_dense_weight": dense_w,
                    }
    else:
        keys = list(grid.keys())
        for values in product(*[grid[k] for k in keys]):
            kw = dict(zip(keys, values))
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
        help="processed (default): raw train/val from data/raw/. raw: single TRAIN_PATH JSONL + 80/20 row split.",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "tfidf", "embedding", "hybrid"),
        default="bm25",
        help="Lexical (bm25, tfidf), dense (embedding), or hybrid RRF of BM25+embedding.",
    )
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-base",
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
        default=30,
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
        default=0.4,
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
    parser.add_argument(
        "--fraction-of-top-score",
        type=float,
        default=0.04,
        help="Relative score threshold for keeping hits (used with --no-tune).",
    )
    parser.add_argument(
        "--max-codes",
        type=int,
        default=2,
        help="Max IR codes per document (used with --no-tune).",
    )
    parser.add_argument(
        "--write-predictions",
        type=str,
        default="outputs/predictions/information_retrieval/predictions.jsonl",
        help="Path to write per-patient predictions JSONL (empty string to skip).",
    )
    args = parser.parse_args()

    labelset = load_labelset(str(LABELSET_PATH))
    term_map = load_term_code_csv(str(TERM_CODE_CSV))
    term_map = build_automaton(term_map)
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
        if not IR_RAW_TRAIN_PATH.exists() or not IR_RAW_VAL_PATH.exists():
            print("Raw train/val JSONL not found (IR uses raw text). Expected:")
            print(" ", IR_RAW_TRAIN_PATH)
            print(" ", IR_RAW_VAL_PATH)
            return
        train_proc = load_jsonl(str(IR_RAW_TRAIN_PATH))
        val_proc = load_jsonl(str(IR_RAW_VAL_PATH))
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

    if not args.no_tune:
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
            pred_top25[resolve_patient_id(rec)] = [h.code for h in r0.search(rec.get("text", ""), 25)]
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
        m_old_summary = {k: m_old[k] for k in ("micro_f1", "precision", "recall")}
        m_new_summary = {k: m_new[k] for k in ("micro_f1", "precision", "recall")}
    else:
        print("Skipping baseline and default evals (--no-tune).")
        m_old_summary = {}
        m_new_summary = {}

    if args.no_tune:
        print(f"\n=== Skipping val grid (--no-tune); using provided/default IRPredictionParams ===")
        best_p = IRPredictionParams(
            fraction_of_top_score=args.fraction_of_top_score,
            max_codes=args.max_codes,
        )
        print("Params:", best_p)
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

    out_dir = PROJECT_ROOT / "outputs" / "experiments" / "information_retrieval"
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
        "baseline_top25_plain_corpus": m_old_summary,
        "default_expanded_hybrid": m_new_summary,
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

    if args.write_predictions:
        pred_path = Path(args.write_predictions)
        pred_dict = predict_ir_codes_for_records(
            val_recs,
            r_final,
            params=best_p,
            term_code_map=term_map,
            strategy=strategy,
            fallback_to_standard_if_no_dictionary=fallback_to_standard_if_no_dictionary,
        )
        save_predictions_jsonl(pred_dict, str(pred_path))
        print("\nWrote predictions to", pred_path)


if __name__ == "__main__":
    main()
