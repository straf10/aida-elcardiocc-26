"""Write IR predictions JSONL for a labeled split (default: processed test set)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _score_after_write(
    ground_truth: str,
    pred: str,
    labelset_path: str | None,
    metrics_json: str | None,
) -> None:
    from preprocessing.io_utils import load_labelset

    from evaluation.scoring import evaluate_file, print_metrics_short

    label_space = load_labelset(labelset_path) if labelset_path else None
    metrics = evaluate_file(ground_truth, pred, label_space=label_space)
    print_metrics_short(metrics, metrics_json)


def _run_ir_predict(args: argparse.Namespace) -> None:
    from dictionary.dictionary import build_automaton, load_term_code_csv
    from evaluation.io_utils import save_predictions_jsonl

    from .corpus import build_code_documents_with_mention_expansion
    from .evaluate import (
        RetrieverKind,
        PredictionStrategy,
        fit_retriever,
        predict_ir_codes_for_records,
        tune_ir_hyperparams,
    )

    from preprocessing.io_utils import (
        LABELSET_PATH,
        RAW_TRAIN_PATH,
        RAW_VAL_PATH,
        load_jsonl,
        load_labelset,
    )

    labelset = load_labelset(args.labelset or str(LABELSET_PATH))
    term_map = build_automaton(load_term_code_csv(args.term_code_csv))

    train_proc = load_jsonl(str(RAW_TRAIN_PATH))
    val_proc = load_jsonl(str(RAW_VAL_PATH))

    docs_exp_train_split = build_code_documents_with_mention_expansion(train_proc, codes=labelset)
    docs_exp_full = build_code_documents_with_mention_expansion(train_proc + val_proc, codes=labelset)

    kind: RetrieverKind = args.retriever  # type: ignore[assignment]
    strategy: PredictionStrategy = args.prediction_strategy  # type: ignore[assignment]
    fallback = not args.no_fallback_standard_when_no_dict

    hybrid_rrf_k = args.hybrid_rrf_k
    hybrid_bm25_weight = args.hybrid_bm25_weight
    hybrid_dense_weight = args.hybrid_dense_weight

    if args.tune:
        best_p, best_val = tune_ir_hyperparams(
            val_proc,
            labelset,
            docs_exp_train_split,
            kind=kind,
            embedding_model=args.embedding_model,
            term_code_map=term_map,
            strategy=strategy,
            fallback_to_standard_if_no_dictionary=fallback,
        )
        print(f"[IR] Tuned val micro_f1={best_val['micro_f1']:.4f} params={best_p}")
        if kind == "hybrid":
            tuned = best_val.get("_best_hybrid_kwargs") or {}
            hybrid_rrf_k = int(tuned.get("hybrid_rrf_k", hybrid_rrf_k))
            hybrid_bm25_weight = float(tuned.get("hybrid_bm25_weight", hybrid_bm25_weight))
            hybrid_dense_weight = float(tuned.get("hybrid_dense_weight", hybrid_dense_weight))
    else:
        from .prediction import IRPredictionParams

        best_p = IRPredictionParams(
            fraction_of_top_score=args.fraction_of_top_score,
            max_codes=args.max_codes,
            include_dictionary=not args.no_dictionary,
        )

    r_final = fit_retriever(
        kind,
        docs_exp_full,
        embedding_model=args.embedding_model,
        hybrid_rrf_k=hybrid_rrf_k,
        hybrid_bm25_weight=hybrid_bm25_weight,
        hybrid_dense_weight=hybrid_dense_weight,
    )

    if getattr(args, "export_standard_splits_dir", None):
        from preprocessing.io_utils import (
            PROCESSED_TEST_PATH,
            PROCESSED_VAL_PATH,
            RAW_SUBMISSION_TEST_PATH,
        )

        out_dir = Path(args.export_standard_splits_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        splits: list[tuple[str, str, str]] = [
            ("val", str(PROCESSED_VAL_PATH), "val_predictions.jsonl"),
            ("test", str(PROCESSED_TEST_PATH), "test_predictions.jsonl"),
            ("blind", str(RAW_SUBMISSION_TEST_PATH), "blind_predictions.jsonl"),
        ]
        for label, jsonl_path, out_name in splits:
            p = Path(jsonl_path)
            if not p.is_file():
                print(f"[IR] Skip {label}: missing input {jsonl_path}", flush=True)
                continue
            recs = load_jsonl(str(p))
            pred_map = predict_ir_codes_for_records(
                recs,
                r_final,
                params=best_p,
                term_code_map=term_map,
                strategy=strategy,
                fallback_to_standard_if_no_dictionary=fallback,
            )
            dest = out_dir / out_name
            save_predictions_jsonl(pred_map, str(dest))
            print(f"[IR] Wrote {len(pred_map)} predictions -> {dest}", flush=True)
        test_out = out_dir / "test_predictions.jsonl"
        if not args.no_score and test_out.is_file():
            _score_after_write(
                str(PROCESSED_TEST_PATH),
                str(test_out),
                args.labelset or str(LABELSET_PATH),
                args.metrics_json,
            )
        return

    test_records = load_jsonl(args.test_jsonl)
    pred = predict_ir_codes_for_records(
        test_records,
        r_final,
        params=best_p,
        term_code_map=term_map,
        strategy=strategy,
        fallback_to_standard_if_no_dictionary=fallback,
    )
    save_predictions_jsonl(pred, args.output)
    print(f"[IR] Wrote {len(pred)} predictions -> {args.output}")

    if not args.no_score:
        gt_path = args.ground_truth or args.test_jsonl
        _score_after_write(gt_path, args.output, args.labelset or str(LABELSET_PATH), args.metrics_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IR retrieval and write predictions JSONL.")
    parser.add_argument(
        "--test-jsonl",
        default=None,
        help="Input JSONL. Default: data/processed/test.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL. Default: outputs/predictions/information_retrieval/test_predictions.jsonl",
    )
    parser.add_argument("--ground-truth", dest="ground_truth", default=None)
    parser.add_argument("--labelset", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--no-score", action="store_true")

    ir = parser.add_argument_group("IR")
    ir.add_argument("--retriever", choices=("bm25", "tfidf", "embedding", "hybrid"), default="bm25")
    ir.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ir.add_argument("--tune", action="store_true")
    ir.add_argument("--fraction-of-top-score", type=float, default=0.22)
    ir.add_argument("--max-codes", type=int, default=12)
    ir.add_argument("--no-dictionary", action="store_true")
    ir.add_argument("--no-fallback-standard-when-no-dict", action="store_true")
    ir.add_argument("--prediction-strategy", choices=("standard", "dict-rerank"), default="standard")
    ir.add_argument("--term-code-csv", default=None)
    ir.add_argument("--hybrid-rrf-k", type=int, default=60)
    ir.add_argument("--hybrid-bm25-weight", type=float, default=1.0)
    ir.add_argument("--hybrid-dense-weight", type=float, default=1.0)
    ir.add_argument(
        "--export-standard-splits-dir",
        default=None,
        metavar="DIR",
        help=(
            "After fitting the retriever, write val_predictions.jsonl, test_predictions.jsonl, "
            "and blind_predictions.jsonl (when inputs exist) under DIR. Uses one tuning pass with --tune."
        ),
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.no_score = bool(getattr(args, "no_score", False))
    from preprocessing.io_utils import PROCESSED_TEST_PATH, TERM_CODE_CSV

    if getattr(args, "export_standard_splits_dir", None):
        args.export_standard_splits_dir = str(Path(args.export_standard_splits_dir))
    else:
        if not args.test_jsonl:
            args.test_jsonl = PROCESSED_TEST_PATH
        if not args.output:
            args.output = "outputs/predictions/information_retrieval/test_predictions.jsonl"
    if args.term_code_csv is None:
        args.term_code_csv = str(TERM_CODE_CSV)

    _run_ir_predict(args)


if __name__ == "__main__":
    main()
