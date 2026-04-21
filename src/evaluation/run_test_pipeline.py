"""
Unified evaluation CLI (JSONL-only metrics; optional IR/NER prediction).

Subcommands
-----------

**predict** — run IR or NER+EL, write ``predictions.jsonl``, optional immediate **score**.

**score** — micro/macro F1 from existing gold + prediction JSONLs (``evaluation.evaluator``).

**compare** — table of F1 for many methods: either ``--config`` (``experiment.yaml`` lists
``predictions_path`` per model) or ``--ground-truth`` + repeated ``--pair pred.jsonl:Name``.

Examples::

    PYTHONPATH=src python -m evaluation.run_test_pipeline predict \\
        --backend ir --test-jsonl data/processed/test.jsonl \\
        --output outputs/predictions/ir/test.jsonl --retriever bm25

    PYTHONPATH=src python -m evaluation.run_test_pipeline score \\
        --ground-truth data/processed/test.jsonl \\
        --pred outputs/predictions/ir/test.jsonl \\
        --labelset data/raw/labelset.txt

    PYTHONPATH=src python -m evaluation.run_test_pipeline compare \\
        --config src/evaluation/experiment.yaml

    PYTHONPATH=src python -m evaluation.run_test_pipeline compare \\
        --ground-truth data/processed/val.jsonl \\
        --pair outputs/predictions/ir/val.jsonl:IR \\
        --pair outputs/predictions/ner_el/predictions.jsonl:NER \\
        --labelset data/raw/labelset.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _run_compare(args: argparse.Namespace) -> None:
    from preprocessing.io_utils import LABELSET_PATH, load_labelset

    from evaluation.config_utils import get_cfg, load_config
    from evaluation.evaluator import evaluate_file

    rows: list[dict] = []

    if args.config:
        cfg = load_config(args.config)
        gt_path = args.ground_truth or get_cfg(cfg, "data.val_path")
        if not gt_path or not Path(gt_path).is_file():
            raise SystemExit(f"Ground truth missing or not a file: {gt_path!r}")
        default_ls = args.labelset
        for m in get_cfg(cfg, "models", []) or []:
            name = str(m.get("name", "?"))
            pred_path = m.get("predictions_path")
            ls_path = default_ls or m.get("labelset_path") or str(LABELSET_PATH)
            if not pred_path:
                rows.append({"name": name, "error": "no predictions_path in config"})
                continue
            if not Path(pred_path).is_file():
                rows.append({"name": name, "error": f"missing file {pred_path}"})
                continue
            label_space = load_labelset(ls_path)
            metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
            rows.append(
                {
                    "name": name,
                    "predictions_path": pred_path,
                    "micro_f1": metrics["micro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1_present": metrics.get("macro_f1_present_labels"),
                }
            )
    elif args.ground_truth and args.pair:
        gt_path = args.ground_truth
        if not Path(gt_path).is_file():
            raise SystemExit(f"Ground truth not found: {gt_path}")
        ls_path = args.labelset or str(LABELSET_PATH)
        label_space = load_labelset(ls_path)
        for raw in args.pair:
            if ":" not in raw:
                raise SystemExit(f"--pair must be PRED.jsonl:Name, got {raw!r}")
            pred_path, _, name = raw.partition(":")
            pred_path = pred_path.strip()
            name = name.strip() or Path(pred_path).stem
            if not Path(pred_path).is_file():
                rows.append({"name": name, "error": f"missing file {pred_path}"})
                continue
            metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
            rows.append(
                {
                    "name": name,
                    "predictions_path": pred_path,
                    "micro_f1": metrics["micro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1_present": metrics.get("macro_f1_present_labels"),
                }
            )
    else:
        raise SystemExit("compare: pass --config, or --ground-truth with one or more --pair PRED.jsonl:Name")

    col_w = max(22, max((len(r.get("name", "")) for r in rows), default=10) + 2)
    header = f"{'Method':<{col_w}} {'Micro-F1':>9} {'Precision':>10} {'Recall':>8} {'Macro-F1*':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<{col_w}}  ERROR: {r['error']}")
        else:
            mf = r.get("macro_f1_present")
            mf_s = f"{mf:.4f}" if mf is not None else "n/a"
            print(
                f"{r['name']:<{col_w}} {r['micro_f1']:>9.4f} {r['precision']:>10.4f}"
                f" {r['recall']:>8.4f} {mf_s:>10}"
            )
    print("\n*Macro-F1 over labels with support in gold.\n")

    if args.metrics_json:
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote compare table JSON -> {out}")


def _score(ground_truth: str, pred: str, labelset_path: str | None, metrics_json: str | None) -> None:
    from preprocessing.io_utils import load_labelset

    from evaluation.evaluator import evaluate_file

    label_space = load_labelset(labelset_path) if labelset_path else None
    metrics = evaluate_file(ground_truth, pred, label_space=label_space)
    print(f"Documents: {metrics['docs_evaluated']}")
    print(f"Micro-F1:  {metrics['micro_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"TP={metrics['total_tp']} FP={metrics['total_fp']} FN={metrics['total_fn']}")
    if "macro_f1_present_labels" in metrics:
        print(f"Macro-F1 (labels with support): {metrics['macro_f1_present_labels']:.4f}")
    if metrics.get("missing_prediction_ids"):
        print(f"Missing prediction patient_ids: {len(metrics['missing_prediction_ids'])}")
    if metrics.get("extra_prediction_ids"):
        print(f"Extra prediction patient_ids:   {len(metrics['extra_prediction_ids'])}")
    if metrics_json:
        out = Path(metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote metrics JSON -> {out}")


def _predict_ir(args: argparse.Namespace) -> None:
    from dictionary.dictionary import build_automaton, load_term_code_csv
    from information_retrieval.evaluate import (
        RetrieverKind,
        PredictionStrategy,
        fit_retriever,
        predict_ir_codes_for_records,
        tune_ir_hyperparams,
    )
    from preprocessing.io_utils import LABELSET_PATH, RAW_TRAIN_PATH, RAW_VAL_PATH, load_jsonl, load_labelset

    from evaluation.io_utils import save_predictions_jsonl

    labelset = load_labelset(args.labelset or str(LABELSET_PATH))
    term_map = build_automaton(load_term_code_csv(args.term_code_csv))

    train_proc = load_jsonl(str(RAW_TRAIN_PATH))
    val_proc = load_jsonl(str(RAW_VAL_PATH))
    from information_retrieval.corpus import build_code_documents_with_mention_expansion

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
        from information_retrieval.prediction import IRPredictionParams

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
        _score(gt_path, args.output, args.labelset or str(LABELSET_PATH), args.metrics_json)


def _predict_ner(args: argparse.Namespace) -> None:
    from ner_el.config import PredictConfig
    from ner_el.io_utils import load_documents, save_jsonl as ner_save_jsonl
    from ner_el.service import NERELService

    if not args.model_dir:
        raise SystemExit("--model-dir is required for --backend ner")

    cfg = PredictConfig(
        model_dir=args.model_dir,
        tokenizer_name=args.tokenizer_name or PredictConfig.tokenizer_name,
        max_length=args.max_length,
        use_dictionary_fusion=not args.no_dictionary_fusion,
        dictionary_doc_boost=not args.no_dictionary_doc_boost,
        train_path_for_linker=args.train_jsonl,
    )
    service = NERELService.from_config(cfg)
    docs = load_documents(args.test_jsonl)
    outputs = service.predict_many(docs)
    rows = [o.doc_prediction for o in outputs]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ner_save_jsonl(args.output, rows)
    print(f"[NER] Wrote {len(rows)} predictions -> {args.output}")

    if not args.no_score:
        gt_path = args.ground_truth or args.test_jsonl
        from preprocessing.io_utils import LABELSET_PATH

        _score(gt_path, args.output, args.labelset or str(LABELSET_PATH), args.metrics_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on a JSONL test split and/or score predictions.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pred = sub.add_parser("predict", help="Run IR or NER+EL and write predictions JSONL.")
    p_pred.add_argument("--backend", choices=("ir", "ner"), required=True)
    p_pred.add_argument("--test-jsonl", required=True, help="Input records (patient_id, text, …).")
    p_pred.add_argument("--output", required=True, help="Output JSONL path.")
    p_pred.add_argument(
        "--ground-truth",
        default=None,
        help="Gold JSONL for scoring (defaults to --test-jsonl when labels present).",
    )
    p_pred.add_argument(
        "--labelset",
        default=None,
        help="labelset.txt path for macro-F1 (defaults to data/raw/labelset.txt).",
    )
    p_pred.add_argument("--metrics-json", default=None, help="Optional path to dump full metrics dict.")
    p_pred.add_argument("--no-score", action="store_true", help="Skip micro/macro F1 after predict.")

    ir = p_pred.add_argument_group("IR (information retrieval)")
    ir.add_argument("--retriever", choices=("bm25", "tfidf", "embedding", "hybrid"), default="bm25")
    ir.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ir.add_argument("--tune", action="store_true", help="Grid-tune on raw val (slower). Default: fixed IRPredictionParams.")
    ir.add_argument("--fraction-of-top-score", type=float, default=0.22)
    ir.add_argument("--max-codes", type=int, default=12)
    ir.add_argument("--no-dictionary", action="store_true", help="Disable dictionary union (IR only).")
    ir.add_argument(
        "--no-fallback-standard-when-no-dict",
        action="store_true",
        help="With dict-rerank: do not fall back to standard IR when dictionary is empty.",
    )
    ir.add_argument("--prediction-strategy", choices=("standard", "dict-rerank"), default="standard")
    ir.add_argument("--term-code-csv", default=None, help="Override dictionary CSV path.")
    ir.add_argument("--hybrid-rrf-k", type=int, default=60)
    ir.add_argument("--hybrid-bm25-weight", type=float, default=1.0)
    ir.add_argument("--hybrid-dense-weight", type=float, default=1.0)

    ner = p_pred.add_argument_group("NER+EL (HuggingFace model dir with .bin / safetensors)")
    ner.add_argument("--model-dir", default=None, help="HF save dir (e.g. outputs/models/NER_EL).")
    ner.add_argument("--tokenizer-name", default=None, help="Override tokenizer if not in config.json.")
    ner.add_argument("--max-length", type=int, default=512)
    ner.add_argument(
        "--train-jsonl",
        default=None,
        help="Train JSONL to rebuild linker priors if linker_prior.json is missing in model dir.",
    )
    ner.add_argument("--no-dictionary-fusion", action="store_true")
    ner.add_argument("--no-dictionary-doc-boost", action="store_true")

    p_score = sub.add_parser("score", help="Score an existing predictions JSONL against gold.")
    p_score.add_argument("--ground-truth", required=True)
    p_score.add_argument("--pred", required=True)
    p_score.add_argument("--labelset", default=None)
    p_score.add_argument("--metrics-json", default=None)

    p_cmp = sub.add_parser(
        "compare",
        help="Print F1 table for multiple prediction JSONLs vs one ground-truth JSONL.",
    )
    p_cmp.add_argument(
        "--config",
        default=None,
        help="YAML with data.val_path and models[].predictions_path / labelset_path.",
    )
    p_cmp.add_argument(
        "--ground-truth",
        default=None,
        help="Gold JSONL (defaults to data.val_path from --config when using --config).",
    )
    p_cmp.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="PRED_JSONL:NAME",
        help="Ad-hoc compare (repeat per method). Requires --ground-truth.",
    )
    p_cmp.add_argument("--labelset", default=None, help="Override labelset.txt for macro-F1.")
    p_cmp.add_argument("--metrics-json", default=None, help="Write table rows as JSON.")

    args = parser.parse_args()
    if args.command == "score":
        _score(args.ground_truth, args.pred, args.labelset, args.metrics_json)
        return
    if args.command == "compare":
        _run_compare(args)
        return

    args.no_score = bool(getattr(args, "no_score", False))
    if args.backend == "ir":
        from preprocessing.io_utils import TERM_CODE_CSV

        if args.term_code_csv is None:
            args.term_code_csv = str(TERM_CODE_CSV)
        _predict_ir(args)
    elif args.backend == "ner":
        _predict_ner(args)
    else:
        raise SystemExit(f"Unknown backend {args.backend!r}")


if __name__ == "__main__":
    main()
