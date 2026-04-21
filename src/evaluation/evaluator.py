from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .config_utils import get_cfg, load_config
from .io_utils import load_ground_truth, load_predictions

DEFAULT_EVAL_CONFIG = "src/evaluation/config.yaml"


def score_document(ground_truth_groups: List[List[str]], pred_codes: List[str]) -> Tuple[int, int, int]:
    pred_set = set(pred_codes)
    tp = 0

    for group in ground_truth_groups:
        if pred_set.intersection(set(group)):
            tp += 1

    fn = len(ground_truth_groups) - tp
    all_ground_truth_codes = {code for group in ground_truth_groups for code in group}
    fp = len([code for code in pred_set if code not in all_ground_truth_codes])
    return tp, fp, fn


def micro_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def per_class_report(
    ground_truth_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    label_space: Sequence[str],
) -> List[dict]:
    support = {label: 0 for label in label_space}
    groups_hit = {label: 0 for label in label_space}
    fp_count = {label: 0 for label in label_space}

    for patient_id, ground_truth_groups in ground_truth_data.items():
        pred_set = set(pred_data.get(patient_id, []))
        all_ground_truth_codes = {c for grp in ground_truth_groups for c in grp}

        for group in ground_truth_groups:
            group_set = set(group)
            hit = bool(pred_set.intersection(group_set))
            for code in group_set:
                if code in support:
                    support[code] += 1
                    if hit:
                        groups_hit[code] += 1

        for code in pred_set:
            if code in fp_count and code not in all_ground_truth_codes:
                fp_count[code] += 1

    rows: List[dict] = []
    for code in label_space:
        s = support[code]
        gh = groups_hit[code]
        fp = fp_count[code]
        precision = gh / (gh + fp) if (gh + fp) > 0 else 0.0
        recall = gh / s if s > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "code": code,
                "support": s,
                "groups_hit": gh,
                "fp_count": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def evaluate_data(
    ground_truth_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    label_space: Sequence[str] | None = None,
) -> Dict:
    total_tp = total_fp = total_fn = 0
    doc_breakdown: List[dict] = []

    ground_truth_ids = set(ground_truth_data.keys())
    pred_ids = set(pred_data.keys())
    missing_pred_ids = sorted(ground_truth_ids - pred_ids)
    extra_pred_ids = sorted(pred_ids - ground_truth_ids)

    for patient_id, ground_truth_groups in ground_truth_data.items():
        pred_codes = pred_data.get(patient_id, [])
        pred_set = set(pred_codes)
        tp, fp, fn = score_document(ground_truth_groups, pred_codes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        missed_groups = [
            group for group in ground_truth_groups if not pred_set.intersection(set(group))
        ]
        all_ground_truth_codes = {c for g in ground_truth_groups for c in g}
        wrong_codes = sorted([code for code in pred_set if code not in all_ground_truth_codes])

        doc_breakdown.append(
            {
                "patient_id": patient_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "missed_groups": missed_groups,
                "wrong_codes": wrong_codes,
            }
        )

    precision, recall, f1 = micro_f1(total_tp, total_fp, total_fn)
    result = {
        "micro_f1": f1,
        "precision": precision,
        "recall": recall,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "docs_evaluated": len(ground_truth_data),
        "missing_prediction_ids": missing_pred_ids,
        "extra_prediction_ids": extra_pred_ids,
        "doc_breakdown": doc_breakdown,
    }

    if label_space:
        class_rows = per_class_report(ground_truth_data, pred_data, label_space)
        present = [row["f1"] for row in class_rows if row["support"] > 0]
        all_rows = [row["f1"] for row in class_rows]
        result["per_class"] = class_rows
        result["macro_f1_present_labels"] = mean(present) if present else 0.0
        result["macro_f1_all_labels"] = mean(all_rows) if all_rows else 0.0

    return result


def evaluate_file(
    ground_truth_jsonl_path: str,
    pred_jsonl_path: str,
    label_space: Sequence[str] | None = None,
) -> Dict:
    """Evaluate metrics using **only** ground-truth and predictions JSONL on disk."""
    ground_truth_data = load_ground_truth(ground_truth_jsonl_path)
    pred_data = load_predictions(pred_jsonl_path)
    return evaluate_data(ground_truth_data, pred_data, label_space=label_space)


def evaluate_from_prediction_files(
    ground_truth_jsonl_path: str,
    predictions_jsonl_path: str,
    *,
    label_space: Sequence[str] | None = None,
) -> Dict:
    """
    Public alias for file-only evaluation (same as ``evaluate_file``).

    Workflow for pipelines: materialize ``predictions_jsonl_path`` (e.g. via
    ``save_predictions_jsonl``), then call this — metrics are always derived from the JSONL.
    """
    return evaluate_file(ground_truth_jsonl_path, predictions_jsonl_path, label_space=label_space)


def _parse_label_space(path: str | None) -> List[str]:
    if not path:
        return []
    import json

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("label_names JSON must be a list of code strings.")
    return [str(item) for item in data]


def _print_score_metrics(metrics: Dict, metrics_json: str | None) -> None:
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


def _score(ground_truth: str, pred: str, labelset_path: str | None, metrics_json: str | None) -> None:
    from preprocessing.io_utils import load_labelset

    label_space = load_labelset(labelset_path) if labelset_path else None
    metrics = evaluate_file(ground_truth, pred, label_space=label_space)
    _print_score_metrics(metrics, metrics_json)


def _run_compare(args: argparse.Namespace) -> None:
    from preprocessing.io_utils import LABELSET_PATH, load_labelset

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


def _predict_ir(args: argparse.Namespace) -> None:
    from dictionary.dictionary import build_automaton, load_term_code_csv
    from information_retrieval.corpus import build_code_documents_with_mention_expansion
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


def _cmd_score(args: argparse.Namespace) -> None:
    from preprocessing.io_utils import load_labelset

    config = load_config(args.config)
    ground_truth_path = (
        args.ground_truth
        or get_cfg(config, "ground_truth_path")
        or get_cfg(config, "data.val_path")
    )
    pred_path = args.pred or get_cfg(config, "prediction_path")
    if not ground_truth_path or not pred_path:
        raise SystemExit("score: provide --ground-truth and --pred (or set them in --config YAML).")

    label_space: Sequence[str] | None = None
    if args.labelset:
        label_space = load_labelset(args.labelset)
    elif args.labels:
        label_space = _parse_label_space(args.labels)
    else:
        labels_path = get_cfg(config, "label_names_path")
        if labels_path:
            label_space = _parse_label_space(labels_path)

    metrics = evaluate_file(ground_truth_path, pred_path, label_space=label_space)
    print(f"Evaluated {metrics['docs_evaluated']} documents.")
    print(f"Micro-F1:  {metrics['micro_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if "macro_f1_present_labels" in metrics:
        print(f"Macro-F1 (present labels): {metrics['macro_f1_present_labels']:.4f}")
        print(f"Macro-F1 (all labels):     {metrics['macro_f1_all_labels']:.4f}")
    print(f"TP: {metrics['total_tp']} | FP: {metrics['total_fp']} | FN: {metrics['total_fn']}")

    if args.show_missing:
        print(f"Missing prediction IDs: {len(metrics['missing_prediction_ids'])}")
        print(f"Extra prediction IDs:   {len(metrics['extra_prediction_ids'])}")

    if args.metrics_json:
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote metrics JSON -> {out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ELCardioCC evaluation CLI: score, compare (multi-method), predict (IR / NER+EL)."
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand (optional for legacy score flags).")

    p_score = sub.add_parser("score", help="Micro/macro F1 for one gold + one predictions JSONL.")
    p_score.add_argument("--config", default=None, help=f"YAML (default paths); often {DEFAULT_EVAL_CONFIG}")
    p_score.add_argument("--ground-truth", dest="ground_truth", default=None)
    p_score.add_argument("--pred", default=None)
    p_score.add_argument("--labels", help="Optional JSON list of ICD-10 labels for per-class metrics")
    p_score.add_argument("--labelset", default=None, help="labelset.txt path for macro-F1 over full label space")
    p_score.add_argument("--metrics-json", default=None)
    p_score.add_argument("--show-missing", action="store_true")

    p_pred = sub.add_parser("predict", help="Run IR or NER+EL and write predictions JSONL.")
    p_pred.add_argument("--backend", choices=("ir", "ner"), required=True)
    p_pred.add_argument(
        "--test-jsonl",
        default=None,
        help="Input JSONL. Default: data/processed/test.jsonl",
    )
    p_pred.add_argument(
        "--output",
        default=None,
        help="Output JSONL. Default: outputs/predictions/<ir|ner_el>/test_predictions.jsonl",
    )
    p_pred.add_argument("--ground-truth", default=None)
    p_pred.add_argument("--labelset", default=None)
    p_pred.add_argument("--metrics-json", default=None)
    p_pred.add_argument("--no-score", action="store_true")

    ir = p_pred.add_argument_group("IR")
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

    ner = p_pred.add_argument_group("NER+EL")
    ner.add_argument("--model-dir", default=None)
    ner.add_argument("--tokenizer-name", default=None)
    ner.add_argument("--max-length", type=int, default=512)
    ner.add_argument("--train-jsonl", default=None)
    ner.add_argument("--no-dictionary-fusion", action="store_true")
    ner.add_argument("--no-dictionary-doc-boost", action="store_true")

    p_cmp = sub.add_parser("compare", help="F1 table: --config (models list) or --ground-truth + --pair.")
    p_cmp.add_argument("--config", default=None, help=f"YAML with data.val_path and models[]. Default: {DEFAULT_EVAL_CONFIG}")
    p_cmp.add_argument("--ground-truth", default=None)
    p_cmp.add_argument("--pair", action="append", default=[], metavar="PRED_JSONL:NAME")
    p_cmp.add_argument("--labelset", default=None)
    p_cmp.add_argument("--metrics-json", default=None)

    return parser


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    subcommands = ("score", "compare", "predict")
    if argv and argv[0] not in subcommands and argv[0] not in ("-h", "--help"):
        argv = ["score"] + argv

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        raise SystemExit(2)

    if args.command == "score":
        _cmd_score(args)
        return
    if args.command == "compare":
        if not args.config and not args.ground_truth and not args.pair:
            args.config = DEFAULT_EVAL_CONFIG
        _run_compare(args)
        return

    args.no_score = bool(getattr(args, "no_score", False))
    from preprocessing.io_utils import PROCESSED_TEST_PATH, TERM_CODE_CSV

    if not args.test_jsonl:
        args.test_jsonl = PROCESSED_TEST_PATH
    if not args.output:
        if args.backend == "ir":
            args.output = "outputs/predictions/information_retrieval/test_predictions.jsonl"
        else:
            args.output = "outputs/predictions/ner_el/test_predictions.jsonl"

    if args.backend == "ir":
        if args.term_code_csv is None:
            args.term_code_csv = str(TERM_CODE_CSV)
        _predict_ir(args)
    elif args.backend == "ner":
        _predict_ner(args)
    else:
        raise SystemExit(f"Unknown backend {args.backend!r}")


if __name__ == "__main__":
    main()
