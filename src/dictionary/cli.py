"""CLI for dictionary baseline: reports, evaluation, JSONL export."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from preprocessing.io_utils import (
    load_jsonl,
    load_labelset,
    PROJECT_ROOT,
    DICTIONARY_CONFIG_PATH,
)

from .build import build_mention_dictionary
from .config import load_dictionary_config
from .export import export_code_lookup, export_predictions_jsonl, load_code_description_csv
from .matcher import build_automaton, load_term_code_csv
from .reports import (
    coverage_report,
    fp_fn_analysis,
    label_frequency_report,
    official_evaluation,
    tokenization_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ELCardioCC dictionary baseline")
    parser.add_argument(
        "--config",
        default=DICTIONARY_CONFIG_PATH,
        help="Path to dictionary YAML (default: src/dictionary/dictionary.yaml)",
    )
    parser.add_argument(
        "--build-from",
        dest="build_from",
        default=None,
        help="Optional JSONL source used only to build a term-code CSV and exit.",
    )
    parser.add_argument(
        "--out",
        dest="out_csv",
        default=None,
        help="Output CSV path for --build-from mode.",
    )
    args = parser.parse_args()

    cfg = load_dictionary_config(args.config if Path(args.config).exists() else None)

    if args.build_from or args.out_csv:
        if not args.build_from or not args.out_csv:
            raise ValueError("--build-from and --out must be provided together.")
        build_records = load_jsonl(args.build_from)
        b = cfg.build or {}
        build_mention_dictionary(
            build_records,
            args.out_csv,
            min_term_len=int(b.get("min_term_len", 4)),
            min_term_count=int(b.get("min_term_count", 1)),
            min_code_ratio=float(b.get("min_code_ratio", 0.20)),
        )
        print(f"Train-only dictionary written to {args.out_csv}")
        return

    train_path = cfg.paths["train_jsonl"]
    labelset_path = cfg.paths["labelset"]
    term_code_csv = cfg.paths["term_code_csv"]
    test_path = cfg.paths["test_jsonl"]
    code_desc_path = cfg.paths["code_description_csv"]
    output_dir = Path(cfg.paths["output_dir"])

    print("Loading data...")
    records = load_jsonl(train_path)
    labelset = load_labelset(labelset_path)
    print(f"Loaded {len(records)} records, {len(labelset)} labels")

    if not Path(term_code_csv).exists():
        print(f"\n{term_code_csv} not found -> building from mention_level_annotations...")
        b = cfg.build or {}
        build_mention_dictionary(
            records,
            term_code_csv,
            min_term_len=int(b.get("min_term_len", 4)),
            min_term_count=int(b.get("min_term_count", 1)),
            min_code_ratio=float(b.get("min_code_ratio", 0.20)),
        )

    print("\nLoading term-code dictionary...")
    term_code_map = load_term_code_csv(term_code_csv, blacklist=cfg.blacklist)
    print(f"Dictionary size: {len(term_code_map)} terms (after blacklist)")
    wb = bool((cfg.matching or {}).get("word_boundary", False))
    matcher = build_automaton(term_code_map, word_boundary=wb)

    print("\nTokenization / normalization report...")
    tok_report = tokenization_report(
        records,
        str(output_dir / "tokenization_report.json"),
        output_dir=output_dir,
    )
    print(f"  Avg tokens/doc: {tok_report['avg_tokens_per_doc']}")
    print(f"  Docs over 512 tokens: {tok_report['docs_over_512_tokens']} ({tok_report['docs_over_512_tokens_pct']}%)")

    print("\nLabel frequency report...")
    label_frequency_report(records, output_dir=output_dir)

    code_desc_map = load_code_description_csv(code_desc_path)

    print("Coverage report...")
    cov = coverage_report(
        records,
        matcher,
        labelset,
        config=cfg,
        labelset_for_fuzzy=labelset,
        code_desc_map=code_desc_map,
        output_dir=output_dir,
    )
    print(f"  Records with predictions: {cov['records_with_any_prediction']}/{cov['num_records']}")
    print(f"  Codes covered: {cov['num_codes_covered_by_dictionary']}/{cov['num_codes_in_labelset']}")
    print(f"  Missing codes: {cov['missing_codes']}")

    has_gold = any(rec.get("document_level_annotations") for rec in records)
    if has_gold:
        print("\nOfficial evaluation (group-level micro-F1, full training set)...")
        official_metrics = official_evaluation(
            records,
            matcher,
            labelset,
            config=cfg,
            labelset_for_fuzzy=labelset,
            code_desc_map=code_desc_map,
            output_dir=output_dir,
        )
        print(json.dumps(official_metrics, ensure_ascii=False, indent=2))
        print(f"  Micro-F1:  {official_metrics['micro_f1']:.4f}")
        print(f"  Precision: {official_metrics['precision']:.4f}")
        print(f"  Recall:    {official_metrics['recall']:.4f}")
        if "macro_f1_present_labels" in official_metrics:
            print(f"  Macro-F1 (present labels): {official_metrics['macro_f1_present_labels']:.4f}")
            print(f"  Macro-F1 (all labels):     {official_metrics['macro_f1_all_labels']:.4f}")

        print("\nFP/FN analysis...")
        fp_c, fn_c = fp_fn_analysis(
            records,
            matcher,
            labelset=labelset,
            config=cfg,
            labelset_for_fuzzy=labelset,
            code_desc_map=code_desc_map,
            output_dir=output_dir,
        )
        print(f"  Top FP: {fp_c.most_common(5)}")
        print(f"  Top FN: {fn_c.most_common(5)}")

    print("\nCode description lookup...")
    if code_desc_map:
        export_code_lookup(
            code_desc_map,
            str(output_dir / "icd10_lookup.json"),
            output_dir=output_dir,
        )
        print(f"  Loaded {len(code_desc_map)} code descriptions")

    export_predictions_jsonl(
        records,
        matcher,
        str(output_dir / "dictionary_predictions_train.jsonl"),
        config=cfg,
        labelset=labelset,
        code_desc_map=code_desc_map,
        output_dir=output_dir,
    )

    if Path(test_path).exists():
        print(f"\nTest set found: {test_path}")
        test_records = load_jsonl(test_path)
        print(f"Loaded {len(test_records)} test records")
        export_predictions_jsonl(
            test_records,
            matcher,
            str(output_dir / "dictionary_predictions_test.jsonl"),
            config=cfg,
            labelset=labelset,
            code_desc_map=code_desc_map,
            output_dir=output_dir,
        )
    else:
        print(f"\nTest set not found ({test_path}) — skipping.")

    compare_eval = cfg.paths.get("compare_eval_test_jsonl")
    compare_out = cfg.paths.get("compare_predictions_jsonl")
    if compare_eval and compare_out and Path(compare_eval).exists():
        print(f"\nCompare split (labeled test): {compare_eval}")
        eval_records = load_jsonl(compare_eval)
        print(f"Loaded {len(eval_records)} records for compare/export")
        out_p = Path(compare_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        export_predictions_jsonl(
            eval_records,
            matcher,
            str(out_p),
            config=cfg,
            labelset=labelset,
            code_desc_map=code_desc_map,
            output_dir=output_dir,
        )
        bundled = output_dir / "predictions.jsonl"
        try:
            shutil.copy2(out_p, bundled)
            print(f"Also copied -> {bundled} (for ``run_predictions`` bundled copy)")
        except OSError as exc:
            print(f"Note: could not copy to {bundled}: {exc}")
    elif compare_eval and not Path(compare_eval).exists():
        print(f"\nCompare eval test not found ({compare_eval}) — skipping compare export.")

    print(f"\nDone. Outputs in: {output_dir.resolve()}")
    print("\nNext steps:")
    print("  1. Send dictionary_predictions_train.jsonl to Stanimeros")
    print("  2. Send dictionary_predictions_test.jsonl to Stanimeros and Strafiotis")
    print("  3. Send full_dictionary.csv to Stelios")
    print("  4. Send icd10_lookup.json to Panagiotis")


if __name__ == "__main__":
    main()
