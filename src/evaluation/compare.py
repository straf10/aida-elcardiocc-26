"""Multi-method compare: score each model's predictions JSONL against gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .config_utils import get_cfg, load_config
from .scoring import evaluate_file


def _parse_splits_arg(raw: str | None) -> List[str]:
    if raw is None or not str(raw).strip():
        return ["test"]
    out: list[str] = []
    for part in str(raw).split(","):
        s = part.strip().lower()
        if s in ("test", "val", "blind"):
            out.append(s)
    return out or ["test"]


def _gold_path_for_cfg_split(cfg: dict, split: str) -> str | None:
    keys = {"test": "data.test_path", "val": "data.val_path", "blind": "data.blind_path"}
    return get_cfg(cfg, keys[split], None)


def _pred_path_for_cfg_model(m: dict, split: str) -> str | None:
    p = m.get("predictions_path")
    if not p:
        return None
    base = Path(p)
    if split == "test":
        return str(base) if base.is_file() else None
    sibling = base.parent / ("val_predictions.jsonl" if split == "val" else "blind_predictions.jsonl")
    return str(sibling) if sibling.is_file() else None


def _blind_gold_has_codes(path: str) -> bool:
    from preprocessing.io_utils import load_jsonl

    for rec in load_jsonl(path):
        for g in rec.get("document_level_annotations") or []:
            if isinstance(g, list) and any(str(c).strip() for c in g):
                return True
    return False


def _split_predictions_basename(split: str) -> str:
    return {
        "test": "test_predictions.jsonl",
        "val": "val_predictions.jsonl",
        "blind": "blind_predictions.jsonl",
    }[split]


def _predictions_dir_root(cfg: dict) -> Path:
    """Root folder scanned for extra JSONL (default ``outputs/predictions``)."""
    raw = get_cfg(cfg, "data.predictions_root", None)
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    return Path("outputs/predictions")


def _display_name_under_predictions_root(pred_file: Path, root: Path) -> str:
    rel = pred_file.parent.resolve().relative_to(root.resolve())
    if rel == Path("."):
        return root.resolve().name
    return rel.as_posix()


def _tier_for_predictions_subpath(display_name: str) -> str:
    """Stacking / metaheuristic export folders → ``ensemble_strategy`` tier (not base models)."""
    if display_name == "ensemble_metaheuristic" or display_name.startswith("ensemble_metaheuristic/"):
        return "ensemble_strategy"
    if display_name == "ensemble_stacking" or display_name.startswith("ensemble_stacking/"):
        return "ensemble_strategy"
    if display_name == "ensemble_committee_mlp" or display_name.startswith("ensemble_committee_mlp/"):
        return "ensemble_strategy"
    if display_name == "ensemble_kfold_stacking" or display_name.startswith("ensemble_kfold_stacking/"):
        return "ensemble_strategy"
    return "individual"


def _display_name_ensemble_from_path(cfg: dict, pred_path: str, config_name: str) -> str:
    """Config rows for ensemble exports → path under predictions root.

    e.g. ``.../ensemble_metaheuristic/weighted/test_predictions.jsonl`` →
    ``ensemble_metaheuristic/weighted`` (no bare parent name in tables).
    """
    if config_name not in (
        "ensemble_metaheuristic",
        "ensemble_stacking",
        "ensemble_committee_mlp",
        "ensemble_kfold_stacking",
    ):
        return config_name
    root = _predictions_dir_root(cfg)
    try:
        parent = Path(pred_path).resolve().parent
        rel = parent.relative_to(root.resolve())
        return rel.as_posix()
    except (ValueError, OSError):
        return config_name


def _append_disk_predictions_not_in_config(
    *,
    cfg: dict,
    models: list[dict],
    split: str,
    gold: str,
    default_ls: str | None,
    rows: list[dict],
) -> None:
    """Score every ``{split}_predictions.jsonl`` under ``data.predictions_root`` not already in ``rows``."""
    from preprocessing.io_utils import LABELSET_PATH, load_labelset

    seen: set[Path] = set()
    for r in rows:
        pp = r.get("predictions_path")
        if isinstance(pp, str) and pp.strip():
            try:
                seen.add(Path(pp).resolve())
            except OSError:
                seen.add(Path(pp))

    root = _predictions_dir_root(cfg)
    if not root.is_dir():
        return

    ens_m = next((m for m in models if str(m.get("name")) == "ensemble_metaheuristic"), None)
    ls_path = default_ls or (ens_m or {}).get("labelset_path") or str(LABELSET_PATH)
    label_space = load_labelset(ls_path)

    basename = _split_predictions_basename(split)
    root_resolved = root.resolve()
    hits = sorted(root.rglob(basename), key=lambda p: str(p))
    for pred in hits:
        if "__pycache__" in pred.parts:
            continue
        if not pred.is_file():
            continue
        try:
            key = pred.resolve()
        except OSError:
            key = pred
        if key in seen:
            continue
        seen.add(key)
        display = _display_name_under_predictions_root(pred, root)
        tier = _tier_for_predictions_subpath(display)
        try:
            metrics = evaluate_file(gold, str(pred), label_space=label_space)
        except Exception as exc:
            rows.append(
                {
                    "name": display,
                    "split": split,
                    "ensemble_tier": tier,
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "name": display,
                "split": split,
                "ensemble_tier": tier,
                "predictions_path": str(key),
                "micro_f1": metrics["micro_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1_present": metrics.get("macro_f1_present_labels"),
            }
        )


def gather_compare_rows(args: argparse.Namespace) -> List[dict]:
    """Build one result dict per method (or error row) for compare / compare_methods.

    With ``--config``, default ``--splits test`` scores ``models[].predictions_path`` against
    ``data.test_path``. Use ``--splits val,test,blind`` for more splits. Rows are tagged
    ``ensemble_tier``: ``ensemble_metaheuristic``, ``ensemble_stacking``, ``ensemble_committee_mlp``,
    and ``ensemble_kfold_stacking`` (plus disk paths under those folders) are **ensemble_strategy**;
    other config models are individuals.
    After config models, every ``{split}_predictions.jsonl`` under ``data.predictions_root`` (default
    ``outputs/predictions``) is scored if not already covered by a config row (same resolved path).
    Names use the path relative to that root (e.g. ``ensemble_metaheuristic/merge_and_weighted_correction``).
    """
    from preprocessing.io_utils import LABELSET_PATH, load_labelset

    rows: list[dict] = []

    if args.config:
        cfg = load_config(args.config)
        default_ls = args.labelset
        models = get_cfg(cfg, "models", []) or []

        if args.ground_truth:
            gt_path = args.ground_truth
            if not gt_path or not Path(gt_path).is_file():
                raise SystemExit(f"Ground truth missing or not a file: {gt_path!r}")
            for m in models:
                name = str(m.get("name", "?"))
                pred_path = m.get("predictions_path")
                ls_path = default_ls or m.get("labelset_path") or str(LABELSET_PATH)
                if not pred_path:
                    rows.append(
                        {
                            "name": name,
                            "split": "override",
                            "ensemble_tier": "individual",
                            "error": "no predictions_path in config",
                        }
                    )
                    continue
                if not Path(pred_path).is_file():
                    rows.append(
                        {
                            "name": name,
                            "split": "override",
                            "ensemble_tier": "individual",
                            "error": f"missing file {pred_path}",
                        }
                    )
                    continue
                label_space = load_labelset(ls_path)
                metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
                rows.append(
                    {
                        "name": name,
                        "split": "override",
                        "ensemble_tier": "individual",
                        "predictions_path": pred_path,
                        "micro_f1": metrics["micro_f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "macro_f1_present": metrics.get("macro_f1_present_labels"),
                    }
                )
        else:
            splits = _parse_splits_arg(getattr(args, "splits", None))
            for split in splits:
                gold = _gold_path_for_cfg_split(cfg, split)
                if not gold or not Path(gold).is_file():
                    rows.append(
                        {
                            "name": "(config)",
                            "split": split,
                            "ensemble_tier": "individual",
                            "error": f"missing gold for split {split!r}: {gold!r}",
                        }
                    )
                    continue
                if split == "blind" and not _blind_gold_has_codes(gold):
                    rows.append(
                        {
                            "name": "(split blind)",
                            "split": split,
                            "ensemble_tier": "individual",
                            "error": "blind JSONL has no document_level_annotations; metrics n/a",
                        }
                    )
                    continue
                for m in models:
                    name = str(m.get("name", "?"))
                    tier = (
                        "ensemble_strategy"
                        if name
                        in (
                            "ensemble_metaheuristic",
                            "ensemble_stacking",
                            "ensemble_committee_mlp",
                            "ensemble_kfold_stacking",
                        )
                        else "individual"
                    )
                    pred_path = _pred_path_for_cfg_model(m, split)
                    ls_path = default_ls or m.get("labelset_path") or str(LABELSET_PATH)
                    if not m.get("predictions_path"):
                        rows.append(
                            {"name": name, "split": split, "ensemble_tier": tier, "error": "no predictions_path in config"}
                        )
                        continue
                    if not pred_path:
                        exp = m.get("predictions_path") if split == "test" else str(Path(m["predictions_path"]).parent / f"{split}_predictions.jsonl")
                        rows.append(
                            {
                                "name": name,
                                "split": split,
                                "ensemble_tier": tier,
                                "error": f"missing predictions file (expected {exp!r})",
                            }
                        )
                        continue
                    label_space = load_labelset(ls_path)
                    metrics = evaluate_file(gold, pred_path, label_space=label_space)
                    row_name = _display_name_ensemble_from_path(cfg, pred_path, name)
                    rows.append(
                        {
                            "name": row_name,
                            "split": split,
                            "ensemble_tier": tier,
                            "predictions_path": pred_path,
                            "micro_f1": metrics["micro_f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "macro_f1_present": metrics.get("macro_f1_present_labels"),
                        }
                    )
                _append_disk_predictions_not_in_config(
                    cfg=cfg,
                    models=models,
                    split=split,
                    gold=gold,
                    default_ls=default_ls,
                    rows=rows,
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
                rows.append(
                    {"name": name, "split": "pair", "ensemble_tier": "individual", "error": f"missing file {pred_path}"}
                )
                continue
            metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
            rows.append(
                {
                    "name": name,
                    "split": "pair",
                    "ensemble_tier": "individual",
                    "predictions_path": pred_path,
                    "micro_f1": metrics["micro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1_present": metrics.get("macro_f1_present_labels"),
                }
            )
    else:
        raise SystemExit("compare: pass --config, or --ground-truth with one or more --pair PRED.jsonl:Name")

    return rows


def _tier(r: dict) -> str:
    return str(r.get("ensemble_tier", "individual"))


def _dedupe_identical_metrics(rows: List[dict]) -> List[dict]:
    """One row per distinct (micro_f1, precision, recall, macro); keep shortest display ``name``.

    Used for ensemble strategy folders that export **identical** predictions (same scores on gold).
    """
    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    buckets: Dict[tuple, dict] = {}
    for r in ok:
        mac = r.get("macro_f1_present")
        key = (
            round(float(r["micro_f1"]), 4),
            round(float(r["precision"]), 4),
            round(float(r["recall"]), 4),
            round(float(mac), 4) if mac is not None else None,
        )
        prev = buckets.get(key)
        if prev is None:
            buckets[key] = r
            continue
        na, nb = str(prev["name"]), str(r["name"])
        if len(na) > len(nb) or (len(na) == len(nb) and na > nb):
            buckets[key] = r
    return errors + list(buckets.values())


def _dedupe_rows_by_predictions_path(rows: List[dict]) -> List[dict]:
    """Keep one row per resolved ``predictions_path``; tie-break by shorter display name then lexicographic.

    Drops duplicate disk scans / symlinks that point at the same JSONL (common under
    ``ensemble_metaheuristic/*/``). Error rows are kept as-is.
    """
    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    chosen: Dict[str, dict] = {}
    for r in ok:
        pp = r.get("predictions_path")
        if isinstance(pp, str) and pp.strip():
            try:
                key = str(Path(pp).resolve())
            except OSError:
                key = pp.strip()
        else:
            key = f"name:{r.get('name', '')}"
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = r
            continue
        a, b = str(prev["name"]), str(r["name"])
        if len(a) > len(b) or (len(a) == len(b) and a > b):
            chosen[key] = r
    return errors + list(chosen.values())


def _table_row_sort_key(r: dict) -> tuple:
    """Errors last; then micro-F1 descending; then name ascending."""
    if "error" in r:
        return (1, 0.0, str(r.get("name", "")))
    return (0, -float(r["micro_f1"]), str(r.get("name", "")))


def _print_method_table(title: str, chunk: List[dict]) -> None:
    if not chunk:
        return
    print(f"\n--- {title} ---")
    col_w = max(22, max((len(r.get("name", "")) for r in chunk), default=10) + 2)
    header = f"{'Method':<{col_w}} {'Micro-F1':>9} {'Precision':>10} {'Recall':>8} {'Macro-F1*':>10}"
    print(header)
    print("-" * len(header))
    for r in sorted(chunk, key=_table_row_sort_key):
        if "error" in r:
            print(f"{r['name']:<{col_w}}  ERROR: {r['error']}")
        else:
            mf = r.get("macro_f1_present")
            mf_s = f"{mf:.4f}" if mf is not None else "n/a"
            print(
                f"{r['name']:<{col_w}} {r['micro_f1']:>9.4f} {r['precision']:>10.4f}"
                f" {r['recall']:>8.4f} {mf_s:>10}"
            )
    print()


def print_compare_report(rows: List[dict]) -> None:
    """Grouped by ``split``, then **individual models** vs **ensemble strategies** (config + scan)."""
    from itertools import groupby

    def _split_key(r: dict) -> str:
        return str(r.get("split", ""))

    rows_sorted = sorted(rows, key=lambda r: (_split_key(r), _tier(r), str(r.get("name", ""))))
    for split, group_it in groupby(rows_sorted, key=_split_key):
        chunk = list(group_it)
        label = split if split else "default"
        print(f"\n=== Split: {label} ===")
        individuals = _dedupe_rows_by_predictions_path(
            [r for r in chunk if _tier(r) != "ensemble_strategy"],
        )
        strategies = _dedupe_rows_by_predictions_path(
            [r for r in chunk if _tier(r) == "ensemble_strategy"],
        )
        strategies = _dedupe_identical_metrics(strategies)
        _print_method_table("Individual models / methods", individuals)
        _print_method_table("Ensemble strategies (config + subfolders)", strategies)
    print("*Macro-F1 over labels with support in gold.\n")


def run_compare(args: argparse.Namespace) -> List[dict]:
    """Evaluate every model from config or --pair list; print report; optional JSON."""
    rows = gather_compare_rows(args)
    print_compare_report(rows)
    if args.metrics_json:
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote compare table JSON -> {out}")
    return rows
