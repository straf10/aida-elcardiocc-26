from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from src.analysis.common import load_model_artifacts
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.evaluation.config_utils import load_config, get_cfg
except ImportError:
    from analysis.common import load_model_artifacts  # type: ignore
    from evaluation.evaluator import evaluate_data  # type: ignore
    from evaluation.io_utils import load_ground_truth  # type: ignore
    from evaluation.config_utils import load_config, get_cfg  # type: ignore

from visualisation.src.config import EXCLUDED_MODELS

Pair = Tuple[str, str]


@dataclass
class DocWrongEdge:
    """One (patient_id, predicted_fp, missed_true) for a specific source model."""

    pid: int
    predicted: str
    missed: str
    source_model: str


@dataclass
class CrossModelBundle:
    """Holds GT, predictions, per-model metrics, wrong-pair counts, and doc-level wrong edges."""

    cfg: Dict[str, Any]
    gt_data: Dict[int, List[List[str]]]
    patient_ids: List[int]
    model_names: List[str]
    label_names: List[str]
    pred_by_model: Dict[str, Dict[int, List[str]]]
    metrics_by_model: Dict[str, Dict[str, Any]]
    wrong_pairs_by_model: Dict[str, Counter[Pair]]
    pooled_wrong_pairs: Counter[Pair]
    doc_edges_by_model: Dict[str, List[DocWrongEdge]] = field(default_factory=dict)

    def rescuers_for_edge(self, edge: DocWrongEdge) -> List[str]:
        """Models (full names) other than source that predict `missed` on this pid."""
        out: List[str] = []
        for name in self.model_names:
            if name == edge.source_model:
                continue
            preds = set(self.pred_by_model.get(name, {}).get(edge.pid, []))
            if edge.missed in preds:
                out.append(name)
        return sorted(out)


def _doc_wrong_edges_for_model(model_name: str, metrics: Dict[str, Any]) -> List[DocWrongEdge]:
    edges: List[DocWrongEdge] = []
    for row in metrics.get("doc_breakdown", []):
        pid = int(row["patient_id"])
        wrong_codes = row.get("wrong_codes", [])
        missed_groups = row.get("missed_groups", [])
        for p in wrong_codes:
            for group in missed_groups:
                for t in group:
                    edges.append(DocWrongEdge(pid=pid, predicted=p, missed=t, source_model=model_name))
    return edges


def _pooled_counter(wrong_by_model: Dict[str, Counter[Pair]]) -> Counter[Pair]:
    out: Counter[Pair] = Counter()
    for ctr in wrong_by_model.values():
        out.update(ctr)
    return out


def load_cross_model_bundle(config_path: Path) -> CrossModelBundle:
    """Load validation GT and all non-base models from analysis.yaml; build pooled wrong-pair stats."""
    cfg = load_config(str(config_path))
    val_path = get_cfg(cfg, "data.val_path")
    gt_data = load_ground_truth(val_path)
    patient_ids = sorted(gt_data.keys())

    model_cfgs = [m for m in get_cfg(cfg, "models", []) if m.get("name") not in EXCLUDED_MODELS]
    model_names = [m["name"] for m in model_cfgs]
    if not model_names:
        raise ValueError("No models left after excluding xlm_r_base.")

    out_root = Path(get_cfg(cfg, "output.dir", "outputs/analysis"))
    artifacts_list = [load_model_artifacts(m, patient_ids, analysis_out_dir=out_root) for m in model_cfgs]

    # Single label space: prefer first artifact's label_names (aligned across competition)
    label_names = list(artifacts_list[0].label_names)
    pred_by_model: Dict[str, Dict[int, List[str]]] = {}
    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    wrong_pairs_by_model: Dict[str, Counter[Pair]] = {}
    doc_edges_by_model: Dict[str, List[DocWrongEdge]] = {}

    try:
        from src.analysis.error_analysis import build_confusion_views
    except ImportError:
        from analysis.error_analysis import build_confusion_views  # type: ignore

    for art in artifacts_list:
        name = art.name
        pred_by_model[name] = art.pred_data
        metrics = evaluate_data(gt_data, art.pred_data, label_space=art.label_names)
        metrics_by_model[name] = metrics
        views = build_confusion_views(metrics)
        wrong_pairs_by_model[name] = views["wrong_pairs"]
        doc_edges_by_model[name] = _doc_wrong_edges_for_model(name, metrics)

    pooled = _pooled_counter(wrong_pairs_by_model)

    return CrossModelBundle(
        cfg=cfg,
        gt_data=gt_data,
        patient_ids=patient_ids,
        model_names=model_names,
        label_names=label_names,
        pred_by_model=pred_by_model,
        metrics_by_model=metrics_by_model,
        wrong_pairs_by_model=wrong_pairs_by_model,
        pooled_wrong_pairs=pooled,
        doc_edges_by_model=doc_edges_by_model,
    )


def top_pairs_subset(
    pooled: Counter[Pair],
    important_pairs: Set[Pair],
    top_n: int,
) -> List[Pair]:
    """Order: pooled count desc; tie-break by pair string; intersect important if possible."""
    ranked = [p for p, _ in pooled.most_common()]
    in_imp = [p for p in ranked if p in important_pairs]
    if len(in_imp) >= top_n:
        return in_imp[:top_n]
    # fall back: add highest pooled pairs not in important set
    rest = [p for p in ranked if p not in important_pairs]
    merged = in_imp + rest
    return merged[:top_n]


def per_class_fn(row: dict) -> int:
    return int(row.get("support", 0)) - int(row.get("groups_hit", 0))
