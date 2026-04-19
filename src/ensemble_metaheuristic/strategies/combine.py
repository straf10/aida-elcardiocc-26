"""OR / AND / k-of-n fusion of flat prediction dicts from other strategies."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List


def merge_preds_union(
    a: Dict[int, List[str]],
    b: Dict[int, List[str]],
    all_pids: List[int],
) -> Dict[int, List[str]]:
    return {pid: sorted(set(a.get(pid, [])) | set(b.get(pid, []))) for pid in all_pids}


def merge_preds_intersection(
    a: Dict[int, List[str]],
    b: Dict[int, List[str]],
    all_pids: List[int],
) -> Dict[int, List[str]]:
    return {pid: sorted(set(a.get(pid, [])) & set(b.get(pid, []))) for pid in all_pids}


def merge_preds_k_of_n(
    pred_list: List[Dict[int, List[str]]],
    all_pids: List[int],
    k: int,
) -> Dict[int, List[str]]:
    """Predict a label if it appears in at least ``k`` of the strategy outputs (per document)."""
    if not pred_list:
        return {pid: [] for pid in all_pids}
    k = min(max(int(k), 1), len(pred_list))
    out: Dict[int, List[str]] = {}
    for pid in all_pids:
        cnt: Counter[str] = Counter()
        for d in pred_list:
            for lab in d.get(pid, []):
                cnt[lab] += 1
        out[pid] = sorted(lab for lab, c in cnt.items() if c >= k)
    return out
