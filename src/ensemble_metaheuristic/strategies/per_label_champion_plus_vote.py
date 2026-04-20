"""Per-label champion plus agreement among non-champion models (OR gate)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def per_label_champion_plus_other_vote_predict(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: List[str],
    all_pids: List[int],
    all_labels: List[str],
    label_routing: Dict[str, str],
    *,
    score_cutoff: float = 1.0,
    binary_cutoff: float = 0.5,
    min_other_votes: int = 2,
) -> Dict[int, List[str]]:
    """
    Per-label **champion as base**, **non-champions vote** to add labels.

    For document ``i`` and label ``L`` with champion model ``C``:

    - **Base:** include ``L`` if ``C``'s score for ``L`` passes ``C``'s cutoff (same as pure per-label routing).
    - **Vote:** else include ``L`` if at least ``max(1, min_other_votes)`` *other* models pass their cutoffs
      for ``L`` (so every model can contribute, not only the champion).

    This is an OR between the champion gate and agreement among the rest; it tends to raise recall
    vs strict per-label routing when ``min_other_votes`` is low.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    min_other_votes = max(0, min(int(min_other_votes), max(0, len(names) - 1)))
    pred_data: Dict[int, List[str]] = {pid: [] for pid in all_pids}

    for j, label in enumerate(all_labels):
        champion = label_routing.get(label, names[0])
        ci = name_to_idx[champion]
        cut_c = score_cutoff if is_score_model[ci] else binary_cutoff
        col_c = matrices[ci][:, j]

        other_masks: List[np.ndarray] = []
        for mi, _nm in enumerate(names):
            if mi == ci:
                continue
            cut_m = score_cutoff if is_score_model[mi] else binary_cutoff
            other_masks.append(matrices[mi][:, j] >= cut_m)
        if not other_masks:
            votes_others = np.zeros(len(all_pids), dtype=np.int32)
            vote_ok = np.zeros(len(all_pids), dtype=bool)
        else:
            votes_others = np.sum(np.stack(other_masks, axis=0), axis=0).astype(np.int32)
            thr_other = max(1, int(min_other_votes))
            vote_ok = votes_others >= thr_other

        champ_on = col_c >= cut_c
        accept = champ_on | vote_ok
        for i, pid in enumerate(all_pids):
            if accept[i]:
                pred_data[pid].append(label)

    return pred_data
