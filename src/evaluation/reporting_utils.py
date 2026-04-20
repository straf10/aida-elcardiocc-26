"""Sparse binary matrices for sklearn-style metrics (``metrics_engine``)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


def build_binary_matrices(
    gt_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    patient_ids: List[int],
    label_names: List[str],
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build scipy sparse CSR matrices for GT and Predictions (N_docs x n_labels)."""
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    n_docs = len(patient_ids)
    n_labels = len(label_names)

    y_true = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)
    y_pred = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)

    for i, pid in enumerate(patient_ids):
        gt_groups = gt_data.get(pid, [])
        for group in gt_groups:
            for code in group:
                if code in label_to_idx:
                    y_true[i, label_to_idx[code]] = 1

        preds = pred_data.get(pid, [])
        for code in preds:
            if code in label_to_idx:
                y_pred[i, label_to_idx[code]] = 1

    return y_true.tocsr(), y_pred.tocsr()
