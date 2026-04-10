import numpy as np


def aggregate_scores_by_patient(pid_to_logits: dict) -> tuple[list, np.ndarray]:
    """
    Max-pool logits over chunks per patient, then apply sigmoid.
    Returns (sorted_patient_ids, scores array of shape [n_patients, n_labels]).
    """
    unique_pids = sorted(pid_to_logits.keys())
    rows = []
    for pid in unique_pids:
        chunk_logits = np.array(pid_to_logits[pid])
        max_logits = np.max(chunk_logits, axis=0)
        rows.append(1.0 / (1.0 + np.exp(-max_logits)))
    return unique_pids, np.array(rows)
