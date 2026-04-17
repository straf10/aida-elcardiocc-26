import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _aggregate_logits(
    chunk_logits: np.ndarray,
    strategy: str = "max",
    temperature: float = 1.0,
) -> np.ndarray:
    if strategy == "max":
        return np.max(chunk_logits, axis=0)
    if strategy == "mean":
        return np.mean(chunk_logits, axis=0)
    if strategy == "logsumexp":
        temp = max(float(temperature), 1e-6)
        scaled = chunk_logits / temp
        max_scaled = np.max(scaled, axis=0, keepdims=True)
        lse = max_scaled + np.log(
            np.mean(np.exp(scaled - max_scaled), axis=0, keepdims=True)
        )
        return (temp * lse).squeeze(0)
    raise ValueError(
        f"Unknown aggregation strategy '{strategy}'. Choose from: max, mean, logsumexp."
    )


def aggregate_scores_by_patient(
    pid_to_logits: dict,
    strategy: str = "max",
    temperature: float = 1.0,
) -> tuple[list, np.ndarray]:
    """
    Aggregate chunk-level logits per patient, then apply sigmoid.
    Returns (sorted_patient_ids, scores array of shape [n_patients, n_labels]).
    """
    unique_pids = sorted(pid_to_logits.keys())
    rows = []
    for pid in unique_pids:
        chunk_logits = np.array(pid_to_logits[pid])
        aggregated_logits = _aggregate_logits(
            chunk_logits,
            strategy=strategy,
            temperature=temperature,
        )
        rows.append(_sigmoid(aggregated_logits))
    return unique_pids, np.array(rows)
