from __future__ import annotations

import numpy as np
import torch


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _aggregate_logits(
    chunk_logits: np.ndarray,
    strategy: str = "max",
    temperature: float = 1.0,
    alpha: float = 0.5,
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
    if strategy == "mean_max":
        w = float(np.clip(alpha, 0.0, 1.0))
        mean_logits = np.mean(chunk_logits, axis=0)
        max_logits = np.max(chunk_logits, axis=0)
        return (w * mean_logits) + ((1.0 - w) * max_logits)
    raise ValueError(
        f"Unknown aggregation strategy '{strategy}'. Choose from: max, mean, logsumexp, mean_max."
    )


def aggregate_scores_by_patient(
    pid_to_logits: dict,
    strategy: str = "max",
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> tuple[list, np.ndarray]:
    """
    Aggregate chunk-level logits per patient, then apply sigmoid.
    Note: pid_to_logits[pid] may contain chunks from multiple documents
    of the same patient. This pools across all chunks and documents for a patient,
    which is the desired behavior for ELCardioCC patient-level evaluation.
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
            alpha=alpha,
        )
        rows.append(_sigmoid(aggregated_logits))
    return unique_pids, np.array(rows)


def aggregate_scores_by_patient_torch(
    logits: torch.Tensor,
    patient_ids: torch.Tensor,
    strategy: str = "mean_max",
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> tuple[list[int], np.ndarray]:
    """
    Math-equivalent to aggregate_scores_by_patient, but fuses all chunks in one
    pass on the device where logits live (avoids per-chunk host sync).
    """
    if strategy == "logsumexp":
        n = logits.shape[0]
        pid_to: dict[int, list[np.ndarray]] = {}
        lcpu = logits.detach().float().cpu().numpy()
        pids_cpu = patient_ids.view(-1).long().cpu().numpy()
        for i in range(n):
            pid = int(pids_cpu[i])
            pid_to.setdefault(pid, []).append(lcpu[i])
        return aggregate_scores_by_patient(
            {k: pid_to[k] for k in sorted(pid_to.keys())},
            strategy="logsumexp",
            temperature=temperature,
            alpha=alpha,
        )

    if strategy not in ("max", "mean", "mean_max"):
        raise ValueError(
            f"Unknown aggregation strategy '{strategy}'. Choose from: max, mean, logsumexp, mean_max."
        )

    device = logits.device
    L = int(logits.shape[1])
    if logits.shape[0] == 0:
        return [], np.empty((0, L), dtype=np.float64)
    patient_ids = patient_ids.to(device=device, dtype=torch.long).contiguous().view(-1)
    logits_f = logits.float()
    unique_pids, inv = torch.unique(patient_ids, sorted=True, return_inverse=True)
    n = int(unique_pids.shape[0])
    if strategy in ("max", "mean_max"):
        max_agg = torch.full(
            (n, L), float("-inf"), device=device, dtype=torch.float32
        )
        max_agg.scatter_reduce_(
            0,
            inv.unsqueeze(1).expand(-1, L),
            logits_f,
            reduce="amax",
            include_self=True,
        )
    if strategy in ("mean", "mean_max"):
        sum_agg = torch.zeros((n, L), device=device, dtype=torch.float32)
        counts = torch.zeros((n,), device=device, dtype=torch.float32)
        ones = torch.ones((inv.shape[0],), device=device, dtype=torch.float32)
        sum_agg.scatter_add_(0, inv.unsqueeze(1).expand(-1, L), logits_f)
        counts.scatter_add_(0, inv, ones)
        mean_agg = sum_agg / counts.unsqueeze(1).clamp_min(1e-8)
    w = float(max(0.0, min(1.0, float(alpha))))
    if strategy == "mean_max":
        agg = w * mean_agg + (1.0 - w) * max_agg
    elif strategy == "max":
        agg = max_agg
    else:
        agg = mean_agg
    scores = torch.sigmoid(agg).float().cpu().numpy()
    return [int(p) for p in unique_pids.tolist()], np.asarray(
        scores, dtype=np.float64
    )
