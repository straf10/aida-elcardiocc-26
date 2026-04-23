"""Meta-classifiers for the stacking ensemble.

Four learner families are supported:

* ``logistic_regression`` — fast, interpretable, well-regularised; CPU only.
* ``random_forest``       — captures non-linear interactions; CPU, multi-core.
* ``gradient_boosting``  — ``HistGradientBoostingClassifier``; strong but slower; CPU.
* ``pytorch_mlp``        — shared MLP trained on GPU (CUDA / MPS / CPU fallback).
                           Treats every (doc, label) pair as one training example
                           so the whole dataset fits in a single GPU forward/backward
                           pass — much faster than per-label sklearn loops when a GPU
                           is available. Optional **val BCE early stopping** (see
                           ``mlp_early_stop_patience``) restores the best checkpoint when
                           validation loss plateaus.

``LEARNER_NAMES`` lists all options.  Pass ``device="auto"`` (default) to let
PyTorch auto-detect CUDA → MPS → CPU.  Pass ``device="cuda"`` to force GPU.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from .meta_features import build_target_matrix, extract_label_features

LEARNER_NAMES = ("logistic_regression", "random_forest", "gradient_boosting", "pytorch_mlp")
_SKLEARN_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")
MIN_POSITIVE_DEFAULT = 3


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_device(spec: str) -> torch.device:
    """Resolve a device specifier string to a ``torch.device``.

    ``"auto"`` picks CUDA if available, then MPS (Apple Silicon), then CPU.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# sklearn per-label stacker
# ---------------------------------------------------------------------------

def _make_sklearn_learner(
    name: str,
    seed: int,
    *,
    logreg_max_iter: int = 1000,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 6,
    hgb_max_iter: int = 100,
    hgb_max_depth: int = 4,
):
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=int(logreg_max_iter),
            solver="lbfgs",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(rf_n_estimators),
            max_depth=int(rf_max_depth),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(hgb_max_iter),
            max_depth=int(hgb_max_depth),
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"Unknown sklearn learner {name!r}. Choose from {_SKLEARN_NAMES}.")


class PerLabelStackingEnsemble:
    """Train one sklearn binary classifier per label using base-model scores as features.

    Attributes
    ----------
    classifiers_:
        Dict mapping ``label_idx`` → trained sklearn classifier.
        Only populated for labels with enough positive training examples.
    """

    def __init__(
        self,
        meta_learner: str = "logistic_regression",
        seed: int = 42,
        include_aggregates: bool = True,
        min_positive: int = MIN_POSITIVE_DEFAULT,
        *,
        logreg_max_iter: int = 1000,
        rf_n_estimators: int = 100,
        rf_max_depth: int = 6,
        hgb_max_iter: int = 100,
        hgb_max_depth: int = 4,
    ) -> None:
        if meta_learner not in _SKLEARN_NAMES:
            raise ValueError(
                f"meta_learner must be one of {_SKLEARN_NAMES}, got {meta_learner!r}. "
                "For GPU, use PyTorchMLPStacker (or make_stacker('pytorch_mlp', ...)).",
            )
        self.meta_learner = meta_learner
        self.seed = seed
        self.include_aggregates = include_aggregates
        self.min_positive = min_positive
        self.logreg_max_iter = int(logreg_max_iter)
        self.rf_n_estimators = int(rf_n_estimators)
        self.rf_max_depth = int(rf_max_depth)
        self.hgb_max_iter = int(hgb_max_iter)
        self.hgb_max_depth = int(hgb_max_depth)
        self.classifiers_: Dict[int, object] = {}

    def fit(
        self,
        train_matrices: List[np.ndarray],
        train_gt: Dict,
        train_pids: List[int],
        all_labels: List[str],
        **_: Any,
    ) -> "PerLabelStackingEnsemble":
        self.classifiers_ = {}
        Y = build_target_matrix(train_gt, train_pids, all_labels)
        n_labels = len(all_labels)
        trained = skipped = 0

        for j in range(n_labels):
            X_j = extract_label_features(
                train_matrices, j, include_aggregates=self.include_aggregates,
            )
            y_j = Y[:, j]
            if int(y_j.sum()) < self.min_positive:
                skipped += 1
                continue
            clf = _make_sklearn_learner(
                self.meta_learner,
                self.seed + j,
                logreg_max_iter=self.logreg_max_iter,
                rf_n_estimators=self.rf_n_estimators,
                rf_max_depth=self.rf_max_depth,
                hgb_max_iter=self.hgb_max_iter,
                hgb_max_depth=self.hgb_max_depth,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    clf.fit(X_j, y_j)
                    self.classifiers_[j] = clf
                    trained += 1
                except Exception:
                    skipped += 1

        print(
            f"    [{self.meta_learner}] trained={trained}  "
            f"skipped={skipped}  (of {n_labels} labels)",
        )
        return self

    def predict_proba(
        self,
        matrices: List[np.ndarray],
        pids: List[int],
        all_labels: List[str],
    ) -> np.ndarray:
        n_docs = len(pids)
        n_labels = len(all_labels)
        proba = np.zeros((n_docs, n_labels), dtype=np.float32)

        for j in range(n_labels):
            X_j = extract_label_features(
                matrices, j, include_aggregates=self.include_aggregates,
            )
            if j in self.classifiers_:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        proba[:, j] = self.classifiers_[j].predict_proba(X_j)[:, 1]
                    except Exception:
                        proba[:, j] = _fallback_col(matrices, j)
            else:
                proba[:, j] = _fallback_col(matrices, j)

        return proba

    @property
    def n_trained(self) -> int:
        return len(self.classifiers_)


# ---------------------------------------------------------------------------
# PyTorch MLP stacker (GPU-accelerated)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Small fully-connected network for binary classification of (doc, label) pairs."""

    def __init__(self, in_features: int, hidden_dims: Tuple[int, ...]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PyTorchMLPStacker:
    """GPU-accelerated MLP stacking ensemble.

    Architecture
    ------------
    A single shared binary MLP is trained over **all** ``(doc, label)`` pairs at
    once.  For ``n_train`` documents and ``n_labels`` labels the training set has
    ``n_train × n_labels`` examples; each example carries the ``K + 3`` feature
    vector produced by ``extract_label_features`` (K model scores + max/mean/vote).

    The shared model learns which patterns of model agreement predict a positive
    label, regardless of which label it is.  This is more GPU-efficient than
    fitting a separate network per label.

    Device selection
    ----------------
    Pass ``device="auto"`` (default) to auto-detect CUDA → MPS → CPU.
    Pass ``device="cuda"`` or ``device="mps"`` to force a specific backend.
    The resolved device is printed at fit time.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 512,
        device: str = "auto",
        seed: int = 42,
        include_aggregates: bool = True,
        mlp_early_stop_patience: int = 0,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device_spec = device
        self.seed = seed
        self.include_aggregates = include_aggregates
        self.mlp_early_stop_patience = max(0, int(mlp_early_stop_patience))
        self.model_: Optional[_MLP] = None
        self._device: Optional[torch.device] = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = resolve_device(self.device_spec)
        return self._device

    def _build_feature_matrix(
        self,
        matrices: List[np.ndarray],
        n_labels: int,
    ) -> np.ndarray:
        """Stack per-label features into (n_docs × n_labels, K+3)."""
        parts = [
            extract_label_features(matrices, j, include_aggregates=self.include_aggregates)
            for j in range(n_labels)
        ]
        return np.concatenate(parts, axis=0).astype(np.float32)

    def fit(
        self,
        train_matrices: List[np.ndarray],
        train_gt: Dict,
        train_pids: List[int],
        all_labels: List[str],
        *,
        val_matrices: Optional[List[np.ndarray]] = None,
        val_gt: Optional[Dict] = None,
        val_pids: Optional[List[int]] = None,
    ) -> "PyTorchMLPStacker":
        torch.manual_seed(self.seed)
        dev = self.device
        n_labels = len(all_labels)
        n_train = len(train_pids)

        print(f"    [pytorch_mlp] device={dev}  building {n_train}×{n_labels} training pairs…")

        # Feature matrix: (n_train * n_labels, K+3)
        X_all = self._build_feature_matrix(train_matrices, n_labels)

        # Target: (n_train, n_labels) → column-major ravel → (n_labels * n_train,)
        Y = build_target_matrix(train_gt, train_pids, all_labels)
        y_all = Y.T.ravel().astype(np.float32)  # same ordering as X_all

        # Positive-class weight for class imbalance
        n_pos = float(y_all.sum())
        n_neg = float(len(y_all)) - n_pos
        pos_w = n_neg / n_pos if n_pos > 0 else 1.0

        X_t = torch.from_numpy(X_all).to(dev)
        y_t = torch.from_numpy(y_all).to(dev)
        pos_weight = torch.tensor([pos_w], device=dev)

        n_features = X_all.shape[1]
        self.model_ = _MLP(n_features, self.hidden_dims).to(dev)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        use_val_es = (
            self.mlp_early_stop_patience > 0
            and val_matrices is not None
            and val_gt is not None
            and val_pids is not None
        )
        if use_val_es:
            print(
                f"    [pytorch_mlp] val early-stop patience={self.mlp_early_stop_patience} "
                f"(restore weights on best val BCE)",
                flush=True,
            )
        best_vloss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stall = 0

        self.model_.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)
            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / len(y_all)
                print(f"    [pytorch_mlp] epoch {epoch + 1:3d}/{self.n_epochs}  loss={avg:.4f}")

            if use_val_es:
                self.model_.eval()
                with torch.no_grad():
                    Xv = self._build_feature_matrix(val_matrices, n_labels)
                    Yv = build_target_matrix(val_gt, val_pids, all_labels)
                    yv = Yv.T.ravel().astype(np.float32)
                    v_logits = self.model_(torch.from_numpy(Xv).to(dev))
                    v_loss = float(criterion(v_logits, torch.from_numpy(yv).to(dev)).item())
                self.model_.train()
                if v_loss < best_vloss - 1e-7:
                    best_vloss = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    stall = 0
                else:
                    stall += 1
                    if stall >= self.mlp_early_stop_patience:
                        print(
                            f"    [pytorch_mlp] early stop at epoch {epoch + 1}  "
                            f"best_val_bce={best_vloss:.4f}",
                            flush=True,
                        )
                        break

        if use_val_es and best_state is not None:
            self.model_.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
            print(f"    [pytorch_mlp] loaded best val BCE checkpoint ({best_vloss:.4f})", flush=True)

        print(
            f"    [pytorch_mlp] training complete  "
            f"examples={n_train * n_labels:,}  device={dev}",
        )
        return self

    def predict_proba(
        self,
        matrices: List[np.ndarray],
        pids: List[int],
        all_labels: List[str],
    ) -> np.ndarray:
        """Return (n_docs × n_labels) probability matrix via GPU forward pass."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        dev = self.device
        n_docs = len(pids)
        n_labels = len(all_labels)

        X_all = self._build_feature_matrix(matrices, n_labels)
        X_t = torch.from_numpy(X_all).to(dev)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            proba_flat = torch.sigmoid(logits).cpu().numpy()

        # Reshape (n_labels * n_docs,) → (n_labels, n_docs) → (n_docs, n_labels)
        return proba_flat.reshape(n_labels, n_docs).T.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared fallback and factory
# ---------------------------------------------------------------------------

def _fallback_col(matrices: List[np.ndarray], j: int) -> np.ndarray:
    """Max normalised score across models mapped to [0, 1].  Used when no classifier exists."""
    raw = np.column_stack([mat[:, j] for mat in matrices])
    return np.clip(raw.max(axis=1) / 2.0, 0.0, 1.0).astype(np.float32)


def make_stacker(
    name: str,
    *,
    seed: int = 42,
    device: str = "auto",
    include_aggregates: bool = True,
    min_positive: int = MIN_POSITIVE_DEFAULT,
    mlp_epochs: int = 50,
    mlp_lr: float = 1e-3,
    mlp_batch_size: int = 512,
    mlp_hidden_dims: Tuple[int, ...] = (64, 32),
    logreg_max_iter: int = 1000,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 6,
    hgb_max_iter: int = 100,
    hgb_max_depth: int = 4,
    mlp_early_stop_patience: int = 0,
) -> Union[PerLabelStackingEnsemble, PyTorchMLPStacker]:
    """Factory: return the right stacker class for ``name``.

    Parameters
    ----------
    name:
        One of ``LEARNER_NAMES``.
    device:
        PyTorch device spec (only used for ``pytorch_mlp``).
        ``"auto"`` picks CUDA → MPS → CPU automatically.
    mlp_epochs / mlp_lr / mlp_batch_size / mlp_hidden_dims:
        Training budget for ``pytorch_mlp``.
    logreg_max_iter / rf_* / hgb_*:
        Training budget for sklearn meta-learners (ignored for ``pytorch_mlp``).
    mlp_early_stop_patience:
        If >0, monitor val BCE each epoch (pass ``val_*`` into ``fit``) and stop when
        val loss does not improve for this many epochs; restore best weights.
    """
    if name in _SKLEARN_NAMES:
        return PerLabelStackingEnsemble(
            meta_learner=name,
            seed=seed,
            include_aggregates=include_aggregates,
            min_positive=min_positive,
            logreg_max_iter=logreg_max_iter,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            hgb_max_iter=hgb_max_iter,
            hgb_max_depth=hgb_max_depth,
        )
    if name == "pytorch_mlp":
        return PyTorchMLPStacker(
            hidden_dims=mlp_hidden_dims,
            lr=float(mlp_lr),
            n_epochs=int(mlp_epochs),
            batch_size=int(mlp_batch_size),
            device=device,
            seed=seed,
            include_aggregates=include_aggregates,
            mlp_early_stop_patience=int(mlp_early_stop_patience),
        )
    raise ValueError(f"Unknown stacker {name!r}. Choose from {LEARNER_NAMES}.")
