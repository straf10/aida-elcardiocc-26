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
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from .meta_features import MetaFeatureMode, build_target_matrix, extract_label_features
from .patient_clusters import (
    fit_train_patient_clusters,
    patient_cluster_onehot,
)

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
    logreg_c: float = 1.0,
    logreg_max_iter: int = 1000,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 6,
    hgb_max_iter: int = 100,
    hgb_max_depth: int = 4,
):
    if name == "logistic_regression":
        return LogisticRegression(
            C=float(logreg_c),
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
        meta_features: MetaFeatureMode = "default",
        min_positive: int = MIN_POSITIVE_DEFAULT,
        *,
        logreg_c: float = 1.0,
        logreg_max_iter: int = 1000,
        rf_n_estimators: int = 100,
        rf_max_depth: int = 6,
        hgb_max_iter: int = 100,
        hgb_max_depth: int = 4,
        patient_cluster_k: int = 0,
    ) -> None:
        if meta_learner not in _SKLEARN_NAMES:
            raise ValueError(
                f"meta_learner must be one of {_SKLEARN_NAMES}, got {meta_learner!r}. "
                "For GPU, use PyTorchMLPStacker (or make_stacker('pytorch_mlp', ...)).",
            )
        self.meta_learner = meta_learner
        self.seed = seed
        self.include_aggregates = include_aggregates
        self.meta_features = cast(MetaFeatureMode, meta_features)
        self.min_positive = min_positive
        self.logreg_max_iter = int(logreg_max_iter)
        self.rf_n_estimators = int(rf_n_estimators)
        self.rf_max_depth = int(rf_max_depth)
        self.hgb_max_iter = int(hgb_max_iter)
        self.hgb_max_depth = int(hgb_max_depth)
        self.logreg_c = float(logreg_c)
        self.patient_cluster_k = max(0, int(patient_cluster_k))
        self._patient_kmeans: Optional[object] = None
        self._patient_cluster_active: int = 0
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
        self._patient_kmeans = None
        self._patient_cluster_active = 0
        extra_train: Optional[np.ndarray] = None
        if self.patient_cluster_k >= 2:
            km, eff_k = fit_train_patient_clusters(
                train_matrices, self.patient_cluster_k, seed=self.seed,
            )
            if km is not None:
                self._patient_kmeans = km
                self._patient_cluster_active = eff_k
                extra_train = patient_cluster_onehot(train_matrices, km, eff_k)
                print(
                    f"    [patient_clusters] K={eff_k}  appended one-hot to meta-features",
                    flush=True,
                )
            else:
                print(
                    f"    [patient_clusters] disabled (need n_train ≥ K; K={self.patient_cluster_k})",
                    flush=True,
                )

        Y = build_target_matrix(train_gt, train_pids, all_labels)
        n_labels = len(all_labels)
        trained = skipped = 0

        for j in range(n_labels):
            X_j = extract_label_features(
                train_matrices,
                j,
                include_aggregates=self.include_aggregates,
                meta_features=self.meta_features,
            )
            if extra_train is not None:
                X_j = np.hstack([X_j, extra_train]).astype(np.float32, copy=False)
            y_j = Y[:, j]
            if int(y_j.sum()) < self.min_positive:
                skipped += 1
                continue
            clf = _make_sklearn_learner(
                self.meta_learner,
                self.seed + j,
                logreg_c=self.logreg_c,
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

        extra: Optional[np.ndarray] = None
        if self._patient_kmeans is not None and self._patient_cluster_active >= 2:
            extra = patient_cluster_onehot(
                matrices, self._patient_kmeans, self._patient_cluster_active,
            )

        for j in range(n_labels):
            X_j = extract_label_features(
                matrices,
                j,
                include_aggregates=self.include_aggregates,
                meta_features=self.meta_features,
            )
            if extra is not None:
                X_j = np.hstack([X_j, extra]).astype(np.float32, copy=False)
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

class _StackingMLP(nn.Module):
    """FC net for binary (doc, label) classification; optional label embedding."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Tuple[int, ...],
        *,
        n_labels: int,
        label_emb_dim: int,
    ) -> None:
        super().__init__()
        self.label_emb_dim = int(label_emb_dim)
        if self.label_emb_dim > 0:
            self.label_emb = nn.Embedding(int(n_labels), self.label_emb_dim)
            din = in_features + self.label_emb_dim
        else:
            self.label_emb = None
            din = in_features
        layers: List[nn.Module] = []
        prev = din
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, label_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.label_emb is not None:
            if label_ids is None:
                raise ValueError("label_ids required when label embedding is enabled")
            e = self.label_emb(label_ids)
            x = torch.cat([x, e], dim=-1)
        return self.net(x).squeeze(-1)


class PyTorchMLPStacker:
    """GPU-accelerated MLP stacking ensemble.

    Architecture
    ------------
    A single shared binary MLP is trained over **all** ``(doc, label)`` pairs at
    once.  For ``n_train`` documents and ``n_labels`` labels the training set has
    ``n_train × n_labels`` examples; each row uses ``extract_label_features`` for
    that label (width set by ``--meta-features``).

    With ``--mlp-label-emb D`` (D>0), a learned embedding of the label index is
    concatenated to meta-features so the network can learn **label-specific**
    fusion (the default shared MLP without embedding treats every code the same
    up to its score vector).  This is more GPU-efficient than one sklearn model
    per label.

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
        meta_features: MetaFeatureMode = "default",
        label_emb_dim: int = 0,
        patient_cluster_k: int = 0,
        mlp_early_stop_patience: int = 0,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device_spec = device
        self.seed = seed
        self.include_aggregates = include_aggregates
        self.meta_features = cast(MetaFeatureMode, meta_features)
        self.label_emb_dim = max(0, int(label_emb_dim))
        self.patient_cluster_k = max(0, int(patient_cluster_k))
        self._patient_kmeans: Optional[object] = None
        self._patient_cluster_active: int = 0
        self.mlp_early_stop_patience = max(0, int(mlp_early_stop_patience))
        self.model_: Optional[_StackingMLP] = None
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
        """Stack per-label features into (n_docs × n_labels, n_meta_feat)."""
        ex: Optional[np.ndarray] = None
        if self._patient_kmeans is not None and self._patient_cluster_active >= 2:
            ex = patient_cluster_onehot(
                matrices, self._patient_kmeans, self._patient_cluster_active,
            )
        parts: List[np.ndarray] = []
        for j in range(n_labels):
            block = extract_label_features(
                matrices,
                j,
                include_aggregates=self.include_aggregates,
                meta_features=self.meta_features,
            )
            if ex is not None:
                block = np.hstack([block, ex]).astype(np.float32, copy=False)
            parts.append(block.astype(np.float32, copy=False))
        return np.concatenate(parts, axis=0).astype(np.float32)

    @staticmethod
    def _label_ids_flat(n_docs: int, n_labels: int) -> np.ndarray:
        """Row r corresponds to label r // n_docs (blocks of n_docs per label)."""
        return (np.arange(n_docs * n_labels, dtype=np.int64) // n_docs)

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

        self._patient_kmeans = None
        self._patient_cluster_active = 0
        if self.patient_cluster_k >= 2:
            km, eff_k = fit_train_patient_clusters(
                train_matrices, self.patient_cluster_k, seed=self.seed,
            )
            if km is not None:
                self._patient_kmeans = km
                self._patient_cluster_active = eff_k
                print(
                    f"    [pytorch_mlp] patient_clusters K={eff_k} (train-only KMeans)",
                    flush=True,
                )
            else:
                print(
                    f"    [pytorch_mlp] patient_clusters disabled "
                    f"(n_train < K={self.patient_cluster_k})",
                    flush=True,
                )

        # Feature matrix: (n_train * n_labels, n_meta_feat)
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
        if self.label_emb_dim > 0:
            print(
                f"    [pytorch_mlp] meta_feat_dim={n_features}  label_emb_dim={self.label_emb_dim}",
                flush=True,
            )
        self.model_ = _StackingMLP(
            n_features,
            self.hidden_dims,
            n_labels=n_labels,
            label_emb_dim=self.label_emb_dim,
        ).to(dev)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if self.label_emb_dim > 0:
            lbl_flat = self._label_ids_flat(n_train, n_labels)
            lbl_t = torch.from_numpy(lbl_flat).to(dev)
            loader = DataLoader(
                TensorDataset(X_t, lbl_t, y_t),
                batch_size=self.batch_size,
                shuffle=True,
            )
        else:
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
            for batch in loader:
                if self.label_emb_dim > 0:
                    X_batch, lbl_batch, y_batch = batch
                    logits = self.model_(X_batch, lbl_batch)
                else:
                    X_batch, y_batch = batch
                    logits = self.model_(X_batch, None)
                optimizer.zero_grad()
                loss = criterion(logits, y_batch)
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
                    xv_t = torch.from_numpy(Xv).to(dev)
                    if self.label_emb_dim > 0:
                        n_val = len(val_pids)
                        lv = torch.from_numpy(self._label_ids_flat(n_val, n_labels)).to(dev)
                        v_logits = self.model_(xv_t, lv)
                    else:
                        v_logits = self.model_(xv_t, None)
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
            if self.label_emb_dim > 0:
                lid = torch.from_numpy(self._label_ids_flat(n_docs, n_labels)).to(dev)
                logits = self.model_(X_t, lid)
            else:
                logits = self.model_(X_t, None)
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
    meta_features: MetaFeatureMode = "default",
    min_positive: int = MIN_POSITIVE_DEFAULT,
    mlp_epochs: int = 50,
    mlp_lr: float = 1e-3,
    mlp_batch_size: int = 512,
    mlp_hidden_dims: Tuple[int, ...] = (64, 32),
    mlp_label_emb_dim: int = 0,
    logreg_c: float = 1.0,
    logreg_max_iter: int = 1000,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 6,
    hgb_max_iter: int = 100,
    hgb_max_depth: int = 4,
    mlp_early_stop_patience: int = 0,
    patient_cluster_k: int = 0,
) -> Union[PerLabelStackingEnsemble, PyTorchMLPStacker]:
    """Factory: return the right stacker class for ``name``.

    Parameters
    ----------
    name:
        One of ``LEARNER_NAMES``.
    device:
        PyTorch device spec (only used for ``pytorch_mlp``).
        ``"auto"`` picks CUDA → MPS → CPU automatically.
    meta_features:
        ``default`` | ``rich`` | ``full`` — passed to ``extract_label_features``.
    mlp_label_emb_dim:
        For ``pytorch_mlp`` only: embedding size for label indices (0 disables).
    logreg_c:
        Inverse L2 strength for ``logistic_regression`` (ignored for other sklearn names).
    mlp_epochs / mlp_lr / mlp_batch_size / mlp_hidden_dims:
        Training budget for ``pytorch_mlp``.
    logreg_max_iter / rf_* / hgb_*:
        Training budget for sklearn meta-learners (ignored for ``pytorch_mlp``).
    mlp_early_stop_patience:
        If >0, monitor val BCE each epoch (pass ``val_*`` into ``fit``) and stop when
        val loss does not improve for this many epochs; restore best weights.
    patient_cluster_k:
        If ≥2, fit ``KMeans`` on train rows of concatenated score matrices and append
        cluster one-hot to every meta-feature row (train-only centroids).
    """
    if name in _SKLEARN_NAMES:
        return PerLabelStackingEnsemble(
            meta_learner=name,
            seed=seed,
            include_aggregates=include_aggregates,
            meta_features=cast(MetaFeatureMode, meta_features),
            min_positive=min_positive,
            logreg_c=float(logreg_c),
            logreg_max_iter=logreg_max_iter,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            hgb_max_iter=hgb_max_iter,
            hgb_max_depth=hgb_max_depth,
            patient_cluster_k=int(patient_cluster_k),
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
            meta_features=cast(MetaFeatureMode, meta_features),
            label_emb_dim=int(mlp_label_emb_dim),
            patient_cluster_k=int(patient_cluster_k),
            mlp_early_stop_patience=int(mlp_early_stop_patience),
        )
    raise ValueError(f"Unknown stacker {name!r}. Choose from {LEARNER_NAMES}.")
