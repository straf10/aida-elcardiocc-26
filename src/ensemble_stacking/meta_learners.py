"""Per-label meta-classifiers for the stacking ensemble.

Three learner families are supported:

* ``logistic_regression`` — fast, interpretable, well-regularised; good first choice.
* ``random_forest``       — captures non-linear interactions between model scores.
* ``gradient_boosting``  — ``HistGradientBoostingClassifier``; strong but slower.

For each label ``j`` in ``all_labels``, one binary classifier is trained on the
(n_train × K+3) feature matrix produced by ``meta_features.extract_label_features``.
Labels with fewer than ``min_positive`` positive training examples fall back to a
raw max-score heuristic (no classifier is stored for them).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .meta_features import build_target_matrix, extract_label_features

LEARNER_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")
MIN_POSITIVE_DEFAULT = 3


def _make_learner(name: str, seed: int):
    """Instantiate a fresh sklearn classifier for the given learner name."""
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"Unknown meta-learner {name!r}. Choose from {LEARNER_NAMES}.")


class PerLabelStackingEnsemble:
    """Train one binary classifier per label using base-model scores as features.

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
    ) -> None:
        if meta_learner not in LEARNER_NAMES:
            raise ValueError(f"meta_learner must be one of {LEARNER_NAMES}, got {meta_learner!r}.")
        self.meta_learner = meta_learner
        self.seed = seed
        self.include_aggregates = include_aggregates
        self.min_positive = min_positive
        self.classifiers_: Dict[int, object] = {}

    def fit(
        self,
        train_matrices: List[np.ndarray],
        train_gt: Dict,
        train_pids: List[int],
        all_labels: List[str],
    ) -> "PerLabelStackingEnsemble":
        """Train one meta-classifier per label on the training split.

        Parameters
        ----------
        train_matrices:
            List of (n_train × n_labels) score matrices, one per base model.
        train_gt:
            ``Dict[patient_id, List[List[str]]]`` — training ground truth.
        train_pids:
            Ordered patient IDs matching ``train_matrices`` rows.
        all_labels:
            Full ordered label list (column order of matrices).
        """
        self.classifiers_ = {}
        Y = build_target_matrix(train_gt, train_pids, all_labels)
        n_labels = len(all_labels)
        trained = skipped = 0

        for j in range(n_labels):
            X_j = extract_label_features(
                train_matrices, j, include_aggregates=self.include_aggregates,
            )
            y_j = Y[:, j]
            n_pos = int(y_j.sum())
            if n_pos < self.min_positive:
                skipped += 1
                continue

            clf = _make_learner(self.meta_learner, self.seed + j)
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
        """Return an (n_docs × n_labels) stacking probability matrix.

        For labels with a trained classifier, the classifier's ``predict_proba``
        is used.  For labels without a classifier (rare / unseen in training),
        a heuristic fallback maps the max normalised score to [0, 1].
        """
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
                        proba[:, j] = self._fallback_col(matrices, j, n_docs)
            else:
                proba[:, j] = self._fallback_col(matrices, j, n_docs)

        return proba

    @staticmethod
    def _fallback_col(
        matrices: List[np.ndarray],
        j: int,
        n_docs: int,
    ) -> np.ndarray:
        """Max normalised score across models, clipped to [0, 1].

        A normalised score of 1.0 = threshold → maps to 0.5 probability.
        """
        raw = np.column_stack([mat[:, j] for mat in matrices])  # (n_docs, K)
        return np.clip(raw.max(axis=1) / 2.0, 0.0, 1.0).astype(np.float32)

    @property
    def n_trained(self) -> int:
        return len(self.classifiers_)
