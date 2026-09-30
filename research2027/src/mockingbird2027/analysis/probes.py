"""Training-only selection of standardized, regularized linear probes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ProbeConfig:
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    cv_folds: int = 3
    seed: int = 20260930
    max_iter: int = 2000
    tol: float = 1e-4

    def __post_init__(self) -> None:
        if not self.c_grid or any(not np.isfinite(c) or c <= 0 for c in self.c_grid):
            raise ValueError("regularization strengths must be finite and positive")
        if tuple(sorted(set(self.c_grid))) != tuple(self.c_grid):
            raise ValueError("C grid must be unique and increasing (ties prefer smaller C)")
        if self.cv_folds < 2 or self.max_iter < 1 or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("invalid cross-validation or optimizer configuration")


def inner_splits(
    labels: np.ndarray, config: ProbeConfig, groups: np.ndarray | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct validation folds entirely inside the caller's training set."""
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < config.cv_folds:
        raise ValueError("every training class needs at least cv_folds observations")
    features = np.zeros((len(labels), 1))
    if groups is None:
        splitter = StratifiedKFold(
            n_splits=config.cv_folds, shuffle=True, random_state=config.seed
        )
        folds = list(splitter.split(features, labels))
    else:
        if len(groups) != len(labels) or len(np.unique(groups)) < config.cv_folds:
            raise ValueError("insufficient or misaligned training speaker groups")
        splitter = StratifiedGroupKFold(
            n_splits=config.cv_folds, shuffle=True, random_state=config.seed
        )
        folds = list(splitter.split(features, labels, groups))
    for train, validation in folds:
        if set(train) & set(validation) or len(train) + len(validation) != len(labels):
            raise ValueError("inner folds must partition training observations")
        if not np.array_equal(np.unique(labels[train]), classes):
            raise ValueError("inner training fold is missing classes")
        if not np.array_equal(np.unique(labels[validation]), classes):
            raise ValueError("inner validation fold is missing classes")
        if groups is not None and set(groups[train]) & set(groups[validation]):
            raise ValueError("inner cross-validation leaks speakers")
    return folds


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    config: ProbeConfig,
    *,
    groups: np.ndarray | None = None,
) -> GridSearchCV:
    """Select C by macro-F1, then refit using only the outer training data."""
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all():
        raise ValueError("probe features must be finite and aligned with training labels")
    folds = inner_splits(labels, config, groups)
    pipeline = Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=config.max_iter,
                    tol=config.tol,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"classifier__C": list(config.c_grid)},
        scoring=make_scorer(
            f1_score, average="macro", labels=np.unique(labels), pos_label=None, zero_division=0
        ),
        cv=folds,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    # Keep the scaler inside CV and fail instead of accepting unconverged fits.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(np.asarray(features, dtype=np.float64), labels)
    return search


def classification_metrics(
    gold: np.ndarray, predicted: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    """Use the full training label space, including never-predicted classes."""
    if len(gold) == 0 or len(gold) != len(predicted):
        raise ValueError("metric inputs must be nonempty and aligned")
    if not set(gold).issubset(set(classes)) or not set(predicted).issubset(set(classes)):
        raise ValueError("metric labels are outside the training label space")
    return {
        "macro_f1": float(f1_score(gold, predicted, labels=classes, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
    }
