"""Affine mean-contrast projection and minimum-MSE LEACE in float64."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits


class UnidentifiableErasure(ValueError):
    """The training data contain no estimable confounder contrast."""


@dataclass(frozen=True)
class Numerics:
    covariance_rtol: float = 1e-10
    rank_rtol: float = 1e-10
    rank_atol: float = 1e-12
    invariant_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) or not 0 < value < 1 for value in vars(self).values()):
            raise ValueError("erasure tolerances must be finite and between zero and one")


def feature_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.isfinite(matrix).all():
        raise ValueError("erasure requires a nonempty finite feature matrix")
    return matrix


@dataclass(frozen=True)
class AffineRemoval:
    mean: np.ndarray
    left: np.ndarray
    right: np.ndarray
    method: str
    tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if (
            self.mean.ndim != 1 or self.left.ndim != 2
            or self.left.shape != self.right.shape
            or self.left.shape[0] != len(self.mean)
            or not 0 < self.left.shape[1] <= len(self.mean)
            or not all(np.isfinite(value).all() for value in (self.mean, self.left, self.right))
        ):
            raise ValueError("invalid low-rank affine removal factors")
        error = np.linalg.norm(self.right.T @ self.left - np.eye(self.removed_rank))
        if error > self.tolerance * max(1, self.removed_rank):
            raise ValueError("removal is not an idempotent true-rank eraser")

    @property
    def removed_rank(self) -> int:
        return self.left.shape[1]

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = feature_matrix(features)
        if values.shape[1] != len(self.mean):
            raise ValueError("transform feature dimension differs from fitted eraser")
        transformed = values - ((values - self.mean) @ self.right) @ self.left.T
        if not np.isfinite(transformed).all():
            raise ValueError("erasure produced nonfinite features")
        return transformed

    def squared_operator_norm(self) -> float:
        return float(np.sum((self.left.T @ self.left) * (self.right.T @ self.right)))

    def relative_operator_difference(self, other: AffineRemoval) -> float:
        cross = float(np.sum((self.left.T @ other.left) * (self.right.T @ other.right)))
        norm_squared = self.squared_operator_norm()
        difference = norm_squared + other.squared_operator_norm() - 2 * cross
        if difference < -self.tolerance * max(1.0, norm_squared):
            raise ValueError("operator-distance calculation is numerically inconsistent")
        return float(np.sqrt(max(0.0, difference) / norm_squared))

    def perturbation(self, features: np.ndarray) -> dict[str, float]:
        values = feature_matrix(features)
        lengths = np.linalg.norm(self.transform(values) - values, axis=1)
        return {"rms": float(np.sqrt(np.mean(lengths ** 2))), "mean_l2": float(lengths.mean())}


class LinearEraser:
    def __init__(self, method: str, numerics: Numerics = Numerics()):
        if method not in {"mp", "leace"}:
            raise ValueError(f"unknown erasure method: {method}")
        self.method = method
        self.numerics = numerics
        self._transformation: AffineRemoval | None = None

    @property
    def transformation(self) -> AffineRemoval:
        if self._transformation is None:
            raise RuntimeError("fit the eraser before transforming or reading its rank")
        return self._transformation

    @property
    def removed_rank(self) -> int:
        return self.transformation.removed_rank

    def fit(self, features: np.ndarray, confounder: np.ndarray) -> LinearEraser:
        self._transformation = None
        values = feature_matrix(features)
        labels = np.asarray(confounder)
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("confounder labels must align with training features")
        if any(label is None or label == "" for label in labels.tolist()):
            raise ValueError("confounder labels must not be missing")
        if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
            raise ValueError("confounder labels must be finite")
        classes, indices = np.unique(labels, return_inverse=True)
        if len(classes) < 2:
            raise UnidentifiableErasure("one training label cannot identify a confounder contrast")
        mean = values.mean(axis=0)
        centered = values - mean
        one_hot = np.eye(len(classes))[indices]
        one_hot -= one_hot.mean(axis=0)
        with threadpool_limits(limits=1):
            covariance = centered.T @ centered / len(values)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            cutoff = max(eigenvalues[-1] * self.numerics.covariance_rtol, 0.0)
            if eigenvalues[0] < -max(cutoff, self.numerics.rank_atol):
                raise ValueError("training covariance has material negative eigenvalues")
            keep = eigenvalues > cutoff
            self.eigenvalues_ = eigenvalues[keep]
            self.eigenvectors_ = eigenvectors[:, keep]
            if len(self.eigenvalues_) == 0:
                raise UnidentifiableErasure("training features have no covariance support")
            cross = centered.T @ one_hot / len(values)
            roots = np.sqrt(self.eigenvalues_)
            concept = cross
            if self.method == "leace":
                concept = self.eigenvectors_ @ ((self.eigenvectors_.T @ cross) / roots[:, None])
            basis, singular_values, _ = np.linalg.svd(concept, full_matrices=False)
            rank = int(np.sum(singular_values > max(
                self.numerics.rank_atol, singular_values[0] * self.numerics.rank_rtol
            )))
            if rank == 0:
                raise UnidentifiableErasure("training cross-covariance identifies no concept direction")
            if rank > len(classes) - 1:
                raise ValueError("concept rank exceeds the centered label-space rank")
            basis = basis[:, :rank]
            if self.method == "mp":
                left = right = basis
            else:
                coordinates = self.eigenvectors_.T @ basis
                left = self.eigenvectors_ @ (roots[:, None] * coordinates)
                right = self.eigenvectors_ @ (coordinates / roots[:, None])
            transformation = AffineRemoval(mean, left, right, self.method, self.numerics.invariant_tolerance)
            relative_residual = float(
                np.linalg.norm(cross - left @ (right.T @ cross)) / np.linalg.norm(cross)
            )
            if relative_residual > self.numerics.invariant_tolerance:
                raise ValueError("eraser failed the training cross-covariance invariant")
        self._transformation = transformation
        self.covariance_ = covariance
        self.classes_ = classes
        self.training_cross_covariance_residual_ = relative_residual
        self.training_rms_ = transformation.perturbation(values)["rms"]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.transformation.transform(features)


class MeanProjection(LinearEraser):
    def __init__(self, numerics: Numerics = Numerics()):
        super().__init__("mp", numerics)


class LEACE(LinearEraser):
    def __init__(self, numerics: Numerics = Numerics()):
        super().__init__("leace", numerics)
