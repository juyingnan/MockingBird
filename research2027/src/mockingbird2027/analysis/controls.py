"""True-rank random removals matched to training RMS, without outcome tuning."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .erasure import AffineRemoval, LinearEraser


class ControlMatchError(ValueError):
    """A predeclared random draw could not satisfy the matching contract."""


def control_seed(base_seed: int, condition_id: str, draw: int) -> int:
    identity = f"{base_seed}:{condition_id}:{draw}".encode("ascii")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")


def matched_random_removal(
    eraser: LinearEraser,
    training_features: np.ndarray,
    *,
    seed: int,
    settings: dict[str, Any],
) -> tuple[AffineRemoval, dict[str, Any]]:
    target = eraser.transformation
    values = eraser.eigenvalues_
    vectors = eraser.eigenvectors_
    rank = target.removed_rank
    if rank >= len(values):
        raise ControlMatchError("no distinct matched random subspace at full covariance-support rank")
    gaussian = np.random.default_rng(seed).normal(size=(len(values), rank))
    log_values = np.log(values)
    target_rms = eraser.training_rms_
    tolerance = settings["relative_rms_tolerance"]

    def candidate(power: float) -> tuple[np.ndarray, float]:
        exponents = power * log_values
        weights = np.exp(exponents - exponents.max())
        coordinates, _ = np.linalg.qr(weights[:, None] * gaussian, mode="reduced")
        rms = float(np.sqrt(np.sum(values[:, None] * coordinates ** 2)))
        return coordinates, rms

    low, high = settings["power_bounds"]
    _, low_rms = candidate(low)
    _, high_rms = candidate(high)
    if target_rms < low_rms * (1 - tolerance) or target_rms > high_rms * (1 + tolerance):
        raise ControlMatchError("target RMS lies outside the predeclared random-power bracket")
    for _ in range(settings["bisection_steps"]):
        power = (low + high) / 2
        coordinates, rms = candidate(power)
        if abs(rms / target_rms - 1) <= tolerance:
            break
        if rms < target_rms:
            low = power
        else:
            high = power
    else:
        raise ControlMatchError("random-power bisection did not reach the RMS tolerance")
    if eraser.method == "mp":
        left = right = vectors @ coordinates
    else:
        roots = np.sqrt(values)
        left = vectors @ (roots[:, None] * coordinates)
        right = vectors @ (coordinates / roots[:, None])
    removal = AffineRemoval(target.mean.copy(), left, right, eraser.method, target.tolerance)
    perturbation = removal.perturbation(training_features)
    relative_error = abs(perturbation["rms"] / target_rms - 1)
    difference = target.relative_operator_difference(removal)
    if removal.removed_rank != rank or relative_error > tolerance:
        raise ControlMatchError("random control failed measured true-rank/RMS matching")
    if difference <= settings["minimum_operator_difference"]:
        raise ControlMatchError("random control is indistinguishable from the targeted operator")
    return removal, {
        "seed": seed,
        "power": power,
        "removed_rank": rank,
        "train_rms": perturbation["rms"],
        "train_mean_l2": perturbation["mean_l2"],
        "relative_rms_error": relative_error,
        "relative_operator_difference": difference,
    }
