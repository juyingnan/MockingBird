"""Figures from saved validity-only metrics and bounds."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..analysis.validity_reporting import typed_metrics


def generate_validity_figures(output_dir: Path) -> None:
    metrics = typed_metrics(output_dir / "metrics.csv")
    arms = ["original", "leace", "pca64", "pca64_leace"]
    names = ["Original", "LEACE", "PCA-64", "PCA-64 + LEACE"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("macro_f1", "accuracy"), ("Statement macro-F1", "Statement accuracy")):
        for probe, offset, color in (("linear", -0.07, "#0072B2"), ("rbf_svm", 0.07, "#D55E00")):
            rows = [next(row for row in metrics if row["arm"] == arm and row["probe"] == probe and row["attribute"] == "statement") for arm in arms]
            points = np.array([row[metric] for row in rows])
            upper = np.array([row[f"{metric}_upper_95"] for row in rows])
            x = np.arange(len(arms)) + offset
            axis.plot(x, points, "o-", color=color, label=probe)
            axis.plot(x, upper, "_", color=color, markersize=12)
            axis.vlines(x, points, upper, color=color)
        axis.axhline(0.55, color="black", linestyle="--", label="Frozen ceiling 0.55")
        axis.axhline(0.5, color="gray", linestyle=":", label="Chance 0.50")
        axis.set_xticks(range(len(arms)), names, rotation=12)
        axis.set(ylim=(0, 1.03), ylabel=title, title=title)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Locked C evaluation; caps mark one-sided 95% actor-bootstrap upper bounds")
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"statement_validity.{extension}", dpi=180)
    plt.close(figure)

    attributes = ["emotion", "speaker", "statement", "gender", "intensity"]
    matrix = np.array([
        [next(row["macro_f1"] for row in metrics if row["arm"] == arm and row["probe"] == "linear" and row["attribute"] == attribute) for arm in arms]
        for attribute in attributes
    ])
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(arms)), names)
    axis.set_yticks(range(len(attributes)), attributes)
    for row in range(len(attributes)):
        for column in range(len(arms)):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.55 else "black")
    axis.set_title("Fresh linear-probe macro-F1 on C (not M3/M4 task-head performance)")
    figure.colorbar(image, ax=axis, label="Macro-F1")
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"attribute_information.{extension}", dpi=180)
    plt.close(figure)
