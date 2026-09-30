"""M3 figures generated exclusively from saved result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..analysis.m3 import ATTRIBUTES, LEXICAL, REFERENCE, read_csv


def generate_figures(output_dir: Path) -> None:
    probes = read_csv(output_dir / "probes.csv")
    generalization = read_csv(output_dir / "generalization.csv")
    layers = sorted({int(row["layer"]) for row in probes})
    attributes = list(ATTRIBUTES)
    lookup = {(row["attribute"], int(row["layer"])): row for row in probes}
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("macro_f1", "normalized_macro_f1"),
        ("Held-out macro-F1", "Gain above majority: (F1 - baseline F1) / (1 - baseline F1)"),
    ):
        matrix = np.array([
            [float(lookup[(attribute, layer)][metric]) for layer in layers]
            for attribute in attributes
        ])
        image = axis.imshow(matrix, aspect="auto", vmin=min(0.0, matrix.min()), vmax=1.0, cmap="viridis")
        axis.set_xticks(range(len(layers)), layers)
        axis.set_yticks(range(len(attributes)), attributes)
        axis.set_xlabel("WavLM hidden state (0 = initial state)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
        for row in range(len(attributes)):
            for column in range(len(layers)):
                value = matrix[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value < 0.55 else "black")
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"encoding_heatmap.{extension}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    labels = {
        REFERENCE: "Repetition reference",
        LEXICAL[0]: "Statement 1 to 2",
        LEXICAL[1]: "Statement 2 to 1",
        "speaker_shift": "Held-out speaker mean",
    }
    colors = dict(zip(labels, ("black", "#0072B2", "#E69F00", "#D55E00")))
    for name, label in labels.items():
        groups = [
            [row for row in generalization if row["split_name"] == name and int(row["layer"]) == layer]
            for layer in layers
        ]
        for axis, metric in zip(axes, ("macro_f1", "macro_f1_gap")):
            values = [[float(row[metric]) for row in group] for group in groups]
            axis.plot(layers, [np.mean(group) for group in values], label=label, color=colors[name], marker="o", markersize=3)
            if name == "speaker_shift":
                axis.fill_between(layers, [min(group) for group in values], [max(group) for group in values], color=colors[name], alpha=0.15, label="Speaker fold range (not CI)")
                for fold in sorted({int(row["fold"]) for row in groups[0]}):
                    axis.plot(
                        layers,
                        [float(next(row for row in group if int(row["fold"]) == fold)[metric]) for group in groups],
                        color=colors[name], linewidth=0.6, alpha=0.35,
                    )
            axis.set_xlabel("WavLM hidden state")
            axis.set_xticks(layers)
            axis.grid(alpha=0.2)
    axes[0].set(ylabel="Emotion macro-F1", ylim=(0, 1), title="Emotion generalization")
    axes[1].set(ylabel="Reference F1 minus shifted F1", title="Controlled-shift gaps")
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[0].legend(fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"emotion_generalization.{extension}", dpi=180)
    plt.close(figure)
