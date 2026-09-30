"""M4 diagnostic figures derived only from persisted verified tables."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..analysis.m3 import ATTRIBUTES, read_csv


def generate_m4_figures(output_dir: Path) -> None:
    conditions = {row["condition_id"]: row for row in read_csv(output_dir / "conditions.csv")}
    validation = read_csv(output_dir / "targeted_validation.csv")
    fixed = {row["condition_id"]: row for row in read_csv(output_dir / "fixed_head_reliance.csv") if row["metric"] == "macro_f1"}
    controls = read_csv(output_dir / "random_controls.csv")
    collateral = read_csv(output_dir / "collateral_damage.csv")
    names = [row["condition_id"] for row in validation]
    labels = [
        f"{conditions[name]['target']} L{conditions[name]['layer']} {conditions[name]['method'].upper()}\n"
        f"{conditions[name]['split_name']} f{conditions[name]['fold']}"
        for name in names
    ]

    def save(figure, name):
        for extension in ("png", "pdf"):
            figure.savefig(output_dir / f"{name}.{extension}", dpi=180)
        plt.close(figure)

    width = max(9, len(names) * 2.4)
    figure, axis = plt.subplots(figsize=(width, 5), constrained_layout=True)
    x = np.arange(len(names))
    axis.bar(x - 0.18, [float(row["target_before_macro_f1"]) for row in validation], width=0.36, label="Before")
    axis.bar(x + 0.18, [float(row["target_after_macro_f1"]) for row in validation], width=0.36, label="After fresh held-out probe")
    axis.set_xticks(x, labels, fontsize=8)
    axis.set(ylabel="Target macro-F1", ylim=(0, 1.05), title="Target removal validation (not training-set covariance)")
    axis.legend()
    save(figure, "target_decodability")

    figure, axis = plt.subplots(figsize=(width, 5), constrained_layout=True)
    for index, name in enumerate(names):
        draws = [float(row["macro_f1_effect"]) for row in controls if row["condition_id"] == name and row["status"] == "matched"]
        if draws:
            axis.scatter(np.full(len(draws), index), draws, color="#0072B2", alpha=0.4, label="Matched random draws" if index == 0 else None)
        axis.scatter(index, float(fixed[name]["targeted_effect"]), marker="D", color="#D55E00", label="Targeted effect" if index == 0 else None)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(x, labels, fontsize=8)
    axis.set(ylabel="Fixed-head macro-F1 after minus before", title="Fixed-head sensitivity; invalid erasure is not reliance evidence")
    axis.legend()
    save(figure, "fixed_head_controls")

    figure, axis = plt.subplots(figsize=(width, 5), constrained_layout=True)
    lookup = {(row["condition_id"], row["attribute"]): float(row["delta_macro_f1"]) for row in collateral}
    matrix = np.array([[lookup[(name, attribute)] for name in names] for attribute in ATTRIBUTES])
    image = axis.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="RdBu")
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_yticks(range(len(ATTRIBUTES)), list(ATTRIBUTES))
    for row in range(len(ATTRIBUTES)):
        for column in range(len(names)):
            axis.text(column, row, f"{matrix[row, column]:+.3f}", ha="center", va="center", fontsize=9)
    axis.set_title("Collateral damage: fresh audit macro-F1 after minus before")
    figure.colorbar(image, ax=axis)
    save(figure, "collateral_damage")

    retrained = [row for row in read_csv(output_dir / "retrained_heads.csv") if row["status"] == "complete"]
    for name, metric, title in (
        ("retraining_performance", "performance", "M4B: separately retrained emotion heads"),
        ("ood_gap_change", "gap", "M4B: OOD-gap change (negative means smaller gap)"),
    ):
        figure, axis = plt.subplots(figsize=(max(9, len(retrained) * 1.5), 4.5), constrained_layout=True)
        if not retrained:
            axis.text(0.5, 0.5, "BLOCKED BY PREDECLARED VALIDITY / IDENTIFIABILITY GATES\nNo M4B scores; missing outcomes are not zero effects.", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
        else:
            positions = np.arange(len(retrained))
            if metric == "performance":
                axis.bar(positions - 0.18, [float(row["original_macro_f1"]) for row in retrained], width=0.36, label="Original")
                axis.bar(positions + 0.18, [float(row["erased_macro_f1"]) for row in retrained], width=0.36, label="Erased + retrained")
                axis.set_ylim(0, 1)
                axis.legend()
            else:
                axis.bar(positions, [float(row["delta_macro_f1_gap"]) for row in retrained])
                axis.axhline(0, color="black", linewidth=0.7)
            axis.set_xticks(positions, [row["condition_id"] for row in retrained], rotation=45, ha="right", fontsize=7)
        axis.set_title(title)
        save(figure, name)
