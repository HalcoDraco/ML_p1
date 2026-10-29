"""Render paper figures from archived, protocol-labelled measurements."""

import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlp1-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "mlp1-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
INK = "#24364B"
TEAL = "#167D8D"
ORANGE = "#B66A25"
PURPLE = "#706198"


def configure_style():
    """Set a compact, print-readable Seaborn theme with embedded PDF fonts."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="STIXGeneral",
        font_scale=1.0,
        rc={
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": "#E4E9EF",
            "grid.linewidth": 0.55,
            "axes.edgecolor": "#C8D1DB",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        },
    )


def plot_measurements(rows, filename, heading, colors, limits, ticks):
    """Draw point estimates without inventing unrecorded fold distributions."""
    positions = np.arange(len(rows))
    scores = np.array([row["f1"] for row in rows])
    labels = [row["label"] for row in rows]
    fig, ax = plt.subplots(figsize=(3.45, 1.98 if len(rows) == 4 else 2.20))
    fig.subplots_adjust(left=0.47, right=0.985, bottom=0.23, top=0.85)
    fig.text(0.025, 0.96, heading, fontsize=8, fontweight="bold", va="top")
    for y, score, color in zip(positions, scores, colors):
        ax.hlines(y, limits[0], score, color=color, linewidth=1.2, alpha=0.28)
    sns.scatterplot(
        x=scores,
        y=positions,
        hue=labels,
        palette=dict(zip(labels, colors)),
        s=42,
        linewidth=0.65,
        edgecolor="white",
        legend=False,
        ax=ax,
        zorder=3,
    )
    for y, score, color in zip(positions, scores, colors):
        ax.annotate(
            f"{score:.4f}",
            (score, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=color,
            annotation_clip=False,
        )
    ax.set(yticks=positions, yticklabels=labels, xlim=limits, xticks=ticks)
    ax.set_ylim(len(rows) - 0.55, -0.45)
    ax.set_xlabel("Positive-class F1", labelpad=6)
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", pad=7)
    sns.despine(ax=ax, left=True, bottom=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            ROOT / "figures" / f"{filename}.{suffix}",
            facecolor="white",
            metadata={"Creator": "Seaborn / Matplotlib"} if suffix == "pdf" else None,
        )
    plt.close(fig)


def write_table(rows):
    """Generate the manuscript table from the same source data as the figures."""
    lines = [
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Method & F1 (mean $\pm$ SD) & AUC\\",
        r"\midrule",
    ]
    for row in rows:
        value = f"{row['f1']:.4f} \\pm {row['std']:.4f}"
        if row["id"] == "logistic":
            value = f"\\mathbf{{{row['f1']:.4f}}} \\pm {row['std']:.4f}"
        label = row["label"]
        if row["id"] == "baseline":
            label = "Raw inputs / LS"
        lines.append(f"{label} & ${value}$ & {row['auc']:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT / "generated" / "baseline_table.tex").write_text("\n".join(lines) + "\n")


def main():
    """Build both vector figures, preview images, and numerical table inputs."""
    evidence = json.loads((ROOT / "data" / "results.json").read_text())
    for folder in ("figures", "generated"):
        (ROOT / folder).mkdir(exist_ok=True)
    configure_style()
    plot_measurements(
        evidence["encoding"]["rows"],
        "representation",
        "REPRESENTATION ABLATION  /  EXPLORATORY HOLDOUT",
        ["#65758A", "#496B8A", "#397F89", TEAL],
        (0.411, 0.451),
        [0.415, 0.425, 0.435, 0.445],
    )
    plot_measurements(
        evidence["advanced"]["rows"],
        "model_families",
        "MODEL CAPACITY  /  REPEATED FIVE-FOLD CV",
        [INK, ORANGE, PURPLE, "#8A739D", TEAL],
        (0.436, 0.453),
        [0.436, 0.441, 0.446, 0.451],
    )
    write_table(evidence["baseline"]["rows"])
    with open(ROOT / "generated" / "measurements.csv", "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["protocol", "experiment", "f1", "fold_std", "source"])
        for block in ("baseline", "encoding", "advanced"):
            for row in evidence[block]["rows"]:
                writer.writerow(
                    [
                        block,
                        row["id"],
                        row["f1"],
                        row.get("std", ""),
                        row.get("source", evidence[block].get("source", "")),
                    ]
                )
    print("Rendered two Seaborn figures (PDF and PNG) and the numerical table.")


if __name__ == "__main__":
    main()
