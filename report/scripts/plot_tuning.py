"""Draw the archived one-factor GBDT tuning results for the report."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlp1-mpl"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

# docs/part3_results.md, "GBDT tuning", round 1. Each change is separate
# from the default; values are best-stage five-fold mean validation log-loss.
RESULTS = [
    ("Default", 0.21474),
    ("Ordered code splits", 0.21403),
    ("Feature fraction 0.3", 0.21445),
    ("Leaf penalty 10", 0.21463),
]


def main():
    sns.set_theme(style="whitegrid", context="paper", font="STIXGeneral")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.26, top=0.96)
    for row, (label, loss) in enumerate(RESULTS):
        y = len(RESULTS) - row - 1
        color = "#65758A" if row == 0 else "#167D8D"
        ax.scatter(loss, y, s=31, color=color, zorder=3)
        ax.annotate(
            f"{loss:.5f}",
            (loss, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=color,
        )
    ax.axvline(RESULTS[0][1], color="#BDC8D3", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(RESULTS)), [row[0] for row in reversed(RESULTS)])
    ax.set_xlim(0.21388, 0.21498)
    ax.set_ylim(-0.35, len(RESULTS) - 0.65)
    ax.set_xticks([0.2140, 0.2144, 0.2148])
    ax.set_xlabel("Validation log-loss (lower is better)", labelpad=3)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color="#E4E9EF", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            ROOT / "figures" / f"gbdt_tuning_draft.{suffix}",
            dpi=300,
            facecolor="white",
            metadata={"Creator": "Matplotlib"} if suffix == "pdf" else None,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
