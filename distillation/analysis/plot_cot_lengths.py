"""CoT length comparison for distilled models across all datasets."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["math_distill", "code_distill"]
MODEL_LABELS = {"math_distill": "Math Distill", "code_distill": "Code Distill"}
MODEL_COLORS = {"math_distill": "#fd8d3c", "code_distill": "#74c476"}

ALL_DATASETS = [
    "math_500", "aime", "omni_math",
    "kodcode_500", "codeforces_500",
    "fuzzy_philosophy", "fuzzy_weird_qs", "fuzzy_futuristic_tech",
]
DATASET_LABELS = {
    "math_500": "MATH-500", "aime": "AIME", "omni_math": "Omni-MATH",
    "kodcode_500": "KodCode", "codeforces_500": "Codeforces",
    "fuzzy_philosophy": "Philosophy", "fuzzy_weird_qs": "Weird Qs",
    "fuzzy_futuristic_tech": "Future Tech",
}


def get_cot_lengths(model: str, dataset: str) -> np.ndarray | None:
    path = RESULTS_DIR / model / f"results_{dataset}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "cot" not in df.columns:
        return None
    # CoT length in characters (None = no CoT = 0)
    lengths = df["cot"].apply(lambda x: len(x) if isinstance(x, str) else 0).values
    return lengths


def main():
    print("Generating CoT length comparison plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Subplot 1: Box plot of CoT lengths per model per dataset ---
    ax = axes[0]
    positions = []
    tick_positions = []
    tick_labels = []
    box_colors = []

    group_width = 2.5
    box_width = 0.8

    for i, ds in enumerate(ALL_DATASETS):
        center = i * group_width
        for j, model in enumerate(MODELS):
            lengths = get_cot_lengths(model, ds)
            if lengths is None:
                continue
            # Convert to thousands of characters for readability
            lengths_k = lengths / 1000
            pos = center + (j - 0.5) * 1.0
            bp = ax.boxplot([lengths_k], positions=[pos], widths=box_width,
                           patch_artist=True, showfliers=False,
                           medianprops={"color": "black", "linewidth": 1.5})
            bp["boxes"][0].set_facecolor(MODEL_COLORS[model])
            bp["boxes"][0].set_alpha(0.7)
            positions.append(pos)
            box_colors.append(MODEL_COLORS[model])

        tick_positions.append(center)
        tick_labels.append(DATASET_LABELS[ds])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("CoT Length (thousands of chars)", fontsize=10)
    ax.set_title("Chain-of-Thought Length Distribution by Dataset", fontsize=12, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=MODEL_COLORS[m], alpha=0.7, label=MODEL_LABELS[m])
                       for m in MODELS]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Subplot 2: Mean CoT length bar chart ---
    ax2 = axes[1]
    n_datasets = len(ALL_DATASETS)
    bar_width = 0.35
    x = np.arange(n_datasets)

    for j, model in enumerate(MODELS):
        means = []
        medians = []
        for ds in ALL_DATASETS:
            lengths = get_cot_lengths(model, ds)
            if lengths is None:
                means.append(0)
                medians.append(0)
            else:
                means.append(lengths.mean() / 1000)
                medians.append(np.median(lengths) / 1000)

        offset = (j - 0.5) * bar_width
        bars = ax2.bar(x + offset, means, bar_width,
                       label=f"{MODEL_LABELS[model]} (mean)",
                       color=MODEL_COLORS[model], alpha=0.8,
                       edgecolor="white", linewidth=0.5)
        # Add median markers
        ax2.scatter(x + offset, medians, color="black", s=20, zorder=5, marker="_",
                   linewidths=2)

        for bar, m in zip(bars, means):
            if m > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{m:.1f}k", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_LABELS[ds] for ds in ALL_DATASETS], fontsize=9)
    ax2.set_ylabel("CoT Length (thousands of chars)", fontsize=10)
    ax2.set_title("Mean CoT Length by Dataset (black markers = median)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    out = PLOT_DIR / "cot_lengths.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
