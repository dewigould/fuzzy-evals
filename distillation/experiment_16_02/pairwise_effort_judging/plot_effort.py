"""Plot pairwise effort judging results.

Plot 1: Effort win-rate heatmap per benchmark x correctness bucket
Plot 2: Effort breakdown bar chart, faceted by model pair x correctness bucket

Usage:
    python plot_effort.py                  # Base model variant (default)
    python plot_effort.py --variant it     # Instruct model variant
    python plot_effort.py --variant all    # Both variants
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
PLOT_DIR = Path(__file__).parent.parent / "analysis" / "plots"

BENCHMARKS = ["MATH-500", "AIME", "MBPP+", "HumanEval+", "LiveCodeBench"]
BUCKETS = ["both_correct", "both_wrong"]
BUCKET_LABELS = {"both_correct": "both \u2713", "both_wrong": "both \u2717"}
MIN_N = 10  # Gray out cells with fewer samples

# Variant configs: (file_prefix, row_labels, plot_suffix, title_suffix)
VARIANTS = {
    "base": {
        "files": {
            "base_vs_math": "base_vs_math.parquet",
            "base_vs_code": "base_vs_code.parquet",
        },
        "rows": [
            ("Math-distilled vs Base", "base_vs_math", "B"),
            ("Code-distilled vs Base", "base_vs_code", "B"),
        ],
        "bar_pairs": [
            ("Math-distilled vs Base", "base_vs_math", "B", "Math-distilled", "Base"),
            ("Code-distilled vs Base", "base_vs_code", "B", "Code-distilled", "Base"),
        ],
        "plot_suffix": "",
        "title_suffix": " (Base Model)",
    },
    "it": {
        "files": {
            "base_vs_math": "it_base_vs_math.parquet",
            "base_vs_code": "it_base_vs_code.parquet",
        },
        "rows": [
            ("Math-distilled vs Instruct", "base_vs_math", "B"),
            ("Code-distilled vs Instruct", "base_vs_code", "B"),
        ],
        "bar_pairs": [
            ("Math-distilled vs Instruct", "base_vs_math", "B", "Math-distilled", "Instruct"),
            ("Code-distilled vs Instruct", "base_vs_code", "B", "Code-distilled", "Instruct"),
        ],
        "plot_suffix": "_it",
        "title_suffix": " (Instruct Model)",
    },
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_variant(variant_key):
    cfg = VARIANTS[variant_key]
    return {k: pd.read_parquet(OUTPUT_DIR / v) for k, v in cfg["files"].items()}


def compute_cell(df, benchmark, bucket, distilled_side):
    """Compute effort stats for a benchmark x bucket cell.

    distilled_side: "A" or "B" -- which side is the distilled model.
    Returns (distilled_pct, same_pct, base_pct, n)
    """
    mask = (df["benchmark"] == benchmark) & (df["correctness_bucket"] == bucket)
    sub = df[mask]
    n = len(sub)
    if n == 0:
        return None, None, None, 0

    a_ct = (sub["effort_judgment"] == "A").sum()
    b_ct = (sub["effort_judgment"] == "B").sum()
    tie_ct = (sub["effort_judgment"] == "Neither").sum()

    if distilled_side == "B":
        return b_ct / n * 100, tie_ct / n * 100, a_ct / n * 100, n
    else:
        return a_ct / n * 100, tie_ct / n * 100, b_ct / n * 100, n


# ── Plot 1: Heatmap ────────────────────────────────────────────────────────

def plot_heatmap(data, variant_key):
    """Effort win-rate heatmap: benchmark x bucket columns, model pair rows."""
    cfg = VARIANTS[variant_key]
    rows = cfg["rows"]

    # Build columns: benchmark x bucket
    col_defs = []
    for bench in BENCHMARKS:
        for bucket in BUCKETS:
            label = f"{bench}\n({BUCKET_LABELS[bucket]})"
            col_defs.append((label, bench, bucket))

    n_rows = len(rows)
    n_cols = len(col_defs)
    values = np.full((n_rows, n_cols), np.nan)
    counts = np.zeros((n_rows, n_cols), dtype=int)

    for i, (_, key, side) in enumerate(rows):
        df = data[key]
        for j, (_, bench, bucket) in enumerate(col_defs):
            pct, _, _, n = compute_cell(df, bench, bucket, side)
            if pct is not None:
                values[i, j] = pct
                counts[i, j] = n

    fig, ax = plt.subplots(figsize=(16, 4.5))

    # Diverging colormap centered at 50%
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    cmap = plt.cm.RdBu

    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            n = counts[i, j]
            if n == 0 or np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
            elif n < MIN_N:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True, facecolor="#e0e0e0",
                    edgecolor="white", linewidth=1.5, zorder=2))
                ax.text(j, i, f"n<{MIN_N}", ha="center", va="center",
                        fontsize=9, color="#888888", zorder=3)
            else:
                color = "white" if abs(v - 50) > 30 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
                ax.text(j, i + 0.28, f"(n={n})", ha="center", va="center",
                        fontsize=8, color=color, alpha=0.8)

    # Labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c[0] for c in col_defs], fontsize=8.5, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)

    # Vertical separators between benchmark groups (every 2 columns)
    for sep_x in [1.5, 3.5, 5.5, 7.5]:
        ax.axvline(sep_x, color="white", linewidth=2.5)

    # Thicker separator between math (cols 0-3) and code (cols 4-9)
    ax.axvline(3.5, color="#333333", linewidth=2.5, linestyle="-")

    # Domain group labels
    trans = ax.get_xaxis_transform()
    ax.text(1.5, 1.15, "Math benchmarks", transform=trans,
            fontsize=11, fontweight="bold", color="#555555",
            ha="center", va="bottom")
    ax.text(7, 1.15, "Code benchmarks", transform=trans,
            fontsize=11, fontweight="bold", color="#555555",
            ha="center", va="bottom")

    ax.set_title(f"Effort Win Rate: Does Distillation Make Models Try Harder?{cfg['title_suffix']}",
                 fontsize=14, fontweight="bold", pad=50)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% distilled model tries harder", fontsize=10)
    cbar.set_ticks([0, 25, 50, 75, 100])

    plt.tight_layout()

    out_path = PLOT_DIR / f"effort_heatmap{cfg['plot_suffix']}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Plot 2: Bar chart ──────────────────────────────────────────────────────

def plot_bar_chart(data, variant_key):
    """Effort breakdown bar chart: facet rows = model pair, facet cols = bucket."""
    cfg = VARIANTS[variant_key]
    pairs = cfg["bar_pairs"]

    fig, axes = plt.subplots(
        len(pairs), len(BUCKETS),
        figsize=(14, 3.5 * len(pairs)),
        sharey=True,
    )

    bar_width = 0.25
    x = np.arange(len(BENCHMARKS))
    colors_map = {"distilled": "#4393c3", "same": "#999999", "base": "#d6604d"}

    for row_idx, (row_title, key, side, dist_label, base_label) in enumerate(pairs):
        df = data[key]
        for col_idx, bucket in enumerate(BUCKETS):
            ax = axes[row_idx, col_idx]

            dist_vals, same_vals, base_vals, ns = [], [], [], []
            for bench in BENCHMARKS:
                d, s, b, n = compute_cell(df, bench, bucket, side)
                dist_vals.append(d or 0)
                same_vals.append(s or 0)
                base_vals.append(b or 0)
                ns.append(n)

            bars_d = ax.bar(x - bar_width, dist_vals, bar_width,
                            label=f"{dist_label} tries harder",
                            color=colors_map["distilled"], edgecolor="white", linewidth=0.5)
            bars_s = ax.bar(x, same_vals, bar_width,
                            label="Similar effort",
                            color=colors_map["same"], edgecolor="white", linewidth=0.5)
            bars_b = ax.bar(x + bar_width, base_vals, bar_width,
                            label=f"{base_label} tries harder",
                            color=colors_map["base"], edgecolor="white", linewidth=0.5)

            # Add n labels above tallest bar in each group
            for i, (bar, n) in enumerate(zip(bars_d, ns)):
                label_text = f"n={n}" if n >= MIN_N else f"n={n}*"
                label_color = "gray" if n >= MIN_N else "red"
                ax.text(bar.get_x() + bar.get_width() / 2, max(dist_vals[i], same_vals[i], base_vals[i]) + 2,
                        label_text, ha="center", va="bottom", fontsize=7.5, color=label_color)

            ax.set_ylim(0, 110)
            ax.set_xticks(x)
            ax.set_xticklabels(BENCHMARKS, fontsize=8.5, rotation=30, ha="right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            bucket_label = "Both Correct" if bucket == "both_correct" else "Both Wrong"
            if row_idx == 0:
                ax.set_title(bucket_label, fontsize=11, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n\nPercentage", fontsize=9)
            else:
                ax.set_ylabel("")

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    fig.suptitle(f"Effort Judgment Distribution by Benchmark and Correctness{cfg['title_suffix']}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = PLOT_DIR / f"effort_bar_chart{cfg['plot_suffix']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base", choices=["base", "it", "all"],
                        help="Which model variant to plot")
    args = parser.parse_args()

    variants_to_plot = ["base", "it"] if args.variant == "all" else [args.variant]

    for variant_key in variants_to_plot:
        print(f"\n=== Plotting {variant_key} variant ===")
        data = load_variant(variant_key)
        plot_heatmap(data, variant_key)
        plot_bar_chart(data, variant_key)


if __name__ == "__main__":
    main()
