"""Generate R1 vs Sonnet comparison charts and tables for the report."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

OUTPUT_DIR = "/workspace/fuzzy-evals/distillation/experiment_16_02/analysis/plots"


def plot_bar_chart():
    """Chart 1: Best checkpoint accuracy — R1 vs Sonnet distilled models."""

    # Data: best checkpoint per config, selected by in-domain metric
    # R1 Math best by MATH-500: step200
    # R1 Code best by MBPP+: step400
    # Sonnet Math best by MATH-500: step300
    # Sonnet Code best by MBPP+: step200
    models = [
        ("Baseline\n(Qwen Base)", 82.6, 77.0, "#9e9e9e"),
        ("R1\nMath-dist", 72.0, 37.0, "#ef5350"),
        ("Sonnet\nMath-dist", 91.0, 73.0, "#42a5f5"),
        ("R1\nCode-dist", 75.0, 56.0, "#ef5350"),
        ("Sonnet\nCode-dist", 83.0, 74.0, "#42a5f5"),
    ]

    labels = [m[0] for m in models]
    math_acc = [m[1] for m in models]
    code_acc = [m[2] for m in models]
    colors = [m[3] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(labels))
    width = 0.35

    bars_math = ax.bar(x - width / 2, math_acc, width, label="MATH-500",
                       color=[c for c in colors], edgecolor="white", linewidth=1.5,
                       alpha=0.9)
    bars_code = ax.bar(x + width / 2, code_acc, width, label="MBPP+",
                       color=[c for c in colors], edgecolor="white", linewidth=1.5,
                       alpha=0.55, hatch="//")

    # Add value labels
    for bar in bars_math:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars_code:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    # Vertical separator between math-dist and code-dist groups
    ax.axvline(x=2.5, color="#bdbdbd", linewidth=1, linestyle="--", alpha=0.7)

    # Group labels
    ax.text(1.0, -0.18, "Math-distilled", ha="center", fontsize=11,
            fontstyle="italic", color="#555", transform=ax.get_xaxis_transform())
    ax.text(3.5, -0.18, "Code-distilled", ha="center", fontsize=11,
            fontstyle="italic", color="#555", transform=ax.get_xaxis_transform())

    # Legend
    import matplotlib.patches as mpatches
    solid_patch = mpatches.Patch(facecolor="#777", alpha=0.9, label="MATH-500")
    hatch_patch = mpatches.Patch(facecolor="#777", alpha=0.55, hatch="//",
                                  edgecolor="white", label="MBPP+")
    r1_patch = mpatches.Patch(facecolor="#ef5350", label="R1 (DeepSeek-R1)")
    sonnet_patch = mpatches.Patch(facecolor="#42a5f5", label="Sonnet 4.5")
    baseline_patch = mpatches.Patch(facecolor="#9e9e9e", label="Baseline")
    ax.legend(handles=[solid_patch, hatch_patch, mpatches.Patch(alpha=0), r1_patch, sonnet_patch, baseline_patch],
              loc="upper right", fontsize=9, ncol=2)

    ax.set_title("Best Checkpoint Accuracy: R1 vs Sonnet Distillation (Qwen Base)", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Baseline reference lines
    ax.axhline(y=82.6, color="#9e9e9e", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axhline(y=77.0, color="#9e9e9e", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(4.75, 83.5, "baseline MATH", fontsize=8, color="#999", ha="right")
    ax.text(4.75, 74.5, "baseline MBPP+", fontsize=8, color="#999", ha="right")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/r1_vs_sonnet_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_token_table():
    """Chart 2: Token count table for training traces."""

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis("off")

    col_labels = ["", "Median\ntokens", "Mean\ntokens", "P95\ntokens", "Under\n4096"]
    row_data = [
        ["R1 Math",       "4,754",   "5,821",  "14,071",   "42%"],
        ["R1 Code",       "4,244",   "4,966",  "11,151",   "48%"],
        ["Sonnet Math",   "915",     "927",    "1,570",    "99.8%"],
        ["Sonnet Code",   "654",     "712",    "1,318",    "99.8%"],
    ]

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#37474f")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)

    # Style rows — alternate shading and highlight R1 vs Sonnet
    for i in range(len(row_data)):
        row_idx = i + 1
        for j in range(len(col_labels)):
            cell = table[row_idx, j]
            if i < 2:  # R1 rows
                cell.set_facecolor("#ffebee" if i % 2 == 0 else "#fce4ec")
            else:  # Sonnet rows
                cell.set_facecolor("#e3f2fd" if i % 2 == 0 else "#e1f5fe")
            if j == 0:
                cell.set_text_props(fontweight="bold")
            # Highlight "Under 4096" column for R1 (poor filter pass rate)
            if j == 4 and i < 2:
                cell.set_text_props(color="#c62828", fontweight="bold")

    ax.set_title("Training Trace Token Lengths (Qwen Tokenizer)", fontsize=13,
                 fontweight="bold", pad=20)

    path = f"{OUTPUT_DIR}/trace_token_lengths.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_forgetting_table():
    """Chart 3: Table 4.1 — Cross-domain transfer deltas."""

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")

    col_labels = ["Configuration", "MATH-500", "AIME", "MBPP+", "HumanEval+", "LCB"]
    row_data = [
        ["Base + Math dist (s300)",       "+8.4",  "+5.5",  "-4.0",  "+0.4",  "-2.0"],
        ["Base + Code dist (s200)",       "+0.4",  "+10.0", "-3.0",  "-4.6",  "0.0"],
        ["Instruct + Math dist (s200)",   "+2.2",  "-6.7",  "-9.5",  "-23.0", "-10.0"],
        ["Instruct + Code dist (final)",  "+2.8",  "+2.2",  "+2.4",  "-5.0",  "-3.0"],
    ]

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#37474f")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        if j == 0:
            cell.set_width(0.32)

    # Color-code cells by value
    for i, row in enumerate(row_data):
        row_idx = i + 1
        for j, val in enumerate(row):
            cell = table[row_idx, j]
            if j == 0:
                cell.set_text_props(fontweight="bold", fontsize=10)
                cell.set_facecolor("#f5f5f5")
                cell.set_width(0.32)
                continue
            try:
                v = float(val)
                if v > 5:
                    cell.set_facecolor("#c8e6c9")  # strong green
                    cell.set_text_props(fontweight="bold", color="#2e7d32")
                elif v > 0:
                    cell.set_facecolor("#e8f5e9")  # light green
                    cell.set_text_props(color="#2e7d32")
                elif v == 0:
                    cell.set_facecolor("#fafafa")
                    cell.set_text_props(color="#757575")
                elif v > -5:
                    cell.set_facecolor("#ffebee")  # light red
                    cell.set_text_props(color="#c62828")
                else:
                    cell.set_facecolor("#ffcdd2")  # strong red
                    cell.set_text_props(fontweight="bold", color="#c62828")
            except ValueError:
                pass

    ax.set_title("Accuracy Delta vs Baseline (pp) — Sonnet Distillation", fontsize=13,
                 fontweight="bold", pad=20)

    path = f"{OUTPUT_DIR}/cross_domain_forgetting.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_bar_chart()
    plot_token_table()
    plot_forgetting_table()
    print("\nDone!")
