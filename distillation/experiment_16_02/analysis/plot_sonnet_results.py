"""Plot Qwen3-30B-A3B distillation results — all 4 training configs."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"

# ── Data ─────────────────────────────────────────────────────────────────────

STEPS = [100, 200, 300, 400, 500]
STEP_LABELS = ["100\n(5K)", "200\n(10K)", "300\n(15K)", "400\n(20K)", "500\n(25K)"]

BASE = {"math": 80.6, "mbpp": 75.7, "heplus": 78.7}

DATA = {
    "r1_math":     {"math": [58, 72, 65, 65, 67], "mbpp": [10, 31, 41, 48, 44]},
    "r1_code":     {"math": [66, 73, 75, 74, 75], "mbpp": [15, 43, 48, 56, 49]},
    "sonnet_math": {"math": [85, 82, 88, 85, 86], "mbpp": [65, 62, 57, 63, 62],
                    "heplus": [80, 69, 67, 76, 74]},
    "sonnet_code": {"math": [82, 79, 77, 84, 79], "mbpp": [69, 74, 74, 71, 73],
                    "heplus": [82, 85, 85, 82, 84]},
}

BLUE = "#2563eb"
ORANGE = "#ea580c"
TEAL = "#0d9488"
GRAY = "#6b7280"
GREEN = "#16a34a"
RED = "#dc2626"


def plot_checkpoint_bars(ax, data_key, title):
    """Grouped bar chart: MATH-500, MBPP+, and optionally HumanEval+ at each checkpoint."""
    d = DATA[data_key]
    has_heplus = "heplus" in d
    x = np.arange(len(STEPS))

    if has_heplus:
        width = 0.25
        offsets = [-width, 0, width]
        series = [
            (d["math"], BLUE, "MATH-500"),
            (d["mbpp"], ORANGE, "MBPP+"),
            (d["heplus"], TEAL, "HumanEval+"),
        ]
        bases = [
            (BASE["math"], BLUE),
            (BASE["mbpp"], ORANGE),
            (BASE["heplus"], TEAL),
        ]
    else:
        width = 0.35
        offsets = [-width / 2, width / 2]
        series = [
            (d["math"], BLUE, "MATH-500"),
            (d["mbpp"], ORANGE, "MBPP+"),
        ]
        bases = [
            (BASE["math"], BLUE),
            (BASE["mbpp"], ORANGE),
        ]

    for offset, (values, color, label) in zip(offsets, series):
        bars = ax.bar(x + offset, values, width, color=color, alpha=0.85,
                      label=label, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7, color=color, fontweight="bold")

    for base_val, color in bases:
        ax.axhline(base_val, color=color, linestyle="--", alpha=0.4, linewidth=1.2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Step (samples)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(STEP_LABELS, fontsize=8)
    ax.set_ylim(0, 102)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25, axis="y")


def plot_delta_matrix(ax, teacher_label, math_data_key, code_data_key):
    """2x2 heatmap: rows = {math-trained, code-trained}, cols = {MATH-500, MBPP+}.

    Each cell = delta from base at the best in-domain checkpoint.
    Math-trained -> best MATH checkpoint; Code-trained -> best MBPP+ checkpoint.
    """
    md = DATA[math_data_key]
    cd = DATA[code_data_key]

    # Best in-domain checkpoint indices
    best_math_idx = int(np.argmax(md["math"]))   # best MATH for math-trained
    best_code_idx = int(np.argmax(cd["mbpp"]))    # best MBPP+ for code-trained

    # Build 2x2 delta matrix
    matrix = np.array([
        [md["math"][best_math_idx] - BASE["math"],
         md["mbpp"][best_math_idx] - BASE["mbpp"]],
        [cd["math"][best_code_idx] - BASE["math"],
         cd["mbpp"][best_code_idx] - BASE["mbpp"]],
    ])

    # Color: green for positive, red for negative, intensity by magnitude
    max_abs = max(abs(matrix.min()), abs(matrix.max()), 1)
    colors = np.zeros((*matrix.shape, 4))
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            intensity = min(abs(val) / max_abs, 1.0) * 0.55 + 0.1
            if val >= 0:
                colors[i, j] = (*matplotlib.colors.to_rgb(GREEN), intensity)
            else:
                colors[i, j] = (*matplotlib.colors.to_rgb(RED), intensity)

    # Draw cells
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, 1 - i), 1, 1, facecolor=colors[i, j],
                                 edgecolor="white", linewidth=3)
            ax.add_patch(rect)
            val = matrix[i, j]
            sign = "+" if val >= 0 else ""
            ax.text(j + 0.5, 1.5 - i, f"{sign}{val:.1f}pp",
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="black")
            # Subtext: absolute accuracy and step
            if i == 0:
                step = STEPS[best_math_idx]
                abs_val = md["math"][best_math_idx] if j == 0 else md["mbpp"][best_math_idx]
            else:
                step = STEPS[best_code_idx]
                abs_val = cd["math"][best_code_idx] if j == 0 else cd["mbpp"][best_code_idx]
            ax.text(j + 0.5, 1.22 - i, f"({abs_val}% @ s{step})",
                    ha="center", va="center", fontsize=9, color=GRAY)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["MATH-500", "MBPP+"], fontsize=11, fontweight="bold")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Code-trained", "Math-trained"], fontsize=11, fontweight="bold")
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    ax.set_title(f"{teacher_label} \u2014 Best Checkpoint \u0394 from Base",
                 fontsize=12, fontweight="bold", pad=12)
    ax.spines[:].set_visible(False)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Qwen3-30B-A3B \u2014 Distillation Results",
                 fontsize=16, fontweight="bold", y=0.98)

    # Layout: 3 rows
    # Row 0 (bar charts): R1 math, R1 code
    # Row 1 (bar charts): Sonnet math, Sonnet code  (with HumanEval+)
    # Row 2 (matrices):   R1 delta matrix, Sonnet delta matrix
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.25,
                          left=0.06, right=0.97, top=0.93, bottom=0.04,
                          height_ratios=[1, 1, 1.1])

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])

    # Row 0: R1 distilled
    plot_checkpoint_bars(ax00, "r1_math", "R1 Math-Distilled")
    plot_checkpoint_bars(ax01, "r1_code", "R1 Code-Distilled")

    # Row 1: Sonnet distilled (with HumanEval+)
    plot_checkpoint_bars(ax10, "sonnet_math", "Sonnet Math-Distilled")
    plot_checkpoint_bars(ax11, "sonnet_code", "Sonnet Code-Distilled")

    # Row 2: Delta matrices
    plot_delta_matrix(ax20, "R1 Traces", "r1_math", "r1_code")
    plot_delta_matrix(ax21, "Sonnet Traces", "sonnet_math", "sonnet_code")

    out_path = PLOTS_DIR / "qwen_distillation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
