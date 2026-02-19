"""Plot Sonnet distillation: Qwen Base vs Qwen Instruct across 4 benchmarks.

Uses best checkpoint for each config:
- Math-distilled: checkpoint with best MATH-500
- Code-distilled: checkpoint with best MBPP+
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"

# ── Data ─────────────────────────────────────────────────────────────────────
# Updated with two-stage MATH grading (grade_answer + math_verify + LLM judge)
# and fixed code extraction (prefer last def-containing block, skip empty blocks).

BENCHMARKS = ["MATH-500", "MBPP+", "HumanEval+", "LiveCodeBench"]

# Qwen3-30B-A3B-Base
# Baseline: MATH 82.6% (413/500), MBPP+ 77.0% (291/378), HE+ 88% (first-100), LCB 12%
# Math-dist best MATH: step 300 (91% MATH)
# Code-dist best MBPP+: step 200 (74% MBPP+)
BASE = {
    "baseline": [82.6, 77,   88,   12],
    "math":     [91,   73,   90,   10],   # step 300
    "code":     [83,   74,   85,   12],   # step 200
}

# Qwen3-30B-A3B-Instruct-2507
# Baseline: MATH 89.4% (447/500), MBPP+ 77.8% (294/378), HE+ 91%, LCB 27%
# Math-dist best MATH: step 200 (91.6% MATH)
# Code-dist best MBPP+: final (80.2% MBPP+)
INSTRUCT = {
    "baseline": [89.4, 77.8, 91,   27],
    "math":     [91.6, 68.3, 60,   16],   # step 200
    "code":     [92.2, 80.2, 86,   24],   # final
}

# Checkpoint labels for subtitle
BASE_LABELS = {"math": "s300", "code": "s200"}
INSTRUCT_LABELS = {"math": "s200", "code": "final"}

GRAY = "#9ca3af"
BLUE = "#2563eb"
ORANGE = "#ea580c"


def plot_model(ax, data, title, ckpt_labels):
    """Grouped bar chart: 4 benchmarks x 3 variants (baseline, math-dist, code-dist)."""
    x = np.arange(len(BENCHMARKS))
    width = 0.24

    ax.bar(x - width, data["baseline"], width, color=GRAY, alpha=0.7,
           label="Baseline", zorder=3)
    ax.bar(x, data["math"], width, color=BLUE, alpha=0.85,
           label=f"+ Sonnet Math ({ckpt_labels['math']})", zorder=3)
    ax.bar(x + width, data["code"], width, color=ORANGE, alpha=0.85,
           label=f"+ Sonnet Code ({ckpt_labels['code']})", zorder=3)

    # Value labels + delta annotations
    all_bars = list(zip(
        [x - width, x, x + width],
        [data["baseline"], data["math"], data["code"]],
        ["baseline", "math", "code"],
    ))
    for positions, values, key in all_bars:
        for i, (pos, val) in enumerate(zip(positions, values)):
            # Value on top
            label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
            ax.text(pos, val + 1.2, label,
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
            # Delta below value for non-baseline
            if key != "baseline":
                delta = val - data["baseline"][i]
                sign = "+" if delta >= 0 else ""
                color = "#16a34a" if delta >= 0 else "#dc2626"
                ax.text(pos, val - 3,
                        f"{sign}{delta:.1f}", ha="center", va="top",
                        fontsize=6.5, color=color, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("Sonnet 4.5 Distillation \u2014 Best Checkpoint per Config",
                 fontsize=15, fontweight="bold", y=1.02)

    plot_model(ax1, BASE, "Qwen3-30B-A3B-Base", BASE_LABELS)
    plot_model(ax2, INSTRUCT, "Qwen3-30B-A3B-Instruct", INSTRUCT_LABELS)

    plt.tight_layout()

    out_path = PLOTS_DIR / "base_vs_instruct.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
