"""Plot Sonnet distillation: Qwen Base vs Qwen Instruct across 4 benchmarks."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"

# ── Data (final checkpoint, step 500) ────────────────────────────────────────

BENCHMARKS = ["MATH-500", "MBPP+", "HumanEval+", "LiveCodeBench"]

# Qwen3-30B-A3B-Base
BASE = {
    "baseline": [80.6, 75.7, 78.7, 12],
    "math":     [86,   62,   74,   12],
    "code":     [79,   73,   84,   14],
}

# Qwen3-30B-A3B-Instruct-2507
INSTRUCT = {
    "baseline": [86.8, 77.5, 91,  27],
    "math":     [87.4, 65.9, 70,  23],
    "code":     [89.4, 80.2, 86,  24],
}

GRAY = "#9ca3af"
BLUE = "#2563eb"
ORANGE = "#ea580c"


def plot_model(ax, data, title):
    """Grouped bar chart: 4 benchmarks x 3 variants (baseline, math-dist, code-dist)."""
    x = np.arange(len(BENCHMARKS))
    width = 0.24

    bars_base = ax.bar(x - width, data["baseline"], width, color=GRAY, alpha=0.7,
                       label="Baseline", zorder=3)
    bars_math = ax.bar(x, data["math"], width, color=BLUE, alpha=0.85,
                       label="+ Sonnet Math", zorder=3)
    bars_code = ax.bar(x + width, data["code"], width, color=ORANGE, alpha=0.85,
                       label="+ Sonnet Code", zorder=3)

    # Value labels + delta annotations
    for bars, key in [(bars_base, "baseline"), (bars_math, "math"), (bars_code, "code")]:
        for i, bar in enumerate(bars):
            h = bar.get_height()
            # Value on top
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.0f}" if h == int(h) else f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            # Delta below value for non-baseline
            if key != "baseline":
                delta = data[key][i] - data["baseline"][i]
                sign = "+" if delta >= 0 else ""
                color = "#16a34a" if delta >= 0 else "#dc2626"
                ax.text(bar.get_x() + bar.get_width() / 2, h - 3.5,
                        f"{sign}{delta:.1f}", ha="center", va="top",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("Sonnet 4.5 Distillation \u2014 Base vs Instruct (Final Checkpoint)",
                 fontsize=15, fontweight="bold", y=1.02)

    plot_model(ax1, BASE, "Qwen3-30B-A3B-Base")
    plot_model(ax2, INSTRUCT, "Qwen3-30B-A3B-Instruct")

    plt.tight_layout()

    out_path = PLOTS_DIR / "base_vs_instruct.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
