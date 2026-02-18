"""Plot Sonnet 4.5 distillation results across all checkpoints and models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

# ── Accuracy data ────────────────────────────────────────────────────────────

STEPS = [100, 200, 300, 400, 500]
SAMPLES = [f"{s//1000}K" for s in [5000, 10000, 15000, 20000, 25000]]

DATA = {
    "sonnet_math_qwen": {"math": [85, 82, 88, 85, 86], "mbpp": [65, 62, 57, 63, 62]},
    "sonnet_math_llama": {"math": [45, 46, 44, 45, 50], "mbpp": [36, 47, 48, 49, 53]},
    "sonnet_code_qwen": {"math": [82, 79, 77, 84, 79], "mbpp": [69, 74, 74, 71, 73]},
    "sonnet_code_llama": {"math": [0, 0, 0, 1, 0], "mbpp": [57, 56, 55, 58, 57]},
}

BASE = {
    "qwen": {"math": 80.6, "mbpp": 75.7},
    "llama": {"math": 40.0, "mbpp": 46.0},
}

BLUE = "#2563eb"
ORANGE = "#ea580c"
BLUE_LIGHT = "#93bbfd"
ORANGE_LIGHT = "#fdba74"
GRAY = "#6b7280"


def plot_training_curve(ax, data_key, model_key, title):
    """Plot MATH-500 and MBPP+ accuracy vs training step."""
    d = DATA[data_key]
    base = BASE[model_key]

    ax.plot(STEPS, d["math"], "o-", color=BLUE, linewidth=2, markersize=6, label="MATH-500", zorder=3)
    ax.plot(STEPS, d["mbpp"], "s-", color=ORANGE, linewidth=2, markersize=6, label="MBPP+", zorder=3)

    ax.axhline(base["math"], color=BLUE, linestyle="--", alpha=0.5, linewidth=1.5, label=f"Base MATH ({base['math']}%)")
    ax.axhline(base["mbpp"], color=ORANGE, linestyle="--", alpha=0.5, linewidth=1.5, label=f"Base MBPP+ ({base['mbpp']}%)")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}\n({sa})" for s, sa in zip(STEPS, SAMPLES)], fontsize=9)
    ax.set_ylim(-2, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def compute_completion_rates():
    """Compute completion rates from parquet files for final checkpoints."""
    configs = [
        ("sonnet_math_qwen_4k_final", "Qwen\nMath"),
        ("sonnet_math_llama_4k_final", "Llama\nMath"),
        ("sonnet_code_qwen_4k_final", "Qwen\nCode"),
        ("sonnet_code_llama_4k_final", "Llama\nCode"),
    ]

    math_rates = []
    code_rates = []

    for dirname, label in configs:
        result_dir = RESULTS_DIR / dirname

        # Math completion: has think tags + boxed answer
        math_path = result_dir / "results_math_500.parquet"
        if math_path.exists():
            df = pd.read_parquet(math_path)
            has_think = df["cot"].notna()
            has_boxed = df["predicted_answer"] != "no_boxed"
            rate = (has_think & has_boxed).mean() * 100
            math_rates.append(rate)
        else:
            math_rates.append(0)

        # Code completion: has think tags + code block
        code_path = result_dir / "results_mbppplus.parquet"
        if code_path.exists():
            df = pd.read_parquet(code_path)
            has_think = df["cot"].notna()
            has_code = df["user_output"].str.contains("```", na=False)
            rate = (has_think & has_code).mean() * 100
            code_rates.append(rate)
        else:
            code_rates.append(0)

    return configs, math_rates, code_rates


def plot_best_checkpoint_summary(ax):
    """2x2 grouped bar chart: best in-domain checkpoint, showing in-domain + cross-domain."""
    # Best in-domain checkpoints:
    # Math-distilled: best MATH checkpoint
    # Code-distilled: best MBPP+ checkpoint
    groups = [
        ("Qwen\nMath-dist", 88, 57, "math"),    # step 300: best MATH
        ("Qwen\nCode-dist", 74, 77, "code"),     # step 200: best MBPP+  (math at that step = 79, but showing cross)
        ("Llama\nMath-dist", 50, 53, "math"),    # step 500: best MATH
        ("Llama\nCode-dist", 58, 1, "code"),     # step 400: best MBPP+  (math at that step = 1)
    ]

    labels = [g[0] for g in groups]
    in_domain = []
    cross_domain = []
    in_labels = []
    cross_labels = []

    for label, val1, val2, train_type in groups:
        if train_type == "math":
            in_domain.append(val1)
            cross_domain.append(val2)
            in_labels.append("MATH")
            cross_labels.append("MBPP+")
        else:
            in_domain.append(val1)
            cross_domain.append(val2)
            in_labels.append("MBPP+")
            cross_labels.append("MATH")

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, in_domain, width, color=[BLUE if t == "math" else ORANGE for _, _, _, t in groups],
                   label="In-domain", alpha=0.9, zorder=3)
    bars2 = ax.bar(x + width/2, cross_domain, width, color=[ORANGE if t == "math" else BLUE for _, _, _, t in groups],
                   label="Cross-domain", alpha=0.5, zorder=3)

    # Annotate bars
    for bar, val in zip(bars1, in_domain):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, cross_domain):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val}%",
                ha="center", va="bottom", fontsize=9)

    # Base model lines
    # Qwen base
    ax.plot([-0.5, 1.5], [80.6, 80.6], "--", color=BLUE, alpha=0.4, linewidth=1)
    ax.plot([-0.5, 1.5], [75.7, 75.7], "--", color=ORANGE, alpha=0.4, linewidth=1)
    # Llama base
    ax.plot([1.5, 3.5], [40, 40], "--", color=BLUE, alpha=0.4, linewidth=1)
    ax.plot([1.5, 3.5], [46, 46], "--", color=ORANGE, alpha=0.4, linewidth=1)

    ax.axvline(1.5, color=GRAY, linestyle=":", alpha=0.3)

    ax.set_title("Best Checkpoint: In-domain vs Cross-domain", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.3, axis="y")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE, alpha=0.9, label="MATH-500"),
        Patch(facecolor=ORANGE, alpha=0.9, label="MBPP+"),
        Patch(facecolor=GRAY, alpha=0.5, label="Cross-domain (lighter)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")


def plot_completion_rates(ax):
    """Grouped bar chart of completion rates for final checkpoints."""
    configs, math_rates, code_rates = compute_completion_rates()
    labels = [c[1] for c in configs]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, math_rates, width, color=BLUE, alpha=0.8, label="Math completion", zorder=3)
    bars2 = ax.bar(x + width/2, code_rates, width, color=ORANGE, alpha=0.8, label="Code completion", zorder=3)

    for bar, val in zip(bars1, math_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, code_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axvline(1.5, color=GRAY, linestyle=":", alpha=0.3)

    ax.set_title("Format Completion Rate (Final Checkpoint)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model × Training", fontsize=11)
    ax.set_ylabel("Completion Rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add subtitle explaining what completion means
    ax.text(0.5, -0.18, "Math: has <think>...</think> + \\boxed{}  |  Code: has <think>...</think> + ```python```",
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle("Sonnet 4.5 Distillation Results", fontsize=16, fontweight="bold", y=0.98)

    # Row 1: Math-distilled
    plot_training_curve(axes[0, 0], "sonnet_math_qwen", "qwen", "Qwen — Math-Distilled (Sonnet traces)")
    plot_training_curve(axes[0, 1], "sonnet_math_llama", "llama", "Llama — Math-Distilled (Sonnet traces)")

    # Row 2: Code-distilled
    plot_training_curve(axes[1, 0], "sonnet_code_qwen", "qwen", "Qwen — Code-Distilled (Sonnet traces)")
    plot_training_curve(axes[1, 1], "sonnet_code_llama", "llama", "Llama — Code-Distilled (Sonnet traces)")

    # Row 3: Summary + Completion
    plot_best_checkpoint_summary(axes[2, 0])
    plot_completion_rates(axes[2, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = PLOTS_DIR / "sonnet_distillation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
