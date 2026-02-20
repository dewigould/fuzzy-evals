"""Plot histograms of training example response lengths.

Row 1: R1 traces (OpenR1-Math / OpenCodeReasoning)
Row 2: Sonnet 4.5 traces (generated reasoning traces)

Usage:
    python plot_training_lengths.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DISTILLATION_DIR = Path(__file__).parent.parent.parent

# R1 training data
R1_MATH_DATA = DISTILLATION_DIR / "filtered_data" / "clean.jsonl"
R1_CODE_DATA = DISTILLATION_DIR / "filtered_data_kodcode" / "clean.jsonl"

# Sonnet training data
SONNET_MATH_DATA = DISTILLATION_DIR / "generate_reasoning_traces" / "data" / "correct_only.jsonl"
SONNET_CODE_DATA = DISTILLATION_DIR / "generate_reasoning_traces" / "data_code" / "correct_only.jsonl"

PLOTS_DIR = Path(__file__).parent / "plots"

BLUE = "#2563eb"
ORANGE = "#ea580c"


def load_r1_lengths(path: Path) -> np.ndarray:
    lengths = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "messages" in d:
                resp = d["messages"][-1]["content"]
            else:
                resp = d["conversations"][-1]["value"]
            lengths.append(len(resp))
    return np.array(lengths)


def load_sonnet_lengths(path: Path) -> np.ndarray:
    lengths = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            lengths.append(len(d["generation"]))
    return np.array(lengths)


def plot_row(ax_math, ax_code, math_lens, code_lens, math_title, code_title, bins):
    math_k = math_lens / 1000
    code_k = code_lens / 1000

    ax_math.hist(math_k, bins=bins, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax_math.axvline(np.median(math_k), color="#1e40af", linestyle="--", linewidth=1.5,
                    label=f"Median: {np.median(math_k):.1f}k")
    ax_math.axvline(math_k.mean(), color="#1e40af", linestyle=":", linewidth=1.5,
                    label=f"Mean: {math_k.mean():.1f}k")
    ax_math.set_title(math_title, fontsize=11, fontweight="bold")
    ax_math.set_xlabel("Response length (k chars)", fontsize=9)
    ax_math.set_ylabel("Count", fontsize=9)
    ax_math.legend(fontsize=8)
    ax_math.grid(True, alpha=0.2, axis="y")

    ax_code.hist(code_k, bins=bins, color=ORANGE, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax_code.axvline(np.median(code_k), color="#9a3412", linestyle="--", linewidth=1.5,
                    label=f"Median: {np.median(code_k):.1f}k")
    ax_code.axvline(code_k.mean(), color="#9a3412", linestyle=":", linewidth=1.5,
                    label=f"Mean: {code_k.mean():.1f}k")
    ax_code.set_title(code_title, fontsize=11, fontweight="bold")
    ax_code.set_xlabel("Response length (k chars)", fontsize=9)
    ax_code.legend(fontsize=8)
    ax_code.grid(True, alpha=0.2, axis="y")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading R1 math data...")
    r1_math = load_r1_lengths(R1_MATH_DATA)
    print(f"  {len(r1_math):,} examples, mean={r1_math.mean():.0f}, median={np.median(r1_math):.0f}")

    print("Loading R1 code data...")
    r1_code = load_r1_lengths(R1_CODE_DATA)
    print(f"  {len(r1_code):,} examples, mean={r1_code.mean():.0f}, median={np.median(r1_code):.0f}")

    print("Loading Sonnet math data...")
    s_math = load_sonnet_lengths(SONNET_MATH_DATA)
    print(f"  {len(s_math):,} examples, mean={s_math.mean():.0f}, median={np.median(s_math):.0f}")

    print("Loading Sonnet code data...")
    s_code = load_sonnet_lengths(SONNET_CODE_DATA)
    print(f"  {len(s_code):,} examples, mean={s_code.mean():.0f}, median={np.median(s_code):.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                              sharex="col")
    fig.suptitle("Training Example Lengths (Response Only)", fontsize=14, fontweight="bold")

    bins = np.linspace(0, 80, 80)

    # Row 1: R1 traces
    plot_row(axes[0, 0], axes[0, 1], r1_math, r1_code,
             f"R1 Math (n={len(r1_math):,})",
             f"R1 Code (n={len(r1_code):,})",
             bins)

    # Row 2: Sonnet traces — same x-axis as row 1
    plot_row(axes[1, 0], axes[1, 1], s_math, s_code,
             f"Sonnet 4.5 Math (n={len(s_math):,})",
             f"Sonnet 4.5 Code (n={len(s_code):,})",
             bins)

    plt.tight_layout()
    out_path = PLOTS_DIR / "training_lengths.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
