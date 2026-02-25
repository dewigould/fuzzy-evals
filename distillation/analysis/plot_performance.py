"""Performance comparison: base vs math_distill vs code_distill.

Three subplots: math performance, code performance, fuzzy performance.
Each has grouped bars with 95% clustered bootstrap CIs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["base", "math_distill", "code_distill"]
MODEL_LABELS = {"base": "Base", "math_distill": "Math Distill", "code_distill": "Code Distill"}
MODEL_COLORS = {"base": "#6baed6", "math_distill": "#fd8d3c", "code_distill": "#74c476"}

MATH_DATASETS = ["math_500", "aime", "omni_math"]
CODE_DATASETS = ["kodcode_500", "codeforces_500"]
FUZZY_DATASETS = ["fuzzy_philosophy", "fuzzy_weird_qs", "fuzzy_futuristic_tech"]

DATASET_LABELS = {
    "math_500": "MATH-500",
    "aime": "AIME",
    "omni_math": "Omni-MATH",
    "kodcode_500": "KodCode",
    "codeforces_500": "Codeforces",
    "fuzzy_philosophy": "Philosophy",
    "fuzzy_weird_qs": "Weird Qs",
    "fuzzy_futuristic_tech": "Future Tech",
}

N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


def load_results(model: str, dataset: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / model / f"results_{dataset}.parquet"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return None
    return pd.read_parquet(path)


def bootstrap_ci_accuracy(correct: np.ndarray, n_boot: int = N_BOOTSTRAP,
                          ci: float = 0.95) -> tuple[float, float, float]:
    """Clustered bootstrap CI for binary accuracy."""
    mean = correct.mean()
    boot_means = np.array([
        RNG.choice(correct, size=len(correct), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return mean, lo, hi


def bootstrap_ci_mean(scores: np.ndarray, n_boot: int = N_BOOTSTRAP,
                      ci: float = 0.95) -> tuple[float, float, float]:
    """Clustered bootstrap CI for continuous mean score."""
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    mean = scores.mean()
    boot_means = np.array([
        RNG.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return mean, lo, hi


def compute_stats(model: str, dataset: str) -> tuple[float, float, float] | None:
    """Return (mean, ci_lo, ci_hi) for a model/dataset pair."""
    df = load_results(model, dataset)
    if df is None:
        return None
    if "correct" in df.columns:
        arr = df["correct"].astype(float).values
        return bootstrap_ci_accuracy(arr)
    elif "passed" in df.columns:
        arr = df["passed"].astype(float).values
        return bootstrap_ci_accuracy(arr)
    elif "total_score" in df.columns:
        arr = df["total_score"].astype(float).values
        return bootstrap_ci_mean(arr)
    return None


def plot_group(ax, datasets, models, is_accuracy=True):
    """Plot grouped bars for a set of datasets on a single axes."""
    n_datasets = len(datasets)
    n_models = len(models)
    bar_width = 0.22
    x = np.arange(n_datasets)

    for j, model in enumerate(models):
        means, ci_los, ci_his = [], [], []
        for ds in datasets:
            stats = compute_stats(model, ds)
            if stats is None:
                means.append(0)
                ci_los.append(0)
                ci_his.append(0)
            else:
                m, lo, hi = stats
                if is_accuracy:
                    m, lo, hi = m * 100, lo * 100, hi * 100
                means.append(m)
                ci_los.append(lo)
                ci_his.append(hi)

        means = np.array(means)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)
        yerr = np.array([means - ci_los, ci_his - means])
        yerr = np.clip(yerr, 0, None)

        offset = (j - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=yerr,
                      label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                      capsize=3, edgecolor="white", linewidth=0.5,
                      error_kw={"linewidth": 1.2})

        # Value labels on bars
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets], fontsize=9)
    ax.set_xlim(-0.5, n_datasets - 0.5)
    if is_accuracy:
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("Mean Judge Score", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("Generating performance comparison plot...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    print("  Math performance...")
    plot_group(axes[0], MATH_DATASETS, MODELS, is_accuracy=True)
    axes[0].set_title("Math Performance", fontsize=12, fontweight="bold")

    print("  Code performance...")
    plot_group(axes[1], CODE_DATASETS, MODELS, is_accuracy=True)
    axes[1].set_title("Code Performance", fontsize=12, fontweight="bold")

    print("  Fuzzy performance...")
    plot_group(axes[2], FUZZY_DATASETS, MODELS, is_accuracy=False)
    axes[2].set_title("Fuzzy Eval Performance", fontsize=12, fontweight="bold")

    fig.suptitle("Distillation Evaluation: Base vs Math-Distilled vs Code-Distilled",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOT_DIR / "performance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
