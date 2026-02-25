"""CoT length vs performance analysis for distilled models.

1. For math/code tasks: Do models reason longer on problems they get wrong?
2. For fuzzy tasks: Do longer reasoning traces get higher rubric scores?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["math_distill", "code_distill"]
MODEL_LABELS = {"math_distill": "Math Distill", "code_distill": "Code Distill"}
MODEL_COLORS = {"math_distill": "#fd8d3c", "code_distill": "#74c476"}

ACCURACY_DATASETS = ["math_500", "aime", "omni_math", "kodcode_500", "codeforces_500"]
FUZZY_DATASETS = ["fuzzy_philosophy", "fuzzy_weird_qs", "fuzzy_futuristic_tech"]

DATASET_LABELS = {
    "math_500": "MATH-500", "aime": "AIME", "omni_math": "Omni-MATH",
    "kodcode_500": "KodCode", "codeforces_500": "Codeforces",
    "fuzzy_philosophy": "Philosophy", "fuzzy_weird_qs": "Weird Qs",
    "fuzzy_futuristic_tech": "Future Tech",
}


def load_with_cot_len(model: str, dataset: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / model / f"results_{dataset}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["cot_len"] = df["cot"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    return df


def main():
    print("Generating CoT length vs performance plots...")

    # =========================================================================
    # Plot 1: CoT length for correct vs incorrect (accuracy-based datasets)
    # =========================================================================
    fig1, axes1 = plt.subplots(2, len(ACCURACY_DATASETS), figsize=(18, 8),
                                sharey="row")

    for col, ds in enumerate(ACCURACY_DATASETS):
        for row, model in enumerate(MODELS):
            ax = axes1[row, col]
            df = load_with_cot_len(model, ds)
            bool_col = "correct" if "correct" in (df.columns if df is not None else []) else "passed"
            if df is None or bool_col not in df.columns:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                       ha="center", va="center", fontsize=10, color="gray")
                ax.set_title(f"{DATASET_LABELS[ds]}", fontsize=9)
                continue

            correct_lens = df.loc[df[bool_col] == True, "cot_len"].values / 1000
            incorrect_lens = df.loc[df[bool_col] == False, "cot_len"].values / 1000

            parts = ax.violinplot(
                [correct_lens, incorrect_lens] if len(correct_lens) > 0 and len(incorrect_lens) > 0
                else [correct_lens] if len(correct_lens) > 0
                else [incorrect_lens],
                positions=list(range(1, (1 + (len(correct_lens) > 0) + (len(incorrect_lens) > 0)))),
                showmedians=True, showextrema=False,
            )

            colors = []
            if len(correct_lens) > 0:
                colors.append("#2ca02c")
            if len(incorrect_lens) > 0:
                colors.append("#d62728")
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

            labels = []
            if len(correct_lens) > 0:
                labels.append(f"Correct\n(n={len(correct_lens)})\nmed={np.median(correct_lens):.1f}k")
            if len(incorrect_lens) > 0:
                labels.append(f"Wrong\n(n={len(incorrect_lens)})\nmed={np.median(incorrect_lens):.1f}k")

            ax.set_xticks(list(range(1, len(labels) + 1)))
            ax.set_xticklabels(labels, fontsize=7)

            if row == 0:
                ax.set_title(f"{DATASET_LABELS[ds]}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nCoT Length (k chars)", fontsize=9)

            # Mann-Whitney U test if both groups exist
            if len(correct_lens) > 1 and len(incorrect_lens) > 1:
                try:
                    u_stat, p_val = sp_stats.mannwhitneyu(incorrect_lens, correct_lens,
                                                          alternative="greater")
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    ax.text(0.95, 0.95, f"p={p_val:.3f} {sig}",
                           transform=ax.transAxes, ha="right", va="top", fontsize=7,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
                except Exception:
                    pass

            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig1.suptitle("Do Distilled Models Reason Longer on Problems They Get Wrong?",
                  fontsize=13, fontweight="bold", y=1.02)
    fig1.tight_layout()
    out1 = PLOT_DIR / "cot_vs_accuracy.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"  Saved to {out1}")
    plt.close(fig1)

    # =========================================================================
    # Plot 2: CoT length vs fuzzy score (scatter + regression)
    # =========================================================================
    fig2, axes2 = plt.subplots(2, len(FUZZY_DATASETS), figsize=(15, 8))

    for col, ds in enumerate(FUZZY_DATASETS):
        for row, model in enumerate(MODELS):
            ax = axes2[row, col]
            df = load_with_cot_len(model, ds)
            if df is None or "total_score" not in df.columns:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                       ha="center", va="center", fontsize=10, color="gray")
                continue

            valid = df.dropna(subset=["total_score"])
            x = valid["cot_len"].values / 1000
            y = valid["total_score"].values

            ax.scatter(x, y, alpha=0.4, s=20, color=MODEL_COLORS[model], edgecolor="none")

            # Linear regression
            if len(x) > 2:
                slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax.text(0.05, 0.95,
                       f"r={r_val:.2f} {sig}\nslope={slope:.2f}",
                       transform=ax.transAxes, ha="left", va="top", fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

            if row == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nJudge Score", fontsize=9)
            ax.set_xlabel("CoT Length (k chars)", fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig2.suptitle("Does Longer Reasoning Lead to Higher Fuzzy Eval Scores?",
                  fontsize=13, fontweight="bold", y=1.02)
    fig2.tight_layout()
    out2 = PLOT_DIR / "cot_vs_fuzzy_score.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"  Saved to {out2}")
    plt.close(fig2)

    # =========================================================================
    # Print summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: CoT Length vs Performance")
    print("=" * 70)

    for model in MODELS:
        print(f"\n--- {MODEL_LABELS[model]} ---")
        for ds in ACCURACY_DATASETS:
            df = load_with_cot_len(model, ds)
            if df is None:
                continue
            bool_col = "correct" if "correct" in df.columns else "passed"
            if bool_col not in df.columns:
                continue
            correct_med = df.loc[df[bool_col] == True, "cot_len"].median()
            wrong_med = df.loc[df[bool_col] == False, "cot_len"].median()
            n_correct = (df[bool_col] == True).sum()
            n_wrong = (df[bool_col] == False).sum()
            print(f"  {DATASET_LABELS[ds]:15s}: correct(n={n_correct}) median={correct_med/1000:.1f}k | "
                  f"wrong(n={n_wrong}) median={wrong_med/1000:.1f}k | "
                  f"ratio={wrong_med/correct_med:.2f}x" if correct_med > 0 else
                  f"  {DATASET_LABELS[ds]:15s}: correct(n={n_correct}) | wrong(n={n_wrong})")

        for ds in FUZZY_DATASETS:
            df = load_with_cot_len(model, ds)
            if df is None or "total_score" not in df.columns:
                continue
            valid = df.dropna(subset=["total_score"])
            r, p = sp_stats.pearsonr(valid["cot_len"], valid["total_score"]) if len(valid) > 2 else (0, 1)
            print(f"  {DATASET_LABELS[ds]:15s}: r={r:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
