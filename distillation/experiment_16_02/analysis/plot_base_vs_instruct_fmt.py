"""Plot Sonnet distillation with format-SFT'd checkpoints.

Same layout as base_vs_instruct.py but uses the format-SFT'd checkpoints:
- Math-distilled + code formatting SFT
- Code-distilled + math formatting SFT

Top row: accuracy bars with delta annotations and 95% bootstrap CIs.
Bottom row: format compliance bars.

All values computed directly from parquet result files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

BENCHMARKS = ["MATH-500", "MBPP+"]
BENCH_FILES = ["math_500", "mbppplus"]

# ── Model configs ─────────────────────────────────────────────────────────────
BASE_DIRS = {
    "baseline": "qwen3_base",
    "math":     "sonnet_math_qwen_4k_step300_fmt_code",
    "code":     "sonnet_code_qwen_4k_step200_fmt_math",
}
INSTRUCT_DIRS = {
    "baseline": "qwen3_instruct_base",
    "math":     "sonnet_math_qwen_instruct_4k_step200_fmt_code",
    "code":     "sonnet_code_qwen_instruct_4k_final_fmt_math",
}
BASE_LABELS = {"math": "s300+fmt", "code": "s200+fmt"}
INSTRUCT_LABELS = {"math": "s200+fmt", "code": "final+fmt"}

GRAY = "#9ca3af"
BLUE = "#2563eb"
ORANGE = "#ea580c"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_accuracy(results_name: str, bench_file: str) -> tuple[float, np.ndarray]:
    """Load parquet, return (accuracy%, per-problem bool array)."""
    path = RESULTS_DIR / results_name / f"results_{bench_file}.parquet"
    df = pd.read_parquet(path)
    col = "correct" if "correct" in df.columns else "passed"
    arr = df[col].values.astype(float)
    return arr.mean() * 100, arr


def load_format_compliance(results_name: str, bench_file: str) -> float:
    """Return % of responses with correct output format."""
    path = RESULTS_DIR / results_name / f"results_{bench_file}.parquet"
    df = pd.read_parquet(path)
    if bench_file == "math_500":
        compliant = (df["predicted_answer"] != "no_boxed").sum()
    else:
        compliant = df["raw_output"].str.contains(r"```", regex=False, na=False).sum()
    return compliant / len(df) * 100


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    n = len(arr)
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(arr, n, replace=True).mean() for _ in range(n_boot)]) * 100
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def build_data(dir_map: dict) -> tuple[dict, dict, dict, dict]:
    acc = {}
    ci_lo = {}
    ci_hi = {}
    fmt = {}
    for variant, results_name in dir_map.items():
        acc_vals, lo_vals, hi_vals, fmt_vals = [], [], [], []
        for bench_file in BENCH_FILES:
            mean, arr = load_accuracy(results_name, bench_file)
            lo, hi = bootstrap_ci(arr)
            fc = load_format_compliance(results_name, bench_file)
            acc_vals.append(round(mean, 1))
            lo_vals.append(round(lo, 1))
            hi_vals.append(round(hi, 1))
            fmt_vals.append(round(fc, 1))
        acc[variant] = acc_vals
        ci_lo[variant] = lo_vals
        ci_hi[variant] = hi_vals
        fmt[variant] = fmt_vals
    return acc, ci_lo, ci_hi, fmt


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_accuracy(ax, data, ci_lo, ci_hi, title, ckpt_labels):
    x = np.arange(len(BENCHMARKS))
    width = 0.24

    variants = ["baseline", "math", "code"]
    colors = [GRAY, BLUE, ORANGE]
    alphas = [0.7, 0.85, 0.85]
    labels = ["Baseline", f"+ Sonnet Math ({ckpt_labels['math']})",
              f"+ Sonnet Code ({ckpt_labels['code']})"]
    offsets = [-width, 0, width]

    for v, color, alpha, label, off in zip(variants, colors, alphas, labels, offsets):
        vals = np.array(data[v])
        lo = np.array(ci_lo[v])
        hi = np.array(ci_hi[v])
        yerr_lo = vals - lo
        yerr_hi = hi - vals
        ax.bar(x + off, vals, width, color=color, alpha=alpha,
               label=label, zorder=3,
               yerr=[yerr_lo, yerr_hi], error_kw=dict(
                   capsize=3, capthick=1, elinewidth=1, ecolor="#444444", zorder=4))

    for v, off in zip(variants, offsets):
        for i in range(len(BENCHMARKS)):
            val = data[v][i]
            top = ci_hi[v][i]
            pos = x[i] + off
            label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
            ax.text(pos, top + 1.5, label,
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
            if v != "baseline":
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
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")


def plot_format_compliance(ax, fmt_data, ckpt_labels):
    x = np.arange(len(BENCHMARKS))
    width = 0.24

    ax.bar(x - width, fmt_data["baseline"], width, color=GRAY, alpha=0.5, zorder=3)
    ax.bar(x, fmt_data["math"], width, color=BLUE, alpha=0.6, zorder=3)
    ax.bar(x + width, fmt_data["code"], width, color=ORANGE, alpha=0.6, zorder=3)

    all_bars = list(zip(
        [x - width, x, x + width],
        [fmt_data["baseline"], fmt_data["math"], fmt_data["code"]],
    ))
    for positions, values in all_bars:
        for pos, val in zip(positions, values):
            if val < 98:
                label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
                color = "#dc2626" if val < 90 else "#b45309"
                ax.text(pos, val + 1.5, label,
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                        color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=9)
    ax.set_ylabel("Format compliance (%)", fontsize=9)
    ax.set_ylim(50, 105)
    ax.set_yticks([60, 80, 100])
    ax.axhline(y=100, color="#16a34a", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(True, alpha=0.2, axis="y")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Base data...")
    base_acc, base_ci_lo, base_ci_hi, base_fmt = build_data(BASE_DIRS)
    print("Loading Instruct data...")
    inst_acc, inst_ci_lo, inst_ci_hi, inst_fmt = build_data(INSTRUCT_DIRS)

    for name, acc in [("BASE", base_acc), ("INSTRUCT", inst_acc)]:
        print(f"\n{name}:")
        for v, vals in acc.items():
            print(f"  {v}: {vals}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                              height_ratios=[3, 1.2],
                              gridspec_kw={"hspace": 0.25})

    fig.suptitle("Sonnet 4.5 Distillation + Format SFT (1 epoch, 19 examples)",
                 fontsize=14, fontweight="bold", y=0.98)

    plot_accuracy(axes[0, 0], base_acc, base_ci_lo, base_ci_hi,
                  "Qwen3-30B-A3B-Base", BASE_LABELS)
    plot_accuracy(axes[0, 1], inst_acc, inst_ci_lo, inst_ci_hi,
                  "Qwen3-30B-A3B-Instruct", INSTRUCT_LABELS)

    plot_format_compliance(axes[1, 0], base_fmt, BASE_LABELS)
    plot_format_compliance(axes[1, 1], inst_fmt, INSTRUCT_LABELS)

    out_path = PLOTS_DIR / "base_vs_instruct_fmt.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
