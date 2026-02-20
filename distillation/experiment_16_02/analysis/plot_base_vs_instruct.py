"""Plot Sonnet distillation: Qwen Base vs Qwen Instruct across 5 benchmarks.

Row 1: accuracy bars with delta annotations and 95% bootstrap CIs.
Row 2: mean response length (correct vs incorrect) with 95% bootstrap CIs.
Row 3: format compliance bars.

All values computed directly from parquet result files.

Usage:
    python plot_base_vs_instruct.py                  # best checkpoint per config
    python plot_base_vs_instruct.py --variant step200
    python plot_base_vs_instruct.py --variant final
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

BENCHMARKS = ["MATH-500", "AIME", "MBPP+", "HumanEval+", "LiveCodeBench"]
BENCH_FILES = ["math_500", "aime", "mbppplus", "humanevalplus", "livecodebench_v5"]

# ── Variant configs ───────────────────────────────────────────────────────────
VARIANTS = {
    "best": {
        "title": "Sonnet 4.5 Distillation \u2014 Best Checkpoint per Config",
        "output": "base_vs_instruct.png",
        "base_dirs": {
            "baseline": "qwen3_base",
            "math":     "sonnet_math_qwen_4k_step300",
            "code":     "sonnet_code_qwen_4k_step200",
        },
        "instruct_dirs": {
            "baseline": "qwen3_instruct_base",
            "math":     "sonnet_math_qwen_instruct_4k_step200",
            "code":     "sonnet_code_qwen_instruct_4k_final",
        },
        "base_labels": {"math": "s300", "code": "s200"},
        "instruct_labels": {"math": "s200", "code": "final"},
    },
    "step200": {
        "title": "Sonnet 4.5 Distillation \u2014 All Step 200 Checkpoints",
        "output": "base_vs_instruct_step200.png",
        "base_dirs": {
            "baseline": "qwen3_base",
            "math":     "sonnet_math_qwen_4k_step200",
            "code":     "sonnet_code_qwen_4k_step200",
        },
        "instruct_dirs": {
            "baseline": "qwen3_instruct_base",
            "math":     "sonnet_math_qwen_instruct_4k_step200",
            "code":     "sonnet_code_qwen_instruct_4k_step200",
        },
        "base_labels": {"math": "s200", "code": "s200"},
        "instruct_labels": {"math": "s200", "code": "s200"},
    },
    "final": {
        "title": "Sonnet 4.5 Distillation \u2014 All Final Checkpoints (500 steps)",
        "output": "base_vs_instruct_final.png",
        "base_dirs": {
            "baseline": "qwen3_base",
            "math":     "sonnet_math_qwen_4k_final",
            "code":     "sonnet_code_qwen_4k_final",
        },
        "instruct_dirs": {
            "baseline": "qwen3_instruct_base",
            "math":     "sonnet_math_qwen_instruct_4k_final",
            "code":     "sonnet_code_qwen_instruct_4k_final",
        },
        "base_labels": {"math": "final", "code": "final"},
        "instruct_labels": {"math": "final", "code": "final"},
    },
}

GRAY = "#9ca3af"
BLUE = "#2563eb"
ORANGE = "#ea580c"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_parquet(results_name: str, bench_file: str) -> pd.DataFrame:
    path = RESULTS_DIR / results_name / f"results_{bench_file}.parquet"
    return pd.read_parquet(path)


def load_accuracy(results_name: str, bench_file: str) -> tuple[float, np.ndarray]:
    df = _load_parquet(results_name, bench_file)
    col = "correct" if "correct" in df.columns else "passed"
    arr = df[col].values.astype(float)
    return arr.mean() * 100, arr


def load_response_lengths_split(results_name: str, bench_file: str) -> tuple[np.ndarray, np.ndarray]:
    df = _load_parquet(results_name, bench_file)
    col = "correct" if "correct" in df.columns else "passed"
    lengths = df["raw_output"].str.len().fillna(0).values.astype(float)
    mask = df[col].values.astype(bool)
    return lengths[mask], lengths[~mask]


def load_format_compliance(results_name: str, bench_file: str) -> float:
    df = _load_parquet(results_name, bench_file)
    if bench_file in ("math_500", "aime"):
        compliant = (df["predicted_answer"] != "no_boxed").sum()
    else:
        compliant = df["raw_output"].str.contains(r"```", regex=False, na=False).sum()
    return compliant / len(df) * 100


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10000, ci: float = 0.95,
                 scale: float = 100) -> tuple[float, float]:
    n = len(arr)
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(arr, n, replace=True).mean() for _ in range(n_boot)]) * scale
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def build_data(dir_map: dict) -> tuple[dict, dict, dict, dict]:
    acc, ci_lo, ci_hi, fmt = {}, {}, {}, {}
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


def build_length_data(dir_map: dict) -> dict:
    data = {}
    for variant, results_name in dir_map.items():
        entries = []
        for bench_file in BENCH_FILES:
            corr, incorr = load_response_lengths_split(results_name, bench_file)
            c_mean = corr.mean() / 1000 if len(corr) > 0 else 0
            c_ci = bootstrap_ci(corr, scale=1) if len(corr) > 1 else (corr.mean(), corr.mean())
            c_ci = (c_ci[0] / 1000, c_ci[1] / 1000)
            i_mean = incorr.mean() / 1000 if len(incorr) > 0 else 0
            i_ci = bootstrap_ci(incorr, scale=1) if len(incorr) > 1 else (i_mean * 1000, i_mean * 1000)
            i_ci = (i_ci[0] / 1000, i_ci[1] / 1000)
            entries.append((c_mean, c_ci, i_mean, i_ci))
        data[variant] = entries
    return data


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
        ax.bar(x + off, vals, width, color=color, alpha=alpha,
               label=label, zorder=3,
               yerr=[vals - lo, hi - vals], error_kw=dict(
                   capsize=3, capthick=1, elinewidth=1, ecolor="#444444", zorder=4))

    for v, off in zip(variants, offsets):
        for i in range(len(BENCHMARKS)):
            val = data[v][i]
            top = ci_hi[v][i]
            pos = x[i] + off
            label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
            ax.text(pos, top + 1.5, label,
                    ha="center", va="bottom", fontsize=6.5, fontweight="bold")
            if v != "baseline":
                delta = val - data["baseline"][i]
                sign = "+" if delta >= 0 else ""
                color = "#16a34a" if delta >= 0 else "#dc2626"
                ax.text(pos, val - 2.5,
                        f"{sign}{delta:.1f}", ha="center", va="top",
                        fontsize=5.5, color=color, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=9, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")


def plot_response_length(ax, length_data, ckpt_labels):
    x = np.arange(len(BENCHMARKS))
    full_width = 0.24
    half = full_width / 2

    variants = ["baseline", "math", "code"]
    colors = [GRAY, BLUE, ORANGE]
    offsets = [-full_width, 0, full_width]

    for v, color, off in zip(variants, colors, offsets):
        for i in range(len(BENCHMARKS)):
            c_mean, c_ci, i_mean, i_ci = length_data[v][i]
            pos = x[i] + off
            ax.bar(pos - half / 2, c_mean, half, color=color, alpha=0.7, zorder=3,
                   yerr=[[c_mean - c_ci[0]], [c_ci[1] - c_mean]],
                   error_kw=dict(capsize=2, capthick=0.8, elinewidth=0.8, ecolor="#444444", zorder=4))
            ax.bar(pos + half / 2, i_mean, half, color=color, alpha=0.4,
                   hatch="//", edgecolor=color, linewidth=0.5, zorder=3,
                   yerr=[[i_mean - i_ci[0]], [i_ci[1] - i_mean]],
                   error_kw=dict(capsize=2, capthick=0.8, elinewidth=0.8, ecolor="#444444", zorder=4))

    for v, off in zip(variants, offsets):
        for i in range(len(BENCHMARKS)):
            c_mean, c_ci, i_mean, i_ci = length_data[v][i]
            pos = x[i] + off
            top = max(c_ci[1], i_ci[1])
            ax.text(pos, top + 0.3, f"{c_mean:.1f}/{i_mean:.1f}",
                    ha="center", va="bottom", fontsize=4.5, fontweight="bold")

    ax.bar([], [], 0, color=GRAY, alpha=0.7, label="Correct")
    ax.bar([], [], 0, color=GRAY, alpha=0.4, hatch="//", edgecolor=GRAY, label="Incorrect")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=8)
    ax.set_ylabel("Mean response length (k chars)", fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2, axis="y")


def plot_format_compliance(ax, fmt_data, ckpt_labels):
    x = np.arange(len(BENCHMARKS))
    width = 0.24

    ax.bar(x - width, fmt_data["baseline"], width, color=GRAY, alpha=0.5, zorder=3)
    ax.bar(x, fmt_data["math"], width, color=BLUE, alpha=0.6, zorder=3)
    ax.bar(x + width, fmt_data["code"], width, color=ORANGE, alpha=0.6, zorder=3)

    for positions, values in zip(
        [x - width, x, x + width],
        [fmt_data["baseline"], fmt_data["math"], fmt_data["code"]],
    ):
        for pos, val in zip(positions, values):
            if val < 98:
                label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
                color = "#dc2626" if val < 90 else "#b45309"
                ax.text(pos, val + 1.5, label,
                        ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                        color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=8)
    ax.set_ylabel("Format compliance (%)", fontsize=9)
    ax.set_ylim(30, 105)
    ax.set_yticks([40, 60, 80, 100])
    ax.axhline(y=100, color="#16a34a", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(True, alpha=0.2, axis="y")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_plot(variant_name: str):
    cfg = VARIANTS[variant_name]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Base data ({variant_name})...")
    base_acc, base_ci_lo, base_ci_hi, base_fmt = build_data(cfg["base_dirs"])
    base_len = build_length_data(cfg["base_dirs"])
    print(f"Loading Instruct data ({variant_name})...")
    inst_acc, inst_ci_lo, inst_ci_hi, inst_fmt = build_data(cfg["instruct_dirs"])
    inst_len = build_length_data(cfg["instruct_dirs"])

    for name, acc, lens in [("BASE", base_acc, base_len), ("INSTRUCT", inst_acc, inst_len)]:
        print(f"\n{name}:")
        for v in acc:
            corr = [f"{e[0]:.1f}" for e in lens[v]]
            incorr = [f"{e[2]:.1f}" for e in lens[v]]
            print(f"  {v}: acc={acc[v]}, len_correct={corr}, len_incorrect={incorr}")

    fig, axes = plt.subplots(3, 2, figsize=(18, 13),
                              height_ratios=[3, 1.5, 1.2],
                              gridspec_kw={"hspace": 0.28})
    fig.suptitle(cfg["title"], fontsize=15, fontweight="bold", y=0.98)

    plot_accuracy(axes[0, 0], base_acc, base_ci_lo, base_ci_hi,
                  "Qwen3-30B-A3B-Base", cfg["base_labels"])
    plot_accuracy(axes[0, 1], inst_acc, inst_ci_lo, inst_ci_hi,
                  "Qwen3-30B-A3B-Instruct", cfg["instruct_labels"])

    plot_response_length(axes[1, 0], base_len, cfg["base_labels"])
    plot_response_length(axes[1, 1], inst_len, cfg["instruct_labels"])

    plot_format_compliance(axes[2, 0], base_fmt, cfg["base_labels"])
    plot_format_compliance(axes[2, 1], inst_fmt, cfg["instruct_labels"])

    out_path = PLOTS_DIR / cfg["output"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default=None,
                        help="Which variant to plot (default: all)")
    args = parser.parse_args()

    if args.variant:
        generate_plot(args.variant)
    else:
        for v in VARIANTS:
            generate_plot(v)


if __name__ == "__main__":
    main()
