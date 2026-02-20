"""Plot Sonnet distillation into Llama-3.1-8B-Instruct.

Same 3-row style as plot_base_vs_instruct.py (accuracy, response length,
format compliance) with 4 variants: baseline, math distill, code distill,
code distill + format SFT.

Usage:
    python plot_llama_distillation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

BENCHMARKS = ["MATH-500", "MBPP+", "HumanEval+"]
BENCH_FILES = ["math_500", "mbppplus", "humanevalplus"]

DIR_MAP = {
    "baseline":  "llama8b_base_full",
    "math":      "sonnet_math_llama_4k_final_full",
    "code":      "sonnet_code_llama_4k_step400_full",
}

GRAY = "#9ca3af"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#16a34a"

VARIANT_CONFIG = {
    "baseline":  {"color": GRAY,   "alpha": 0.7,  "label": "Baseline"},
    "math":      {"color": BLUE,   "alpha": 0.85, "label": "+ Sonnet Math (final)"},
    "code":      {"color": ORANGE, "alpha": 0.85, "label": "+ Sonnet Code (s400)"},
}
VARIANTS = list(VARIANT_CONFIG.keys())


def _load_parquet(results_name: str, bench_file: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / results_name / f"results_{bench_file}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_accuracy(results_name: str, bench_file: str) -> tuple[float | None, np.ndarray | None]:
    df = _load_parquet(results_name, bench_file)
    if df is None:
        return None, None
    col = "correct" if "correct" in df.columns else "passed"
    arr = df[col].values.astype(float)
    return arr.mean() * 100, arr


def load_response_lengths_split(results_name: str, bench_file: str) -> tuple[np.ndarray, np.ndarray] | None:
    df = _load_parquet(results_name, bench_file)
    if df is None:
        return None
    col = "correct" if "correct" in df.columns else "passed"
    lengths = df["raw_output"].str.len().fillna(0).values.astype(float)
    mask = df[col].values.astype(bool)
    return lengths[mask], lengths[~mask]


def load_format_compliance(results_name: str, bench_file: str) -> float | None:
    df = _load_parquet(results_name, bench_file)
    if df is None:
        return None
    if bench_file in ("math_500", "aime"):
        compliant = (df["predicted_answer"] != "no_boxed").sum()
    else:
        compliant = df["raw_output"].str.contains(r"```", regex=False, na=False).sum()
    return compliant / len(df) * 100


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10000, ci: float = 0.95,
                 scale: float = 100) -> tuple[float, float]:
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
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
            fc = load_format_compliance(results_name, bench_file)
            if mean is not None:
                lo, hi = bootstrap_ci(arr)
                acc_vals.append(round(mean, 1))
                lo_vals.append(round(lo, 1))
                hi_vals.append(round(hi, 1))
            else:
                acc_vals.append(None)
                lo_vals.append(None)
                hi_vals.append(None)
            fmt_vals.append(round(fc, 1) if fc is not None else None)
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
            result = load_response_lengths_split(results_name, bench_file)
            if result is None:
                entries.append(None)
                continue
            corr, incorr = result
            c_mean = corr.mean() / 1000 if len(corr) > 0 else 0
            c_ci = bootstrap_ci(corr, scale=1) if len(corr) > 1 else (c_mean * 1000, c_mean * 1000)
            c_ci = (c_ci[0] / 1000, c_ci[1] / 1000)
            i_mean = incorr.mean() / 1000 if len(incorr) > 0 else 0
            i_ci = bootstrap_ci(incorr, scale=1) if len(incorr) > 1 else (i_mean * 1000, i_mean * 1000)
            i_ci = (i_ci[0] / 1000, i_ci[1] / 1000)
            entries.append((c_mean, c_ci, i_mean, i_ci))
        data[variant] = entries
    return data


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_accuracy(ax, data, ci_lo, ci_hi):
    n_bench = len(BENCHMARKS)
    n_vars = len(VARIANTS)
    width = 0.18
    x = np.arange(n_bench)
    offsets = [width * (i - (n_vars - 1) / 2) for i in range(n_vars)]

    for v, off in zip(VARIANTS, offsets):
        cfg = VARIANT_CONFIG[v]
        vals = [d if d is not None else 0 for d in data[v]]
        lo = [d if d is not None else 0 for d in ci_lo[v]]
        hi = [d if d is not None else 0 for d in ci_hi[v]]
        mask = [d is not None for d in data[v]]
        vals, lo, hi = np.array(vals), np.array(lo), np.array(hi)
        # Only plot bars where data exists
        idx = np.where(mask)[0]
        if len(idx) > 0:
            ax.bar(x[idx] + off, vals[idx], width, color=cfg["color"], alpha=cfg["alpha"],
                   label=cfg["label"], zorder=3,
                   yerr=[vals[idx] - lo[idx], hi[idx] - vals[idx]], error_kw=dict(
                       capsize=2.5, capthick=0.8, elinewidth=0.8, ecolor="#444444", zorder=4))

    for v, off in zip(VARIANTS, offsets):
        for i in range(n_bench):
            val = data[v][i]
            if val is None:
                continue
            top = ci_hi[v][i]
            pos = x[i] + off
            label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
            ax.text(pos, top + 1.2, label,
                    ha="center", va="bottom", fontsize=6, fontweight="bold")
            if v != "baseline" and data["baseline"][i] is not None:
                delta = val - data["baseline"][i]
                sign = "+" if delta >= 0 else ""
                color = "#16a34a" if delta >= 0 else "#dc2626"
                ax.text(pos, val - 2,
                        f"{sign}{delta:.1f}", ha="center", va="top",
                        fontsize=5, color=color, fontweight="bold")

    ax.set_title("Llama-3.1-8B-Instruct", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=9, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")


def plot_response_length(ax, length_data):
    n_bench = len(BENCHMARKS)
    n_vars = len(VARIANTS)
    full_width = 0.18
    half = full_width / 2
    x = np.arange(n_bench)
    offsets = [full_width * (i - (n_vars - 1) / 2) for i in range(n_vars)]

    for v, off in zip(VARIANTS, offsets):
        cfg = VARIANT_CONFIG[v]
        for i in range(n_bench):
            entry = length_data[v][i]
            if entry is None:
                continue
            c_mean, c_ci, i_mean, i_ci = entry
            pos = x[i] + off
            ax.bar(pos - half / 2, c_mean, half, color=cfg["color"], alpha=0.7, zorder=3,
                   yerr=[[c_mean - c_ci[0]], [c_ci[1] - c_mean]],
                   error_kw=dict(capsize=1.5, capthick=0.6, elinewidth=0.6, ecolor="#444444", zorder=4))
            ax.bar(pos + half / 2, i_mean, half, color=cfg["color"], alpha=0.4,
                   hatch="//", edgecolor=cfg["color"], linewidth=0.5, zorder=3,
                   yerr=[[i_mean - i_ci[0]], [i_ci[1] - i_mean]],
                   error_kw=dict(capsize=1.5, capthick=0.6, elinewidth=0.6, ecolor="#444444", zorder=4))

    for v, off in zip(VARIANTS, offsets):
        for i in range(n_bench):
            entry = length_data[v][i]
            if entry is None:
                continue
            c_mean, c_ci, i_mean, i_ci = entry
            pos = x[i] + off
            top = max(c_ci[1], i_ci[1])
            ax.text(pos, top + 0.2, f"{c_mean:.1f}/{i_mean:.1f}",
                    ha="center", va="bottom", fontsize=4, fontweight="bold")

    ax.bar([], [], 0, color=GRAY, alpha=0.7, label="Correct")
    ax.bar([], [], 0, color=GRAY, alpha=0.4, hatch="//", edgecolor=GRAY, label="Incorrect")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=8)
    ax.set_ylabel("Mean response length (k chars)", fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2, axis="y")


def plot_format_compliance(ax, fmt_data):
    n_bench = len(BENCHMARKS)
    n_vars = len(VARIANTS)
    width = 0.18
    x = np.arange(n_bench)
    offsets = [width * (i - (n_vars - 1) / 2) for i in range(n_vars)]

    for v, off in zip(VARIANTS, offsets):
        cfg = VARIANT_CONFIG[v]
        vals = [d if d is not None else 0 for d in fmt_data[v]]
        mask = [d is not None for d in fmt_data[v]]
        idx = np.where(mask)[0]
        if len(idx) > 0:
            ax.bar(x[idx] + off, np.array(vals)[idx], width, color=cfg["color"], alpha=0.6, zorder=3)

    for v, off in zip(VARIANTS, offsets):
        for i in range(n_bench):
            val = fmt_data[v][i]
            if val is None:
                continue
            pos = x[i] + off
            if val < 98:
                label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
                color = "#dc2626" if val < 90 else "#b45309"
                ax.text(pos, val + 1.2, label,
                        ha="center", va="bottom", fontsize=6, fontweight="bold",
                        color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=8)
    ax.set_ylabel("Format compliance (%)", fontsize=9)
    ax.set_ylim(30, 105)
    ax.set_yticks([40, 60, 80, 100])
    ax.axhline(y=100, color="#16a34a", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(True, alpha=0.2, axis="y")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Llama data...")
    acc, ci_lo, ci_hi, fmt = build_data(DIR_MAP)
    length_data = build_length_data(DIR_MAP)

    for v in acc:
        corr = [f"{e[0]:.1f}" if e else "?" for e in length_data[v]]
        incorr = [f"{e[2]:.1f}" if e else "?" for e in length_data[v]]
        print(f"  {v}: acc={acc[v]}, len_correct={corr}, len_incorrect={incorr}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 14),
                              height_ratios=[3, 1.5, 1.2],
                              gridspec_kw={"hspace": 0.28})
    fig.suptitle("Sonnet 4.5 Distillation \u2014 Llama-3.1-8B-Instruct (Full Eval)",
                 fontsize=15, fontweight="bold", y=0.98)

    plot_accuracy(axes[0], acc, ci_lo, ci_hi)
    plot_response_length(axes[1], length_data)
    plot_format_compliance(axes[2], fmt)

    out_path = PLOTS_DIR / "llama_distillation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
