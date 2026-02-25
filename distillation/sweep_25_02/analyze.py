"""Analysis and plotting for the 25_02 sweep.

Plot 1: Base model comparison (4 base models x 6 benchmarks)
Plot 2: Trace source comparison (per base model, grouped bars)
Plot 3: Training curves (accuracy at each checkpoint from mid-training evals)

Usage:
  python analyze.py                       # all plots
  python analyze.py --plot base           # base model comparison only
  python analyze.py --plot traces         # trace source comparison only
  python analyze.py --plot curves         # training curves only
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sweep_25_02.config import (
    BASE_MODELS, TRACE_SOURCES, TASKS,
    DEFAULT_MAX_LENGTH, DEFAULT_LORA_RANK, DEFAULT_LR,
    TRAINING_BASE, RESULTS_BASE,
    build_run_name, build_base_eval_name, all_run_combos,
)


RESULTS_DIR = RESULTS_BASE
PLOTS_DIR = Path(__file__).parent / "plots"

BENCHMARKS = ["MATH-500", "AIME", "OmniMath", "MBPP+", "HumanEval+", "LiveCodeBench"]
BENCH_FILES = ["math_500", "aime", "omni_math", "mbppplus", "humanevalplus", "livecodebench_v5"]

# Colors
GRAY = "#9ca3af"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#16a34a"
PURPLE = "#7c3aed"

BASE_COLORS = {
    "llama8b": BLUE,
    "llama70b": ORANGE,
    "qwen_base": GREEN,
    "qwen_it": PURPLE,
}

TRACE_COLORS = {
    "sonnet": BLUE,
    "qwen": ORANGE,
    "kimi": GREEN,
}


# ── Data loading (reused from experiment_16_02) ─────────────────────────────

def _load_parquet(results_name: str, bench_file: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / results_name / f"results_{bench_file}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_accuracy(results_name: str, bench_file: str) -> tuple[float, np.ndarray] | None:
    df = _load_parquet(results_name, bench_file)
    if df is None:
        return None
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
    if bench_file in ("math_500", "aime", "omni_math"):
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


# ── Plot 1: Base model comparison ───────────────────────────────────────────

def plot_base_model_comparison(max_length: int = DEFAULT_MAX_LENGTH):
    """4 base models x 6 benchmarks: accuracy bars with 95% CIs."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                              height_ratios=[3, 1.5, 1.2],
                              gridspec_kw={"hspace": 0.3})

    x = np.arange(len(BENCHMARKS))
    n_models = len(BASE_MODELS)
    width = 0.8 / n_models

    # Row 1: Accuracy
    ax = axes[0]
    for j, (short, full) in enumerate(BASE_MODELS.items()):
        results_name = f"{build_base_eval_name(short, max_length)}"
        accs, los, his = [], [], []
        for bench_file in BENCH_FILES:
            result = load_accuracy(results_name, bench_file)
            if result:
                mean, arr = result
                lo, hi = bootstrap_ci(arr)
                accs.append(mean)
                los.append(lo)
                his.append(hi)
            else:
                accs.append(0)
                los.append(0)
                his.append(0)
        accs = np.array(accs)
        los = np.array(los)
        his = np.array(his)
        offset = (j - n_models / 2 + 0.5) * width
        color = BASE_COLORS[short]
        ax.bar(x + offset, accs, width, color=color, alpha=0.8,
               label=short, zorder=3,
               yerr=[accs - los, his - accs],
               error_kw=dict(capsize=2, capthick=0.8, elinewidth=0.8, ecolor="#444", zorder=4))
        for i in range(len(BENCHMARKS)):
            if accs[i] > 0:
                ax.text(x[i] + offset, his[i] + 1, f"{accs[i]:.1f}",
                        ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    ax.set_title(f"Base Model Comparison — Accuracy (max_tokens={max_length})",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")

    # Row 2: Response lengths
    ax = axes[1]
    for j, (short, full) in enumerate(BASE_MODELS.items()):
        results_name = f"{build_base_eval_name(short, max_length)}"
        offset = (j - n_models / 2 + 0.5) * width
        color = BASE_COLORS[short]
        for i, bench_file in enumerate(BENCH_FILES):
            result = load_response_lengths_split(results_name, bench_file)
            if result is None:
                continue
            corr, incorr = result
            c_mean = corr.mean() / 1000 if len(corr) > 0 else 0
            i_mean = incorr.mean() / 1000 if len(incorr) > 0 else 0
            half = width / 2
            pos = x[i] + offset
            ax.bar(pos - half / 2, c_mean, half, color=color, alpha=0.7, zorder=3)
            ax.bar(pos + half / 2, i_mean, half, color=color, alpha=0.3,
                   hatch="//", edgecolor=color, linewidth=0.5, zorder=3)

    ax.bar([], [], 0, color=GRAY, alpha=0.7, label="Correct")
    ax.bar([], [], 0, color=GRAY, alpha=0.3, hatch="//", edgecolor=GRAY, label="Incorrect")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=9)
    ax.set_ylabel("Response length (k chars)", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 3: Format compliance
    ax = axes[2]
    for j, (short, full) in enumerate(BASE_MODELS.items()):
        results_name = f"{build_base_eval_name(short, max_length)}"
        fmts = []
        for bench_file in BENCH_FILES:
            fc = load_format_compliance(results_name, bench_file)
            fmts.append(fc if fc is not None else 0)
        offset = (j - n_models / 2 + 0.5) * width
        color = BASE_COLORS[short]
        ax.bar(x + offset, fmts, width, color=color, alpha=0.6, zorder=3)
        for i in range(len(BENCHMARKS)):
            if fmts[i] < 98:
                color_txt = "#dc2626" if fmts[i] < 90 else "#b45309"
                ax.text(x[i] + offset, fmts[i] + 1, f"{fmts[i]:.0f}",
                        ha="center", va="bottom", fontsize=5.5, fontweight="bold",
                        color=color_txt)

    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=9)
    ax.set_ylabel("Format compliance (%)", fontsize=9)
    ax.set_ylim(30, 105)
    ax.axhline(y=100, color="#16a34a", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.2, axis="y")

    out_path = PLOTS_DIR / "base_model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ── Plot 2: Trace source comparison ─────────────────────────────────────────

def plot_trace_source_comparison(max_length: int = DEFAULT_MAX_LENGTH,
                                  rank: int = DEFAULT_LORA_RANK,
                                  lr: float = DEFAULT_LR):
    """For each base model: baseline vs 3 trace sources, for math and code tasks."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for base_short in BASE_MODELS:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                  gridspec_kw={"hspace": 0.35})

        for task_idx, task in enumerate(["math", "code"]):
            ax = axes[task_idx]
            x = np.arange(len(BENCHMARKS))

            # Baseline (base model)
            base_results_name = f"{build_base_eval_name(base_short, max_length)}"
            variants = {"baseline": base_results_name}

            # Distilled variants (use best checkpoint — look for latest step)
            for traces in TRACE_SOURCES:
                run_name = build_run_name(base_short, task, traces, max_length, rank, lr)
                # Find the latest evaluated checkpoint
                best_step = _find_best_eval_step(run_name)
                if best_step is not None:
                    variants[f"{traces}"] = f"{run_name}_step{best_step}"
                else:
                    variants[f"{traces}"] = None

            n_variants = len(variants)
            width = 0.8 / n_variants
            colors_list = [GRAY] + [TRACE_COLORS.get(t, GRAY) for t in TRACE_SOURCES]

            for j, (var_name, results_name) in enumerate(variants.items()):
                if results_name is None:
                    continue
                accs = []
                for bench_file in BENCH_FILES:
                    result = load_accuracy(results_name, bench_file)
                    accs.append(result[0] if result else 0)

                offset = (j - n_variants / 2 + 0.5) * width
                color = colors_list[j]
                label = var_name if var_name == "baseline" else f"+ {task}_{var_name}"
                ax.bar(x + offset, accs, width, color=color, alpha=0.8,
                       label=label, zorder=3)
                for i in range(len(BENCHMARKS)):
                    if accs[i] > 0:
                        ax.text(x[i] + offset, accs[i] + 1, f"{accs[i]:.1f}",
                                ha="center", va="bottom", fontsize=5, fontweight="bold")

            ax.set_title(f"{base_short} — {task.upper()} distillation",
                         fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(BENCHMARKS, fontsize=9, fontweight="bold")
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_ylim(0, 108)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.25, axis="y")

        out_path = PLOTS_DIR / f"trace_comparison_{base_short}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()


def _find_best_eval_step(run_name: str) -> int | None:
    """Find the checkpoint step with best MATH-500 accuracy from mid-training evals."""
    best_step = None
    best_acc = -1

    # Scan for evaluated checkpoint directories
    for d in RESULTS_DIR.glob(f"{run_name}_step*"):
        # Extract step number
        parts = d.name.replace(run_name + "_step", "")
        try:
            step = int(parts)
        except ValueError:
            continue

        # Check for math_500 results
        result = load_accuracy(d.name, "math_500")
        if result and result[0] > best_acc:
            best_acc = result[0]
            best_step = step

    return best_step


# ── Plot 3: Training curves ─────────────────────────────────────────────────

def plot_training_curves(max_length: int = DEFAULT_MAX_LENGTH,
                          rank: int = DEFAULT_LORA_RANK,
                          lr: float = DEFAULT_LR):
    """Plot accuracy at each checkpoint from mid-training evals."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    combos = all_run_combos(max_length, rank, lr)

    for base_short in BASE_MODELS:
        base_combos = [c for c in combos if c["base_short"] == base_short]
        if not base_combos:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                  gridspec_kw={"hspace": 0.35, "wspace": 0.3})
        fig.suptitle(f"{base_short} — Training Curves",
                     fontsize=14, fontweight="bold")

        # 2x2 grid: (math task, code task) x (math_500 metric, mbppplus metric)
        for task_idx, task in enumerate(["math", "code"]):
            for metric_idx, (bench_name, bench_file) in enumerate([
                ("MATH-500", "math_500"), ("MBPP+", "mbppplus")
            ]):
                ax = axes[task_idx][metric_idx]

                for traces, color in TRACE_COLORS.items():
                    run_name = build_run_name(base_short, task, traces, max_length, rank, lr)

                    # Find all evaluated steps
                    steps = []
                    accs = []
                    if RESULTS_DIR.exists():
                        for d in sorted(RESULTS_DIR.glob(f"{run_name}_step*")):
                            parts = d.name.replace(run_name + "_step", "")
                            try:
                                step = int(parts)
                            except ValueError:
                                continue
                            result = load_accuracy(d.name, bench_file)
                            if result:
                                steps.append(step)
                                accs.append(result[0])

                    if steps:
                        ax.plot(steps, accs, "o-", color=color, label=traces,
                                markersize=4, linewidth=1.5)

                # Add baseline
                base_results = f"{build_base_eval_name(base_short, max_length)}"
                base_result = load_accuracy(base_results, bench_file)
                if base_result:
                    ax.axhline(y=base_result[0], color=GRAY, linestyle="--",
                               alpha=0.5, label="baseline")

                ax.set_title(f"{task} distill → {bench_name}", fontsize=10)
                ax.set_xlabel("Training step", fontsize=9)
                ax.set_ylabel("Accuracy (%)", fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.2)

        out_path = PLOTS_DIR / f"training_curves_{base_short}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate sweep 25_02 analysis plots")
    parser.add_argument("--plot", type=str, default=None,
                        choices=["base", "traces", "curves"],
                        help="Generate only this plot type (default: all)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot is None or args.plot == "base":
        print("\nPlot 1: Base model comparison")
        try:
            plot_base_model_comparison(args.max_length)
        except Exception as e:
            print(f"  Error: {e}")

    if args.plot is None or args.plot == "traces":
        print("\nPlot 2: Trace source comparison")
        try:
            plot_trace_source_comparison(args.max_length, args.lora_rank, args.lr)
        except Exception as e:
            print(f"  Error: {e}")

    if args.plot is None or args.plot == "curves":
        print("\nPlot 3: Training curves")
        try:
            plot_training_curves(args.max_length, args.lora_rank, args.lr)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
