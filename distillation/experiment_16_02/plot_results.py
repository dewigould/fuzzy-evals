"""Plot accuracy trajectories for experiment 16/02.

Left panel:  Accuracy (all) vs training samples
Right panel: Accuracy (completed only) vs training samples, with completion rate annotations

Usage:
  python plot_results.py
  python plot_results.py --output plots/trajectory.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent))

from experiment_16_02.analyze_results import analyze_folder


# ── Data collection ──────────────────────────────────────────────────────────

# Define model runs: (label, color, linestyle, results_prefix, steps, samples_per_step)
MODELS = {
    "math_default": {
        "label": "Math 4K",
        "color": "#2563eb",  # blue
        "prefix": "math_default",
        "steps": [100, 200, 300, 400, 500],
        "samples_per_step": 50,  # batch_size * step = samples
    },
    "math_2k": {
        "label": "Math 2K",
        "color": "#7c3aed",  # purple
        "prefix": "math_2k",
        "steps": [50, 100, 150, 200],
        "samples_per_step": 50,
    },
    "code_default": {
        "label": "Code 4K",
        "color": "#dc2626",  # red
        "prefix": "code_default",
        "steps": [100, 200, 300, 400, 500],
        "samples_per_step": 50,
    },
    "code_2k": {
        "label": "Code 2K",
        "color": "#ea580c",  # orange
        "prefix": "code_2k",
        "steps": [50, 100, 150, 200],
        "samples_per_step": 50,
    },
}

DATASETS = ["math_500", "mbppplus"]
DATASET_LABELS = {"math_500": "MATH-500", "mbppplus": "MBPP+"}


def collect_data():
    """Collect metrics for all models and datasets."""
    results_dir = EXPERIMENT_DIR / "results"

    # Base model
    base = analyze_folder(results_dir / "qwen3_base", is_trained=False)

    # Trained models
    model_data = {}
    for model_key, cfg in MODELS.items():
        model_data[model_key] = {"cfg": cfg, "steps": {}}
        for step in cfg["steps"]:
            step_str = f"step{step:03d}"
            folder = results_dir / f"{cfg['prefix']}_{step_str}"
            if folder.exists():
                analysis = analyze_folder(folder, is_trained=True)
                model_data[model_key]["steps"][step] = analysis

    return base, model_data


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_plot(output_path: Path):
    base, model_data = collect_data()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), gridspec_kw={"width_ratios": [1, 1, 0.6]})

    for row_idx, ds in enumerate(DATASETS):
        ax_all = axes[row_idx, 0]
        ax_cmplt = axes[row_idx, 1]
        ax_cr = axes[row_idx, 2]
        ds_label = DATASET_LABELS[ds]

        # Base model accuracy
        base_acc = None
        if ds in base and "accuracy_all" in base[ds]:
            base_acc = base[ds]["accuracy_all"] * 100

        for ax in [ax_all, ax_cmplt]:
            if base_acc is not None:
                ax.axhline(y=base_acc, color="gray", linestyle="--", linewidth=1.5,
                           label="Base model", zorder=1)

        # Plot each model
        for model_key, mdata in model_data.items():
            cfg = mdata["cfg"]
            samples_list = []
            acc_all_list = []
            acc_cmplt_list = []
            completion_list = []

            for step in cfg["steps"]:
                if step not in mdata["steps"]:
                    continue
                analysis = mdata["steps"][step]
                if ds not in analysis:
                    continue
                ds_data = analysis[ds]
                if "error" in ds_data:
                    continue

                samples = step * cfg["samples_per_step"]
                samples_list.append(samples)
                acc_all_list.append(ds_data["accuracy_all"] * 100)
                acc_cmplt_list.append(ds_data["accuracy_completed"] * 100)
                completion_list.append(ds_data["completion_rate"] * 100)

            if not samples_list:
                continue

            # Accuracy (all)
            ax_all.plot(samples_list, acc_all_list,
                        color=cfg["color"], marker="o", markersize=5,
                        linewidth=2, label=cfg["label"], zorder=2)

            # Accuracy (completed)
            ax_cmplt.plot(samples_list, acc_cmplt_list,
                          color=cfg["color"], marker="o", markersize=5,
                          linewidth=2, label=cfg["label"], zorder=2)

            # Completion rate
            ax_cr.plot(samples_list, completion_list,
                       color=cfg["color"], marker="s", markersize=5,
                       linewidth=2, label=cfg["label"], zorder=2)

        # Format axes
        for ax in [ax_all, ax_cmplt, ax_cr]:
            ax.set_xlabel("Training samples (K)", fontsize=11)
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax_all.set_ylabel(f"{ds_label}\nAccuracy", fontsize=12, fontweight="bold")
        ax_all.set_title("Accuracy (all)" if row_idx == 0 else "", fontsize=13)
        ax_cmplt.set_title("Accuracy (completed only)" if row_idx == 0 else "", fontsize=13)
        ax_cr.set_title("Completion rate" if row_idx == 0 else "", fontsize=13)

        # Base completion rate line
        if ds in base and "completion_rate" in base[ds]:
            base_cr = base[ds]["completion_rate"] * 100
            ax_cr.axhline(y=base_cr, color="gray", linestyle="--", linewidth=1.5,
                          label="Base model", zorder=1)

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.98), frameon=True)

    fig.suptitle("Experiment 16/02 — Distillation Training Trajectories",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment 16/02 results")
    parser.add_argument("--output", type=str,
                        default=str(EXPERIMENT_DIR / "plots" / "trajectory.png"),
                        help="Output path for the plot")
    args = parser.parse_args()
    make_plot(Path(args.output))


if __name__ == "__main__":
    main()
