"""Analysis module. Per-experiment plots, cross-experiment comparison, summaries.

Usage:
    from analyze import plot_checkpoint_curve, compare, generate_summary

    plot_checkpoint_curve(checkpoint_results, Path("plots/curve.png"))
    compare([log_dir_1, log_dir_2, log_dir_3], Path("plots/"))
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Per-experiment plots ─────────────────────────────────────────────────────


def plot_checkpoint_curve(
    checkpoint_results: dict[int, dict[str, dict]],
    output_path: Path,
    title: str = "Checkpoint Evaluation",
    base_scores: dict[str, float] | None = None,
):
    """Plot accuracy vs training step for each dataset.

    Args:
        checkpoint_results: {step: {dataset: {accuracy: ...}}}
        output_path: Where to save the PNG.
        title: Plot title.
        base_scores: Optional base model scores as horizontal lines.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets_seen = set()
    for results in checkpoint_results.values():
        datasets_seen.update(results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    steps = sorted(checkpoint_results.keys())

    for ds in sorted(datasets_seen):
        ys = [checkpoint_results[s].get(ds, {}).get("accuracy") for s in steps]
        valid = [(s, y) for s, y in zip(steps, ys) if y is not None]
        if valid:
            xs, ys_valid = zip(*valid)
            ax.plot(xs, [y * 100 for y in ys_valid], "o-", label=ds, markersize=6)

    if base_scores:
        for ds, score in base_scores.items():
            ax.axhline(y=score * 100, linestyle="--", alpha=0.5, label=f"base ({ds})")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved checkpoint curve to {output_path}")


def plot_training_loss(metrics_path: Path, output_path: Path, title: str = "Training Loss"):
    """Plot NLL loss vs training step from metrics.jsonl."""
    if not metrics_path.exists():
        print(f"  No metrics file at {metrics_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps, train_nll, test_nll = [], [], []

    with open(metrics_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = d.get("step", d.get("progress/batch"))
            if step is None:
                continue
            if "train_mean_nll" in d:
                steps.append(step)
                train_nll.append(d["train_mean_nll"])
            if "test/nll" in d:
                test_nll.append((step, d["test/nll"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    if steps and train_nll:
        ax.plot(steps, train_nll, alpha=0.5, label="Train NLL", linewidth=0.8)
    if test_nll:
        test_steps, test_vals = zip(*test_nll)
        ax.plot(test_steps, test_vals, "o-", label="Test NLL", markersize=5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("NLL")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training loss plot to {output_path}")


# ── Summary generation ───────────────────────────────────────────────────────


def generate_summary(
    experiment_name: str,
    full_eval_results: dict[str, dict],
    training_config: dict | None = None,
    selected_step: int | None = None,
) -> dict:
    """Generate a JSON-serializable experiment summary."""
    summary = {
        "experiment": experiment_name,
        "results": {},
    }
    if training_config:
        summary["training"] = training_config
    if selected_step is not None:
        summary["selected_checkpoint_step"] = selected_step

    for ds, result in full_eval_results.items():
        if "accuracy" in result:
            summary["results"][ds] = {
                "metric": "accuracy",
                "value": result["accuracy"],
                "n_correct": result.get("n_correct"),
                "total": result.get("total"),
            }
        elif "mean_score" in result:
            summary["results"][ds] = {
                "metric": "mean_score",
                "value": result["mean_score"],
            }

    return summary


# ── Cross-experiment comparison ──────────────────────────────────────────────


def compare(log_dirs: list[Path], output_dir: Path):
    """Compare results across multiple experiments.

    Reads summary.json from each log dir and generates:
    - comparison_accuracy.png: grouped bar chart
    - comparison_table.csv: tabular summary

    Args:
        log_dirs: List of experiment log directories (each must have summary.json).
        output_dir: Where to save comparison plots and CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summaries
    summaries = []
    for d in log_dirs:
        summary_path = d / "summary.json"
        if not summary_path.exists():
            print(f"  WARNING: No summary.json in {d.name}, skipping")
            continue
        with open(summary_path) as f:
            s = json.load(f)
        s["_log_dir"] = str(d)
        summaries.append(s)

    if not summaries:
        print("  No summaries found. Nothing to compare.")
        return

    # Collect all datasets and models
    all_datasets = set()
    for s in summaries:
        all_datasets.update(s.get("results", {}).keys())
    all_datasets = sorted(all_datasets)

    model_names = [s["experiment"] for s in summaries]

    # Build accuracy matrix
    accuracy_matrix = []
    for s in summaries:
        row = []
        for ds in all_datasets:
            result = s.get("results", {}).get(ds, {})
            value = result.get("value")
            row.append(value)
        accuracy_matrix.append(row)

    # Generate grouped bar chart
    _plot_comparison_bars(model_names, all_datasets, accuracy_matrix, output_dir)

    # Generate CSV
    csv_path = output_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + all_datasets)
        for name, row in zip(model_names, accuracy_matrix):
            formatted = [f"{v:.3f}" if v is not None else "" for v in row]
            writer.writerow([name] + formatted)
    print(f"  Saved comparison table to {csv_path}")


def _plot_comparison_bars(
    model_names: list[str],
    datasets: list[str],
    values: list[list[float | None]],
    output_dir: Path,
):
    """Generate a grouped bar chart comparing models across datasets."""
    n_models = len(model_names)
    n_datasets = len(datasets)

    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 2), 7))
    x = np.arange(n_datasets)
    width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    for i, (name, row) in enumerate(zip(model_names, values)):
        vals = []
        for v in row:
            if v is not None:
                # Detect if it's accuracy (0-1) or score (0-48)
                vals.append(v * 100 if v <= 1.0 else v)
            else:
                vals.append(0)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i])
        # Add value labels on bars
        for bar, v, raw in zip(bars, vals, row):
            if raw is not None and v > 0:
                label = f"{v:.1f}" if v > 1 else f"{raw:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / "comparison_accuracy.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison chart to {output_path}")
