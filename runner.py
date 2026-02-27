"""Experiment orchestrator. Runs the full pipeline for one or more experiments.

Pipeline for a training experiment:
  1. Filter data (tokenize traces, cache in filtered_data/)
  2. Train (LoRA SFT via tinker)
  3. Eval checkpoints (fast eval on small subset)
  4. Select best checkpoint (by primary metric accuracy)
  5. Full eval (comprehensive benchmark suite)
  6. Generate plots + summary

Usage:
    from config import Experiment, BaseModelEval, TrainingConfig
    from runner import run_experiment, run_base_eval, run_batch

    result = run_experiment(experiment)
    result = run_base_eval(base_eval)
    results = run_batch([base_eval, experiment1, experiment2], max_parallel=2)
"""

import asyncio
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import LOGS_DIR, BaseModelEval, Experiment

# Default max_tokens when no training config is available (base model evals)
_DEFAULT_BASE_EVAL_MAX_TOKENS = 8192

_print_lock = threading.Lock()
from train import train_model
from evaluate import evaluate_checkpoint, evaluate_base_model, select_best_checkpoint
from analyze import (
    plot_checkpoint_curve,
    plot_training_loss,
    generate_summary,
    compare,
)


class _RunLog:
    """Per-experiment logger that writes to both stdout and run.log."""

    def __init__(self, log_dir: Path):
        self._file = open(log_dir / "run.log", "w")

    def __call__(self, msg: str = ""):
        line = msg + "\n"
        self._file.write(line)
        self._file.flush()
        with _print_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def close(self):
        self._file.close()


def _make_log_dir(name: str) -> Path:
    """Create timestamped log directory."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    dirname = f"{ts}_{name}"
    log_dir = LOGS_DIR / dirname
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def run_base_eval(base_eval: BaseModelEval, log_dir: Path | None = None) -> dict:
    """Evaluate a base model (no training).

    Creates a log directory with config + full_eval results + summary.
    If log_dir is provided (e.g. from run_batch), uses it directly.
    """
    if log_dir is None:
        log_dir = _make_log_dir(base_eval.name)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / "config.json", "w") as f:
        f.write(base_eval.to_json())

    # Resolve max_tokens
    eval_config = base_eval.eval
    max_tokens = eval_config.max_tokens or _DEFAULT_BASE_EVAL_MAX_TOKENS

    log = _RunLog(log_dir)
    log(f"\n{'#' * 70}")
    log(f"BASE MODEL EVAL: {base_eval.name}")
    log(f"  Model: {base_eval.base_model}")
    log(f"  Max tokens: {max_tokens}")
    log(f"  Log dir: {log_dir}")
    log(f"{'#' * 70}")

    t0 = time.time()
    results_dir = log_dir / "full_eval"

    results = asyncio.run(evaluate_base_model(
        base_model=base_eval.base_model,
        datasets=list(eval_config.datasets),
        results_dir=results_dir,
        model_label=base_eval.name,
        max_problems=eval_config.max_problems,
        max_tokens=max_tokens,
        serialize_code_eval=base_eval.serialize_code_eval,
    ))

    elapsed = time.time() - t0

    # Generate summary
    summary = generate_summary(base_eval.name, results)
    summary["elapsed_seconds"] = round(elapsed)
    summary["log_dir"] = str(log_dir)
    summary["type"] = "base_eval"

    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_results_table(base_eval.name, results, elapsed, log=log)
    log.close()
    return summary


def run_experiment(experiment: Experiment, log_dir: Path | None = None) -> dict:
    """Run a complete experiment: train -> checkpoint eval -> select -> full eval.

    Returns the final summary dict.
    If log_dir is provided (e.g. from run_batch), uses it directly.
    """
    if log_dir is None:
        log_dir = _make_log_dir(experiment.name)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
    t_config = experiment.training

    # Save config
    with open(log_dir / "config.json", "w") as f:
        f.write(experiment.to_json())

    # Resolve max_tokens: None means inherit from training.max_length
    ce_config = experiment.checkpoint_eval
    ce_max_tokens = ce_config.max_tokens or t_config.max_length
    fe_config = experiment.full_eval
    fe_max_tokens = fe_config.max_tokens or t_config.max_length

    log = _RunLog(log_dir)
    log(f"\n{'#' * 70}")
    log(f"EXPERIMENT: {experiment.name}")
    log(f"  Training: {t_config.task}/{t_config.trace_source}, "
        f"max_length={t_config.max_length}, rank={t_config.lora_rank}")
    log(f"  Checkpoint eval max_tokens: {ce_max_tokens}")
    log(f"  Full eval max_tokens: {fe_max_tokens}")
    log(f"  Log dir: {log_dir}")
    log(f"{'#' * 70}")

    t0 = time.time()

    # ── Phase 1: Training ────────────────────────────────────────────────
    if experiment.skip_training and experiment.sampler_path:
        log("\n--- Skipping training (using provided sampler_path) ---")
        checkpoints = [{"batch": 0, "sampler_path": experiment.sampler_path}]
    else:
        log("\n--- Phase 1: Training ---")
        training_dir = log_dir / "training"
        training_result = train_model(t_config, training_dir)
        checkpoints = training_result.checkpoints
        if not checkpoints:
            log("ERROR: Training produced no checkpoints")
            log.close()
            return {"experiment": experiment.name, "error": "no checkpoints"}

    # ── Phase 2: Checkpoint evaluation ───────────────────────────────────
    log("\n--- Phase 2: Checkpoint Evaluation ---")
    checkpoint_results = {}  # step -> {dataset: result}

    for ckpt in checkpoints:
        step = ckpt.get("batch")
        sampler_path = ckpt.get("sampler_path")
        if step is None or not sampler_path:
            continue

        ckpt_results_dir = log_dir / "checkpoint_eval" / f"step{step}"
        log(f"\n  Evaluating checkpoint step {step}...")

        results = asyncio.run(evaluate_checkpoint(
            sampler_path=sampler_path,
            base_model=t_config.base_model,
            datasets=list(ce_config.datasets),
            results_dir=ckpt_results_dir,
            model_label=f"{experiment.name}_step{step}",
            task=t_config.task,
            max_problems=ce_config.max_problems,
            max_tokens=ce_max_tokens,
            serialize_code_eval=experiment.serialize_code_eval,
        ))
        checkpoint_results[step] = results

    # Plot checkpoint curve
    if len(checkpoint_results) > 1:
        plot_checkpoint_curve(
            checkpoint_results,
            log_dir / "plots" / "checkpoint_curve.png",
            title=f"{experiment.name}: Accuracy vs Training Step",
        )

    # Plot training loss
    training_metrics = log_dir / "training" / "metrics.jsonl"
    if training_metrics.exists():
        plot_training_loss(
            training_metrics,
            log_dir / "plots" / "training_loss.png",
            title=f"{experiment.name}: Training Loss",
        )

    # ── Phase 3: Select best checkpoint ──────────────────────────────────
    log("\n--- Phase 3: Checkpoint Selection ---")
    if experiment.skip_training and experiment.sampler_path:
        best_step = 0
        best_sampler = experiment.sampler_path
        reason = "provided directly"
    elif len(checkpoints) == 1:
        best_step = checkpoints[0]["batch"]
        best_sampler = checkpoints[0]["sampler_path"]
        reason = "only checkpoint"
    else:
        best_step, reason = select_best_checkpoint(
            checkpoint_results,
            task=t_config.task,
        )
        best_sampler = next(
            c["sampler_path"] for c in checkpoints if c["batch"] == best_step
        )

    log(f"  Selected: step {best_step} ({reason})")
    with open(log_dir / "selected_checkpoint.json", "w") as f:
        json.dump(
            {"step": best_step, "sampler_path": best_sampler, "reason": reason},
            f,
            indent=2,
        )

    # ── Phase 4: Full evaluation ─────────────────────────────────────────
    log("\n--- Phase 4: Full Evaluation ---")
    full_results_dir = log_dir / "full_eval"

    full_results = asyncio.run(evaluate_checkpoint(
        sampler_path=best_sampler,
        base_model=t_config.base_model,
        datasets=list(fe_config.datasets),
        results_dir=full_results_dir,
        model_label=experiment.name,
        task=t_config.task,
        max_problems=fe_config.max_problems,
        max_tokens=fe_max_tokens,
        serialize_code_eval=experiment.serialize_code_eval,
    ))

    # ── Generate summary ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = generate_summary(
        experiment.name,
        full_results,
        training_config=asdict(t_config),
        selected_step=best_step,
    )
    summary["elapsed_seconds"] = round(elapsed)
    summary["log_dir"] = str(log_dir)
    summary["type"] = "experiment"

    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _print_results_table(experiment.name, full_results, elapsed, best_step, log=log)
    log.close()
    return summary


def run_batch(
    items: list[Experiment | BaseModelEval],
    max_parallel: int = 1,
    batch_name: str | None = None,
) -> list[dict]:
    """Run multiple experiments/base evals grouped under one batch directory.

    With max_parallel=1 (default), runs sequentially.
    With max_parallel>1, runs concurrently using threads.
    Code evals are serialized via semaphore regardless.

    Returns list of summary dicts.
    """
    t0 = time.time()

    # Create batch parent directory
    if batch_name is None:
        batch_name = "batch"
    batch_dir = _make_log_dir(batch_name)

    # Save batch manifest
    manifest = {
        "batch_name": batch_name,
        "started_at": datetime.now().isoformat(),
        "max_parallel": max_parallel,
        "items": [
            {"name": item.name, "type": "base_eval" if isinstance(item, BaseModelEval) else "experiment"}
            for item in items
        ],
    }
    with open(batch_dir / "batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'#' * 70}")
    print(f"BATCH: {batch_name} ({len(items)} items, max_parallel={max_parallel})")
    print(f"  Log dir: {batch_dir}")
    print(f"{'#' * 70}")

    def run_item(item):
        item_log_dir = batch_dir / item.name
        try:
            if isinstance(item, BaseModelEval):
                return run_base_eval(item, log_dir=item_log_dir)
            else:
                return run_experiment(item, log_dir=item_log_dir)
        except Exception as e:
            print(f"ERROR in {item.name}: {e}")
            return {"experiment": item.name, "error": str(e)}

    if max_parallel <= 1:
        summaries = []
        for i, item in enumerate(items):
            print(f"\n{'=' * 70}")
            print(f"BATCH: Item {i + 1}/{len(items)}: {item.name}")
            print(f"{'=' * 70}")
            summaries.append(run_item(item))
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            summaries = list(executor.map(run_item, items))

    elapsed = time.time() - t0

    # Save batch summary
    batch_summary = {
        "batch_name": batch_name,
        "batch_dir": str(batch_dir),
        "started_at": manifest["started_at"],
        "elapsed_seconds": round(elapsed),
        "max_parallel": max_parallel,
        "total": len(summaries),
        "succeeded": sum(1 for s in summaries if "error" not in s),
        "failed": sum(1 for s in summaries if "error" in s),
        "items": summaries,
    }
    with open(batch_dir / "batch_summary.json", "w") as f:
        json.dump(batch_summary, f, indent=2, default=str)

    _print_batch_summary(batch_dir, summaries, elapsed)
    return summaries


def _print_results_table(name: str, results: dict, elapsed: float,
                         step: int | None = None, log: _RunLog | None = None):
    """Print a formatted results table."""
    out = log or (lambda msg="": print(msg))
    step_str = f", step {step}" if step is not None else ""
    out(f"\n{'=' * 60}")
    out(f"RESULTS: {name} ({elapsed:.0f}s{step_str})")
    out(f"{'=' * 60}")
    for ds, result in results.items():
        if "accuracy" in result:
            out(f"  {ds:<25} {result['accuracy'] * 100:.1f}%")
        elif "mean_score" in result:
            out(f"  {ds:<25} {result['mean_score']:.2f}")


def _print_batch_summary(batch_dir: Path, summaries: list[dict], elapsed: float):
    """Print batch completion summary."""
    n_ok = sum(1 for s in summaries if "error" not in s)
    n_err = sum(1 for s in summaries if "error" in s)
    print(f"\n{'#' * 70}")
    print(f"BATCH COMPLETE: {len(summaries)} items in {elapsed:.0f}s ({n_ok} OK, {n_err} errors)")
    print(f"  Batch dir: {batch_dir}")
    print(f"  Summary:   {batch_dir / 'batch_summary.json'}")
    print(f"{'#' * 70}")

    # Collect all log dirs for easy comparison
    log_dirs = [s.get("log_dir") for s in summaries if "log_dir" in s]
    if log_dirs:
        print("\nTo compare results:")
        print(f"  python run.py compare {' '.join(log_dirs)}")
