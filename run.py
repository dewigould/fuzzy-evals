"""CLI entry point for running experiments.

Usage:
    python run.py single --name math_sonnet --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
        --task math --traces sonnet --max-length 4096

    python run.py base-eval --name qwen30b_base \
        --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507"

    python run.py batch my_overnight.py --max-parallel 4

    python run.py eval --sampler-path "tinker://..." \
        --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507" --task math

    python run.py compare logs/2026-02-26_* --output plots/
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from config import (
    Experiment,
    BaseModelEval,
    TrainingConfig,
    CheckpointEvalConfig,
    FullEvalConfig,
)
from runner import run_experiment, run_base_eval, run_batch
from analyze import compare


def cmd_single(args):
    """Run a single training experiment."""
    training = TrainingConfig(
        base_model=args.base_model,
        task=args.task,
        trace_source=args.traces,
        data_path=args.data_path,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )

    ckpt_eval = CheckpointEvalConfig(
        max_problems=args.checkpoint_eval_max_problems,
        max_tokens=args.full_eval_max_tokens,  # explicit override, or None to match training
    )

    full_eval = FullEvalConfig(
        max_tokens=args.full_eval_max_tokens,  # explicit override, or None to match training
    )

    experiment = Experiment(
        name=args.name,
        training=training,
        checkpoint_eval=ckpt_eval,
        full_eval=full_eval,
    )

    run_experiment(experiment)


def cmd_base_eval(args):
    """Evaluate a base model (no training)."""
    datasets = None
    if args.datasets:
        datasets = tuple(args.datasets.split(","))

    eval_config = FullEvalConfig(
        max_tokens=args.max_tokens,
        max_problems=args.max_problems,
    )
    if datasets:
        eval_config = FullEvalConfig(
            datasets=datasets,
            max_tokens=args.max_tokens,
            max_problems=args.max_problems,
        )

    base_eval = BaseModelEval(
        name=args.name,
        base_model=args.base_model,
        eval=eval_config,
    )

    run_base_eval(base_eval)


def cmd_batch(args):
    """Run a batch from a Python file.

    The file must define EXPERIMENTS: list[Experiment | BaseModelEval].
    """
    spec = importlib.util.spec_from_file_location("batch_config", args.batch_file)
    mod = importlib.util.module_from_spec(spec)

    # Make config importable from the batch file
    sys.path.insert(0, str(Path(__file__).parent))
    spec.loader.exec_module(mod)

    if hasattr(mod, "EXPERIMENTS"):
        items = mod.EXPERIMENTS
    elif hasattr(mod, "BATCH"):
        items = mod.BATCH
    else:
        print("ERROR: Batch file must define EXPERIMENTS or BATCH")
        sys.exit(1)

    if not isinstance(items, list):
        items = list(items)

    # Derive batch name from filename (e.g. "overnight_sweep.py" -> "overnight_sweep")
    # or use explicit --name if provided
    batch_name = args.name or Path(args.batch_file).stem

    run_batch(items, max_parallel=args.max_parallel, batch_name=batch_name)


def cmd_eval(args):
    """Evaluate an existing checkpoint (skip training)."""
    training = TrainingConfig(
        base_model=args.base_model,
        task=args.task,
        trace_source="none",
        max_length=4096,
    )

    datasets = tuple(args.datasets.split(",")) if args.datasets else None
    full_eval_kwargs = {"max_tokens": args.max_tokens}
    if datasets:
        full_eval_kwargs["datasets"] = datasets
    if args.max_problems:
        full_eval_kwargs["max_problems"] = args.max_problems
    full_eval = FullEvalConfig(**full_eval_kwargs)

    experiment = Experiment(
        name=args.name or "adhoc_eval",
        training=training,
        full_eval=full_eval,
        skip_training=True,
        sampler_path=args.sampler_path,
    )

    run_experiment(experiment)


def cmd_compare(args):
    """Compare results across experiment log directories."""
    log_dirs = [Path(d) for d in args.log_dirs]
    output_dir = Path(args.output) if args.output else Path("plots")
    compare(log_dirs, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Distillation experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── single: run one training experiment ──
    p_single = sub.add_parser("single", help="Run a single training experiment")
    p_single.add_argument("--name", required=True)
    p_single.add_argument("--base-model", required=True)
    p_single.add_argument("--task", required=True, choices=["math", "code"])
    p_single.add_argument("--traces", required=True, help="Trace source: sonnet, qwen, kimi")
    p_single.add_argument("--data-path", default=None, help="Override trace data path")
    p_single.add_argument("--max-length", type=int, default=4096)
    p_single.add_argument("--lora-rank", type=int, default=32)
    p_single.add_argument("--learning-rate", type=float, default=2e-4)
    p_single.add_argument("--lr-schedule", default="cosine")
    p_single.add_argument("--batch-size", type=int, default=50)
    p_single.add_argument("--max-samples", type=int, default=25000)
    p_single.add_argument("--save-every", type=int, default=100)
    p_single.add_argument("--eval-every", type=int, default=100)
    p_single.add_argument("--checkpoint-eval-max-problems", type=int, default=100)
    p_single.add_argument("--full-eval-max-tokens", type=int, default=None,
                          help="Max generation tokens for evals (default: match --max-length)")

    # ── base-eval: evaluate a base model ──
    p_base = sub.add_parser("base-eval", help="Evaluate a base model (no training)")
    p_base.add_argument("--name", required=True)
    p_base.add_argument("--base-model", required=True)
    p_base.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    p_base.add_argument("--max-tokens", type=int, default=8192)
    p_base.add_argument("--max-problems", type=int, default=None)

    # ── batch: run from Python file ──
    p_batch = sub.add_parser("batch", help="Run experiments from a Python batch file")
    p_batch.add_argument("batch_file", help="Python file defining EXPERIMENTS list")
    p_batch.add_argument("--name", default=None, help="Batch name (default: derived from filename)")
    p_batch.add_argument("--max-parallel", type=int, default=1)

    # ── eval: evaluate existing checkpoint ──
    p_eval = sub.add_parser("eval", help="Evaluate an existing checkpoint")
    p_eval.add_argument("--sampler-path", required=True)
    p_eval.add_argument("--base-model", required=True)
    p_eval.add_argument("--task", required=True, choices=["math", "code"])
    p_eval.add_argument("--name", default=None)
    p_eval.add_argument("--datasets", default=None)
    p_eval.add_argument("--max-tokens", type=int, default=28672)
    p_eval.add_argument("--max-problems", type=int, default=None)

    # ── compare: cross-experiment comparison ──
    p_compare = sub.add_parser("compare", help="Compare results across experiments")
    p_compare.add_argument("log_dirs", nargs="+", help="Log directories to compare")
    p_compare.add_argument("--output", default=None, help="Output directory for plots")

    args = parser.parse_args()

    if args.command == "single":
        cmd_single(args)
    elif args.command == "base-eval":
        cmd_base_eval(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
