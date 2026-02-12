"""Sweep orchestrator: launches all configs, monitors, picks winner.

Usage:
  python sweep.py --task math
  python sweep.py --task code
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config import TRAINING_BASE
from sweep_configs import MATH_CONFIGS, CODE_CONFIGS


def parse_metrics(config_dir: Path) -> list[dict]:
    """Parse metrics.jsonl from a config directory."""
    metrics_file = config_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    results = []
    with open(metrics_file) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def get_best_nll(metrics: list[dict]) -> tuple[float, int]:
    """Find lowest test NLL and the step it occurred at."""
    best_nll = float("inf")
    best_step = -1
    for d in metrics:
        if "test/nll" in d:
            nll = d["test/nll"]
            step = d.get("step", d.get("progress/batch", -1))
            if nll < best_nll:
                best_nll = nll
                best_step = step
    return best_nll, best_step


def main():
    parser = argparse.ArgumentParser(description="Run distillation sweep")
    parser.add_argument("--task", choices=["math", "code"], required=True,
                        help="Which distillation task to sweep")
    args = parser.parse_args()

    if args.task == "math":
        configs = MATH_CONFIGS
        train_script = "train_math.py"
    else:
        configs = CODE_CONFIGS
        train_script = "train_code.py"

    task_dir = Path(TRAINING_BASE) / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Distillation Sweep: {args.task}")
    print(f"Configs: {', '.join(configs.keys())}")
    print(f"{'='*60}")

    # Launch all configs as subprocesses
    processes = {}
    for config_name in configs:
        log_file = task_dir / config_name / "sweep_runner.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, train_script, config_name]
        print(f"  Launching {config_name}: {' '.join(cmd)}")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent),
            )
        processes[config_name] = proc

    # Monitor until all complete
    print(f"\nMonitoring {len(processes)} configs...")
    while True:
        time.sleep(30)
        all_done = True
        status_parts = []
        for name, proc in processes.items():
            rc = proc.poll()
            if rc is None:
                all_done = False
                metrics = parse_metrics(task_dir / name)
                n_steps = len([m for m in metrics if "train/nll" in m])
                status_parts.append(f"{name}: step {n_steps}")
            else:
                status_parts.append(f"{name}: done (rc={rc})")
        print(f"  [{time.strftime('%H:%M:%S')}] {' | '.join(status_parts)}")
        if all_done:
            break

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Config':<12} {'InitNLL':>8} {'Step100':>8} {'Step500':>8} {'BestNLL':>8} {'@Step':>6} {'Delta':>8}")
    print(f"{'-'*80}")

    results = {}
    for config_name in configs:
        metrics = parse_metrics(task_dir / config_name)
        best_nll, best_step = get_best_nll(metrics)

        # Get NLL at specific steps
        nll_by_step = {}
        for d in metrics:
            if "test/nll" in d:
                step = d.get("step", d.get("progress/batch", -1))
                nll_by_step[step] = d["test/nll"]

        init_nll = nll_by_step.get(0, nll_by_step.get(min(nll_by_step.keys()), float("nan"))) if nll_by_step else float("nan")
        step100 = nll_by_step.get(100, float("nan"))
        step500 = nll_by_step.get(500, float("nan"))
        delta = init_nll - best_nll if init_nll != float("nan") else float("nan")

        print(f"{config_name:<12} {init_nll:>8.4f} {step100:>8.4f} {step500:>8.4f} {best_nll:>8.4f} {best_step:>6d} {delta:>8.4f}")

        results[config_name] = {
            "best_nll": best_nll,
            "best_step": best_step,
            "init_nll": init_nll,
            "delta": delta,
        }

    # Pick winner
    winner = min(results, key=lambda k: results[k]["best_nll"])
    print(f"\nWinner: {winner} (best NLL: {results[winner]['best_nll']:.4f})")

    # Save summary
    summary = {
        "task": args.task,
        "winner": winner,
        "results": results,
    }
    summary_path = task_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
