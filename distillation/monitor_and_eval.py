#!/usr/bin/env python3
"""Monitor training runs, pick best checkpoints, run eval sweeps.

This script:
1. Waits for all training processes to finish
2. Analyzes metrics.jsonl to pick best math and code checkpoints
3. Updates eval_config.yaml with the winning sampler_paths
4. Runs evaluate.py for math_distill and code_distill models

Usage:
  python monitor_and_eval.py
"""

import json
import os
import signal
import subprocess
import sys
import time
import yaml
from pathlib import Path

DISTILL_DIR = Path(__file__).parent
TRAINING_BASE = DISTILL_DIR / "training_runs"
SWEEP_CONFIGS = {
    "math": {"A_fast": 500, "B_medium": 1000, "C_gentle": 1000, "D_long": 2000},
    "code": {"A_fast": 500, "B_medium": 1000, "C_gentle": 1000, "D_long": 2000},
}


def is_training_process_running(task: str, config: str) -> bool:
    """Check if a training process is still running for this config."""
    script = f"train_{task}.py"
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{script} {config}"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def is_training_complete(task: str, config: str) -> bool:
    """Check if training completed (has 'final' checkpoint or reached target steps)."""
    ckpt_file = TRAINING_BASE / task / config / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return False
    with open(ckpt_file) as f:
        lines = f.readlines()
    if not lines:
        return False
    try:
        last = json.loads(lines[-1])
        if last.get("name") == "final":
            return True
    except json.JSONDecodeError:
        pass

    # Also check if process exited
    return not is_training_process_running(task, config)


def get_metrics(task: str, config: str) -> list[dict]:
    """Read all metrics for a config."""
    mf = TRAINING_BASE / task / config / "metrics.jsonl"
    if not mf.exists():
        return []
    results = []
    with open(mf) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def get_best_checkpoint(task: str, config: str) -> dict | None:
    """Find the checkpoint with the lowest test/nll."""
    metrics = get_metrics(task, config)
    best_nll = float("inf")
    best_step = -1
    for d in metrics:
        if "test/nll" in d:
            nll = d["test/nll"]
            step = d.get("step", -1)
            if nll < best_nll:
                best_nll = nll
                best_step = step

    if best_step < 0:
        return None

    # Find the corresponding checkpoint
    ckpt_file = TRAINING_BASE / task / config / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return None

    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Find exact match or closest
    best_ckpt = None
    for ckpt in checkpoints:
        if ckpt.get("batch", -1) == best_step and "sampler_path" in ckpt:
            best_ckpt = ckpt
            break

    if best_ckpt is None:
        valid = [c for c in checkpoints if "sampler_path" in c]
        if valid:
            best_ckpt = min(valid, key=lambda c: abs(c.get("batch", 0) - best_step))

    if best_ckpt:
        return {
            "config": config,
            "best_nll": best_nll,
            "best_step": best_step,
            "sampler_path": best_ckpt["sampler_path"],
            "ckpt_step": best_ckpt.get("batch", -1),
        }
    return None


def wait_for_training():
    """Wait for all training processes to complete."""
    print("=" * 60)
    print("PHASE 1: Waiting for training to complete")
    print("=" * 60)

    while True:
        all_done = True
        status_parts = []

        for task in ["math", "code"]:
            for config, target_steps in SWEEP_CONFIGS[task].items():
                running = is_training_process_running(task, config)
                complete = is_training_complete(task, config)

                metrics = get_metrics(task, config)
                current_step = 0
                for d in metrics:
                    if "step" in d:
                        current_step = max(current_step, d["step"])

                if complete:
                    status = f"DONE({current_step})"
                elif running:
                    status = f"{current_step}/{target_steps}"
                    all_done = False
                else:
                    # Not running, not complete - might have crashed
                    status = f"STOPPED({current_step})"

                status_parts.append(f"{task}/{config}: {status}")

        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {' | '.join(status_parts)}")

        if all_done:
            print("\nAll training runs complete!")
            break

        time.sleep(60)  # Check every minute


def pick_winners() -> dict[str, dict]:
    """Analyze all configs and pick the best for each task."""
    print("\n" + "=" * 60)
    print("PHASE 2: Picking best checkpoints")
    print("=" * 60)

    winners = {}
    for task in ["math", "code"]:
        print(f"\n--- {task.upper()} ---")
        print(f"{'Config':<12} {'BestNLL':>10} {'@Step':>8} {'CkptStep':>10} {'SamplerPath'}")
        print("-" * 80)

        best_result = None
        for config in SWEEP_CONFIGS[task]:
            result = get_best_checkpoint(task, config)
            if result is None:
                print(f"{config:<12} {'N/A':>10}")
                continue

            print(f"{config:<12} {result['best_nll']:>10.4f} {result['best_step']:>8d} "
                  f"{result['ckpt_step']:>10d} {result['sampler_path'][:60]}...")

            if best_result is None or result["best_nll"] < best_result["best_nll"]:
                best_result = result

        if best_result:
            print(f"\n  WINNER: {best_result['config']} "
                  f"(NLL={best_result['best_nll']:.4f} @ step {best_result['best_step']})")
            winners[task] = best_result
        else:
            print(f"\n  ERROR: No valid checkpoints found for {task}!")

    return winners


def update_eval_config(winners: dict[str, dict]):
    """Update eval_config.yaml with winning sampler_paths."""
    print("\n" + "=" * 60)
    print("PHASE 3: Updating eval_config.yaml")
    print("=" * 60)

    config_path = DISTILL_DIR / "eval_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "math" in winners:
        config["models"]["math_distill"]["sampler_path"] = winners["math"]["sampler_path"]
        print(f"  math_distill: {winners['math']['sampler_path']}")

    if "code" in winners:
        config["models"]["code_distill"]["sampler_path"] = winners["code"]["sampler_path"]
        print(f"  code_distill: {winners['code']['sampler_path']}")

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Saved to {config_path}")


def wait_for_base_eval():
    """Wait for the base model evaluation to finish if still running."""
    print("\n" + "=" * 60)
    print("Checking if base model eval is still running...")
    print("=" * 60)

    while True:
        result = subprocess.run(
            ["pgrep", "-f", "evaluate.py --model base"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("  Base eval is done (or not running).")
            break
        print(f"  [{time.strftime('%H:%M:%S')}] Base eval still running, waiting 60s...")
        time.sleep(60)


def run_eval(model_name: str, log_suffix: str):
    """Run evaluate.py for a specific model."""
    print(f"\nLaunching eval for {model_name}...")
    log_file = DISTILL_DIR / f"eval_{log_suffix}.log"
    cmd = [sys.executable, "evaluate.py", "--model", model_name]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(DISTILL_DIR),
        )

    return proc


def main():
    start_time = time.time()
    print(f"Monitor & Eval Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {DISTILL_DIR}")

    # Phase 1: Wait for training
    wait_for_training()

    # Phase 2: Pick winners
    winners = pick_winners()

    if not winners:
        print("\nERROR: No winners found! Aborting.")
        sys.exit(1)

    # Save winners summary
    summary_path = DISTILL_DIR / "training_winners.json"
    with open(summary_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nSaved winners to {summary_path}")

    # Phase 3: Update eval config
    update_eval_config(winners)

    # Phase 4: Wait for base eval to finish (it uses the GPU/tinker service)
    wait_for_base_eval()

    # Phase 5: Run evals
    print("\n" + "=" * 60)
    print("PHASE 4: Running evaluation sweeps")
    print("=" * 60)

    # Run math_distill and code_distill evals sequentially
    # (they share the tinker service, running in parallel may cause issues)
    for model_name, log_suffix in [("math_distill", "math_distill"), ("code_distill", "code_distill")]:
        if model_name.replace("_distill", "") not in winners:
            print(f"  Skipping {model_name} (no winner found)")
            continue

        print(f"\n--- Evaluating {model_name} ---")
        proc = run_eval(model_name, log_suffix)

        # Monitor the eval process
        while proc.poll() is None:
            time.sleep(30)
            log_file = DISTILL_DIR / f"eval_{log_suffix}.log"
            if log_file.exists():
                # Show last few lines
                with open(log_file) as f:
                    lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    print(f"  [{time.strftime('%H:%M:%S')}] {model_name}: {last_line[:100]}")

        rc = proc.returncode
        print(f"  {model_name} eval completed with rc={rc}")

        if rc != 0:
            log_file = DISTILL_DIR / f"eval_{log_suffix}.log"
            print(f"  Check log for errors: {log_file}")
            with open(log_file) as f:
                lines = f.readlines()
            # Print last 20 lines on error
            for line in lines[-20:]:
                print(f"    {line.rstrip()}")

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/3600:.1f} hours")
    print(f"{'='*60}")

    # Print results
    for model_name in ["base", "math_distill", "code_distill"]:
        summary_file = DISTILL_DIR / "results" / model_name / "eval_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results = json.load(f)
            print(f"\n{model_name}:")
            for ds_name, ds_result in results.items():
                if "accuracy" in ds_result:
                    print(f"  {ds_name}: {ds_result['accuracy']*100:.1f}%")
                elif "mean_score" in ds_result:
                    print(f"  {ds_name}: {ds_result['mean_score']:.2f}")
        else:
            print(f"\n{model_name}: No results found")


if __name__ == "__main__":
    main()
