"""
Code RLVR Hyperparameter Sweep Orchestrator.

1. Runs calibrate_code_max_tokens.py to determine optimal max_tokens
2. Launches 6 training configs in parallel as subprocesses
3. Monitors progress and prints results as each finishes
4. Prints final comparison table
"""

import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = Path('/workspace/results_10_02')
CALIBRATION_FILE = RESULTS_DIR / 'code_calibration_kodcode.json'
PYTHON = '/usr/bin/python3.13'

CONFIGS = [
    'A_moderate', 'B_conservative', 'C_aggressive',
    'D_baseline_v2', 'E_ppo_moderate', 'F_large_group',
]


def check_memory():
    """Print current memory usage from cgroup."""
    try:
        with open('/sys/fs/cgroup/memory.current') as f:
            current = int(f.read().strip())
        with open('/sys/fs/cgroup/memory.max') as f:
            limit = int(f.read().strip())
        pct = current / limit * 100
        gb = current / 1024**3
        limit_gb = limit / 1024**3
        return f"{pct:.1f}% ({gb:.1f}GB / {limit_gb:.1f}GB)"
    except Exception:
        return "unknown"


def parse_metrics(config_name):
    """Parse metrics.jsonl for a completed run and extract test accuracy curve."""
    metrics_file = RESULTS_DIR / f'code_{config_name}' / 'metrics.jsonl'
    if not metrics_file.exists():
        return None

    eval_points = []
    with open(metrics_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'test/env/all/correct' in d:
                eval_points.append({
                    'step': d.get('progress/batch', -1),
                    'test_accuracy': d['test/env/all/correct'],
                    'entropy': d.get('optim/entropy', None),
                    'train_correct': d.get('env/all/correct', None),
                })

    if not eval_points:
        return None

    peak = max(eval_points, key=lambda x: x['test_accuracy'])
    return {
        'config': config_name,
        'eval_points': eval_points,
        'peak_accuracy': peak['test_accuracy'],
        'peak_step': peak['step'],
        'final_accuracy': eval_points[-1]['test_accuracy'],
        'final_entropy': eval_points[-1].get('entropy'),
        'step0_accuracy': eval_points[0]['test_accuracy'] if eval_points else None,
    }


def print_comparison(all_results):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"CODE RLVR SWEEP COMPARISON TABLE")
    print(f"{'='*90}")

    # Header
    print(f"{'Config':<20} {'Step0':>7} {'Step25':>7} {'Step50':>7} {'Step75':>7} "
          f"{'Peak':>7} {'@Step':>6} {'Delta':>7} {'Entropy':>8}")
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*8}")

    for r in sorted(all_results, key=lambda x: -x['peak_accuracy']):
        # Build step-by-step values
        step_vals = {}
        for ep in r['eval_points']:
            step_vals[ep['step']] = ep['test_accuracy']

        s0 = step_vals.get(0, None)
        s25 = step_vals.get(25, None)
        s50 = step_vals.get(50, None)
        s75 = step_vals.get(75, None)

        delta = (r['peak_accuracy'] - s0) * 100 if s0 is not None else None
        entropy_str = f"{r['final_entropy']:.3f}" if r['final_entropy'] is not None else "?"

        def fmt(v):
            return f"{v*100:.1f}%" if v is not None else "   -  "

        delta_str = f"+{delta:.1f}pp" if delta is not None else "?"

        print(f"{r['config']:<20} {fmt(s0):>7} {fmt(s25):>7} {fmt(s50):>7} {fmt(s75):>7} "
              f"{fmt(r['peak_accuracy']):>7} {'@'+str(r['peak_step']):>6} {delta_str:>7} {entropy_str:>8}")

    # Winner
    if all_results:
        winner = max(all_results, key=lambda x: x['peak_accuracy'])
        print(f"\n{'='*90}")
        print(f"WINNER: {winner['config']} (peak {winner['peak_accuracy']*100:.1f}% at step {winner['peak_step']})")
        step0 = winner.get('step0_accuracy')
        if step0 is not None:
            print(f"  Delta from base: +{(winner['peak_accuracy'] - step0)*100:.1f}pp")
        print(f"{'='*90}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Step 0: Calibrate max_tokens
    print(f"[{datetime.now():%H:%M:%S}] Step 0: Calibrating code max_tokens...")
    print(f"  Memory: {check_memory()}")

    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        max_tokens = cal['recommended_max_tokens']
        print(f"  Using cached calibration: max_tokens={max_tokens}")
    else:
        result = subprocess.run(
            [PYTHON, str(SCRIPTS_DIR / 'calibrate_code_max_tokens.py')],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: Calibration failed with exit code {result.returncode}")
            sys.exit(1)
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        max_tokens = cal['recommended_max_tokens']

    print(f"\n  Calibrated max_tokens = {max_tokens}")
    print(f"  Time: {time.time() - start_time:.0f}s")

    # Step 1: Launch all configs in parallel
    print(f"\n[{datetime.now():%H:%M:%S}] Step 1: Launching {len(CONFIGS)} code RLVR configs in parallel...")
    print(f"  Memory: {check_memory()}")

    processes = {}
    log_files = {}
    for config_name in CONFIGS:
        log_dir = RESULTS_DIR / f'code_{config_name}'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'sweep_runner.log'

        cmd = [
            PYTHON, str(SCRIPTS_DIR / 'run_code_rlvr_single.py'),
            config_name, '--max_tokens', str(max_tokens),
        ]
        lf = open(log_file, 'w')
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        processes[config_name] = proc
        log_files[config_name] = lf
        print(f"  Launched code_{config_name} (PID {proc.pid})")

    # Step 2: Monitor until all complete
    print(f"\n[{datetime.now():%H:%M:%S}] Monitoring {len(processes)} parallel runs...")
    completed = set()
    all_results = []

    while len(completed) < len(CONFIGS):
        time.sleep(30)
        elapsed = time.time() - start_time

        for config_name, proc in processes.items():
            if config_name in completed:
                continue
            ret = proc.poll()
            if ret is not None:
                completed.add(config_name)
                log_files[config_name].close()

                if ret == 0:
                    result = parse_metrics(config_name)
                    if result:
                        all_results.append(result)
                        print(f"\n[{datetime.now():%H:%M:%S}] code_{config_name} COMPLETED "
                              f"(peak {result['peak_accuracy']*100:.1f}% at step {result['peak_step']})")
                    else:
                        print(f"\n[{datetime.now():%H:%M:%S}] code_{config_name} COMPLETED "
                              f"(no eval metrics found)")
                else:
                    print(f"\n[{datetime.now():%H:%M:%S}] code_{config_name} FAILED "
                          f"(exit code {ret})")

                remaining = len(CONFIGS) - len(completed)
                if remaining > 0:
                    print(f"  {remaining} runs still in progress | "
                          f"Elapsed: {elapsed/60:.0f}min | Memory: {check_memory()}")

        # Periodic status (every 5 min)
        if int(elapsed) % 300 < 35:
            running = [c for c in CONFIGS if c not in completed]
            if running:
                print(f"  [{datetime.now():%H:%M:%S}] Running: {', '.join(running)} | "
                      f"Elapsed: {elapsed/60:.0f}min | Memory: {check_memory()}")

    # Step 3: Final comparison
    total_time = time.time() - start_time
    print(f"\n[{datetime.now():%H:%M:%S}] All runs complete in {total_time/60:.1f} minutes")
    print_comparison(all_results)

    # Save summary
    summary_file = RESULTS_DIR / 'code_sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'max_tokens': max_tokens,
            'total_time_seconds': total_time,
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved summary to {summary_file}")

    if all_results:
        winner = max(all_results, key=lambda x: x['peak_accuracy'])
        print(f"\nTo evaluate the winner, run:")
        print(f"  python3.13 {SCRIPTS_DIR}/eval_code_rlvr_winner.py "
              f"--winner {winner['config']}")


if __name__ == '__main__':
    main()
