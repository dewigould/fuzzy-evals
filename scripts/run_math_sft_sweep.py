"""
Math SFT Sweep Orchestrator.

Launches 4 parallel training configs for Math SFT on OpenR1-Math-220k,
monitors progress, prints comparison table, identifies winner.

Usage:
  python run_math_sft_sweep.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = Path('/workspace/results_10_02/math_sft')
PYTHON = '/usr/bin/python3.13'

CONFIGS = ['A_short', 'B_medium', 'C_gentle', 'D_long']


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
    """Parse metrics.jsonl for a completed SFT run and extract test NLL curve."""
    metrics_file = RESULTS_DIR / config_name / 'metrics.jsonl'
    if not metrics_file.exists():
        return None

    eval_points = []
    with open(metrics_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'test/nll' in d:
                eval_points.append({
                    'step': d.get('progress/batch', -1),
                    'test_nll': d['test/nll'],
                    'train_nll': d.get('train_mean_nll', None),
                })

    if not eval_points:
        return None

    best = min(eval_points, key=lambda x: x['test_nll'])
    return {
        'config': config_name,
        'eval_points': eval_points,
        'best_nll': best['test_nll'],
        'best_step': best['step'],
        'final_nll': eval_points[-1]['test_nll'],
        'initial_nll': eval_points[0]['test_nll'] if eval_points else None,
    }


def print_comparison(all_results):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"MATH SFT SWEEP COMPARISON TABLE")
    print(f"{'='*90}")

    # Header
    print(f"{'Config':<15} {'Init NLL':>10} {'Step100':>10} {'Step200':>10} {'Step500':>10} "
          f"{'Best NLL':>10} {'@Step':>6} {'Delta':>10}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*10}")

    for r in sorted(all_results, key=lambda x: x['best_nll']):
        step_vals = {}
        for ep in r['eval_points']:
            step_vals[ep['step']] = ep['test_nll']

        init = r['initial_nll']
        s100 = step_vals.get(100, None)
        s200 = step_vals.get(200, None)
        s500 = step_vals.get(500, None)

        delta = init - r['best_nll'] if init is not None else None

        def fmt(v):
            return f"{v:.4f}" if v is not None else "   -   "

        delta_str = f"-{delta:.4f}" if delta is not None else "?"

        print(f"{r['config']:<15} {fmt(init):>10} {fmt(s100):>10} {fmt(s200):>10} {fmt(s500):>10} "
              f"{fmt(r['best_nll']):>10} {'@'+str(r['best_step']):>6} {delta_str:>10}")

    # Winner
    if all_results:
        winner = min(all_results, key=lambda x: x['best_nll'])
        print(f"\n{'='*90}")
        print(f"WINNER: {winner['config']} (best NLL {winner['best_nll']:.4f} at step {winner['best_step']})")
        if winner['initial_nll'] is not None:
            delta = winner['initial_nll'] - winner['best_nll']
            print(f"  NLL improvement: {winner['initial_nll']:.4f} -> {winner['best_nll']:.4f} (-{delta:.4f})")
        print(f"{'='*90}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"[{datetime.now():%H:%M:%S}] Math SFT Sweep: Launching {len(CONFIGS)} configs in parallel...")
    print(f"  Memory: {check_memory()}")

    # Launch all configs
    processes = {}
    log_files = {}
    for config_name in CONFIGS:
        log_dir = RESULTS_DIR / config_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'sweep_runner.log'

        cmd = [PYTHON, str(SCRIPTS_DIR / 'run_math_sft_single.py'), config_name]
        lf = open(log_file, 'w')
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        processes[config_name] = proc
        log_files[config_name] = lf
        print(f"  Launched {config_name} (PID {proc.pid})")

    # Monitor until all complete
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
                        print(f"\n[{datetime.now():%H:%M:%S}] {config_name} COMPLETED "
                              f"(best NLL {result['best_nll']:.4f} at step {result['best_step']})")
                    else:
                        print(f"\n[{datetime.now():%H:%M:%S}] {config_name} COMPLETED "
                              f"(no eval metrics found)")
                else:
                    print(f"\n[{datetime.now():%H:%M:%S}] {config_name} FAILED "
                          f"(exit code {ret})")

                remaining = len(CONFIGS) - len(completed)
                if remaining > 0:
                    print(f"  {remaining} runs still in progress | "
                          f"Elapsed: {elapsed/60:.0f}min | Memory: {check_memory()}")

        # Periodic status (every 5 min)
        if int(elapsed) % 300 < 35:
            running = [c for c in CONFIGS if c not in completed]
            if running:
                # Show current NLL progress for running configs
                status_parts = []
                for c in running:
                    result = parse_metrics(c)
                    if result:
                        status_parts.append(f"{c}: NLL={result['final_nll']:.4f} @step{result['eval_points'][-1]['step']}")
                    else:
                        status_parts.append(f"{c}: no metrics yet")

                print(f"  [{datetime.now():%H:%M:%S}] Running: {' | '.join(status_parts)} | "
                      f"Elapsed: {elapsed/60:.0f}min | Memory: {check_memory()}")

    # Final comparison
    total_time = time.time() - start_time
    print(f"\n[{datetime.now():%H:%M:%S}] All runs complete in {total_time/60:.1f} minutes")
    print_comparison(all_results)

    # Save summary
    summary = {
        'total_time_seconds': total_time,
        'results': all_results,
    }
    if all_results:
        winner = min(all_results, key=lambda x: x['best_nll'])
        summary['winner'] = winner['config']
        summary['winner_nll'] = winner['best_nll']
        summary['winner_step'] = winner['best_step']

    summary_file = RESULTS_DIR / 'math_sft_sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")

    if all_results:
        winner = min(all_results, key=lambda x: x['best_nll'])
        print(f"\nTo evaluate the winner, run:")
        print(f"  python3.13 {SCRIPTS_DIR}/eval_math_sft_winner.py --winner {winner['config']}")


if __name__ == '__main__':
    main()
