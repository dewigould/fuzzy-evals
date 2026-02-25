"""
Code RLVR Pipeline — wave-based sweep + eval + plots.

Runs 6 configs in 2 waves of 3 to avoid GPU contention,
then evaluates the winner and generates plots.

All output logged to /workspace/results_10_02/pipeline.log
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = Path('/workspace/results_10_02')
CALIBRATION_FILE = RESULTS_DIR / 'code_calibration.json'
LOG_FILE = RESULTS_DIR / 'pipeline.log'
PYTHON = '/usr/bin/python3.13'

WAVE_1 = ['A_moderate', 'B_conservative', 'C_aggressive']
WAVE_2 = ['D_baseline_v2', 'E_ppo_moderate', 'F_large_group']
ALL_CONFIGS = WAVE_1 + WAVE_2


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def parse_metrics(config_name):
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


def run_wave(wave_name, configs, max_tokens):
    """Run a wave of configs in parallel and wait for completion."""
    log(f"=== {wave_name}: Launching {len(configs)} configs ===")
    processes = {}
    log_files = {}

    for config_name in configs:
        log_dir = RESULTS_DIR / f'code_{config_name}'
        log_dir.mkdir(parents=True, exist_ok=True)
        lf_path = log_dir / 'sweep_runner.log'
        cmd = [
            PYTHON, str(SCRIPTS_DIR / 'run_code_rlvr_single.py'),
            config_name, '--max_tokens', str(max_tokens),
        ]
        lf = open(lf_path, 'w')
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        processes[config_name] = proc
        log_files[config_name] = lf
        log(f"  Launched code_{config_name} (PID {proc.pid})")

    # Monitor
    completed = set()
    wave_start = time.time()
    results = []

    while len(completed) < len(configs):
        time.sleep(30)
        elapsed = time.time() - wave_start

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
                        results.append(result)
                        log(f"  code_{config_name} COMPLETED (peak {result['peak_accuracy']*100:.1f}% at step {result['peak_step']})")
                    else:
                        log(f"  code_{config_name} COMPLETED (no eval metrics)")
                else:
                    log(f"  code_{config_name} FAILED (exit {ret})")

        # Progress every 5 min
        if int(elapsed) % 300 < 35:
            running = [c for c in configs if c not in completed]
            if running:
                # Check training progress from logs
                progress_info = []
                for c in running:
                    log_path = RESULTS_DIR / f'code_{c}' / 'sweep_runner.log'
                    try:
                        with open(log_path) as f:
                            lines = f.readlines()
                        # Find last "Sampling batch N" line
                        for line in reversed(lines):
                            if 'Sampling batch' in line or 'run_evals' in line or 'save_checkpoint' in line:
                                progress_info.append(f"{c}: {line.strip()[-80:]}")
                                break
                        else:
                            progress_info.append(f"{c}: starting up")
                    except:
                        progress_info.append(f"{c}: no log")
                log(f"  {wave_name} {elapsed/60:.0f}min | Running: {', '.join(running)}")
                for info in progress_info:
                    log(f"    {info}")

    wave_time = time.time() - wave_start
    log(f"  {wave_name} complete in {wave_time/60:.1f}min")
    return results


def print_comparison(all_results):
    log(f"\n{'='*90}")
    log(f"CODE RLVR SWEEP COMPARISON TABLE")
    log(f"{'='*90}")
    header = f"{'Config':<20} {'Peak':>7} {'@Step':>6} {'Step0':>7}"
    log(header)
    log(f"{'-'*20} {'-'*7} {'-'*6} {'-'*7}")
    for r in sorted(all_results, key=lambda x: -x['peak_accuracy']):
        s0 = r.get('step0_accuracy')
        s0_str = f"{s0*100:.1f}%" if s0 is not None else "?"
        log(f"{r['config']:<20} {r['peak_accuracy']*100:.1f}%  @{r['peak_step']:<5} {s0_str:>7}")
    if all_results:
        winner = max(all_results, key=lambda x: x['peak_accuracy'])
        log(f"\nWINNER: {winner['config']} (peak {winner['peak_accuracy']*100:.1f}% at step {winner['peak_step']})")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()

    log("=" * 70)
    log("CODE RLVR PIPELINE — Wave-based sweep + eval + plots")
    log("=" * 70)

    # Load calibration
    if not CALIBRATION_FILE.exists():
        log("ERROR: code_calibration.json not found. Run calibrate_code_max_tokens.py first.")
        sys.exit(1)
    with open(CALIBRATION_FILE) as f:
        max_tokens = json.load(f)['recommended_max_tokens']
    log(f"Using cached calibration: max_tokens={max_tokens}")

    # Wave 1
    results_1 = run_wave("Wave 1", WAVE_1, max_tokens)

    # Wave 2
    results_2 = run_wave("Wave 2", WAVE_2, max_tokens)

    all_results = results_1 + results_2

    # Summary
    print_comparison(all_results)
    sweep_time = time.time() - pipeline_start

    # Save sweep summary
    summary_file = RESULTS_DIR / 'code_sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'max_tokens': max_tokens,
            'total_time_seconds': sweep_time,
            'results': all_results,
        }, f, indent=2)
    log(f"Saved sweep summary to {summary_file}")

    if not all_results:
        log("ERROR: No configs completed successfully. Aborting.")
        sys.exit(1)

    # Find winner and run eval
    winner = max(all_results, key=lambda x: x['peak_accuracy'])
    winner_name = winner['config']
    log(f"\n{'='*70}")
    log(f"PHASE 2: Evaluating winner code_{winner_name}")
    log(f"{'='*70}")

    eval_cmd = [
        PYTHON, str(SCRIPTS_DIR / 'eval_code_rlvr_winner.py'),
        '--winner', winner_name,
    ]
    eval_log = RESULTS_DIR / 'code_rlvr_eval' / 'eval_runner.log'
    eval_log.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_log, 'w') as lf:
        result = subprocess.run(
            eval_cmd, stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )

    if result.returncode == 0:
        log("Evaluation completed successfully!")
    else:
        log(f"Evaluation failed with exit code {result.returncode}")
        log(f"Check log: {eval_log}")

    total_time = time.time() - pipeline_start
    log(f"\n{'='*70}")
    log(f"PIPELINE COMPLETE in {total_time/3600:.1f} hours ({total_time/60:.0f} min)")
    log(f"{'='*70}")
    log(f"Results: {RESULTS_DIR / 'code_rlvr_eval'}/")
    log(f"Plots: {RESULTS_DIR / 'code_rlvr_eval'}/final_comparison.png")
    log(f"Sweep summary: {summary_file}")


if __name__ == '__main__':
    main()
