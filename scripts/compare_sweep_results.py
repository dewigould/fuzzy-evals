"""
Compare results from the Math RLVR hyperparameter sweep.
Reads metrics.jsonl from each config and prints a formatted comparison table.

Usage: python compare_sweep_results.py [--results_dir /workspace/results_10_02]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_metrics(config_dir: Path):
    """Parse metrics.jsonl and extract test accuracy + entropy at each eval step."""
    metrics_file = config_dir / 'metrics.jsonl'
    if not metrics_file.exists():
        return None

    eval_points = []
    all_steps = []
    with open(metrics_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = d.get('progress/batch', -1)
            entry = {'step': step}

            # Always record entropy
            if 'optim/entropy' in d:
                entry['entropy'] = d['optim/entropy']

            # Record train accuracy
            if 'env/all/correct' in d:
                entry['train_correct'] = d['env/all/correct']

            all_steps.append(entry)

            # Record test eval when available
            if 'test/env/all/correct' in d:
                eval_points.append({
                    'step': step,
                    'test_accuracy': d['test/env/all/correct'],
                    'entropy': d.get('optim/entropy'),
                    'train_correct': d.get('env/all/correct'),
                    'test_format': d.get('test/env/all/format'),
                    'test_reward': d.get('test/env/all/reward/total'),
                })

    if not eval_points:
        return None

    peak = max(eval_points, key=lambda x: x['test_accuracy'])
    total_steps = max(e['step'] for e in all_steps) if all_steps else 0

    return {
        'config': config_dir.name,
        'eval_points': eval_points,
        'peak_accuracy': peak['test_accuracy'],
        'peak_step': peak['step'],
        'final_accuracy': eval_points[-1]['test_accuracy'],
        'final_entropy': eval_points[-1].get('entropy'),
        'step0_accuracy': eval_points[0]['test_accuracy'] if eval_points else None,
        'step0_entropy': eval_points[0].get('entropy') if eval_points else None,
        'total_steps': total_steps,
    }


def load_config_params(config_dir: Path):
    """Try to load config parameters from config.json if it exists."""
    config_file = config_dir / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Compare RLVR sweep results')
    parser.add_argument('--results_dir', type=str, default='/workspace/results_10_02',
                        help='Results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return

    # Find all config directories (those with metrics.jsonl)
    config_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / 'metrics.jsonl').exists()
    ])

    if not config_dirs:
        print(f"No completed runs found in {results_dir}")
        return

    # Parse all results
    all_results = []
    for config_dir in config_dirs:
        result = parse_metrics(config_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("No eval metrics found in any run")
        return

    # Sort by peak accuracy (descending)
    all_results.sort(key=lambda x: -x['peak_accuracy'])

    # Print main comparison table
    print(f"\n{'='*100}")
    print(f"MATH RLVR SWEEP COMPARISON ({len(all_results)} configs)")
    print(f"{'='*100}")
    print(f"{'Config':<20} {'Step0':>7} {'Step25':>7} {'Step50':>7} {'Step75':>7} "
          f"{'Peak':>7} {'@Step':>6} {'Delta':>8} {'Ent(0)':>7} {'Ent(F)':>7}")
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")

    for r in all_results:
        step_vals = {ep['step']: ep['test_accuracy'] for ep in r['eval_points']}
        step_ent = {ep['step']: ep.get('entropy') for ep in r['eval_points']}

        s0 = step_vals.get(0)
        s25 = step_vals.get(25)
        s50 = step_vals.get(50)
        s75 = step_vals.get(75)

        delta = (r['peak_accuracy'] - s0) if s0 is not None else None
        ent0 = r.get('step0_entropy')
        entf = r.get('final_entropy')

        def fmt_acc(v):
            return f"{v*100:.1f}%" if v is not None else "  -  "

        def fmt_ent(v):
            return f"{v:.3f}" if v is not None else "  -  "

        delta_str = f"+{delta*100:.1f}pp" if delta is not None else "   -   "

        print(f"{r['config']:<20} {fmt_acc(s0):>7} {fmt_acc(s25):>7} {fmt_acc(s50):>7} {fmt_acc(s75):>7} "
              f"{fmt_acc(r['peak_accuracy']):>7} {'@'+str(r['peak_step']):>6} {delta_str:>8} "
              f"{fmt_ent(ent0):>7} {fmt_ent(entf):>7}")

    # Print entropy trajectory table
    print(f"\n{'='*100}")
    print(f"ENTROPY TRAJECTORIES")
    print(f"{'='*100}")
    print(f"{'Config':<20}", end='')
    # Collect all eval steps across configs
    all_eval_steps = sorted(set(
        ep['step'] for r in all_results for ep in r['eval_points']
    ))
    for step in all_eval_steps:
        print(f" Step{step:>3}", end='')
    print()
    print(f"{'-'*20}", end='')
    for _ in all_eval_steps:
        print(f" {'-'*7}", end='')
    print()

    for r in all_results:
        step_ent = {ep['step']: ep.get('entropy') for ep in r['eval_points']}
        print(f"{r['config']:<20}", end='')
        for step in all_eval_steps:
            e = step_ent.get(step)
            print(f" {e:7.3f}" if e is not None else "     -  ", end='')
        print()

    # Winner summary
    winner = all_results[0]  # Already sorted by peak accuracy
    print(f"\n{'='*100}")
    print(f"WINNER: {winner['config']}")
    print(f"  Peak accuracy: {winner['peak_accuracy']*100:.1f}% at step {winner['peak_step']}")
    s0 = winner.get('step0_accuracy')
    if s0 is not None:
        print(f"  Delta from base: +{(winner['peak_accuracy'] - s0)*100:.1f}pp")
        print(f"  Base (step 0): {s0*100:.1f}%")
    ent0 = winner.get('step0_entropy')
    entf = winner.get('final_entropy')
    if ent0 is not None and entf is not None:
        print(f"  Entropy: {ent0:.3f} -> {entf:.3f} ({(entf-ent0)/ent0*100:+.0f}%)")
    print(f"{'='*100}")

    # Print eval command
    cal_file = results_dir / 'calibration.json'
    max_tokens_str = ""
    if cal_file.exists():
        with open(cal_file) as f:
            cal = json.load(f)
        max_tokens_str = f" --max_tokens {cal['recommended_max_tokens']}"
    print(f"\nTo evaluate the winner on BigCodeBench + fuzzy tasks:")
    print(f"  python3 /workspace/fuzzy-evals/scripts/eval_sweep_winner.py "
          f"--winner {winner['config']}{max_tokens_str}")


if __name__ == '__main__':
    main()
