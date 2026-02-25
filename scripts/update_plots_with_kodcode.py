"""
Regenerate comparison plots with KodCode-200 bar added.
Reads all existing results and adds the new KodCode-200 benchmark.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path('/workspace/results_10_02')
EVAL_DIR = RESULTS_DIR / 'code_rlvr_eval'

# Load all existing data
with open(RESULTS_DIR / 'humaneval_results.json') as f:
    humaneval = json.load(f)
with open(RESULTS_DIR / 'fuzzy_base.json') as f:
    fuzzy_base = json.load(f)
with open(RESULTS_DIR / 'fuzzy_best_rlvr.json') as f:
    fuzzy_math_rlvr = json.load(f)
with open(EVAL_DIR / 'fuzzy_code_rlvr.json') as f:
    fuzzy_code_rlvr = json.load(f)

# Code RLVR eval results
with open(EVAL_DIR / 'math_results.json') as f:
    code_rlvr_math = json.load(f)

# KodCode 3-model results
with open(EVAL_DIR / 'kodcode_200_3model.json') as f:
    kodcode = json.load(f)

# Known values
math_base_acc = 40.8
math_rlvr_acc = 48.0
code_rlvr_math_acc = sum(1 for r in code_rlvr_math if r['correct']) / len(code_rlvr_math) * 100  # 38.6

he_n = humaneval['n_problems']
he_base_rate = humaneval['base']['pass@1']
he_math_rlvr_rate = humaneval['best_rlvr']['pass@1']

# HumanEval code RLVR
with open(EVAL_DIR / 'humaneval_results.json') as f:
    he_code_rlvr = json.load(f)
he_code_rlvr_rate = he_code_rlvr['code_rlvr']['pass@1']

# KodCode values
kc_base = kodcode['base']['accuracy'] * 100
kc_math = kodcode['math_rlvr']['accuracy'] * 100
kc_code = kodcode['code_rlvr']['accuracy'] * 100
kc_n = kodcode['n_problems']

# ── Plot 1: final_comparison.png — two subplots side by side ─────────────────
colors = {'base': '#4ECDC4', 'math_rlvr': '#FF6B6B', 'code_rlvr': '#6B8EFF'}
model_keys = ['base', 'math_rlvr', 'code_rlvr']
model_display = {
    'base': 'Base (Llama-3.1-8B-Instruct)',
    'math_rlvr': 'Math RLVR (D_baseline_v2)',
    'code_rlvr': 'Code RLVR (winner)',
}

# Left panel: accuracy benchmarks
acc_datasets = ['math', 'code', 'kodcode']
acc_labels = ['MATH-500', 'HumanEval\n(164)', 'KodCode\n(200 held-out)']

# Right panel: fuzzy benchmarks
fuzzy_datasets = ['philosophy', 'weird_questions', 'futuristic_tech']
fuzzy_labels = ['Philosophy\n(10Q x 10S)', 'Weird Questions\n(46Q x 10S)', 'Futuristic Tech\n(10Q x 10S)']

fig, (ax_acc, ax_fuzzy) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 3]})
n_models = len(model_keys)
width = 0.25
offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

# ── Left panel: Accuracy (%) ──
x_acc = np.arange(len(acc_datasets))
for model_idx, model_key in enumerate(model_keys):
    if model_key == 'base':
        means = [math_base_acc, he_base_rate * 100, kc_base]
    elif model_key == 'math_rlvr':
        means = [math_rlvr_acc, he_math_rlvr_rate * 100, kc_math]
    else:
        means = [code_rlvr_math_acc, he_code_rlvr_rate * 100, kc_code]

    ses = [
        np.sqrt(means[0] * (100 - means[0]) / 500),
        np.sqrt(means[1] * (100 - means[1]) / he_n),
        np.sqrt(means[2] * (100 - means[2]) / kc_n),
    ]

    ax_acc.bar(
        x_acc + offsets[model_idx], means, width, yerr=ses,
        label=model_display[model_key], color=colors[model_key],
        capsize=3, edgecolor='black', linewidth=0.5,
    )
    for i, (m, se) in enumerate(zip(means, ses)):
        if m > 0:
            ax_acc.text(
                x_acc[i] + offsets[model_idx], m + se + 0.8, f'{m:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
ax_acc.set_title('Benchmark Accuracy (pass@1)', fontsize=13)
ax_acc.set_xticks(x_acc)
ax_acc.set_xticklabels(acc_labels)
ax_acc.set_ylim(0, 75)
ax_acc.legend(fontsize=9, loc='upper left')
ax_acc.grid(axis='y', alpha=0.3)

# ── Right panel: Fuzzy rubric scores ──
x_fuzzy = np.arange(len(fuzzy_datasets))
for model_idx, model_key in enumerate(model_keys):
    if model_key == 'base':
        fuzzy = fuzzy_base
    elif model_key == 'math_rlvr':
        fuzzy = fuzzy_math_rlvr
    else:
        fuzzy = fuzzy_code_rlvr

    means = [fuzzy.get(ds, {}).get('mean', 0) for ds in fuzzy_datasets]
    ses = []
    for ds_name in fuzzy_datasets:
        ds_fuzzy = fuzzy.get(ds_name, {})
        scores = ds_fuzzy.get('scores', [])
        if len(scores) >= 10:
            n_q = len(scores) // 10
            q_means = [np.mean(scores[j * 10:(j + 1) * 10]) for j in range(n_q)]
            ses.append(np.std(q_means) / np.sqrt(len(q_means)) if len(q_means) > 1 else 0)
        else:
            ses.append(0)

    ax_fuzzy.bar(
        x_fuzzy + offsets[model_idx], means, width, yerr=ses,
        label=model_display[model_key], color=colors[model_key],
        capsize=3, edgecolor='black', linewidth=0.5,
    )
    for i, (m, se) in enumerate(zip(means, ses)):
        if m > 0:
            ax_fuzzy.text(
                x_fuzzy[i] + offsets[model_idx], m + se + 0.1, f'{m:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

ax_fuzzy.set_ylabel('Mean Rubric Score', fontsize=12)
ax_fuzzy.set_title('Fuzzy Eval (GPT-judged, 1-20 scale)', fontsize=13)
ax_fuzzy.set_xticks(x_fuzzy)
ax_fuzzy.set_xticklabels(fuzzy_labels)
ax_fuzzy.set_ylim(0, 16)
ax_fuzzy.legend(fontsize=9, loc='upper right')
ax_fuzzy.grid(axis='y', alpha=0.3)

fig.suptitle('Base vs Math RLVR vs Code RLVR', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plot_path = EVAL_DIR / 'final_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# ── Plot 2: Response length with KodCode ──────────────────────────────────
with open(RESULTS_DIR / 'raw_outputs_for_length.json') as f:
    length_data = json.load(f)
with open(RESULTS_DIR / 'humaneval_completions_base.json') as f:
    he_base_completions = json.load(f)
with open(RESULTS_DIR / 'humaneval_completions_best_rlvr.json') as f:
    he_math_rlvr_completions = json.load(f)
with open(EVAL_DIR / 'fuzzy_completions.json') as f:
    code_rlvr_fuzzy_comps = json.load(f)

# Code RLVR HumanEval + math completions
he_code_rlvr_comps = []
try:
    with open(EVAL_DIR / 'humaneval_completions.json') as f:
        he_code_rlvr_comps = json.load(f)
except:
    pass

math_code_rlvr_comps = []
try:
    with open(EVAL_DIR / 'math_completions.json') as f:
        math_code_rlvr_comps = json.load(f)
except:
    pass

# KodCode completions
with open(EVAL_DIR / 'kodcode_200_completions.json') as f:
    kc_completions = json.load(f)

dataset_order = ['math', 'code', 'kodcode', 'philosophy', 'weird_questions', 'futuristic_tech']
dataset_labels_len = ['MATH-500', 'HumanEval-164', 'KodCode-200', 'Philosophy', 'Weird Questions', 'Futuristic Tech']

fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(dataset_order))

for model_idx, model_key in enumerate(model_keys):
    means = []
    ses = []
    for ds in dataset_order:
        if model_key == 'base':
            if ds == 'math':
                lengths = [len(s) for s in length_data.get('base_math', [])]
            elif ds == 'code':
                lengths = [len(s) for s in he_base_completions]
            elif ds == 'kodcode':
                lengths = [len(c['text']) for c in kc_completions['base']]
            else:
                lengths = [len(s) for s in length_data.get(f'base_{ds}', [])]
        elif model_key == 'math_rlvr':
            if ds == 'math':
                lengths = [len(s) for s in length_data.get('rlvr_math', [])]
            elif ds == 'code':
                lengths = [len(s) for s in he_math_rlvr_completions]
            elif ds == 'kodcode':
                lengths = [len(c['text']) for c in kc_completions['math_rlvr']]
            else:
                lengths = [len(s) for s in length_data.get(f'rlvr_{ds}', [])]
        else:  # code_rlvr
            if ds == 'math':
                lengths = [len(s) for s in math_code_rlvr_comps] if math_code_rlvr_comps else [0]
            elif ds == 'code':
                lengths = [len(s) for s in he_code_rlvr_comps] if he_code_rlvr_comps else [0]
            elif ds == 'kodcode':
                lengths = [len(c['text']) for c in kc_completions['code_rlvr']]
            else:
                comps = code_rlvr_fuzzy_comps.get(ds, [])
                lengths = [len(s) for s in comps] if comps else [0]

        if not lengths:
            lengths = [0]
        m = np.mean(lengths)
        se = np.std(lengths) / np.sqrt(len(lengths)) if len(lengths) > 1 else 0
        means.append(m)
        ses.append(se)

    ax.bar(
        x + offsets[model_idx], means, width, yerr=ses,
        label=model_display[model_key], color=colors[model_key],
        capsize=3, edgecolor='black', linewidth=0.5,
    )
    for i, (m, se) in enumerate(zip(means, ses)):
        if m > 0:
            ax.text(
                x[i] + offsets[model_idx], m + se + 20, f'{m:.0f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
            )

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Average Response Length (characters)', fontsize=12)
ax.set_title('Response Length Comparison: Base vs Math RLVR vs Code RLVR', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels_len)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plot_path = EVAL_DIR / 'response_length_comparison.png'
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved: {plot_path}")

# Print summary
print(f"\n{'='*60}")
print("UPDATED COMPARISON (with KodCode-200)")
print(f"{'='*60}")
print(f"{'Benchmark':<20} {'Base':>8} {'Math RLVR':>10} {'Code RLVR':>10}")
print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*10}")
print(f"{'MATH-500':<20} {math_base_acc:>7.1f}% {math_rlvr_acc:>9.1f}% {code_rlvr_math_acc:>9.1f}%")
print(f"{'HumanEval-164':<20} {he_base_rate*100:>7.1f}% {he_math_rlvr_rate*100:>9.1f}% {he_code_rlvr_rate*100:>9.1f}%")
print(f"{'KodCode-200':<20} {kc_base:>7.1f}% {kc_math:>9.1f}% {kc_code:>9.1f}%")
print(f"{'Philosophy':<20} {fuzzy_base['philosophy']['mean']:>8.1f} {fuzzy_math_rlvr['philosophy']['mean']:>10.1f} {fuzzy_code_rlvr['philosophy']['mean']:>10.1f}")
print(f"{'Weird Questions':<20} {fuzzy_base['weird_questions']['mean']:>8.1f} {fuzzy_math_rlvr['weird_questions']['mean']:>10.1f} {fuzzy_code_rlvr['weird_questions']['mean']:>10.1f}")
print(f"{'Futuristic Tech':<20} {fuzzy_base['futuristic_tech']['mean']:>8.1f} {fuzzy_math_rlvr['futuristic_tech']['mean']:>10.1f} {fuzzy_code_rlvr['futuristic_tech']['mean']:>10.1f}")
