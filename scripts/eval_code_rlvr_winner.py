"""
Evaluate the Code RLVR sweep winner on all benchmarks.
Only evaluates the code RLVR model (base + math RLVR data already exist).

Generates 3-bar comparison plots: Base vs Math RLVR vs Code RLVR.

Usage:
  python eval_code_rlvr_winner.py --winner D_baseline_v2
"""

import argparse
import asyncio
import concurrent.futures
import gc
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')
sys.path.insert(0, '/workspace/human-eval')

RESULTS_DIR = Path('/workspace/results_10_02')
EVAL_DIR = RESULTS_DIR / 'code_rlvr_eval'
LOG_FILE = EVAL_DIR / 'eval_log.md'

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
JUDGE_MODEL = 'openai/gpt-5.2'
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

N_FUZZY_SAMPLES = 10
FUZZY_MAX_TOKENS = 8192

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def log_experiment(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n### {ts}\n{msg}\n")
    log.info(msg)


def check_memory():
    try:
        with open('/sys/fs/cgroup/memory.current') as f:
            current = int(f.read().strip())
        with open('/sys/fs/cgroup/memory.max') as f:
            limit = int(f.read().strip())
        pct = current / limit * 100
        if pct > 80:
            log.warning(f"HIGH MEMORY: {pct:.1f}% ({current/1024**3:.1f}GB / {limit/1024**3:.1f}GB)")
            gc.collect()
        return pct
    except Exception:
        return 0


# ── API helpers ─────────────────────────────────────────────────────────────

async def call_openrouter(session, model, messages, temperature=0.7,
                          max_tokens=4096, semaphore=None):
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
    }
    body = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    retry_delays = [2, 5, 15, 30, 60, 120]
    for delay in retry_delays + [120]:
        try:
            if semaphore:
                async with semaphore:
                    async with session.post(OPENROUTER_URL, headers=headers,
                                           json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        data = await resp.json()
            else:
                async with session.post(OPENROUTER_URL, headers=headers,
                                       json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    data = await resp.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            elif 'error' in data:
                err = data['error']
                err_msg = err.get('message', str(err)) if isinstance(err, dict) else str(err)
                if 'rate' in err_msg.lower() or '429' in err_msg:
                    await asyncio.sleep(delay)
                    continue
                return f"ERROR: {err_msg}"
            return "ERROR: unexpected response"
        except asyncio.TimeoutError:
            await asyncio.sleep(delay)
        except Exception:
            await asyncio.sleep(delay)
    return "ERROR: max retries exceeded"


def parse_judge_score(response_text):
    try:
        text = response_text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            text = text[start:end]
        scores = json.loads(text)
        total = scores.get('total', None)
        if total is not None:
            return int(total), scores
        total = sum(v for v in scores.values() if isinstance(v, (int, float)))
        return total, scores
    except Exception:
        return None, {}


# ── Tinker sampling ─────────────────────────────────────────────────────────

async def sample_tinker(sampling_client, renderer, prompt, num_samples=1,
                        max_tokens=2048, temperature=0.7, messages=None):
    import tinker
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(messages)
    response = await sampling_client.sample_async(
        model_input,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
        ),
    )
    results = []
    for seq in response.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        results.append(parsed_msg['content'])
    return results


# ── Math grading ────────────────────────────────────────────────────────────

def grade_math_answer(model_output, ground_truth):
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
    try:
        predicted = extract_boxed(model_output)
    except (ValueError, Exception):
        return False, "no_boxed"
    try:
        correct = grade_answer(predicted, ground_truth)
        return correct, predicted
    except Exception:
        return False, predicted


# ── HumanEval helpers (reused from run_humaneval_eval.py) ──────────────────

HUMANEVAL_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a function signature "
    "with a docstring describing the task. Write the complete function implementation.\n\n"
    "Return ONLY the completed function inside a single ```python code block. "
    "Do not include tests, examples, or explanation outside the code block."
)

RELIABILITY_GUARD_CODE = r'''
import faulthandler
faulthandler.disable()

import builtins
builtins.exit = None
builtins.quit = None

import os
os.environ["OMP_NUM_THREADS"] = "1"
_disabled_os = [
    "kill", "system", "putenv", "remove", "removedirs", "rmdir",
    "fchdir", "setuid", "fork", "forkpty", "killpg", "rename", "renames",
    "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
    "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir",
]
for _attr in _disabled_os:
    if hasattr(os, _attr):
        setattr(os, _attr, None)

import shutil
shutil.rmtree = None
shutil.move = None
shutil.chown = None

import subprocess as _sp
_sp.Popen = None

if isinstance(__builtins__, dict):
    __builtins__["help"] = None
else:
    __builtins__.help = None

import sys as _sys
for _mod in ["ipdb", "joblib", "resource", "psutil", "tkinter"]:
    _sys.modules[_mod] = None
'''


def humaneval_extract_code(response: str, prompt: str, entry_point: str):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if blocks:
        code = blocks[-1].strip()
    else:
        code = response.strip()
    if not code:
        return None
    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code):
        return code
    return prompt + code


def humaneval_run_test(code, test, entry_point, timeout=10):
    check_program = (
        RELIABILITY_GUARD_CODE
        + "\n\n# ── Generated code ──\n"
        + code
        + "\n\n# ── HumanEval test suite ──\n"
        + test
        + "\n"
        + f"check({entry_point})\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "test.py")
        with open(script_path, "w") as f:
            f.write(check_program)
        try:
            result = __import__('subprocess').run(
                [sys.executable, "-u", script_path],
                capture_output=True, text=True, timeout=timeout + 2,
                cwd=tmpdir,
            )
            if result.returncode == 0:
                return True, "passed"
            else:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                detail = stderr[-500:] if stderr else stdout[-500:]
                return False, detail or f"exit code {result.returncode}"
        except __import__('subprocess').TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:200]


# ── Checkpoint finder ──────────────────────────────────────────────────────

def find_best_checkpoint(config_dir: Path):
    """Find the checkpoint step with highest test accuracy from training metrics."""
    metrics_file = config_dir / 'metrics.jsonl'
    if not metrics_file.exists():
        return None

    best_step = None
    best_acc = -1
    with open(metrics_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'test/env/all/correct' in d:
                acc = d['test/env/all/correct']
                step = d.get('progress/batch', -1)
                if acc > best_acc:
                    best_acc = acc
                    best_step = step

    if best_step is None:
        return None

    ckpt_file = config_dir / 'checkpoints.jsonl'
    if not ckpt_file.exists():
        return None

    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    best_ckpt = None
    for ckpt in checkpoints:
        ckpt_step = ckpt.get('batch', -1)
        if ckpt_step == best_step and 'sampler_path' in ckpt:
            best_ckpt = ckpt
            break

    if best_ckpt is None:
        valid_ckpts = [c for c in checkpoints if 'sampler_path' in c]
        if valid_ckpts:
            best_ckpt = min(valid_ckpts, key=lambda c: abs(c.get('batch', 0) - best_step))

    if best_ckpt is None:
        return None

    return {
        'step': best_step,
        'accuracy': best_acc,
        'sampler_path': best_ckpt['sampler_path'],
        'checkpoint_step': best_ckpt.get('batch', -1),
    }


# ── Evaluation functions ───────────────────────────────────────────────────

async def eval_math(semaphore, sampling_client, renderer, max_tokens):
    """Evaluate code RLVR model on MATH-500."""
    from datasets import load_dataset
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv

    math_ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    fewshot_prefix = MathEnv.standard_fewshot_prefix()
    question_suffix = MathEnv.question_suffix()
    log_experiment(f"Evaluating code_rlvr on MATH-500 ({len(math_ds)} questions, max_tokens={max_tokens})")

    all_completions = []

    async def _eval_one(i):
        question = math_ds[i]['problem'] + question_suffix
        messages = fewshot_prefix + [{"role": "user", "content": question}]
        async with semaphore:
            try:
                completions = await sample_tinker(
                    sampling_client, renderer, None, max_tokens=max_tokens,
                    messages=messages)
                completion = completions[0]
            except Exception as e:
                completion = f"ERROR: {e}"
        correct, predicted = (False, "error") if completion.startswith("ERROR:") else grade_math_answer(completion, math_ds[i]['answer'])
        return {
            'question_id': i,
            'question': math_ds[i]['problem'][:200],
            'raw_output': completion,
            'correct': correct,
        }

    tasks = [_eval_one(i) for i in range(len(math_ds))]
    results = list(await asyncio.gather(*tasks))
    all_completions = [r['raw_output'] for r in results]
    n_correct = sum(1 for r in results if r['correct'])
    accuracy = n_correct / len(math_ds)
    log_experiment(f"  code_rlvr MATH-500: {n_correct}/{len(math_ds)} ({accuracy*100:.1f}%)")
    return results, all_completions, accuracy


async def eval_humaneval(semaphore, sampling_client, renderer):
    """Evaluate code RLVR model on HumanEval-164."""
    from human_eval.data import read_problems, HUMAN_EVAL

    problems_dict = read_problems(HUMAN_EVAL)
    problems = []
    for task_id in sorted(problems_dict.keys(), key=lambda x: int(x.split("/")[1])):
        p = problems_dict[task_id]
        problems.append({
            "task_id": p["task_id"],
            "prompt": p["prompt"],
            "test": p["test"],
            "entry_point": p["entry_point"],
        })

    log_experiment(f"Evaluating code_rlvr on HumanEval ({len(problems)} problems)")
    done = [0]

    async def _gen_one(i):
        messages = [
            {"role": "system", "content": HUMANEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Complete the following Python function:\n\n"
                f"```python\n{problems[i]['prompt']}```"
            )},
        ]
        async with semaphore:
            try:
                completions = await sample_tinker(
                    sampling_client, renderer, None, max_tokens=4096,
                    temperature=0.7, messages=messages)
                done[0] += 1
                if done[0] % 20 == 0:
                    log.info(f"  [code_rlvr] {done[0]}/{len(problems)} generated")
                return completions[0]
            except Exception as e:
                done[0] += 1
                return f"ERROR: {e}"

    gen_tasks = [_gen_one(i) for i in range(len(problems))]
    completions = list(await asyncio.gather(*gen_tasks))

    # Test completions
    log.info(f"  [code_rlvr] Testing {len(completions)} solutions...")
    jobs = []
    for i, completion in enumerate(completions):
        if completion.startswith("ERROR:"):
            continue
        code = humaneval_extract_code(completion, problems[i]["prompt"], problems[i]["entry_point"])
        if code is None:
            continue
        jobs.append((i, code, problems[i]["test"], problems[i]["entry_point"]))

    test_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for idx, code, test, entry_point in jobs:
            future = executor.submit(humaneval_run_test, code, test, entry_point, 10)
            futures[future] = idx
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                passed, detail = future.result(timeout=60)
            except Exception as e:
                passed, detail = False, str(e)[:100]
            test_results[idx] = (passed, detail)

    details = []
    for i, completion in enumerate(completions):
        passed = False
        detail = "no code extracted"
        if i in test_results:
            passed, detail = test_results[i]
        elif completion.startswith("ERROR:"):
            detail = "generation error"
        details.append({
            "task_id": problems[i]["task_id"],
            "passed": passed,
            "detail": detail,
            "raw_output": completion,
        })

    n_passed = sum(1 for d in details if d["passed"])
    accuracy = n_passed / len(problems)
    log_experiment(f"  code_rlvr HumanEval: {n_passed}/{len(problems)} ({accuracy*100:.1f}%)")

    summary = {
        "benchmark": "HumanEval",
        "n_problems": len(problems),
        "code_rlvr": {"passed": n_passed, "total": len(problems), "pass@1": accuracy},
    }
    return summary, completions, details, accuracy


async def eval_fuzzy(session, semaphore, sampling_client, renderer):
    """Evaluate code RLVR model on philosophy + weird questions + futuristic tech."""
    with open('/workspace/fuzzy-evals/dataset_jsons/philosophy_questions.json') as f:
        phil_data = json.load(f)
    with open('/workspace/fuzzy-evals/dataset_jsons/weird_questions.json') as f:
        weird_data = json.load(f)
    with open('/workspace/fuzzy-evals/dataset_jsons/futuristic_tech_questions.json') as f:
        futuristic_data = json.load(f)
    phil_rubric = Path('/workspace/fuzzy-evals/rubrics/philosophy_rubric.md').read_text()
    weird_rubric = Path('/workspace/fuzzy-evals/rubrics/weird_questions_rubric.md').read_text()
    futuristic_rubric = Path('/workspace/fuzzy-evals/rubrics/futuristic_tech_rubric.md').read_text()

    all_results = {}
    all_completions = {}

    for ds_name, questions, rubric, q_key in [
        ('philosophy', phil_data, phil_rubric, 'question'),
        ('weird_questions', weird_data, weird_rubric, 'prompt'),
        ('futuristic_tech', futuristic_data, futuristic_rubric, 'question'),
    ]:
        n_total = len(questions) * N_FUZZY_SAMPLES
        log_experiment(f"Evaluating code_rlvr on {ds_name}: {len(questions)}Q x {N_FUZZY_SAMPLES}S = {n_total}")

        async def _gen_fuzzy(q_idx, q, _q_key=q_key):
            prompt = f"Answer the following question.\n\n{q[_q_key]}"
            async with semaphore:
                try:
                    samples = await sample_tinker(
                        sampling_client, renderer, prompt,
                        num_samples=N_FUZZY_SAMPLES, temperature=0.7,
                        max_tokens=FUZZY_MAX_TOKENS)
                    return list(samples)
                except Exception as e:
                    return [f"ERROR: {e}"] * N_FUZZY_SAMPLES

        fuzzy_tasks = [_gen_fuzzy(q_idx, q) for q_idx, q in enumerate(questions)]
        ds_completions = list(await asyncio.gather(*fuzzy_tasks))

        # Store raw completions (flattened)
        all_completions[ds_name] = [c for q_comps in ds_completions for c in q_comps]

        if ds_name == 'philosophy':
            sys_msg = "You are grading a philosophical answer for quality of argumentation and exposition."
        elif ds_name == 'futuristic_tech':
            sys_msg = "You are grading an answer for quality of thought, scientific rigour, and effort."
        else:
            sys_msg = "You are grading an answer for quality of thought, research, effort, and exposition."

        judge_tasks = []
        judge_map = []
        for q_idx, q in enumerate(questions):
            for s_idx, completion in enumerate(ds_completions[q_idx]):
                if completion.startswith("ERROR:"):
                    async def _err(c=completion):
                        return c
                    judge_tasks.append(_err())
                else:
                    grading_prompt = f"""{sys_msg}

## Question

{q[q_key]}

## Answer to Grade

{completion}

## Grading Rubric

{rubric}"""
                    messages = [{'role': 'user', 'content': grading_prompt}]
                    judge_tasks.append(call_openrouter(session, JUDGE_MODEL, messages,
                                                       temperature=0.0, max_tokens=1024,
                                                       semaphore=semaphore))
                judge_map.append((q_idx, s_idx))

        judge_responses = []
        batch_size = 30
        for i in range(0, len(judge_tasks), batch_size):
            batch = judge_tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            judge_responses.extend(batch_results)
            done = min(i + batch_size, len(judge_tasks))
            if done % 100 == 0 or done == len(judge_tasks):
                log.info(f"  code_rlvr {ds_name} judging: {done}/{len(judge_tasks)}")

        scores = []
        for (q_idx, s_idx), judge_resp in zip(judge_map, judge_responses):
            score, _ = parse_judge_score(judge_resp) if not str(judge_resp).startswith("ERROR:") else (None, {})
            if score is not None:
                scores.append(score)

        mean_score = float(np.mean(scores)) if scores else 0.0
        log_experiment(f"  code_rlvr {ds_name}: mean={mean_score:.2f} (n={len(scores)})")

        all_results[ds_name] = {
            'mean': mean_score,
            'n': len(scores),
            'scores': scores,
        }

    return all_results, all_completions


# ── Plot generation ────────────────────────────────────────────────────────

def generate_plots(code_rlvr_math_acc, code_rlvr_he_acc, code_rlvr_fuzzy,
                   code_rlvr_completions_all):
    """Generate 3-bar plots: Base vs Math RLVR vs Code RLVR.

    Reads existing base + math RLVR data from results_10_02/.
    Writes NEW plot files (code_rlvr_final_comparison.png, code_rlvr_response_length_comparison.png)
    to avoid overwriting existing math RLVR plots.
    """
    # Load existing data (READ ONLY — never overwrite these)
    with open(RESULTS_DIR / 'humaneval_results.json') as f:
        humaneval = json.load(f)
    with open(RESULTS_DIR / 'fuzzy_base.json') as f:
        fuzzy_base = json.load(f)
    with open(RESULTS_DIR / 'fuzzy_best_rlvr.json') as f:
        fuzzy_math_rlvr = json.load(f)

    # Known MATH-500 results from math RLVR experiment
    math_base_acc = 40.8
    math_rlvr_acc = 48.0

    # HumanEval results
    he_n = humaneval['n_problems']
    he_base_rate = humaneval['base']['pass@1']
    he_math_rlvr_rate = humaneval['best_rlvr']['pass@1']

    # ── Plot 1: code_rlvr_final_comparison.png ─────────────────────────────
    colors = {'base': '#4ECDC4', 'math_rlvr': '#FF6B6B', 'code_rlvr': '#6B8EFF'}
    model_keys = ['base', 'math_rlvr', 'code_rlvr']
    model_display = {
        'base': 'Base (Llama-3.1-8B-Instruct)',
        'math_rlvr': 'Math RLVR (D_baseline_v2)',
        'code_rlvr': 'Code RLVR (winner)',
    }

    datasets_list = ['math', 'code', 'philosophy', 'weird_questions', 'futuristic_tech']
    dataset_labels = [
        'MATH-500', 'HumanEval-164',
        'Philosophy\n(10Q x 10S)', 'Weird Questions\n(46Q x 10S)', 'Futuristic Tech\n(10Q x 10S)',
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets_list))
    n_models = len(model_keys)
    width = 0.25
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    for model_idx, model_key in enumerate(model_keys):
        if model_key == 'base':
            fuzzy = fuzzy_base
            math_acc = math_base_acc
            he_rate = he_base_rate
        elif model_key == 'math_rlvr':
            fuzzy = fuzzy_math_rlvr
            math_acc = math_rlvr_acc
            he_rate = he_math_rlvr_rate
        else:  # code_rlvr
            fuzzy = code_rlvr_fuzzy
            math_acc = code_rlvr_math_acc * 100  # convert from fraction
            he_rate = code_rlvr_he_acc

        means = [
            math_acc,
            he_rate * 100,
            fuzzy.get('philosophy', {}).get('mean', 0),
            fuzzy.get('weird_questions', {}).get('mean', 0),
            fuzzy.get('futuristic_tech', {}).get('mean', 0),
        ]

        # Standard errors
        ses = [
            np.sqrt(means[0] * (100 - means[0]) / 500),
            np.sqrt(means[1] * (100 - means[1]) / he_n),
        ]
        for ds_name in ['philosophy', 'weird_questions', 'futuristic_tech']:
            ds_fuzzy = fuzzy.get(ds_name, {})
            scores = ds_fuzzy.get('scores', [])
            if len(scores) >= 10:
                n_q = len(scores) // 10
                q_means = [np.mean(scores[j * 10:(j + 1) * 10]) for j in range(n_q)]
                ses.append(np.std(q_means) / np.sqrt(len(q_means)) if len(q_means) > 1 else 0)
            else:
                ses.append(0)

        ax.bar(
            x + offsets[model_idx], means, width, yerr=ses,
            label=model_display[model_key], color=colors[model_key],
            capsize=3, edgecolor='black', linewidth=0.5,
        )
        for i, (m, se) in enumerate(zip(means, ses)):
            if m > 0:
                suffix = '%' if datasets_list[i] in ['math', 'code'] else ''
                ax.text(
                    x[i] + offsets[model_idx], m + se + 0.5, f'{m:.1f}{suffix}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                )

    ax.set_xlabel('Evaluation Domain', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Base vs Math RLVR vs Code RLVR Comparison\n'
        'Math/Code: Accuracy (%) | Fuzzy: Mean Rubric Score',
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # Save to code_rlvr_eval/ to avoid overwriting existing final_comparison.png
    plot_path = EVAL_DIR / 'final_comparison.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log_experiment(f"Saved plot: {plot_path}")

    # ── Plot 2: code_rlvr_response_length_comparison.png ────────────────────
    # Load existing length data (READ ONLY)
    with open(RESULTS_DIR / 'raw_outputs_for_length.json') as f:
        length_data = json.load(f)
    with open(RESULTS_DIR / 'humaneval_completions_base.json') as f:
        he_base_completions = json.load(f)
    with open(RESULTS_DIR / 'humaneval_completions_best_rlvr.json') as f:
        he_math_rlvr_completions = json.load(f)

    dataset_order = ['math', 'code', 'philosophy', 'weird_questions', 'futuristic_tech']
    dataset_labels_len = ['MATH-500', 'HumanEval-164', 'Philosophy', 'Weird Questions', 'Futuristic Tech']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(dataset_order))

    for model_idx, model_key in enumerate(model_keys):
        means = []
        ses = []
        for ds in dataset_order:
            if model_key == 'base':
                if ds == 'math':
                    key = 'base_math'
                    lengths = [len(s) for s in length_data.get(key, [])]
                elif ds == 'code':
                    lengths = [len(s) for s in he_base_completions]
                else:
                    key = f'base_{ds}'
                    lengths = [len(s) for s in length_data.get(key, [])]
            elif model_key == 'math_rlvr':
                if ds == 'math':
                    key = 'rlvr_math'
                    lengths = [len(s) for s in length_data.get(key, [])]
                elif ds == 'code':
                    lengths = [len(s) for s in he_math_rlvr_completions]
                else:
                    key = f'rlvr_{ds}'
                    lengths = [len(s) for s in length_data.get(key, [])]
            else:  # code_rlvr
                comps = code_rlvr_completions_all.get(ds, [])
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
    # Save to code_rlvr_eval/ to avoid overwriting existing response_length_comparison.png
    plot_path = EVAL_DIR / 'response_length_comparison.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log_experiment(f"Saved plot: {plot_path}")


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    import tinker
    from tinker_cookbook import renderers, model_info
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    parser = argparse.ArgumentParser(description='Evaluate Code RLVR sweep winner')
    parser.add_argument('--winner', type=str, required=True,
                        help='Winning config name (e.g., D_baseline_v2)')
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Find best checkpoint in code_<winner>/ directory
    winner_dir = RESULTS_DIR / f'code_{args.winner}'
    if not winner_dir.exists():
        print(f"ERROR: Winner directory {winner_dir} does not exist")
        sys.exit(1)

    best_ckpt = find_best_checkpoint(winner_dir)
    if best_ckpt is None:
        print(f"ERROR: Could not find best checkpoint for code_{args.winner}")
        sys.exit(1)

    log_experiment(f"# Code RLVR Evaluation: {args.winner}")
    log_experiment(f"Best checkpoint: step {best_ckpt['step']} "
                   f"(train eval acc: {best_ckpt['accuracy']*100:.1f}%, "
                   f"checkpoint at step {best_ckpt['checkpoint_step']})")

    # Read calibrated max_tokens from code calibration
    code_cal_file = RESULTS_DIR / 'code_calibration.json'
    if code_cal_file.exists():
        with open(code_cal_file) as f:
            max_tokens = json.load(f)['recommended_max_tokens']
    else:
        max_tokens = 4096
    log_experiment(f"Using max_tokens={max_tokens} for math eval")

    # Set up renderer
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create sampling client for code RLVR model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=best_ckpt['sampler_path'])
    log_experiment(f"Loaded code RLVR model from {best_ckpt['sampler_path']}")

    semaphore = asyncio.Semaphore(10)

    # Phase 1: MATH-500 eval
    log_experiment("Phase 1: MATH-500 evaluation")
    math_results, math_completions, math_accuracy = await eval_math(
        semaphore, sampling_client, renderer, max_tokens)
    with open(EVAL_DIR / 'math_results.json', 'w') as f:
        json.dump(math_results, f, indent=2)
    with open(EVAL_DIR / 'math_completions.json', 'w') as f:
        json.dump(math_completions, f)
    check_memory()
    gc.collect()

    # Phase 2: HumanEval-164 eval
    log_experiment("Phase 2: HumanEval-164 evaluation")
    he_summary, he_completions, he_details, he_accuracy = await eval_humaneval(
        semaphore, sampling_client, renderer)
    with open(EVAL_DIR / 'humaneval_results.json', 'w') as f:
        json.dump(he_summary, f, indent=2)
    with open(EVAL_DIR / 'humaneval_completions.json', 'w') as f:
        json.dump(he_completions, f)
    with open(EVAL_DIR / 'humaneval_details.json', 'w') as f:
        json.dump(he_details, f, indent=2)
    check_memory()
    gc.collect()

    # Phase 3: Fuzzy eval
    log_experiment("Phase 3: Fuzzy evaluation (3 datasets)")
    async with aiohttp.ClientSession() as session:
        fuzzy_results, fuzzy_completions = await eval_fuzzy(
            session, semaphore, sampling_client, renderer)
    with open(EVAL_DIR / 'fuzzy_code_rlvr.json', 'w') as f:
        json.dump(fuzzy_results, f, indent=2)
    with open(EVAL_DIR / 'fuzzy_completions.json', 'w') as f:
        json.dump(fuzzy_completions, f)
    check_memory()
    gc.collect()

    # Phase 4: Generate plots
    log_experiment("Phase 4: Generating comparison plots")
    # Collect all completions for length analysis
    all_completions = {
        'math': math_completions,
        'code': he_completions,
    }
    all_completions.update(fuzzy_completions)  # adds philosophy, weird_questions, futuristic_tech

    generate_plots(math_accuracy, he_accuracy, fuzzy_results, all_completions)

    # Final summary
    log_experiment("\n## Final Summary")
    log_experiment(f"  MATH-500:       {math_accuracy*100:.1f}%")
    log_experiment(f"  HumanEval-164:  {he_accuracy*100:.1f}%")
    for ds_name in ['philosophy', 'weird_questions', 'futuristic_tech']:
        ds = fuzzy_results.get(ds_name, {})
        log_experiment(f"  {ds_name}: mean={ds.get('mean', 0):.2f} (n={ds.get('n', 0)})")

    log_experiment(f"\nAll outputs saved to {EVAL_DIR}")
    log_experiment("# Code RLVR evaluation complete")


if __name__ == '__main__':
    asyncio.run(main())
