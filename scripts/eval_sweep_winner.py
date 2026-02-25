"""
Evaluate the sweep winner + base model on BigCodeBench and fuzzy tasks.
Generates a comparison plot: base vs best RLVR across all domains.

Usage:
  python eval_sweep_winner.py --winner A_moderate --max_tokens 2048
"""

import argparse
import asyncio
import concurrent.futures
import gc
import json
import logging
import os
import re
import signal
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

import aiohttp
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

RESULTS_DIR = Path('/workspace/results_10_02')
EVAL_DIR = RESULTS_DIR / 'final_eval'
LOG_FILE = RESULTS_DIR / 'final_eval_log.md'

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


# ── Code grading ────────────────────────────────────────────────────────────

def extract_code_from_response(response_text):
    if '```python' in response_text:
        blocks = re.findall(r'```python\s*\n(.*?)```', response_text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
    if '```' in response_text:
        blocks = re.findall(r'```\s*\n(.*?)```', response_text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
    lines = response_text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(('import ', 'from ', 'def ', 'class ')) or in_code:
            in_code = True
            code_lines.append(line)
        elif in_code and line.strip() == '':
            code_lines.append(line)
        elif in_code and not line.strip().startswith('#') and line.strip() and not line[0].isspace():
            break
    if code_lines:
        return '\n'.join(code_lines).strip()
    return response_text.strip()


def run_test_in_subprocess(code, test_code, entry_point, timeout=30):
    """Run test in subprocess with resource limits to prevent OOM."""
    import subprocess as _sp
    import resource as _resource
    full_code = code + "\n\n" + test_code
    resource_wrapper = (
        "import resource, signal\n"
        "resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
        f"signal.alarm({timeout})\n"
    )
    runner_code = (
        resource_wrapper +
        "import unittest, sys, os, json\n"
        "sys.stderr = open(os.devnull, 'w')\n"
        + full_code + "\n"
        "test_class = None\n"
        "for _name, _obj in list(globals().items()):\n"
        "    if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase) and _obj is not unittest.TestCase:\n"
        "        test_class = _obj\n"
        "        break\n"
        "if test_class is None:\n"
        "    print(json.dumps({'error': 'TestCases not found'}))\n"
        "    sys.exit(0)\n"
        "loader = unittest.TestLoader()\n"
        "suite = loader.loadTestsFromTestCase(test_class)\n"
        "runner = unittest.TextTestRunner(stream=open(os.devnull, 'w'), verbosity=0)\n"
        "result = runner.run(suite)\n"
        "print(json.dumps({'passed': result.wasSuccessful(), "
        "'total': result.testsRun, "
        "'failures': len(result.failures) + len(result.errors)}))\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = os.path.join(tmpdir, 'run.py')
        with open(tmppath, 'w') as f:
            f.write(runner_code)
        try:
            import subprocess as _sp
            result = _sp.run(
                [sys.executable, tmppath],
                capture_output=True, text=True, timeout=timeout + 5,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                return False, f"exit {result.returncode}: {result.stderr[:200]}"
            try:
                data = json.loads(result.stdout.strip().split('\n')[-1])
                if 'error' in data:
                    return False, data['error']
                return data['passed'], f"{data['total']-data['failures']}/{data['total']} tests passed"
            except Exception:
                return False, f"parse error: {result.stdout[:200]}"
        except Exception as e:
            if 'TimeoutExpired' in type(e).__name__:
                return False, "timeout"
            return False, f"subprocess error: {str(e)[:200]}"


# ── Evaluation functions ───────────────────────────────────────────────────

async def eval_math(semaphore, model_key, math_ds, sampling_client, renderer, max_tokens):
    """Evaluate on MATH-500 using the same prompt format as training (1-shot strawberry prefix)."""
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv
    fewshot_prefix = MathEnv.standard_fewshot_prefix()
    question_suffix = MathEnv.question_suffix()
    log_experiment(f"Evaluating {model_key} on MATH-500 ({len(math_ds)} questions, max_tokens={max_tokens})")

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
            'model': model_key, 'dataset': 'math', 'question_id': i,
            'question': math_ds[i]['problem'][:200],
            'raw_output': completion, 'correct': correct,
        }

    tasks = [_eval_one(i) for i in range(len(math_ds))]
    results = list(await asyncio.gather(*tasks))
    n_correct = sum(1 for r in results if r['correct'])
    log_experiment(f"  {model_key} math: {n_correct}/{len(math_ds)} ({n_correct/len(math_ds)*100:.1f}%)")
    return results


async def eval_code(semaphore, model_key, code_ds, sampling_client, renderer, max_tokens):
    """Evaluate on BigCodeBench-500."""
    log_experiment(f"Evaluating {model_key} on BigCodeBench ({len(code_ds)} questions, max_tokens={max_tokens})")

    async def _gen_one(i):
        prompt = code_ds[i]['instruct_prompt']
        async with semaphore:
            try:
                results = await sample_tinker(
                    sampling_client, renderer, prompt, max_tokens=max_tokens)
                return results[0]
            except Exception as e:
                return f"ERROR: {e}"

    gen_tasks = [_gen_one(i) for i in range(len(code_ds))]
    completions = list(await asyncio.gather(*gen_tasks))
    log.info(f"  {model_key} code generation: {len(completions)}/{len(code_ds)} done")

    # Run unit tests in parallel with resource limits
    log.info(f"  {model_key} code: running unit tests...")
    test_results = {}
    non_error_jobs = []
    for i, completion in enumerate(completions):
        if not completion.startswith("ERROR:"):
            code = extract_code_from_response(completion)
            non_error_jobs.append((i, code, code_ds[i]['test'], code_ds[i]['entry_point']))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for idx, code, test_code, ep in non_error_jobs:
            future = executor.submit(run_test_in_subprocess, code, test_code, ep)
            futures[future] = (idx, code)
        for future in concurrent.futures.as_completed(futures):
            idx, code = futures[future]
            try:
                passed, details = future.result(timeout=120)
            except Exception as e:
                passed, details = False, f"executor error: {str(e)[:100]}"
            test_results[idx] = (passed, details, code)

    results = []
    for i, completion in enumerate(completions):
        if completion.startswith("ERROR:"):
            results.append({
                'model': model_key, 'dataset': 'code', 'question_id': i,
                'raw_output': completion, 'passed': False, 'test_details': 'API error',
            })
        else:
            passed, details, code = test_results.get(i, (False, 'not tested', ''))
            results.append({
                'model': model_key, 'dataset': 'code', 'question_id': i,
                'raw_output': completion, 'code_extracted': code[:500],
                'passed': passed, 'test_details': details,
            })

    n_passed = sum(1 for r in results if r['passed'])
    log_experiment(f"  {model_key} code: {n_passed}/{len(code_ds)} ({n_passed/len(code_ds)*100:.1f}%)")
    return results


async def eval_fuzzy(session, semaphore, model_key, sampling_client, renderer):
    """Evaluate on philosophy + weird questions + futuristic tech."""
    with open('/workspace/fuzzy-evals/dataset_jsons/philosophy_questions.json') as f:
        phil_data = json.load(f)
    with open('/workspace/fuzzy-evals/dataset_jsons/weird_questions.json') as f:
        weird_data = json.load(f)
    with open('/workspace/fuzzy-evals/dataset_jsons/futuristic_tech_questions.json') as f:
        futuristic_data = json.load(f)
    phil_rubric = Path('/workspace/fuzzy-evals/rubrics/philosophy_rubric.md').read_text()
    weird_rubric = Path('/workspace/fuzzy-evals/rubrics/weird_questions_rubric.md').read_text()
    futuristic_rubric = Path('/workspace/fuzzy-evals/rubrics/futuristic_tech_rubric.md').read_text()

    all_results = []

    for ds_name, questions, rubric, q_key in [
        ('philosophy', phil_data, phil_rubric, 'question'),
        ('weird_questions', weird_data, weird_rubric, 'prompt'),
        ('futuristic_tech', futuristic_data, futuristic_rubric, 'question'),
    ]:
        n_total = len(questions) * N_FUZZY_SAMPLES
        log_experiment(f"Evaluating {model_key} on {ds_name}: {len(questions)}Q x {N_FUZZY_SAMPLES}S = {n_total}")

        async def _gen_fuzzy(q_idx, q):
            prompt = f"Answer the following question.\n\n{q[q_key]}"
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
        all_completions = list(await asyncio.gather(*fuzzy_tasks))

        if ds_name == 'philosophy':
            sys_msg = "You are grading a philosophical answer for quality of argumentation and exposition."
        elif ds_name == 'futuristic_tech':
            sys_msg = "You are grading an answer for quality of thought, scientific rigour, and effort."
        else:
            sys_msg = "You are grading an answer for quality of thought, research, effort, and exposition."

        judge_tasks = []
        judge_map = []
        for q_idx, q in enumerate(questions):
            for s_idx, completion in enumerate(all_completions[q_idx]):
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
                log.info(f"  {model_key} {ds_name} judging: {done}/{len(judge_tasks)}")

        for (q_idx, s_idx), judge_resp in zip(judge_map, judge_responses):
            q = questions[q_idx]
            completion = all_completions[q_idx][s_idx]
            score, sub_scores = parse_judge_score(judge_resp) if not str(judge_resp).startswith("ERROR:") else (None, {})
            result = {
                'model': model_key, 'dataset': ds_name,
                'question_id': q_idx, 'sample_id': s_idx,
                'question': q[q_key][:200],
                'raw_output': completion, 'judge_response': judge_resp,
                'score': score,
            }
            result.update(sub_scores)
            all_results.append(result)

        valid_scores = [r['score'] for r in all_results
                       if r['dataset'] == ds_name and r['model'] == model_key and r['score'] is not None]
        if valid_scores:
            log_experiment(f"  {model_key} {ds_name}: mean={np.mean(valid_scores):.2f} (n={len(valid_scores)})")

    return all_results


# ── Plotting ────────────────────────────────────────────────────────────────

def generate_comparison_plot(df, model_keys, model_display):
    """Generate base vs best_rlvr comparison bar chart."""
    datasets_list = ['math', 'code', 'philosophy', 'weird_questions', 'futuristic_tech']
    dataset_labels = ['Math\n(MATH-500)', 'Code\n(BigCodeBench)', 'Philosophy\n(10Q x 10S)',
                      'Weird Questions\n(46Q x 10S)', 'Futuristic Tech\n(10Q x 10S)']
    colors = {'base': '#4ECDC4', 'best_rlvr': '#FF6B6B'}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets_list))
    n_models = len(model_keys)
    width = 0.30
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)

    for model_idx, model_key in enumerate(model_keys):
        means = []
        ses = []
        for ds in datasets_list:
            model_df = df[(df['dataset'] == ds) & (df['model'] == model_key)]
            if len(model_df) == 0:
                means.append(0); ses.append(0); continue

            if ds in ['math', 'code']:
                col = 'correct' if ds == 'math' else 'passed'
                acc = model_df[col].mean() * 100
                se = np.sqrt(acc * (100 - acc) / len(model_df)) if len(model_df) > 0 else 0
                means.append(acc); ses.append(se)
            else:
                valid = model_df[model_df['score'].notna()]
                if len(valid) == 0:
                    means.append(0); ses.append(0); continue
                q_means = valid.groupby('question_id')['score'].mean()
                means.append(q_means.mean())
                ses.append(q_means.std() / np.sqrt(len(q_means)) if len(q_means) > 1 else 0)

        bars = ax.bar(x + offsets[model_idx], means, width, yerr=ses,
                      label=model_display[model_key], color=colors.get(model_key, '#999999'),
                      capsize=3, edgecolor='black', linewidth=0.5)

        for i, (m, se) in enumerate(zip(means, ses)):
            if m > 0:
                suffix = '%' if datasets_list[i] in ['math', 'code'] else ''
                ax.text(x[i] + offsets[model_idx], m + se + 0.5, f'{m:.1f}{suffix}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Evaluation Domain', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Base vs Best RLVR Model Comparison\n'
                 'Math/Code: Accuracy (%) | Fuzzy: Mean Rubric Score',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plot_path = EVAL_DIR / 'base_vs_rlvr_comparison.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log_experiment(f"Saved plot: {plot_path}")


# ── Main ────────────────────────────────────────────────────────────────────

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

    # Find the checkpoint path for this step
    ckpt_file = config_dir / 'checkpoints.jsonl'
    if not ckpt_file.exists():
        return None

    # Find closest checkpoint at or after the best step
    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Look for exact match or closest checkpoint
    best_ckpt = None
    for ckpt in checkpoints:
        ckpt_step = ckpt.get('batch', -1)
        if ckpt_step == best_step and 'sampler_path' in ckpt:
            best_ckpt = ckpt
            break

    # If no exact match, find closest checkpoint with sampler_path
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


async def main():
    import tinker
    from tinker_cookbook import renderers, model_info
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    parser = argparse.ArgumentParser(description='Evaluate sweep winner + base')
    parser.add_argument('--winner', type=str, required=True,
                        help='Winning config name (e.g., A_moderate)')
    parser.add_argument('--max_tokens', type=int, required=True,
                        help='Max tokens for code eval (same as training)')
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    winner_dir = RESULTS_DIR / args.winner
    if not winner_dir.exists():
        print(f"ERROR: Winner directory {winner_dir} does not exist")
        sys.exit(1)

    # Find best checkpoint
    best_ckpt = find_best_checkpoint(winner_dir)
    if best_ckpt is None:
        print(f"ERROR: Could not find best checkpoint for {args.winner}")
        sys.exit(1)

    log_experiment(f"# Final Evaluation: Base vs {args.winner}")
    log_experiment(f"Best RLVR checkpoint: step {best_ckpt['step']} "
                   f"(train eval acc: {best_ckpt['accuracy']*100:.1f}%, "
                   f"checkpoint at step {best_ckpt['checkpoint_step']})")
    log_experiment(f"Code eval max_tokens: {args.max_tokens}")
    log_experiment(f"Fuzzy eval max_tokens: {FUZZY_MAX_TOKENS}")

    # Load eval datasets
    from datasets import load_dataset
    math_ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    code_ds = load_dataset('bigcode/bigcodebench', split='v0.1.4').select(range(500))
    log_experiment(f"Loaded MATH-500 ({len(math_ds)}) and BigCodeBench ({len(code_ds)})")

    # Set up renderer
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create sampling clients
    service_client = tinker.ServiceClient()

    # Base model
    base_client = service_client.create_sampling_client(base_model=MODEL_NAME)
    log_experiment(f"  base: loaded from {MODEL_NAME}")

    # Best RLVR model
    rlvr_client = service_client.create_sampling_client(model_path=best_ckpt['sampler_path'])
    log_experiment(f"  best_rlvr: loaded from {best_ckpt['sampler_path']}")

    model_keys = ['base', 'best_rlvr']
    model_display = {'base': 'Base (Llama-3.1-8B-Instruct)', 'best_rlvr': f'Best RLVR ({args.winner})'}
    clients = {'base': base_client, 'best_rlvr': rlvr_client}

    semaphore = asyncio.Semaphore(10)
    all_results = []

    # Check for partial results
    partial_path = EVAL_DIR / 'eval_partial.parquet'
    completed = set()
    if partial_path.exists():
        df_partial = pd.read_parquet(partial_path)
        for (model, ds) in df_partial.groupby(['model', 'dataset']).size().index:
            completed.add((model, ds))
        all_results = df_partial.to_dict('records')
        log_experiment(f"Resuming: {len(all_results)} existing results. Completed: {completed}")

    async with aiohttp.ClientSession() as session:
        for model_key in model_keys:
            sc = clients[model_key]

            # MATH-500
            if (model_key, 'math') not in completed:
                results = await eval_math(semaphore, model_key, math_ds, sc, renderer,
                                          max_tokens=args.max_tokens)
                all_results.extend(results)
                pd.DataFrame(all_results).to_parquet(partial_path, index=False)
                check_memory()
                gc.collect()
            else:
                log_experiment(f"  {model_key} math: SKIPPED (done)")

            # BigCodeBench
            if (model_key, 'code') not in completed:
                results = await eval_code(semaphore, model_key, code_ds, sc, renderer,
                                          max_tokens=args.max_tokens)
                all_results.extend(results)
                pd.DataFrame(all_results).to_parquet(partial_path, index=False)
                check_memory()
                gc.collect()
            else:
                log_experiment(f"  {model_key} code: SKIPPED (done)")

            # Fuzzy tasks
            fuzzy_done = all(
                (model_key, ds) in completed
                for ds in ['philosophy', 'weird_questions', 'futuristic_tech']
            )
            if not fuzzy_done:
                results = await eval_fuzzy(session, semaphore, model_key, sc, renderer)
                all_results.extend(results)
                pd.DataFrame(all_results).to_parquet(partial_path, index=False)
                check_memory()
                gc.collect()
            else:
                log_experiment(f"  {model_key} fuzzy: SKIPPED (done)")

    # Save final results
    df = pd.DataFrame(all_results)
    df.to_parquet(EVAL_DIR / 'eval_scores.parquet', index=False)

    # Generate plot
    generate_comparison_plot(df, model_keys, model_display)

    # Print summary
    log_experiment("\n## Final Summary")
    log_experiment(f"{'Model':<30} {'Math':>8} {'Code':>8} {'Phil':>8} {'Weird':>8} {'FuturTech':>10}")
    log_experiment(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for model_key in model_keys:
        model_df = df[df['model'] == model_key]
        parts = []
        for ds in ['math', 'code', 'philosophy', 'weird_questions', 'futuristic_tech']:
            ds_df = model_df[model_df['dataset'] == ds]
            if len(ds_df) == 0:
                parts.append('   -   ')
                continue
            if ds == 'math':
                acc = ds_df['correct'].mean() * 100
                parts.append(f"{acc:6.1f}%")
            elif ds == 'code':
                acc = ds_df['passed'].mean() * 100
                parts.append(f"{acc:6.1f}%")
            else:
                valid = ds_df[ds_df['score'].notna()]
                if len(valid) > 0:
                    q_means = valid.groupby('question_id')['score'].mean()
                    parts.append(f"{q_means.mean():7.1f}")
                else:
                    parts.append('   -   ')

        display = model_display[model_key]
        log_experiment(f"{display:<30} {'  '.join(parts)}")

    # Print deltas
    log_experiment("\n## Deltas (RLVR - Base)")
    for ds in ['math', 'code', 'philosophy', 'weird_questions', 'futuristic_tech']:
        base_df = df[(df['model'] == 'base') & (df['dataset'] == ds)]
        rlvr_df = df[(df['model'] == 'best_rlvr') & (df['dataset'] == ds)]
        if len(base_df) == 0 or len(rlvr_df) == 0:
            continue
        if ds in ['math', 'code']:
            col = 'correct' if ds == 'math' else 'passed'
            base_val = base_df[col].mean() * 100
            rlvr_val = rlvr_df[col].mean() * 100
            delta = rlvr_val - base_val
            log_experiment(f"  {ds}: {base_val:.1f}% -> {rlvr_val:.1f}% ({delta:+.1f}pp)")
        else:
            base_valid = base_df[base_df['score'].notna()]
            rlvr_valid = rlvr_df[rlvr_df['score'].notna()]
            if len(base_valid) > 0 and len(rlvr_valid) > 0:
                base_val = base_valid.groupby('question_id')['score'].mean().mean()
                rlvr_val = rlvr_valid.groupby('question_id')['score'].mean().mean()
                delta = rlvr_val - base_val
                log_experiment(f"  {ds}: {base_val:.2f} -> {rlvr_val:.2f} ({delta:+.2f})")

    log_experiment("\n# Final evaluation complete")
    log_experiment(f"Results saved to {EVAL_DIR}")


if __name__ == '__main__':
    asyncio.run(main())
