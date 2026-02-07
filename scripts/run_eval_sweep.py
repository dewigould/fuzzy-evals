"""
Full evaluation sweep for 5 models across 4 domains.
Models: base, math_rlvr, code_rlvr, math_sft, code_sft (all post-formatting-SFT)
Domains: MATH-500, BigCodeBench-500, philosophy (10Qx10S), weird_questions (46Qx10S)
"""

import asyncio
import concurrent.futures
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

RESULTS_DIR = Path('/workspace/results_06_02_v2')
EVAL_DIR = RESULTS_DIR / 'eval'
EVAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / 'experiment_log.md'

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
JUDGE_MODEL = 'openai/gpt-5.2'
BASE_MODEL_OR = 'meta-llama/llama-3.1-8b-instruct'  # OpenRouter model name

N_FUZZY_SAMPLES = 10

MODEL_KEYS = ['base', 'math_rlvr', 'code_rlvr', 'math_sft', 'code_sft',
               'base_seeded', 'math_sft_seeded', 'code_sft_seeded']
# Map seeded model keys to their base sampling client
SEEDED_MODELS = {'base_seeded': 'base', 'math_sft_seeded': 'math_sft', 'code_sft_seeded': 'code_sft'}
MODEL_DISPLAY = {
    'base': 'Base',
    'math_rlvr': 'Math RLVR',
    'code_rlvr': 'Code RLVR',
    'math_sft': 'Math SFT',
    'code_sft': 'Code SFT',
    'base_seeded': 'Base (seeded)',
    'math_sft_seeded': 'Math SFT (seeded)',
    'code_sft_seeded': 'Code SFT (seeded)',
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / 'eval_runner.log'),
    ]
)
log = logging.getLogger(__name__)


def log_experiment(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n### {ts}\n{msg}\n")
    log.info(msg)


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
    except:
        return None, {}


# ── Tinker sampling ─────────────────────────────────────────────────────────

async def sample_tinker(sampling_client, renderer, prompt, num_samples=1,
                        max_tokens=2048, temperature=0.7, think_prefix=False,
                        tokenizer=None):
    import tinker
    messages = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(messages)

    if think_prefix and tokenizer is not None:
        # Append <think>\n tokens to force model into reasoning mode
        prefix_tokens = tokenizer.encode("<think>\n", add_special_tokens=False)
        for token_id in prefix_tokens:
            model_input.append_int(token_id)

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
        content = parsed_msg['content']
        if think_prefix:
            # Prepend the <think> tag since it was in the prefix (not in response tokens)
            content = "<think>\n" + content
        results.append(content)
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
    import subprocess as _sp
    full_code = code + "\n\n" + test_code
    runner_code = (
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
    fd, tmppath = tempfile.mkstemp(suffix='.py')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(runner_code)
        result = _sp.run(
            [sys.executable, tmppath],
            capture_output=True, text=True, timeout=timeout,
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
    except _sp.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"subprocess error: {str(e)[:200]}"
    finally:
        os.unlink(tmppath)


# ── Generate completions ────────────────────────────────────────────────────

async def generate_completion(session, semaphore, model_key, prompt,
                              sampling_client=None, renderer=None,
                              num_samples=1, temperature=0.7, max_tokens=2048,
                              tokenizer=None):
    """Generate completion(s) from a model. All models use Tinker (post-formatting-SFT)."""
    is_seeded = model_key in SEEDED_MODELS
    return await sample_tinker(sampling_client, renderer, prompt,
                                num_samples=num_samples, max_tokens=max_tokens,
                                temperature=temperature, think_prefix=is_seeded,
                                tokenizer=tokenizer)


# ── Per-domain evaluation ───────────────────────────────────────────────────

async def eval_math(session, semaphore, model_key, math_ds,
                    sampling_client=None, renderer=None, tokenizer=None):
    """Evaluate on MATH-500 with parallel Tinker API calls."""
    log_experiment(f"Evaluating {model_key} on MATH-500 (500 questions)")

    async def _eval_one(i):
        prompt = (f"Solve the following math problem. Show your work step by step and "
                  f"put your final answer in \\boxed{{}}.\n\n{math_ds[i]['problem']}")
        async with semaphore:
            try:
                completions = await sample_tinker(
                    sampling_client, renderer, prompt, max_tokens=2048,
                    think_prefix=(model_key in SEEDED_MODELS), tokenizer=tokenizer)
                completion = completions[0]
            except Exception as e:
                completion = f"ERROR: {e}"
        correct, predicted = (False, "error") if completion.startswith("ERROR:") else grade_math_answer(completion, math_ds[i]['answer'])
        return {
            'model': model_key, 'dataset': 'math', 'question_id': i,
            'question': math_ds[i]['problem'][:200],
            'raw_output': completion, 'correct': correct,
        }

    # Run all questions in parallel (semaphore limits concurrency)
    tasks = [_eval_one(i) for i in range(len(math_ds))]
    results = await asyncio.gather(*tasks)
    results = list(results)
    n_correct = sum(1 for r in results if r['correct'])
    log_experiment(f"  {model_key} math: {n_correct}/500 ({n_correct/5:.1f}%)")
    return results


async def eval_code(session, semaphore, model_key, code_ds,
                    sampling_client=None, renderer=None, tokenizer=None):
    """Evaluate on BigCodeBench-500 with parallel Tinker API calls."""
    log_experiment(f"Evaluating {model_key} on BigCodeBench (500 questions)")

    # Generate completions in parallel
    async def _gen_one(i):
        prompt = code_ds[i]['instruct_prompt']
        async with semaphore:
            try:
                results = await sample_tinker(
                    sampling_client, renderer, prompt, max_tokens=2048,
                    think_prefix=(model_key in SEEDED_MODELS), tokenizer=tokenizer)
                return results[0]
            except Exception as e:
                return f"ERROR: {e}"

    gen_tasks = [_gen_one(i) for i in range(len(code_ds))]
    completions = list(await asyncio.gather(*gen_tasks))
    log.info(f"  {model_key} code generation: {len(completions)}/500 done")

    # Run unit tests in parallel
    log.info(f"  {model_key} code: running unit tests...")
    test_results = {}
    non_error_jobs = []
    for i, completion in enumerate(completions):
        if not completion.startswith("ERROR:"):
            code = extract_code_from_response(completion)
            non_error_jobs.append((i, code, code_ds[i]['test'], code_ds[i]['entry_point']))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
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
            passed, details, code = test_results[i]
            results.append({
                'model': model_key, 'dataset': 'code', 'question_id': i,
                'raw_output': completion, 'code_extracted': code[:500],
                'passed': passed, 'test_details': details,
            })

    n_passed = sum(1 for r in results if r['passed'])
    log_experiment(f"  {model_key} code: {n_passed}/500 ({n_passed/5:.1f}%)")
    return results


async def eval_fuzzy(session, semaphore, model_key,
                     sampling_client=None, renderer=None, tokenizer=None):
    """Evaluate on philosophy + weird questions with N=10 samples per question."""
    with open('/workspace/fuzzy-evals/dataset_jsons/philosophy_questions.json') as f:
        phil_data = json.load(f)
    with open('/workspace/fuzzy-evals/dataset_jsons/weird_questions.json') as f:
        weird_data = json.load(f)
    phil_rubric = Path('/workspace/fuzzy-evals/rubrics/philosophy_rubric.md').read_text()
    weird_rubric = Path('/workspace/fuzzy-evals/rubrics/weird_questions_rubric.md').read_text()

    all_results = []

    for ds_name, questions, rubric, q_key in [
        ('philosophy', phil_data, phil_rubric, 'question'),
        ('weird_questions', weird_data, weird_rubric, 'prompt'),
    ]:
        n_total = len(questions) * N_FUZZY_SAMPLES
        log_experiment(f"Evaluating {model_key} on {ds_name}: {len(questions)}Q x {N_FUZZY_SAMPLES}S = {n_total}")

        # Generate N samples per question (parallel)
        async def _gen_fuzzy(q_idx, q):
            prompt = f"Answer the following question.\n\n{q[q_key]}"
            async with semaphore:
                try:
                    samples = await sample_tinker(
                        sampling_client, renderer, prompt,
                        num_samples=N_FUZZY_SAMPLES, temperature=0.7,
                        think_prefix=(model_key in SEEDED_MODELS), tokenizer=tokenizer)
                    return list(samples)
                except Exception as e:
                    return [f"ERROR: {e}"] * N_FUZZY_SAMPLES

        fuzzy_tasks = [_gen_fuzzy(q_idx, q) for q_idx, q in enumerate(questions)]
        all_completions = list(await asyncio.gather(*fuzzy_tasks))

        # Judge all completions
        if ds_name == 'philosophy':
            sys_msg = "You are grading a philosophical answer for quality of argumentation and exposition."
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

        # Batch judge calls
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
            score, sub_scores = parse_judge_score(judge_resp) if not judge_resp.startswith("ERROR:") else (None, {})
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

def generate_comparison_plot(df):
    """Generate 8-model grouped bar chart (5 normal + 3 seeded)."""
    datasets_list = ['math', 'code', 'philosophy', 'weird_questions']
    dataset_labels = ['Math\n(MATH-500)', 'Code\n(BigCodeBench)', 'Philosophy\n(10Q x 10S)', 'Weird Questions\n(46Q x 10S)']
    # Solid colors for normal, hatched lighter variants for seeded
    colors = {
        'base': '#4ECDC4', 'math_rlvr': '#FF6B6B', 'code_rlvr': '#45B7D1',
        'math_sft': '#96CEB4', 'code_sft': '#FFEAA7',
        'base_seeded': '#4ECDC4', 'math_sft_seeded': '#96CEB4', 'code_sft_seeded': '#FFEAA7',
    }
    hatches = {k: '//' if k in SEEDED_MODELS else '' for k in MODEL_KEYS}

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(datasets_list))
    n_models = len(MODEL_KEYS)
    width = 0.10
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)

    for model_idx, model_key in enumerate(MODEL_KEYS):
        means = []
        ses = []
        for ds in datasets_list:
            model_df = df[(df['dataset'] == ds) & (df['model'] == model_key)]
            if len(model_df) == 0:
                means.append(0); ses.append(0); continue

            if ds in ['math', 'code']:
                col = 'correct' if ds == 'math' else 'passed'
                acc = model_df[col].mean() * 100
                se = np.sqrt(acc * (100 - acc) / len(model_df))
                means.append(acc); ses.append(se)
            else:
                valid = model_df[model_df['score'].notna()]
                if len(valid) == 0:
                    means.append(0); ses.append(0); continue
                q_means = valid.groupby('question_id')['score'].mean()
                means.append(q_means.mean())
                ses.append(q_means.std() / np.sqrt(len(q_means)))

        bars = ax.bar(x + offsets[model_idx], means, width, yerr=ses,
                      label=MODEL_DISPLAY[model_key], color=colors[model_key],
                      hatch=hatches[model_key],
                      capsize=2, edgecolor='black', linewidth=0.3)

        for i, (m, se) in enumerate(zip(means, ses)):
            if m > 0:
                suffix = '%' if datasets_list[i] in ['math', 'code'] else ''
                ax.text(x[i] + offsets[model_idx], m + se + 0.3, f'{m:.1f}{suffix}',
                        ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xlabel('Evaluation Domain', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('RLVR vs SFT: Math & Code Training Transfer (+ <think> Seeded Inference)\n'
                 'Math/Code: Accuracy (%) | Fuzzy: Mean Rubric Score (0-48)\n'
                 'Hatched bars = seeded with <think> prefix', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS_DIR / 'plots' / 'comparison_plot.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log_experiment(f"Saved plot: {plot_path}")


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    import tinker
    from tinker_cookbook import renderers, model_info, checkpoint_utils
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    log_experiment("# Evaluation Sweep: 5 models x 4 domains")

    # Load eval datasets
    from datasets import load_dataset
    math_ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    code_ds = load_dataset('bigcode/bigcodebench', split='v0.1.4').select(range(500))
    log_experiment(f"Loaded MATH-500 ({len(math_ds)}) and BigCodeBench ({len(code_ds)})")

    # Set up renderer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Set up Tinker sampling clients for format-SFT'd models
    service_client = tinker.ServiceClient()
    sampling_clients = {}
    available_models = []
    # Load sampling clients for non-seeded models
    base_model_keys = [k for k in MODEL_KEYS if k not in SEEDED_MODELS]
    for model_key in base_model_keys:
        if model_key == 'base':
            fmt_path = RESULTS_DIR / 'training' / 'format_sft' / 'base'
        else:
            fmt_path = RESULTS_DIR / 'training' / 'format_sft' / model_key

        ckpt_file = fmt_path / 'checkpoints.jsonl'
        if not ckpt_file.exists():
            log_experiment(f"  {model_key}: SKIPPED (formatting SFT not yet done)")
            continue

        ckpt = checkpoint_utils.get_last_checkpoint(str(fmt_path))
        sampler_path = ckpt['sampler_path']
        sampling_clients[model_key] = service_client.create_sampling_client(model_path=sampler_path)
        available_models.append(model_key)
        log_experiment(f"  {model_key}: loaded from {sampler_path}")

    # Seeded models reuse the same sampling client as their base
    for seeded_key, base_key in SEEDED_MODELS.items():
        if base_key in sampling_clients:
            sampling_clients[seeded_key] = sampling_clients[base_key]
            available_models.append(seeded_key)
            log_experiment(f"  {seeded_key}: using same client as {base_key} (with <think> prefix)")
        else:
            log_experiment(f"  {seeded_key}: SKIPPED (base model not available)")

    eval_model_keys = [k for k in MODEL_KEYS if k in available_models]
    log_experiment(f"Evaluating models: {eval_model_keys}")

    semaphore = asyncio.Semaphore(15)
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
        for model_key in eval_model_keys:
            sc = sampling_clients[model_key]

            # Math
            if (model_key, 'math') not in completed:
                results = await eval_math(session, semaphore, model_key, math_ds,
                                          sampling_client=sc, renderer=renderer,
                                          tokenizer=tokenizer)
                all_results.extend(results)
                pd.DataFrame(all_results).to_parquet(partial_path, index=False)
            else:
                log_experiment(f"  {model_key} math: SKIPPED (done)")

            # Code
            if (model_key, 'code') not in completed:
                results = await eval_code(session, semaphore, model_key, code_ds,
                                          sampling_client=sc, renderer=renderer,
                                          tokenizer=tokenizer)
                all_results.extend(results)
                pd.DataFrame(all_results).to_parquet(partial_path, index=False)
            else:
                log_experiment(f"  {model_key} code: SKIPPED (done)")

            # Fuzzy (philosophy + weird_questions)
            fuzzy_done = (model_key, 'philosophy') in completed and (model_key, 'weird_questions') in completed
            if not fuzzy_done:
                results = await eval_fuzzy(session, semaphore, model_key,
                                           sampling_client=sc, renderer=renderer,
                                           tokenizer=tokenizer)
                all_results.extend(results)
                pd.DataFrame(all_results).to_parquet(partial_path, index=False)
            else:
                log_experiment(f"  {model_key} fuzzy: SKIPPED (done)")

    # Save final results
    df = pd.DataFrame(all_results)
    df.to_parquet(EVAL_DIR / 'eval_scores.parquet', index=False)

    # Generate plot
    generate_comparison_plot(df)

    # Summary
    log_experiment("## Summary")
    for model_key in MODEL_KEYS:
        model_df = df[df['model'] == model_key]
        summary_parts = []
        for ds in ['math', 'code', 'philosophy', 'weird_questions']:
            ds_df = model_df[model_df['dataset'] == ds]
            if len(ds_df) == 0:
                continue
            if ds == 'math':
                acc = ds_df['correct'].mean() * 100
                summary_parts.append(f"math={acc:.1f}%")
            elif ds == 'code':
                acc = ds_df['passed'].mean() * 100
                summary_parts.append(f"code={acc:.1f}%")
            else:
                valid = ds_df[ds_df['score'].notna()]
                if len(valid) > 0:
                    q_means = valid.groupby('question_id')['score'].mean()
                    summary_parts.append(f"{ds}={q_means.mean():.1f}±{q_means.std()/np.sqrt(len(q_means)):.1f}")
        log_experiment(f"  {model_key}: {' | '.join(summary_parts)}")

    log_experiment("# Evaluation sweep complete")


if __name__ == '__main__':
    asyncio.run(main())
