"""
Evaluate the Math SFT sweep winner on all benchmarks.

Pre-seeds generation with <think>\\n to trigger R1-style reasoning.
Evaluates on MATH-500, KodCode-200, HumanEval-164, and fuzzy evals.
Generates 4-bar comparison plots: Base vs Math RLVR vs Code RLVR vs Math SFT.

Usage:
  python eval_math_sft_winner.py --winner B_medium
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
MATH_SFT_DIR = RESULTS_DIR / 'math_sft'
EVAL_DIR = RESULTS_DIR / 'math_sft_eval'
LOG_FILE = EVAL_DIR / 'eval_log.md'

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
JUDGE_MODEL = 'openai/gpt-5.2'
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

N_FUZZY_SAMPLES = 10
FUZZY_MAX_TOKENS = 4096

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


PROGRESS_FILE = EVAL_DIR / 'progress.log'


def progress(msg: str):
    """Print with flush and write to file for reliable monitoring."""
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    # Also write directly to progress file for monitoring
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()

_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def log_experiment(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n### {ts}\n{msg}\n")
    progress(msg)


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


# ── Tinker sampling (with <think> seeding) ──────────────────────────────────

async def sample_tinker(sampling_client, renderer, prompt, tokenizer,
                        num_samples=1, max_tokens=8192, temperature=0.0,
                        think_prefix=True, messages=None):
    """Sample from model, optionally seeding with <think>\\n prefix."""
    import tinker
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(messages)

    if think_prefix and tokenizer is not None:
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


# ── Code helpers ────────────────────────────────────────────────────────────

def extract_code_from_response(response_text):
    """Extract code, handling <think>...</think> wrapper."""
    text = response_text
    # If response has </think>, extract content after it
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()

    if '```python' in text:
        blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
    if '```' in text:
        blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
    return None


# ── HumanEval helpers ──────────────────────────────────────────────────────

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
    text = response
    # Handle <think> wrapper
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()

    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        code = blocks[-1].strip()
    else:
        code = text.strip()
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
            import subprocess as _sp
            result = _sp.run(
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
        except _sp.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:200]


# ── KodCode helpers ────────────────────────────────────────────────────────

from run_code_rlvr import extract_code, load_kodcode_tasks, run_kodcode_tests_sync


# ── Checkpoint finder ──────────────────────────────────────────────────────

def find_best_sft_checkpoint(config_dir: Path):
    """Find the checkpoint with lowest test NLL from SFT training metrics."""
    metrics_file = config_dir / 'metrics.jsonl'
    if not metrics_file.exists():
        return None

    best_step = None
    best_nll = float('inf')
    with open(metrics_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'test/nll' in d:
                nll = d['test/nll']
                step = d.get('step', d.get('progress/batch', -1))
                if nll < best_nll:
                    best_nll = nll
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

    # Find exact match
    best_ckpt = None
    for ckpt in checkpoints:
        ckpt_step = ckpt.get('batch', -1)
        if ckpt_step == best_step and 'sampler_path' in ckpt:
            best_ckpt = ckpt
            break

    # Fallback: closest checkpoint
    if best_ckpt is None:
        valid_ckpts = [c for c in checkpoints if 'sampler_path' in c]
        if valid_ckpts:
            best_ckpt = min(valid_ckpts, key=lambda c: abs(c.get('batch', 0) - best_step))

    if best_ckpt is None:
        return None

    return {
        'step': best_step,
        'nll': best_nll,
        'sampler_path': best_ckpt['sampler_path'],
        'checkpoint_step': best_ckpt.get('batch', -1),
    }


# ── Evaluation functions ───────────────────────────────────────────────────

async def eval_math(semaphore, sampling_client, renderer, tokenizer):
    """Evaluate Math SFT model on MATH-500 with <think> seeding."""
    from datasets import load_dataset

    math_ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    total = len(math_ds)
    log_experiment(f"Evaluating math_sft on MATH-500 ({total} questions, max_tokens=8192)")

    counter = {'done': 0, 'correct': 0}
    t0 = time.time()

    async def _eval_one(i):
        prompt = (f"Solve the following math problem. Show your work step by step and "
                  f"put your final answer in \\boxed{{}}.\n\n{math_ds[i]['problem']}")
        async with semaphore:
            try:
                completions = await sample_tinker(
                    sampling_client, renderer, prompt, tokenizer,
                    max_tokens=8192, temperature=0.0, think_prefix=True)
                completion = completions[0]
            except Exception as e:
                completion = f"ERROR: {e}"
        correct, predicted = (False, "error") if completion.startswith("ERROR:") else grade_math_answer(completion, math_ds[i]['answer'])
        counter['done'] += 1
        if correct:
            counter['correct'] += 1
        if counter['done'] % 25 == 0 or counter['done'] == total:
            elapsed = time.time() - t0
            progress(f"  MATH-500: {counter['done']}/{total} done, "
                     f"{counter['correct']} correct ({counter['correct']/counter['done']*100:.1f}%) "
                     f"[{elapsed:.0f}s]")
        return {
            'question_id': i,
            'question': math_ds[i]['problem'][:200],
            'raw_output': completion,
            'correct': correct,
        }

    tasks = [_eval_one(i) for i in range(total)]
    results = list(await asyncio.gather(*tasks))
    all_completions = [r['raw_output'] for r in results]
    n_correct = sum(1 for r in results if r['correct'])
    accuracy = n_correct / total
    log_experiment(f"  math_sft MATH-500 FINAL: {n_correct}/{total} ({accuracy*100:.1f}%)")
    return results, all_completions, accuracy


async def eval_kodcode(semaphore, sampling_client, renderer, tokenizer):
    """Evaluate Math SFT model on 200 held-out KodCode easy problems with <think> seeding."""
    N_TEST = 200
    all_tasks = load_kodcode_tasks(difficulty="easy", max_tasks=7000, seed=42)
    test_problems = all_tasks[-N_TEST:]
    log_experiment(f"Evaluating math_sft on KodCode-200 ({len(test_problems)} problems)")

    loop = asyncio.get_event_loop()
    results = []
    completions = []
    n_correct = 0

    async def eval_one(idx, problem):
        nonlocal n_correct
        prompt = (
            "You are an expert Python programmer. Solve the following problem.\n\n"
            + problem.question
            + "\n\nProvide your solution in a ```python code block."
        )
        async with semaphore:
            try:
                resp = await sample_tinker(
                    sampling_client, renderer, prompt, tokenizer,
                    max_tokens=4096, temperature=0.0, think_prefix=True)
                text = resp[0]
            except Exception as e:
                text = f"ERROR: {e}"

        code = extract_code(text.split('</think>', 1)[-1] if '</think>' in text else text)
        passed = False
        if code is not None:
            passed = await loop.run_in_executor(
                _test_executor,
                run_kodcode_tests_sync, code, problem.test_code,
            )

        if passed:
            n_correct += 1

        results.append({'idx': idx, 'correct': passed})
        completions.append({'idx': idx, 'text': text, 'correct': passed})

    t0 = time.time()
    for batch_start in range(0, len(test_problems), 20):
        batch_end = min(batch_start + 20, len(test_problems))
        await asyncio.gather(*[eval_one(i, test_problems[i]) for i in range(batch_start, batch_end)])
        done = len(results)
        correct_so_far = sum(1 for r in results if r['correct'])
        elapsed = time.time() - t0
        progress(f"  KodCode: {done}/{len(test_problems)} done, "
                 f"{correct_so_far} correct ({correct_so_far/max(done,1)*100:.1f}%) "
                 f"[{elapsed:.0f}s]")

    accuracy = n_correct / len(test_problems)
    log_experiment(f"  math_sft KodCode-200: {n_correct}/{len(test_problems)} ({accuracy*100:.1f}%)")
    return {
        'n_problems': len(test_problems),
        'n_correct': n_correct,
        'accuracy': accuracy,
        'completions': completions,
    }


async def eval_humaneval(semaphore, sampling_client, renderer, tokenizer):
    """Evaluate Math SFT model on HumanEval-164 with <think> seeding."""
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

    log_experiment(f"Evaluating math_sft on HumanEval ({len(problems)} problems)")
    counter = {'done': 0}
    t0 = time.time()

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
                    sampling_client, renderer, None, tokenizer,
                    max_tokens=4096, temperature=0.7,
                    think_prefix=True, messages=messages)
                counter['done'] += 1
                if counter['done'] % 20 == 0 or counter['done'] == len(problems):
                    elapsed = time.time() - t0
                    progress(f"  HumanEval generation: {counter['done']}/{len(problems)} "
                             f"[{elapsed:.0f}s]")
                return completions[0]
            except Exception as e:
                counter['done'] += 1
                return f"ERROR: {e}"

    gen_tasks = [_gen_one(i) for i in range(len(problems))]
    completions = list(await asyncio.gather(*gen_tasks))

    # Test completions
    progress(f"  HumanEval: testing {len(completions)} solutions...")
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
    log_experiment(f"  math_sft HumanEval: {n_passed}/{len(problems)} ({accuracy*100:.1f}%)")

    summary = {
        "benchmark": "HumanEval",
        "n_problems": len(problems),
        "math_sft": {"passed": n_passed, "total": len(problems), "pass@1": accuracy},
    }
    return summary, completions, details, accuracy


async def eval_fuzzy(session, semaphore, sampling_client, renderer, tokenizer):
    """Evaluate Math SFT model on fuzzy evals with <think> seeding."""
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
        log_experiment(f"Evaluating math_sft on {ds_name}: {len(questions)}Q x {N_FUZZY_SAMPLES}S = {n_total}")

        async def _gen_fuzzy(q_idx, q, _q_key=q_key):
            prompt = f"Answer the following question.\n\n{q[_q_key]}"
            async with semaphore:
                try:
                    samples = await sample_tinker(
                        sampling_client, renderer, prompt, tokenizer,
                        num_samples=N_FUZZY_SAMPLES, temperature=0.7,
                        max_tokens=FUZZY_MAX_TOKENS, think_prefix=True)
                    return list(samples)
                except Exception as e:
                    return [f"ERROR: {e}"] * N_FUZZY_SAMPLES

        fuzzy_tasks = [_gen_fuzzy(q_idx, q) for q_idx, q in enumerate(questions)]
        ds_completions = list(await asyncio.gather(*fuzzy_tasks))

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

        progress(f"  Fuzzy {ds_name}: generation done, judging {len(judge_tasks)} responses...")
        judge_responses = []
        batch_size = 30
        for i in range(0, len(judge_tasks), batch_size):
            batch = judge_tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            judge_responses.extend(batch_results)
            done_count = min(i + batch_size, len(judge_tasks))
            if done_count % 30 == 0 or done_count == len(judge_tasks):
                progress(f"  Fuzzy {ds_name} judging: {done_count}/{len(judge_tasks)}")

        scores = []
        for (q_idx, s_idx), judge_resp in zip(judge_map, judge_responses):
            score, _ = parse_judge_score(judge_resp) if not str(judge_resp).startswith("ERROR:") else (None, {})
            if score is not None:
                scores.append(score)

        mean_score = float(np.mean(scores)) if scores else 0.0
        log_experiment(f"  math_sft {ds_name}: mean={mean_score:.2f} (n={len(scores)})")

        all_results[ds_name] = {
            'mean': mean_score,
            'n': len(scores),
            'scores': scores,
        }

    return all_results, all_completions


# ── Plot generation ────────────────────────────────────────────────────────

def generate_plots(math_sft_math_acc, math_sft_he_acc, math_sft_kodcode_acc,
                   math_sft_fuzzy, math_sft_completions_all):
    """Generate 4-bar plots: Base vs Math RLVR vs Code RLVR vs Math SFT."""

    # Load existing data (READ ONLY)
    try:
        with open(RESULTS_DIR / 'humaneval_results.json') as f:
            humaneval = json.load(f)
        with open(RESULTS_DIR / 'fuzzy_base.json') as f:
            fuzzy_base = json.load(f)
        with open(RESULTS_DIR / 'fuzzy_best_rlvr.json') as f:
            fuzzy_math_rlvr = json.load(f)
    except FileNotFoundError as e:
        log_experiment(f"WARNING: Could not load existing results for plots: {e}")
        log_experiment("Skipping plot generation")
        return

    # Load code RLVR fuzzy results if available
    code_rlvr_fuzzy = {}
    code_rlvr_eval_dir = RESULTS_DIR / 'code_rlvr_eval'
    try:
        with open(code_rlvr_eval_dir / 'fuzzy_code_rlvr.json') as f:
            code_rlvr_fuzzy = json.load(f)
    except FileNotFoundError:
        log_experiment("WARNING: No code RLVR fuzzy results found")

    # Load code RLVR math/humaneval results
    code_rlvr_math_acc = None
    code_rlvr_he_acc = None
    try:
        with open(code_rlvr_eval_dir / 'math_results.json') as f:
            code_math = json.load(f)
            n_correct = sum(1 for r in code_math if r.get('correct'))
            code_rlvr_math_acc = n_correct / len(code_math) * 100
    except FileNotFoundError:
        pass
    try:
        with open(code_rlvr_eval_dir / 'humaneval_results.json') as f:
            code_he = json.load(f)
            code_rlvr_he_acc = code_he.get('code_rlvr', {}).get('pass@1', 0)
    except FileNotFoundError:
        pass

    # Load KodCode 3-model results
    kodcode_base_acc = None
    kodcode_math_rlvr_acc = None
    kodcode_code_rlvr_acc = None
    try:
        with open(RESULTS_DIR / 'code_rlvr_eval' / 'kodcode_200_3model.json') as f:
            kodcode_3model = json.load(f)
            kodcode_base_acc = kodcode_3model['base']['accuracy'] * 100
            kodcode_math_rlvr_acc = kodcode_3model['math_rlvr']['accuracy'] * 100
            kodcode_code_rlvr_acc = kodcode_3model['code_rlvr']['accuracy'] * 100
    except FileNotFoundError:
        pass

    # Known results
    math_base_acc = 40.8
    math_rlvr_acc = 48.0
    he_base_rate = humaneval['base']['pass@1']
    he_math_rlvr_rate = humaneval['best_rlvr']['pass@1']

    # ── Plot 1: final_comparison.png ─────────────────────────────────────────
    colors = {'base': '#4ECDC4', 'math_rlvr': '#FF6B6B', 'code_rlvr': '#6B8EFF', 'math_sft': '#96CEB4'}
    model_keys = ['base', 'math_rlvr', 'code_rlvr', 'math_sft']
    model_display = {
        'base': 'Base (Llama-3.1-8B)',
        'math_rlvr': 'Math RLVR',
        'code_rlvr': 'Code RLVR',
        'math_sft': 'Math SFT (R1)',
    }

    datasets_list = ['math', 'humaneval', 'kodcode', 'philosophy', 'weird_questions', 'futuristic_tech']
    dataset_labels = [
        'MATH-500', 'HumanEval-164', 'KodCode-200',
        'Philosophy\n(10Q x 10S)', 'Weird Questions\n(46Q x 10S)', 'Futuristic Tech\n(10Q x 10S)',
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(datasets_list))
    n_models = len(model_keys)
    width = 0.20
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    for model_idx, model_key in enumerate(model_keys):
        if model_key == 'base':
            fuzzy = fuzzy_base
            math_acc = math_base_acc
            he_rate = he_base_rate * 100
            kc_acc = kodcode_base_acc or 0
        elif model_key == 'math_rlvr':
            fuzzy = fuzzy_math_rlvr
            math_acc = math_rlvr_acc
            he_rate = he_math_rlvr_rate * 100
            kc_acc = kodcode_math_rlvr_acc or 0
        elif model_key == 'code_rlvr':
            fuzzy = code_rlvr_fuzzy
            math_acc = code_rlvr_math_acc or 0
            he_rate = (code_rlvr_he_acc or 0) * 100
            kc_acc = kodcode_code_rlvr_acc or 0
        else:  # math_sft
            fuzzy = math_sft_fuzzy
            math_acc = math_sft_math_acc * 100
            he_rate = math_sft_he_acc * 100
            kc_acc = math_sft_kodcode_acc * 100

        means = [
            math_acc,
            he_rate,
            kc_acc,
            fuzzy.get('philosophy', {}).get('mean', 0),
            fuzzy.get('weird_questions', {}).get('mean', 0),
            fuzzy.get('futuristic_tech', {}).get('mean', 0),
        ]

        ax.bar(
            x + offsets[model_idx], means, width,
            label=model_display[model_key], color=colors[model_key],
            capsize=3, edgecolor='black', linewidth=0.5,
        )
        for i, m in enumerate(means):
            if m > 0:
                suffix = '%' if datasets_list[i] in ['math', 'humaneval', 'kodcode'] else ''
                ax.text(
                    x[i] + offsets[model_idx], m + 0.5, f'{m:.1f}{suffix}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
                )

    ax.set_xlabel('Evaluation Domain', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Base vs Math RLVR vs Code RLVR vs Math SFT (R1 Reasoning)\n'
        'Math/Code: Accuracy (%) | Fuzzy: Mean Rubric Score',
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot_path = EVAL_DIR / 'final_comparison.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log_experiment(f"Saved plot: {plot_path}")

    # ── Plot 2: response_length_comparison.png ────────────────────────────────
    # Math SFT with <think> will have much longer responses
    fig, ax = plt.subplots(figsize=(14, 6))

    # Only plot math_sft response lengths (others are from different eval runs)
    ds_names = []
    lengths = []
    for ds_name, comps in math_sft_completions_all.items():
        if comps:
            ds_names.append(ds_name)
            ds_lengths = [len(s) for s in comps if isinstance(s, str)]
            lengths.append(ds_lengths)

    if ds_names:
        positions = np.arange(len(ds_names))
        means = [np.mean(l) for l in lengths]
        stds = [np.std(l) / np.sqrt(len(l)) if len(l) > 1 else 0 for l in lengths]

        ax.bar(positions, means, yerr=stds, color='#96CEB4', capsize=3,
               edgecolor='black', linewidth=0.5)
        for i, (m, se) in enumerate(zip(means, stds)):
            ax.text(i, m + se + 20, f'{m:.0f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Average Response Length (characters)', fontsize=12)
        ax.set_title('Math SFT (R1) Response Length with <think> Seeding', fontsize=13)
        ax.set_xticks(positions)
        ax.set_xticklabels([n.replace('_', '\n') for n in ds_names])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plot_path = EVAL_DIR / 'response_length_comparison.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log_experiment(f"Saved plot: {plot_path}")


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    import tinker
    from tinker_cookbook import renderers, model_info
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    parser = argparse.ArgumentParser(description='Evaluate Math SFT sweep winner')
    parser.add_argument('--winner', type=str, required=True,
                        help='Winning config name (e.g., B_medium)')
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Find best checkpoint
    winner_dir = MATH_SFT_DIR / args.winner
    if not winner_dir.exists():
        print(f"ERROR: Winner directory {winner_dir} does not exist")
        sys.exit(1)

    best_ckpt = find_best_sft_checkpoint(winner_dir)
    if best_ckpt is None:
        print(f"ERROR: Could not find best checkpoint for {args.winner}")
        sys.exit(1)

    log_experiment(f"# Math SFT Evaluation: {args.winner}")
    log_experiment(f"Best checkpoint: step {best_ckpt['step']} "
                   f"(test NLL: {best_ckpt['nll']:.4f}, "
                   f"checkpoint at step {best_ckpt['checkpoint_step']})")

    # Set up renderer and tokenizer
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create sampling client for Math SFT model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=best_ckpt['sampler_path'])
    log_experiment(f"Loaded Math SFT model from {best_ckpt['sampler_path']}")

    semaphore = asyncio.Semaphore(10)

    # Run all 4 eval phases in parallel
    log_experiment("Running all evals in parallel: MATH-500, KodCode-200, HumanEval-164, Fuzzy")

    async def run_math_eval():
        log_experiment("  Starting MATH-500 evaluation (with <think> seeding)")
        results, completions, accuracy = await eval_math(
            semaphore, sampling_client, renderer, tokenizer)
        with open(EVAL_DIR / 'math_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        with open(EVAL_DIR / 'math_completions.json', 'w') as f:
            json.dump(completions, f)
        log_experiment(f"  MATH-500 done: {accuracy*100:.1f}%")
        return results, completions, accuracy

    async def run_kodcode_eval():
        log_experiment("  Starting KodCode-200 evaluation (with <think> seeding)")
        results = await eval_kodcode(
            semaphore, sampling_client, renderer, tokenizer)
        with open(EVAL_DIR / 'kodcode_results.json', 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'completions'}, f, indent=2)
        with open(EVAL_DIR / 'kodcode_completions.json', 'w') as f:
            json.dump(results.get('completions', []), f, indent=2)
        log_experiment(f"  KodCode-200 done: {results['accuracy']*100:.1f}%")
        return results

    async def run_humaneval_eval():
        log_experiment("  Starting HumanEval-164 evaluation (with <think> seeding)")
        summary, completions, details, accuracy = await eval_humaneval(
            semaphore, sampling_client, renderer, tokenizer)
        with open(EVAL_DIR / 'humaneval_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(EVAL_DIR / 'humaneval_completions.json', 'w') as f:
            json.dump(completions, f)
        with open(EVAL_DIR / 'humaneval_details.json', 'w') as f:
            json.dump(details, f, indent=2)
        log_experiment(f"  HumanEval-164 done: {accuracy*100:.1f}%")
        return summary, completions, details, accuracy

    async def run_fuzzy_eval():
        log_experiment("  Starting Fuzzy evaluation (3 datasets, with <think> seeding)")
        async with aiohttp.ClientSession() as session:
            results, completions = await eval_fuzzy(
                session, semaphore, sampling_client, renderer, tokenizer)
        with open(EVAL_DIR / 'fuzzy_math_sft.json', 'w') as f:
            json.dump(results, f, indent=2)
        with open(EVAL_DIR / 'fuzzy_completions.json', 'w') as f:
            json.dump(completions, f)
        for ds_name in ['philosophy', 'weird_questions', 'futuristic_tech']:
            ds = results.get(ds_name, {})
            log_experiment(f"  Fuzzy {ds_name} done: mean={ds.get('mean', 0):.2f}")
        return results, completions

    # Launch all in parallel
    (math_result, kodcode_result, he_result, fuzzy_result) = await asyncio.gather(
        run_math_eval(),
        run_kodcode_eval(),
        run_humaneval_eval(),
        run_fuzzy_eval(),
    )

    math_results, math_completions, math_accuracy = math_result
    kodcode_accuracy = kodcode_result['accuracy']
    he_summary, he_completions, he_details, he_accuracy = he_result
    fuzzy_results, fuzzy_completions = fuzzy_result

    check_memory()
    gc.collect()

    # Phase 5: Generate plots
    log_experiment("Generating comparison plots")
    all_completions = {
        'math': math_completions,
        'humaneval': he_completions,
    }
    all_completions.update(fuzzy_completions)

    generate_plots(math_accuracy, he_accuracy, kodcode_accuracy,
                   fuzzy_results, all_completions)

    # Final summary
    log_experiment("\n## Final Summary")
    log_experiment(f"  MATH-500:       {math_accuracy*100:.1f}%")
    log_experiment(f"  KodCode-200:    {kodcode_accuracy*100:.1f}%")
    log_experiment(f"  HumanEval-164:  {he_accuracy*100:.1f}%")
    for ds_name in ['philosophy', 'weird_questions', 'futuristic_tech']:
        ds = fuzzy_results.get(ds_name, {})
        log_experiment(f"  {ds_name}: mean={ds.get('mean', 0):.2f} (n={ds.get('n', 0)})")

    log_experiment(f"\nAll outputs saved to {EVAL_DIR}")
    log_experiment("# Math SFT evaluation complete")


if __name__ == '__main__':
    asyncio.run(main())
