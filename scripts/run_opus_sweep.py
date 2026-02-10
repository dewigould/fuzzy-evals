"""
Opus 4.6 Effort & Prompting Sweep Experiments.

Usage:
    python3.13 run_opus_sweep.py effort   # Experiment 1: 4 effort levels
    python3.13 run_opus_sweep.py sweep    # Experiment 2: 5 prompt styles at high effort

Memory-safe design for RunPod instances with ~4GB cgroup limit.
"""

import asyncio
import gc
import json
import logging
import os
import sys
import time
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

# ── Constants ─────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
RESULTS_DIR = Path('/workspace/results_opus_sweep')
RESULTS_DIR.mkdir(exist_ok=True)
LOG_FILE = RESULTS_DIR / 'experiment_log.md'

MODEL = 'anthropic/claude-opus-4.6'
JUDGE_MODEL = 'openai/gpt-5.2'
SAMPLING_N = 10

OPUS_SEMAPHORE = 8
JUDGE_SEMAPHORE = 25
COMPLETION_BATCH_SIZE = 20

CGROUP_USAGE_PATH = '/sys/fs/cgroup/memory/memory.usage_in_bytes'
CGROUP_LIMIT_PATH = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
MEMORY_WARN_RATIO = 0.75
MEMORY_ABORT_RATIO = 0.80

EFFORT_LEVELS = ['low', 'medium', 'high', 'max']
PROMPT_STYLES = ['minimal', 'medium', 'detailed', 'think_hard', 'think_hard_rubric']

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / 'opus_sweep_runner.log'),
    ]
)
log = logging.getLogger(__name__)


def log_experiment(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n### {ts}\n{msg}\n")
    log.info(msg)


# ── Memory Monitoring ─────────────────────────────────────────────────────

def get_memory_usage():
    """Returns (usage_bytes, limit_bytes, ratio)."""
    try:
        usage = int(open(CGROUP_USAGE_PATH).read().strip())
        limit = int(open(CGROUP_LIMIT_PATH).read().strip())
        return usage, limit, usage / limit
    except Exception:
        return 0, 1, 0.0


def check_memory(context: str = "") -> bool:
    """Check memory. Returns True if OK to continue, False if must abort."""
    usage, limit, ratio = get_memory_usage()
    usage_mb = usage / (1024 * 1024)
    limit_mb = limit / (1024 * 1024)

    if ratio > MEMORY_ABORT_RATIO:
        log.error(f"MEMORY ABORT: {usage_mb:.0f}/{limit_mb:.0f} MB ({ratio:.1%}) at {context}")
        return False

    if ratio > MEMORY_WARN_RATIO:
        log.warning(f"MEMORY WARNING: {usage_mb:.0f}/{limit_mb:.0f} MB ({ratio:.1%}) at {context}")
        gc.collect()
        usage2, _, ratio2 = get_memory_usage()
        log.warning(f"  After gc: {usage2/(1024*1024):.0f} MB ({ratio2:.1%})")
        if ratio2 > MEMORY_ABORT_RATIO:
            return False

    return True


def log_memory(context: str = ""):
    usage, limit, ratio = get_memory_usage()
    log.info(f"Memory [{context}]: {usage/(1024*1024):.0f}/{limit/(1024*1024):.0f} MB ({ratio:.1%})")


# ── Prompt Templates ──────────────────────────────────────────────────────

PROMPT_MINIMAL = "Answer the following question.\n\n{question}"

PROMPT_MEDIUM = """Think carefully and give a thorough, well-reasoned answer to the following question. Take your time and consider multiple angles before responding.

{question}"""

PROMPT_DETAILED = """You are about to answer a challenging question that rewards depth of thought, originality, and genuine intellectual engagement. Here is what a great answer looks like:

- It takes a clear position rather than hedging or equivocating
- It reasons carefully through mechanisms, second-order effects, and non-obvious implications
- It uses specific, concrete examples, numbers, or scenarios rather than staying abstract
- It considers counterarguments and alternative perspectives, but still commits to a view
- It makes surprising connections or brings a genuinely fresh angle
- It distinguishes clearly between what is known, what is estimated, and what is speculative
- It demonstrates intellectual courage -- willingness to say something that could be wrong rather than retreating to safe platitudes

Do not pad your answer with unnecessary caveats or qualifications. Do not open with "This is a great question" or "There are many perspectives on this." Just think hard and answer directly.

{question}"""

PROMPT_THINK_HARD = """Answer the following question. Give it your absolute best effort -- this is important.

Before writing your answer:
- Plan your response structure before diving in
- Think step by step through your reasoning
- Try multiple approaches and compare them before committing
- Backtrack when you notice a flaw in your logic
- Self-verify: check your claims, arguments, and examples for accuracy
- Self-criticize: actively look for weaknesses, gaps, or blind spots in your own position
- Consider what a thoughtful, well-informed critic would say, and address their strongest objections
- Distinguish clearly between what you know, what you're estimating, and what you're speculating about

Do not hedge or equivocate. Take a clear position and defend it rigorously.

{question}"""

SIMPLE_PROMPTS = {
    'minimal': PROMPT_MINIMAL,
    'medium': PROMPT_MEDIUM,
    'detailed': PROMPT_DETAILED,
    'think_hard': PROMPT_THINK_HARD,
}


# ── Dataset & Rubric Loading ─────────────────────────────────────────────

def load_datasets():
    """Load all three datasets. Returns list of 66 question dicts."""
    base = Path('/workspace/fuzzy-evals/dataset_jsons')

    with open(base / 'philosophy_questions.json') as f:
        phil_raw = json.load(f)
    questions = [
        {'dataset': 'philosophy', 'question_id': i, 'question': q['question']}
        for i, q in enumerate(phil_raw)
    ]

    with open(base / 'weird_questions.json') as f:
        weird_raw = json.load(f)
    questions.extend([
        {'dataset': 'weird_questions', 'question_id': i, 'question': q['prompt']}
        for i, q in enumerate(weird_raw)
    ])

    with open(base / 'futuristic_tech_questions.json') as f:
        future_raw = json.load(f)
    questions.extend([
        {'dataset': 'futuristic_tech', 'question_id': q['id'], 'question': q['question']}
        for q in future_raw
    ])

    return questions


def load_rubrics():
    """Returns {dataset_name: (system_message, rubric_text)}."""
    base = Path('/workspace/fuzzy-evals/rubrics')
    return {
        'philosophy': (
            "You are grading a philosophical answer for quality of argumentation and exposition.",
            (base / 'philosophy_rubric.md').read_text(),
        ),
        'weird_questions': (
            "You are grading an answer for quality of thought, research, effort, and exposition.",
            (base / 'weird_questions_rubric.md').read_text(),
        ),
        'futuristic_tech': (
            "You are grading an answer for quality of thought, scientific rigour, and effort.",
            (base / 'futuristic_tech_rubric.md').read_text(),
        ),
    }


def build_model_prompt(prompt_style: str, question: str, dataset: str,
                       rubrics: dict) -> str:
    """Build the prompt sent to Opus."""
    if prompt_style in SIMPLE_PROMPTS:
        return SIMPLE_PROMPTS[prompt_style].format(question=question)

    if prompt_style == 'think_hard_rubric':
        _, rubric_text = rubrics[dataset]
        return f"""Answer the following question. Give it your absolute best effort -- this is important.

Before writing your answer:
- Plan your response structure before diving in
- Think step by step through your reasoning
- Try multiple approaches and compare them before committing
- Backtrack when you notice a flaw in your logic
- Self-verify: check your claims, arguments, and examples for accuracy
- Self-criticize: actively look for weaknesses, gaps, or blind spots in your own position
- Consider what a thoughtful, well-informed critic would say, and address their strongest objections
- Distinguish clearly between what you know, what you're estimating, and what you're speculating about

Do not hedge or equivocate. Take a clear position and defend it rigorously.

Your answer will be evaluated on the following rubric. Study it carefully and optimize your response accordingly:

{rubric_text}

{question}"""

    raise ValueError(f"Unknown prompt style: {prompt_style}")


def build_grading_prompt(system_msg: str, question: str, answer: str,
                         rubric: str) -> str:
    """Build the prompt sent to the judge model."""
    return f"""{system_msg}

## Question

{question}

## Answer to Grade

{answer}

## Grading Rubric

{rubric}"""


# ── API Helpers ───────────────────────────────────────────────────────────

async def call_openrouter(session, model, messages, temperature=0.7,
                          max_tokens=4096, semaphore=None, extra_body=None):
    """Make API call, return (content_text, usage_dict)."""
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
    if extra_body:
        body.update(extra_body)

    retry_delays = [2, 5, 15, 30, 60, 120, 120]
    for attempt, delay in enumerate(retry_delays + [120]):
        try:
            if semaphore:
                async with semaphore:
                    async with session.post(OPENROUTER_URL, headers=headers,
                                           json=body, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                        data = await resp.json()
            else:
                async with session.post(OPENROUTER_URL, headers=headers,
                                       json=body, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                    data = await resp.json()

            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                usage = data.get('usage', {})
                return content, usage
            elif 'error' in data:
                err = data['error']
                err_msg = err.get('message', str(err)) if isinstance(err, dict) else str(err)
                if 'rate' in err_msg.lower() or '429' in err_msg:
                    log.warning(f"Rate limited on {model}, attempt {attempt+1}, waiting {delay}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    log.error(f"API error for {model}: {err_msg}")
                    return f"ERROR: {err_msg}", {}
            else:
                return "ERROR: unexpected response", {}
        except asyncio.TimeoutError:
            log.warning(f"Timeout for {model}, attempt {attempt+1}")
            await asyncio.sleep(delay)
        except Exception as e:
            log.warning(f"Exception for {model}: {e}, attempt {attempt+1}")
            await asyncio.sleep(delay)

    return "ERROR: max retries exceeded", {}


def extract_token_counts(usage: dict) -> dict:
    """Extract reasoning tokens, output tokens, and totals from usage dict."""
    completion_tokens = usage.get('completion_tokens')
    prompt_tokens = usage.get('prompt_tokens')

    details = usage.get('completion_tokens_details', {})
    reasoning_tokens = None
    if details and isinstance(details, dict):
        rt = details.get('reasoning_tokens')
        if rt is not None:
            reasoning_tokens = int(rt)

    output_tokens = None
    if completion_tokens is not None and reasoning_tokens is not None:
        output_tokens = int(completion_tokens) - reasoning_tokens
    elif completion_tokens is not None:
        output_tokens = int(completion_tokens)

    return {
        'reasoning_tokens': reasoning_tokens,
        'output_tokens': output_tokens,
        'completion_tokens': int(completion_tokens) if completion_tokens is not None else None,
        'prompt_tokens': int(prompt_tokens) if prompt_tokens is not None else None,
    }


def parse_judge_score(response_text):
    """Parse judge JSON response into (total_score, sub_scores_dict)."""
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


# ── Parquet helpers ───────────────────────────────────────────────────────

def safe_write_parquet(df, path: Path):
    """Write parquet atomically via temp file."""
    tmp = path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


def append_to_parquet(partial_path: Path, new_rows: list):
    """Append new result rows to partial parquet file."""
    new_df = pd.DataFrame(new_rows)
    if partial_path.exists():
        existing_df = pd.read_parquet(partial_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        del existing_df
    else:
        combined = new_df
    safe_write_parquet(combined, partial_path)
    del combined, new_df
    gc.collect()


def get_completed_conditions(partial_path: Path, condition_cols: list,
                             n_expected: int) -> set:
    """Return set of completed condition tuples from partial parquet."""
    if not partial_path.exists():
        return set()

    df = pd.read_parquet(partial_path, columns=condition_cols + ['question_id', 'sample_id'])
    completed = set()

    for group_key, group in df.groupby(condition_cols):
        if len(group) >= n_expected:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            completed.add(group_key)

    del df
    gc.collect()
    return completed


# ── Core Processing Loop ─────────────────────────────────────────────────

async def process_condition(session, opus_sem, judge_sem, questions, rubrics,
                            prompt_style, effort_level, raw_output_file,
                            condition_key):
    """
    Process one condition: generate completions and judge them.
    Returns list of score-only result dicts (no raw_output/judge_response).
    Writes raw outputs incrementally to JSONL.
    """
    extra_body = {
        'verbosity': effort_level,
        'reasoning': {'enabled': True},
    }

    results = []

    # Build all (question, sample_id) pairs
    all_pairs = []
    for q in questions:
        for sample_id in range(SAMPLING_N):
            all_pairs.append((q, sample_id))

    total_pairs = len(all_pairs)
    log_experiment(f"  [{condition_key}] Processing {total_pairs} pairs in batches of {COMPLETION_BATCH_SIZE}")

    for batch_start in range(0, total_pairs, COMPLETION_BATCH_SIZE):
        # Memory check
        if not check_memory(f"{condition_key} batch@{batch_start}"):
            log.error(f"Memory abort at {condition_key}, batch {batch_start}. Returning partial.")
            return results

        batch_pairs = all_pairs[batch_start:batch_start + COMPLETION_BATCH_SIZE]

        # 1. Generate completions
        completion_tasks = []
        for q, sample_id in batch_pairs:
            full_prompt = build_model_prompt(prompt_style, q['question'],
                                            q['dataset'], rubrics)
            messages = [{'role': 'user', 'content': full_prompt}]
            completion_tasks.append(call_openrouter(
                session, MODEL, messages,
                temperature=1.0,
                semaphore=opus_sem,
                extra_body=extra_body,
            ))

        completions = await asyncio.gather(*completion_tasks)
        del completion_tasks

        # 2. Judge completions
        judge_tasks = []
        for (q, sample_id), (content, usage) in zip(batch_pairs, completions):
            if isinstance(content, str) and content.startswith("ERROR:"):
                async def _err(c=content):
                    return c, {}
                judge_tasks.append(_err())
                continue

            sys_msg, rubric_text = rubrics[q['dataset']]
            grading_prompt = build_grading_prompt(sys_msg, q['question'],
                                                  content, rubric_text)
            messages = [{'role': 'user', 'content': grading_prompt}]
            judge_tasks.append(call_openrouter(
                session, JUDGE_MODEL, messages,
                temperature=0.0, max_tokens=1024, semaphore=judge_sem,
            ))

        judge_responses = await asyncio.gather(*judge_tasks)
        del judge_tasks

        # 3. Assemble results, write raw outputs to JSONL
        for (q, sample_id), (content, usage), (judge_content, _) in zip(
            batch_pairs, completions, judge_responses
        ):
            # Write raw output to JSONL immediately
            raw_line = json.dumps({
                'condition': condition_key,
                'dataset': q['dataset'],
                'question_id': q['question_id'],
                'sample_id': sample_id,
                'raw_output': content if isinstance(content, str) else str(content),
                'judge_response': judge_content if isinstance(judge_content, str) else str(judge_content),
            })
            with open(raw_output_file, 'a') as f:
                f.write(raw_line + '\n')

            # Parse score
            if isinstance(judge_content, str) and judge_content.startswith("ERROR:"):
                total_score, sub_scores = None, {}
            else:
                total_score, sub_scores = parse_judge_score(judge_content)

            token_counts = extract_token_counts(usage)

            result = {
                'model': MODEL,
                'prompt_style': prompt_style,
                'effort_level': effort_level,
                'dataset': q['dataset'],
                'question_id': q['question_id'],
                'sample_id': sample_id,
                'score': total_score,
                **token_counts,
            }
            result.update(sub_scores)
            results.append(result)

        del completions, judge_responses
        gc.collect()

        done = min(batch_start + COMPLETION_BATCH_SIZE, total_pairs)
        log.info(f"  [{condition_key}] {done}/{total_pairs} done")

    return results


# ── Experiment 1: Effort Sweep ────────────────────────────────────────────

async def run_effort_experiment():
    """4 effort levels x minimal prompt x 66 questions x 10 samples."""
    partial_path = RESULTS_DIR / 'effort_partial.parquet'
    raw_output_file = RESULTS_DIR / 'raw_outputs_effort.jsonl'

    questions = load_datasets()
    rubrics = load_rubrics()
    n_questions = len(questions)
    expected_per_condition = n_questions * SAMPLING_N

    completed = get_completed_conditions(partial_path, ['effort_level'],
                                         expected_per_condition)

    log_experiment("# Effort Experiment: Opus 4.6 effort level sweep")
    log_experiment(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_experiment(f"Questions: {n_questions}, Samples: {SAMPLING_N}")
    log_experiment(f"Effort levels: {EFFORT_LEVELS}")
    log_experiment(f"Already completed: {completed}")
    log_memory("effort_start")

    opus_sem = asyncio.Semaphore(OPUS_SEMAPHORE)
    judge_sem = asyncio.Semaphore(JUDGE_SEMAPHORE)

    async with aiohttp.ClientSession() as session:
        for effort_level in EFFORT_LEVELS:
            if (effort_level,) in completed:
                log_experiment(f"  Skipping effort={effort_level} (already done)")
                continue

            condition_key = f"effort_{effort_level}"
            log_experiment(f"  Starting: effort={effort_level}")
            start = time.time()

            results = await process_condition(
                session, opus_sem, judge_sem, questions, rubrics,
                'minimal', effort_level, raw_output_file, condition_key,
            )

            elapsed = time.time() - start
            valid = sum(1 for r in results if r['score'] is not None)
            log_experiment(f"  Done: effort={effort_level} in {elapsed/60:.1f}m "
                          f"({valid}/{len(results)} valid)")

            append_to_parquet(partial_path, results)
            del results
            gc.collect()
            log_memory(f"after_{effort_level}")

    # Final save + plots
    if partial_path.exists():
        df = pd.read_parquet(partial_path)
        safe_write_parquet(df, RESULTS_DIR / 'effort_scores.parquet')
        log_experiment(f"Effort experiment complete: {len(df)} rows, "
                      f"{df['score'].notna().sum()} valid scores")
        generate_effort_plots(df)
        del df
        gc.collect()


# ── Experiment 2: Prompting Sweep ─────────────────────────────────────────

async def run_sweep_experiment():
    """5 prompt styles x HIGH effort x 66 questions x 10 samples."""
    partial_path = RESULTS_DIR / 'sweep_partial.parquet'
    raw_output_file = RESULTS_DIR / 'raw_outputs_sweep.jsonl'

    questions = load_datasets()
    rubrics = load_rubrics()
    n_questions = len(questions)
    expected_per_condition = n_questions * SAMPLING_N

    completed = get_completed_conditions(partial_path, ['prompt_style'],
                                         expected_per_condition)

    log_experiment("# Prompting Sweep: Opus 4.6 at HIGH effort")
    log_experiment(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_experiment(f"Questions: {n_questions}, Samples: {SAMPLING_N}")
    log_experiment(f"Prompt styles: {PROMPT_STYLES}")
    log_experiment(f"Already completed: {completed}")
    log_memory("sweep_start")

    opus_sem = asyncio.Semaphore(OPUS_SEMAPHORE)
    judge_sem = asyncio.Semaphore(JUDGE_SEMAPHORE)

    async with aiohttp.ClientSession() as session:
        for prompt_style in PROMPT_STYLES:
            if (prompt_style,) in completed:
                log_experiment(f"  Skipping prompt={prompt_style} (already done)")
                continue

            condition_key = f"sweep_{prompt_style}"
            log_experiment(f"  Starting: prompt={prompt_style}, effort=high")
            start = time.time()

            results = await process_condition(
                session, opus_sem, judge_sem, questions, rubrics,
                prompt_style, 'high', raw_output_file, condition_key,
            )

            elapsed = time.time() - start
            valid = sum(1 for r in results if r['score'] is not None)
            log_experiment(f"  Done: prompt={prompt_style} in {elapsed/60:.1f}m "
                          f"({valid}/{len(results)} valid)")

            append_to_parquet(partial_path, results)
            del results
            gc.collect()
            log_memory(f"after_{prompt_style}")

    # Final save + plots
    if partial_path.exists():
        df = pd.read_parquet(partial_path)
        safe_write_parquet(df, RESULTS_DIR / 'sweep_scores.parquet')
        log_experiment(f"Sweep experiment complete: {len(df)} rows, "
                      f"{df['score'].notna().sum()} valid scores")
        generate_sweep_plots(df)
        del df
        gc.collect()


# ── Plotting: Effort ──────────────────────────────────────────────────────

def generate_effort_plots(df):
    """Generate effort experiment plots."""
    df_valid = df[df['score'].notna()].copy()
    if len(df_valid) == 0:
        log.warning("No valid scores for effort plots")
        return

    effort_order = ['low', 'medium', 'high', 'max']
    effort_colors = {'low': '#4ECDC4', 'medium': '#FFD93D', 'high': '#FF6B6B', 'max': '#9B59B6'}

    datasets = ['philosophy', 'weird_questions', 'futuristic_tech', 'all']
    for ds in datasets:
        if ds == 'all':
            df_ds = df_valid
            dtitle = 'All Fuzzy Tasks'
        else:
            df_ds = df_valid[df_valid['dataset'] == ds]
            dtitle = ds.replace('_', ' ').title()

        if len(df_ds) == 0:
            continue

        stats = df_ds.groupby('effort_level')['score'].agg(['mean', 'std', 'count']).reset_index()
        stats['se'] = stats['std'] / np.sqrt(stats['count'])
        stats['effort_level'] = pd.Categorical(stats['effort_level'],
                                                categories=effort_order, ordered=True)
        stats = stats.sort_values('effort_level').dropna(subset=['effort_level'])

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [effort_colors.get(e, '#999') for e in stats['effort_level']]
        bars = ax.bar(range(len(stats)), stats['mean'], yerr=stats['se'],
                      color=colors, capsize=5, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels([e.capitalize() for e in stats['effort_level']])
        ax.set_xlabel('Effort Level', fontsize=12)
        ax.set_ylabel('Mean Rubric Score (0-48)', fontsize=12)
        ax.set_title(f'Opus 4.6 Effort Sweep — {dtitle}', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean, se in zip(bars, stats['mean'], stats['se']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + se + 0.3,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        suffix = 'combined' if ds == 'all' else ds
        plt.savefig(RESULTS_DIR / f'effort_bar_{suffix}.png', dpi=150)
        plt.close()

    # Scatter: score vs reasoning tokens
    df_rt = df_valid[df_valid['reasoning_tokens'].notna() & (df_valid['reasoning_tokens'] > 0)]
    if len(df_rt) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        for effort in effort_order:
            df_e = df_rt[df_rt['effort_level'] == effort]
            if len(df_e) == 0:
                continue
            ax.scatter(df_e['reasoning_tokens'], df_e['score'],
                       color=effort_colors[effort], alpha=0.4, s=30,
                       label=effort.capitalize())
        ax.set_xlabel('Reasoning Tokens', fontsize=12)
        ax.set_ylabel('Rubric Score (0-48)', fontsize=12)
        ax.set_title('Opus 4.6: Score vs Reasoning Tokens by Effort Level', fontsize=13)
        ax.legend(title='Effort Level')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'effort_scatter_reasoning.png', dpi=150)
        plt.close()

    plt.close('all')
    gc.collect()
    log_experiment("Effort plots saved")


# ── Plotting: Sweep ───────────────────────────────────────────────────────

def generate_sweep_plots(df):
    """Generate prompting sweep plots."""
    df_valid = df[df['score'].notna()].copy()
    if len(df_valid) == 0:
        log.warning("No valid scores for sweep plots")
        return

    prompt_order = ['minimal', 'medium', 'detailed', 'think_hard', 'think_hard_rubric']
    prompt_colors = {
        'minimal': '#4ECDC4', 'medium': '#FFD93D', 'detailed': '#FF6B6B',
        'think_hard': '#9B59B6', 'think_hard_rubric': '#3498DB',
    }
    prompt_labels = {
        'minimal': 'Minimal', 'medium': 'Medium', 'detailed': 'Detailed',
        'think_hard': 'Think Hard', 'think_hard_rubric': 'Think Hard\n+ Rubric',
    }

    datasets = ['philosophy', 'weird_questions', 'futuristic_tech', 'all']
    for ds in datasets:
        if ds == 'all':
            df_ds = df_valid
            dtitle = 'All Fuzzy Tasks'
        else:
            df_ds = df_valid[df_valid['dataset'] == ds]
            dtitle = ds.replace('_', ' ').title()

        if len(df_ds) == 0:
            continue

        stats = df_ds.groupby('prompt_style')['score'].agg(['mean', 'std', 'count']).reset_index()
        stats['se'] = stats['std'] / np.sqrt(stats['count'])

        # Order by prompt_order
        present = [p for p in prompt_order if p in stats['prompt_style'].values]
        stats_ordered = stats.set_index('prompt_style').loc[present].reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [prompt_colors.get(p, '#999') for p in stats_ordered['prompt_style']]
        bars = ax.bar(range(len(stats_ordered)), stats_ordered['mean'],
                      yerr=stats_ordered['se'], color=colors, capsize=5,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(stats_ordered)))
        ax.set_xticklabels([prompt_labels.get(p, p) for p in stats_ordered['prompt_style']])
        ax.set_xlabel('Prompt Style', fontsize=12)
        ax.set_ylabel('Mean Rubric Score (0-48)', fontsize=12)
        ax.set_title(f'Opus 4.6 Prompt Sweep (High Effort) — {dtitle}', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean, se in zip(bars, stats_ordered['mean'], stats_ordered['se']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + se + 0.3,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        suffix = 'combined' if ds == 'all' else ds
        plt.savefig(RESULTS_DIR / f'sweep_bar_{suffix}.png', dpi=150)
        plt.close()

    # Scatter: score vs reasoning tokens by prompt style
    df_rt = df_valid[df_valid['reasoning_tokens'].notna() & (df_valid['reasoning_tokens'] > 0)]
    if len(df_rt) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        for style in prompt_order:
            df_s = df_rt[df_rt['prompt_style'] == style]
            if len(df_s) == 0:
                continue
            ax.scatter(df_s['reasoning_tokens'], df_s['score'],
                       color=prompt_colors.get(style, '#999'), alpha=0.4, s=30,
                       label=prompt_labels.get(style, style))
        ax.set_xlabel('Reasoning Tokens', fontsize=12)
        ax.set_ylabel('Rubric Score (0-48)', fontsize=12)
        ax.set_title('Opus 4.6: Score vs Reasoning Tokens by Prompt Style', fontsize=13)
        ax.legend(title='Prompt Style')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'sweep_scatter_reasoning.png', dpi=150)
        plt.close()

    plt.close('all')
    gc.collect()
    log_experiment("Sweep plots saved")


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('effort', 'sweep'):
        print("Usage: python3.13 run_opus_sweep.py [effort|sweep]")
        sys.exit(1)

    mode = sys.argv[1]
    log_memory(f"{mode}_pre_start")
    log_experiment(f"Starting {mode} experiment")

    if mode == 'effort':
        await run_effort_experiment()
    else:
        await run_sweep_experiment()

    log_experiment(f"{mode} experiment finished")
    log_memory(f"{mode}_finished")


if __name__ == '__main__':
    asyncio.run(main())
