"""
Sweep v2: Full prompting sweep with reasoning token tracking.
7 models × 5 prompts × 56 questions × N=10

Results stored in results_06_02/
"""

import asyncio
import json
import logging
import os
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

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
RESULTS_DIR = Path('/workspace/results_06_02')
RESULTS_DIR.mkdir(exist_ok=True)
LOG_FILE = RESULTS_DIR / 'experiment_log.md'

JUDGE_MODEL = 'openai/gpt-5.2'
SAMPLING_N = 10
SAMPLING_TEMP = 0.7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / 'sweep_v2_runner.log'),
    ]
)
log = logging.getLogger(__name__)


def log_experiment(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n### {ts}\n{msg}\n")
    log.info(msg)


# ── Prompts ────────────────────────────────────────────────────────────────

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
- It demonstrates intellectual courage—willingness to say something that could be wrong rather than retreating to safe platitudes

Do not pad your answer with unnecessary caveats or qualifications. Do not open with "This is a great question" or "There are many perspectives on this." Just think hard and answer directly.

{question}"""

PROMPT_THINK_HARD = """Answer the following question. Give it your absolute best effort — this is important.

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

ALL_PROMPT_LEVELS = ['minimal', 'medium', 'detailed', 'think_hard', 'think_hard_rubric']


def build_prompt(prompt_level: str, question: str, dataset: str,
                 phil_rubric: str, weird_rubric: str) -> str:
    if prompt_level in SIMPLE_PROMPTS:
        return SIMPLE_PROMPTS[prompt_level].format(question=question)

    if prompt_level == 'think_hard_rubric':
        rubric = phil_rubric if dataset == 'philosophy' else weird_rubric
        return f"""Answer the following question. Give it your absolute best effort — this is important.

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

{rubric}

{question}"""

    raise ValueError(f"Unknown prompt level: {prompt_level}")


# ── Models ─────────────────────────────────────────────────────────────────

STANDARD_MODELS = [
    'openai/gpt-4o',
    'qwen/qwen-2.5-72b-instruct',
    'qwen/qwen-2.5-7b-instruct',
    'anthropic/claude-sonnet-4.5',
    'google/gemini-2.5-flash',
]

OPUS_MODELS = [
    'anthropic/claude-opus-4.5',
    'anthropic/claude-opus-4.6',
]

ALL_MODELS = STANDARD_MODELS + OPUS_MODELS

MODEL_CONFIG = {}
for m in STANDARD_MODELS:
    MODEL_CONFIG[m] = {
        'temperature': SAMPLING_TEMP,
        'extra_body': {},
        'semaphore_size': 25,
        'batch_size': 60,
        'max_tokens': 4096,
    }
for m in OPUS_MODELS:
    MODEL_CONFIG[m] = {
        'temperature': 1.0,
        'extra_body': {'reasoning': {'enabled': True}, 'verbosity': 'high'},
        'semaphore_size': 12,
        'batch_size': 40,
        'max_tokens': 4096,
    }


# ── API call ───────────────────────────────────────────────────────────────

async def call_openrouter(session, model, messages, temperature=0.7,
                          max_tokens=4096, semaphore=None, extra_body=None):
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


# ── Grading ────────────────────────────────────────────────────────────────

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


def load_datasets():
    base = Path('/workspace/fuzzy-evals/dataset_jsons')
    with open(base / 'philosophy_questions.json') as f:
        phil_raw = json.load(f)
    philosophy = [
        {'dataset': 'philosophy', 'question_id': i, 'question': q['question']}
        for i, q in enumerate(phil_raw)
    ]
    with open(base / 'weird_questions.json') as f:
        weird_raw = json.load(f)
    weird = [
        {'dataset': 'weird_questions', 'question_id': i, 'question': q['prompt']}
        for i, q in enumerate(weird_raw)
    ]
    return philosophy, weird


def load_rubrics():
    base = Path('/workspace/fuzzy-evals/rubrics')
    return (base / 'philosophy_rubric.md').read_text(), (base / 'weird_questions_rubric.md').read_text()


# ── Run one condition ──────────────────────────────────────────────────────

async def run_condition(session, model, prompt_level, all_questions,
                        phil_rubric, weird_rubric, semaphore, config):
    label = f"{model.split('/')[-1]}/{prompt_level}"
    log_experiment(f"  Running: {label}")
    start = time.time()

    extra_body = config['extra_body']
    temperature = config['temperature']
    max_tokens = config['max_tokens']
    batch_size = config['batch_size']

    tasks = []
    task_meta = []
    for q in all_questions:
        full_prompt = build_prompt(prompt_level, q['question'], q['dataset'],
                                   phil_rubric, weird_rubric)
        for sample_id in range(SAMPLING_N):
            messages = [{'role': 'user', 'content': full_prompt}]
            tasks.append(call_openrouter(
                session, model, messages,
                temperature=temperature,
                max_tokens=max_tokens,
                semaphore=semaphore,
                extra_body=extra_body if extra_body else None,
            ))
            task_meta.append({
                'model': model,
                'prompt_level': prompt_level,
                'dataset': q['dataset'],
                'question_id': q['question_id'],
                'question': q['question'],
                'sample_id': sample_id,
            })

    # Run completions
    completions = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        completions.extend(batch_results)
        log.info(f"    [{label}] Completed {min(i+batch_size, len(tasks))}/{len(tasks)}")

    # Judge
    log.info(f"    [{label}] Judging {len(completions)} completions...")
    judge_sem = asyncio.Semaphore(25)
    judge_tasks = []
    for meta, (content, usage) in zip(task_meta, completions):
        if isinstance(content, str) and content.startswith("ERROR:"):
            async def _err(c=content):
                return c, {}
            judge_tasks.append(_err())
            continue

        if meta['dataset'] == 'philosophy':
            rubric = phil_rubric
            system = "You are grading a philosophical answer for quality of argumentation and exposition."
        else:
            rubric = weird_rubric
            system = "You are grading an answer for quality of thought, research, effort, and exposition."

        grading_prompt = f"""{system}

## Question

{meta['question']}

## Answer to Grade

{content}

## Grading Rubric

{rubric}"""
        messages = [{'role': 'user', 'content': grading_prompt}]
        judge_tasks.append(call_openrouter(
            session, JUDGE_MODEL, messages,
            temperature=0.0, max_tokens=1024, semaphore=judge_sem
        ))

    judge_responses = []
    for i in range(0, len(judge_tasks), batch_size):
        batch = judge_tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        judge_responses.extend(batch_results)
        log.info(f"    [{label}] Judged {min(i+batch_size, len(judge_tasks))}/{len(judge_tasks)}")

    results = []
    for meta, (content, usage), (judge_content, _) in zip(task_meta, completions, judge_responses):
        if isinstance(judge_content, str) and judge_content.startswith("ERROR:"):
            total_score, sub_scores = None, {}
        else:
            total_score, sub_scores = parse_judge_score(judge_content)

        token_counts = extract_token_counts(usage)
        result = {
            **meta,
            'raw_output': content if isinstance(content, str) else str(content),
            'judge_response': judge_content if isinstance(judge_content, str) else str(judge_content),
            'score': total_score,
            **token_counts,
        }
        result.update(sub_scores)
        results.append(result)

    elapsed = time.time() - start
    valid = sum(1 for r in results if r['score'] is not None)
    log_experiment(f"  Done: {label} in {elapsed/60:.1f} min ({valid}/{len(results)} valid)")
    return results


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    log_experiment("# Sweep v2: Full prompting sweep (fresh run)")
    log_experiment(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    philosophy, weird = load_datasets()
    phil_rubric, weird_rubric = load_rubrics()
    all_questions = philosophy + weird
    n_questions = len(all_questions)

    # Check for partial results to enable resume
    partial_path = RESULTS_DIR / 'sweep_v2_partial.parquet'
    all_results = []
    completed_conditions = set()
    if partial_path.exists():
        partial_df = pd.read_parquet(partial_path)
        all_results = partial_df.to_dict('records')
        for (m, p), group in partial_df.groupby(['model', 'prompt_level']):
            if len(group) >= n_questions * SAMPLING_N * 0.95:
                completed_conditions.add((m, p))
        log_experiment(f"Resumed: {len(all_results)} existing results, {len(completed_conditions)} completed")

    # Build condition list
    conditions = []
    for model in ALL_MODELS:
        for prompt_level in ALL_PROMPT_LEVELS:
            if (model, prompt_level) not in completed_conditions:
                conditions.append((model, prompt_level))

    total = len(ALL_MODELS) * len(ALL_PROMPT_LEVELS)
    log_experiment(f"Conditions: {len(conditions)} to run out of {total} total")
    log_experiment(f"New completions: {len(conditions) * n_questions * SAMPLING_N}")

    # Run — standard models first (faster), then Opus
    model_order = STANDARD_MODELS + OPUS_MODELS

    async with aiohttp.ClientSession() as session:
        for model in model_order:
            model_conditions = [(m, p) for m, p in conditions if m == model]
            if not model_conditions:
                log_experiment(f"Skipping {model} (all done)")
                continue

            config = MODEL_CONFIG[model]
            semaphore = asyncio.Semaphore(config['semaphore_size'])
            log_experiment(f"## Model: {model} ({len(model_conditions)} conditions)")

            for _, prompt_level in model_conditions:
                results = await run_condition(
                    session, model, prompt_level, all_questions,
                    phil_rubric, weird_rubric, semaphore, config
                )
                all_results.extend(results)
                pd.DataFrame(all_results).to_parquet(partial_path, index=False)
                log.info(f"  Checkpointed {len(all_results)} total results")

    df = pd.DataFrame(all_results)
    df.to_parquet(RESULTS_DIR / 'sweep_v2_scores.parquet', index=False)
    log_experiment(f"Saved final: {len(df)} rows, {df['score'].notna().sum()} valid")

    generate_plots(df)
    log_experiment("All plots generated. Done.")


# ── Plotting ───────────────────────────────────────────────────────────────

def generate_plots(df):
    df_valid = df[df['score'].notna()].copy()
    df_valid['model_short'] = df_valid['model'].apply(lambda x: x.split('/')[-1])

    prompt_order = ['minimal', 'medium', 'detailed', 'think_hard', 'think_hard_rubric']
    prompt_colors = {
        'minimal': '#4ECDC4',
        'medium': '#FFD93D',
        'detailed': '#FF6B6B',
        'think_hard': '#9B59B6',
        'think_hard_rubric': '#3498DB',
    }
    prompt_labels = {
        'minimal': 'Minimal',
        'medium': 'Medium',
        'detailed': 'Detailed',
        'think_hard': 'Think Hard',
        'think_hard_rubric': 'Think Hard\n+ Rubric',
    }

    for dataset_name in ['philosophy', 'weird_questions', 'all']:
        if dataset_name == 'all':
            df_ds = df_valid
            dtitle = 'All Fuzzy Tasks'
        else:
            df_ds = df_valid[df_valid['dataset'] == dataset_name]
            dtitle = dataset_name.replace('_', ' ').title()

        if len(df_ds) == 0:
            continue

        stats = df_ds.groupby(['model_short', 'prompt_level'])['score'].agg(
            ['mean', 'std', 'count']).reset_index()
        stats['se'] = stats['std'] / np.sqrt(stats['count'])

        models = sorted(stats['model_short'].unique())
        prompt_levels_present = [p for p in prompt_order if p in stats['prompt_level'].unique()]
        n_prompts = len(prompt_levels_present)

        fig, ax = plt.subplots(figsize=(max(16, len(models) * 2.2), 7))
        x = np.arange(len(models))
        width = 0.8 / n_prompts

        for i, level in enumerate(prompt_levels_present):
            level_data = stats[stats['prompt_level'] == level].set_index('model_short')
            means = [level_data.loc[m, 'mean'] if m in level_data.index else 0 for m in models]
            ses = [level_data.loc[m, 'se'] if m in level_data.index else 0 for m in models]
            bars = ax.bar(x + i * width - (n_prompts - 1) * width / 2, means, width,
                          yerr=ses, label=prompt_labels.get(level, level),
                          color=prompt_colors.get(level, '#999'),
                          capsize=2, edgecolor='black', linewidth=0.3)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Mean Rubric Score (0-48)', fontsize=12)
        ax.set_title(f'Prompting Sweep v2 — {dtitle}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
        ax.legend(title='Prompt Level', fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        suffix = 'combined' if dataset_name == 'all' else dataset_name
        plt.savefig(RESULTS_DIR / f'sweep_v2_{suffix}.png', dpi=150)
        plt.close()
        log.info(f"Saved sweep_v2_{suffix}.png")

    # Scatter: score vs reasoning tokens (Opus models)
    df_rt = df_valid[df_valid['reasoning_tokens'].notna() & (df_valid['reasoning_tokens'] > 0)]
    if len(df_rt) > 0:
        models_with_rt = sorted(df_rt['model_short'].unique())
        n_models = len(models_with_rt)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

        for idx, model_short in enumerate(models_with_rt):
            ax = axes[idx // cols][idx % cols]
            d = df_rt[df_rt['model_short'] == model_short]
            for level in prompt_order:
                dl = d[d['prompt_level'] == level]
                if len(dl) == 0:
                    continue
                jitter = np.random.default_rng(42).uniform(-0.4, 0.4, size=len(dl))
                ax.scatter(dl['reasoning_tokens'], dl['score'].values + jitter,
                           color=prompt_colors.get(level, '#999'),
                           alpha=0.4, s=20, label=prompt_labels.get(level, level))
            ax.set_title(model_short, fontsize=11, fontweight='bold')
            ax.set_xlabel('Reasoning Tokens', fontsize=9)
            ax.set_ylabel('Score', fontsize=9)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)

        for idx in range(n_models, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle('Score vs Reasoning Tokens by Model', fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(RESULTS_DIR / 'sweep_v2_scatter_reasoning.png', dpi=150)
        plt.close()
        log.info("Saved sweep_v2_scatter_reasoning.png")


if __name__ == '__main__':
    asyncio.run(main())
