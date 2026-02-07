"""
Effort-Level Experiment v2 — with reasoning token logging and prompt-style conditions.

Part 1: 4 effort levels (low/medium/high/max) × minimal prompt
Part 2: 4 effort levels × concise-reasoning prompt + verbose-reasoning prompt

All results stored in /workspace/results_06_02/
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
MODEL = 'anthropic/claude-opus-4.6'
SAMPLING_N = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / 'effort_v2_runner.log'),
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

PROMPT_CONCISE = (
    "Answer the following question. Your goal is to answer the question to the "
    "best of your ability, but to also be efficient and concise with your reasoning. "
    "Avoid wasting tokens on unnecessary reasoning, don't deliberate longer than "
    "necessary.\n\n{question}"
)

PROMPT_VERBOSE = (
    "Answer the following question. Reason through this extensively, consider "
    "every angle. Please be long and rambly in your response, but still answer "
    "it well.\n\n{question}"
)

PROMPT_STYLES = {
    'minimal': PROMPT_MINIMAL,
    'concise': PROMPT_CONCISE,
    'verbose': PROMPT_VERBOSE,
}

EFFORT_LEVELS = ['low', 'medium', 'high', 'max']


# ── API call (returns content + usage) ─────────────────────────────────────

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
    """Extract reasoning tokens, output tokens (visible), and total completion tokens."""
    completion_tokens = usage.get('completion_tokens')
    prompt_tokens = usage.get('prompt_tokens')

    details = usage.get('completion_tokens_details', {})
    reasoning_tokens = None
    if details and isinstance(details, dict):
        rt = details.get('reasoning_tokens')
        if rt is not None:
            reasoning_tokens = int(rt)

    # Output tokens = completion_tokens - reasoning_tokens (the visible response)
    output_tokens = None
    if completion_tokens is not None and reasoning_tokens is not None:
        output_tokens = int(completion_tokens) - reasoning_tokens
    elif completion_tokens is not None:
        output_tokens = int(completion_tokens)  # if no reasoning, all completion is output

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


# ── Main experiment ────────────────────────────────────────────────────────

async def run_condition(session, semaphore, all_questions, phil_rubric, weird_rubric,
                        prompt_style: str, effort_level: str, results: list):
    """Run one (prompt_style, effort_level) condition: all questions × N samples."""
    condition_label = f"{prompt_style}/{effort_level}"
    log_experiment(f"  Running condition: {condition_label}")
    start = time.time()

    prompt_template = PROMPT_STYLES[prompt_style]
    extra_body = {
        'verbosity': effort_level,
        'reasoning': {'enabled': True},
    }

    # Build tasks
    tasks = []
    task_meta = []
    for q in all_questions:
        full_prompt = prompt_template.format(question=q['question'])
        for sample_id in range(SAMPLING_N):
            messages = [{'role': 'user', 'content': full_prompt}]
            tasks.append(call_openrouter(
                session, MODEL, messages,
                temperature=1.0,
                semaphore=semaphore,
                extra_body=extra_body
            ))
            task_meta.append({
                'model': MODEL,
                'prompt_style': prompt_style,
                'effort_level': effort_level,
                'dataset': q['dataset'],
                'question_id': q['question_id'],
                'question': q['question'],
                'sample_id': sample_id,
            })

    # Run completions in batches
    batch_size = 30
    completions = []  # list of (content, usage) tuples
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        completions.extend(batch_results)
        log.info(f"    [{condition_label}] Completed batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")

    # Judge all completions
    log.info(f"    [{condition_label}] Judging {len(completions)} completions...")
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
            temperature=0.0, max_tokens=1024, semaphore=semaphore
        ))

    judge_responses = []
    for i in range(0, len(judge_tasks), batch_size):
        batch = judge_tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        judge_responses.extend(batch_results)
        log.info(f"    [{condition_label}] Judged batch {i//batch_size + 1}/{(len(judge_tasks)-1)//batch_size + 1}")

    # Assemble results
    for meta, (content, usage), (judge_content, _judge_usage) in zip(task_meta, completions, judge_responses):
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
    log_experiment(f"  Condition {condition_label} done in {elapsed/60:.1f} min ({len(task_meta)} samples)")


async def main():
    log_experiment("# Effort Experiment v2 — with reasoning token logging")
    log_experiment(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_experiment(f"Model: {MODEL}")

    philosophy, weird = load_datasets()
    phil_rubric, weird_rubric = load_rubrics()
    all_questions = philosophy + weird
    n_questions = len(all_questions)

    # Check for existing partial results to enable resume
    partial_path = RESULTS_DIR / 'effort_v2_partial.parquet'
    results = []
    completed_conditions = set()
    if partial_path.exists():
        existing_df = pd.read_parquet(partial_path)
        results = existing_df.to_dict('records')
        for (ps, el), group in existing_df.groupby(['prompt_style', 'effort_level']):
            # A condition is complete if it has the expected number of rows
            expected = n_questions * SAMPLING_N
            if len(group) >= expected:
                completed_conditions.add((ps, el))
        log_experiment(f"Resumed: {len(results)} existing results, {len(completed_conditions)} completed conditions")

    # Build condition list — experiment 1: minimal prompt only, 4 effort levels
    conditions = []
    for effort_level in EFFORT_LEVELS:
        conditions.append(('minimal', effort_level))

    total_conditions = len(conditions)
    total_completions = total_conditions * n_questions * SAMPLING_N
    log_experiment(f"Conditions: {total_conditions} (minimal prompt × {len(EFFORT_LEVELS)} effort levels)")
    log_experiment(f"Questions: {n_questions}, Samples per question: {SAMPLING_N}")
    log_experiment(f"Total completions: {total_completions} + {total_completions} judge calls")

    semaphore = asyncio.Semaphore(8)

    async with aiohttp.ClientSession() as session:
        for idx, (prompt_style, effort_level) in enumerate(conditions):
            if (prompt_style, effort_level) in completed_conditions:
                log_experiment(f"  Skipping {prompt_style}/{effort_level} (already done)")
                continue

            log_experiment(f"Condition {idx+1}/{total_conditions}: prompt={prompt_style}, effort={effort_level}")
            await run_condition(
                session, semaphore, all_questions, phil_rubric, weird_rubric,
                prompt_style, effort_level, results
            )

            # Checkpoint after each condition
            df_tmp = pd.DataFrame(results)
            df_tmp.to_parquet(partial_path, index=False)
            log.info(f"  Checkpointed {len(results)} results")

    # Final save
    df = pd.DataFrame(results)
    df.to_parquet(RESULTS_DIR / 'effort_v2_scores.parquet', index=False)
    log_experiment(f"Experiment complete. Total: {len(results)}, Valid scores: {df['score'].notna().sum()}")

    # Generate all plots
    generate_plots(df)


# ── Plotting ───────────────────────────────────────────────────────────────

def generate_plots(df):
    """Generate all experiment plots."""
    df_valid = df[df['score'].notna()].copy()
    if len(df_valid) == 0:
        log.warning("No valid scores — skipping plots")
        return

    # --- Plot 1: Bar plot of effort level vs mean rubric score (minimal prompt only) ---
    df_minimal = df_valid[df_valid['prompt_style'] == 'minimal']
    if len(df_minimal) > 0:
        _bar_plot_effort(df_minimal, 'Effort Level vs Rubric Score (Minimal Prompt)',
                         'effort_bar_minimal')

    # --- Plot 1b: Bar plot grouped by prompt style, for each effort level ---
    _bar_plot_prompt_x_effort(df_valid, 'Rubric Score by Prompt Style × Effort Level',
                              'effort_bar_prompt_x_effort')

    # --- Plot 2: Scatter of rubric score vs reasoning tokens, colored by effort level ---
    df_with_rt = df_valid[df_valid['reasoning_tokens'].notna() & (df_valid['reasoning_tokens'] > 0)]
    if len(df_with_rt) > 0:
        # 2a: minimal prompt only
        df_rt_minimal = df_with_rt[df_with_rt['prompt_style'] == 'minimal']
        if len(df_rt_minimal) > 0:
            _scatter_score_vs_tokens(df_rt_minimal, 'reasoning_tokens',
                                     'Rubric Score vs Reasoning Tokens (Minimal Prompt)',
                                     'scatter_score_vs_rt_minimal', xlabel='Reasoning Tokens')
        # 2b: all prompt styles
        _scatter_score_vs_tokens_by_style(df_with_rt, 'reasoning_tokens',
                                           'Rubric Score vs Reasoning Tokens (All Prompts)',
                                           'scatter_score_vs_rt_all', xlabel='Reasoning Tokens')
    else:
        log.warning("No reasoning token data available — skipping reasoning token scatter plots")

    # --- Plot 3: Scatter of rubric score vs output tokens (visible response) ---
    df_with_ot = df_valid[df_valid['output_tokens'].notna() & (df_valid['output_tokens'] > 0)]
    if len(df_with_ot) > 0:
        df_ot_minimal = df_with_ot[df_with_ot['prompt_style'] == 'minimal']
        if len(df_ot_minimal) > 0:
            _scatter_score_vs_tokens(df_ot_minimal, 'output_tokens',
                                     'Rubric Score vs Output Tokens (Minimal Prompt)',
                                     'scatter_score_vs_ot_minimal', xlabel='Output Tokens')
        _scatter_score_vs_tokens_by_style(df_with_ot, 'output_tokens',
                                           'Rubric Score vs Output Tokens (All Prompts)',
                                           'scatter_score_vs_ot_all', xlabel='Output Tokens')

    log_experiment("All plots saved")


def _bar_plot_effort(df, title, filename):
    """Simple bar plot: effort levels on x-axis, mean score on y-axis."""
    effort_order = ['low', 'medium', 'high', 'max']
    colors = ['#4ECDC4', '#FFD93D', '#FF6B6B', '#9B59B6']

    for dataset_name in ['philosophy', 'weird_questions', 'all']:
        if dataset_name == 'all':
            df_ds = df
            dtitle = 'All Fuzzy Tasks'
        else:
            df_ds = df[df['dataset'] == dataset_name]
            dtitle = dataset_name.replace('_', ' ').title()

        if len(df_ds) == 0:
            continue

        stats = df_ds.groupby('effort_level')['score'].agg(['mean', 'std', 'count']).reset_index()
        stats['se'] = stats['std'] / np.sqrt(stats['count'])
        stats['effort_level'] = pd.Categorical(stats['effort_level'], categories=effort_order, ordered=True)
        stats = stats.sort_values('effort_level')

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(range(len(stats)), stats['mean'], yerr=stats['se'],
                      color=colors[:len(stats)], capsize=5, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels([e.capitalize() for e in stats['effort_level']])
        ax.set_xlabel('Effort Level', fontsize=12)
        ax.set_ylabel('Mean Rubric Score (0-48)', fontsize=12)
        ax.set_title(f'{title} — {dtitle}', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean, se in zip(bars, stats['mean'], stats['se']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + se + 0.3,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        suffix = 'combined' if dataset_name == 'all' else dataset_name
        plt.savefig(RESULTS_DIR / f'{filename}_{suffix}.png', dpi=150)
        plt.close()


def _bar_plot_prompt_x_effort(df, title, filename):
    """Grouped bar plot: effort level on x-axis, bars grouped by prompt style."""
    effort_order = ['low', 'medium', 'high', 'max']
    prompt_styles = ['minimal', 'concise', 'verbose']
    style_colors = {'minimal': '#4ECDC4', 'concise': '#FFD93D', 'verbose': '#FF6B6B'}

    for dataset_name in ['philosophy', 'weird_questions', 'all']:
        if dataset_name == 'all':
            df_ds = df
            dtitle = 'All Fuzzy Tasks'
        else:
            df_ds = df[df['dataset'] == dataset_name]
            dtitle = dataset_name.replace('_', ' ').title()

        if len(df_ds) == 0:
            continue

        stats = df_ds.groupby(['prompt_style', 'effort_level'])['score'].agg(['mean', 'std', 'count']).reset_index()
        stats['se'] = stats['std'] / np.sqrt(stats['count'])

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(effort_order))
        width = 0.25

        for i, style in enumerate(prompt_styles):
            style_data = stats[stats['prompt_style'] == style].copy()
            style_data['effort_level'] = pd.Categorical(style_data['effort_level'],
                                                         categories=effort_order, ordered=True)
            style_data = style_data.sort_values('effort_level')
            means = style_data['mean'].values
            ses = style_data['se'].values
            bars = ax.bar(x + i * width, means, width, yerr=ses,
                          label=style.capitalize(), color=style_colors[style],
                          capsize=3, edgecolor='black', linewidth=0.5)
            for bar, mean, se in zip(bars, means, ses):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + se + 0.2,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x + width)
        ax.set_xticklabels([e.capitalize() for e in effort_order])
        ax.set_xlabel('Effort Level', fontsize=12)
        ax.set_ylabel('Mean Rubric Score (0-48)', fontsize=12)
        ax.set_title(f'{title} — {dtitle}', fontsize=13)
        ax.legend(title='Prompt Style')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        suffix = 'combined' if dataset_name == 'all' else dataset_name
        plt.savefig(RESULTS_DIR / f'{filename}_{suffix}.png', dpi=150)
        plt.close()


def _scatter_score_vs_tokens(df, token_col, title, filename, xlabel='Tokens'):
    """Scatter plot: token count vs rubric score, colored by effort level."""
    effort_order = ['low', 'medium', 'high', 'max']
    effort_colors = {'low': '#4ECDC4', 'medium': '#FFD93D', 'high': '#FF6B6B', 'max': '#9B59B6'}

    fig, ax = plt.subplots(figsize=(10, 7))
    for effort in effort_order:
        df_e = df[df['effort_level'] == effort]
        if len(df_e) == 0:
            continue
        ax.scatter(df_e[token_col], df_e['score'],
                   color=effort_colors[effort], alpha=0.4, s=30, label=effort.capitalize())

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Rubric Score (0-48)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(title='Effort Level')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{filename}.png', dpi=150)
    plt.close()


def _scatter_score_vs_tokens_by_style(df, token_col, title, filename, xlabel='Tokens'):
    """Scatter plot: token count vs rubric score, effort color + style marker."""
    effort_order = ['low', 'medium', 'high', 'max']
    effort_colors = {'low': '#4ECDC4', 'medium': '#FFD93D', 'high': '#FF6B6B', 'max': '#9B59B6'}
    style_markers = {'minimal': 'o', 'concise': 's', 'verbose': '^'}

    fig, ax = plt.subplots(figsize=(10, 7))
    for effort in effort_order:
        for style, marker in style_markers.items():
            df_sub = df[(df['effort_level'] == effort) & (df['prompt_style'] == style)]
            if len(df_sub) == 0:
                continue
            ax.scatter(df_sub[token_col], df_sub['score'],
                       color=effort_colors[effort], marker=marker,
                       alpha=0.35, s=30, label=f'{effort.capitalize()}/{style}')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Rubric Score (0-48)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(title='Effort/Style', fontsize=7, ncol=2, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{filename}.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    asyncio.run(main())
