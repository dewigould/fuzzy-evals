# Experiment: Effort Level & Prompting Sweep for Fuzzy Evals

This document describes two related experiments evaluating how **effort levels** (Claude's `verbosity` parameter) and **prompting strategies** affect LLM response quality on open-ended "fuzzy" tasks. A future Claude Code instance should be able to reproduce these experiments using the scripts and infrastructure described here.

## Overview

There are two experiments:

1. **Effort Experiment** (`run_effort_v2.py`): Tests Claude Opus 4.6 at 4 effort levels (low/medium/high/max) with reasoning enabled, measuring how verbosity affects quality and reasoning token usage.

2. **Prompting Sweep** (`run_sweep_v2.py`): Tests 7 models × 5 prompt styles, measuring how prompt engineering affects response quality across models.

Both use **LLM-as-judge** evaluation (GPT-5.2 at temperature=0) with structured rubrics to score responses on a 0-48 scale (6 criteria × 0-8 each).

---

## Infrastructure

### API
- All model calls go through **OpenRouter** (`https://openrouter.ai/api/v1/chat/completions`)
- API key is in `/workspace/.env` as `OPENROUTER_API_KEY`
- Judge model: `openai/gpt-5.2` at temperature=0

### Datasets (56 questions total)
- `/workspace/fuzzy-evals/dataset_jsons/philosophy_questions.json` — 10 philosophy questions (key: `question`)
- `/workspace/fuzzy-evals/dataset_jsons/weird_questions.json` — 46 speculative/unusual questions (key: `prompt`)

### Rubrics
- `/workspace/fuzzy-evals/rubrics/philosophy_rubric.md` — 6 criteria: thesis_clarity, argumentative_soundness, dialectical_engagement, precision_distinctions, substantive_contribution, example_quality
- `/workspace/fuzzy-evals/rubrics/weird_questions_rubric.md` — 6 criteria: willingness_to_engage, specificity_concreteness, reasoning_depth, intellectual_risk_taking, creative_insight, epistemic_calibration

Each criterion is scored 0-8, total max 48. The judge returns a JSON object with individual scores and a total.

### Dependencies
Use `python3.13`. Required packages: `aiohttp`, `pandas`, `matplotlib`, `numpy`, `python-dotenv`. Install with `pip3.13 install aiohttp pandas matplotlib numpy python-dotenv`.

If you hit `ModuleNotFoundError: No module named 'six.moves'`, run: `pip3.13 install six --upgrade && pip3.13 install python-dateutil --upgrade`.

---

## Experiment 1: Effort Levels (`run_effort_v2.py`)

### What it tests
Claude Opus 4.6 with **reasoning enabled** (`reasoning: {enabled: true}`) at 4 verbosity levels: `low`, `medium`, `high`, `max`. This controls how much effort the model puts into its response (mapped to `output_config.effort` internally).

### Key config
- Model: `anthropic/claude-opus-4.6`
- Temperature: `1.0` (required for thinking/extended output on Opus)
- Samples per question: 10
- Concurrency: semaphore=8, batch_size=30
- Extra body per request: `{'verbosity': '<level>', 'reasoning': {'enabled': True}}`

### Important: `verbosity` vs `reasoning` are SEPARATE parameters
- `verbosity` (low/medium/high/max) controls response detail level — maps to `output_config.effort`
- `reasoning: {enabled: true}` enables adaptive thinking (reasoning tokens)
- You need BOTH to get reasoning tokens. Without `reasoning: {enabled: true}`, reasoning tokens will be 0.
- At low/medium verbosity, the model may adaptively choose NOT to use reasoning tokens for some questions. This is expected behavior, not a bug.

### Token tracking
The `extract_token_counts()` function extracts from the API response `usage` dict:
- `reasoning_tokens`: from `usage.completion_tokens_details.reasoning_tokens`
- `output_tokens`: `completion_tokens - reasoning_tokens` (the visible response)
- `completion_tokens`: total (reasoning + output)
- `prompt_tokens`: input tokens

### Output
- `results_06_02/effort_v2_scores.parquet` — all results (2,240 rows = 56 questions × 10 samples × 4 effort levels)
- Bar plots of effort level vs mean rubric score (per dataset and combined)
- Scatter plots of score vs reasoning tokens, colored by effort level

### Previous results (now lost, for reference)
- LOW: mean score=18.2, mean reasoning_tokens=8, 11% of responses used reasoning
- MEDIUM: mean score=19.4, mean reasoning_tokens=64, 51% used reasoning
- HIGH: mean score=20.9, mean reasoning_tokens=408, 95% used reasoning
- MAX: mean score=17.9, mean reasoning_tokens=358, 59% used reasoning

Key finding: **non-monotonic relationship** — "high" outperforms "max". This was consistent across both experiments.

### How to run
```bash
mkdir -p /workspace/results_06_02
nohup python3.13 /workspace/run_effort_v2.py > /workspace/results_06_02/effort_nohup.log 2>&1 &
```

Use `nohup` because background processes can get killed by workspace restructuring. Monitor with:
```bash
tail -f /workspace/results_06_02/effort_nohup.log
```

The script has **resume capability** — if it crashes, re-running will pick up from the last completed condition via `effort_v2_partial.parquet`.

---

## Experiment 2: Prompting Sweep (`run_sweep_v2.py`)

### What it tests
7 models × 5 prompt styles, all graded by the same judge. Tests whether detailed prompting instructions improve response quality, and whether giving the model the actual grading rubric helps.

### Models

**Standard models** (temperature=0.7, semaphore=25, batch_size=60):
- `openai/gpt-4o`
- `qwen/qwen-2.5-72b-instruct`
- `qwen/qwen-2.5-7b-instruct`
- `anthropic/claude-sonnet-4.5`
- `google/gemini-2.5-flash`

**Opus models** (temperature=1.0, semaphore=12, batch_size=40, extra_body includes reasoning+verbosity):
- `anthropic/claude-opus-4.5`
- `anthropic/claude-opus-4.6`

Opus models use: `{'reasoning': {'enabled': True}, 'verbosity': 'high'}` — always at "high" effort since that was found optimal in Experiment 1.

### Prompt styles

1. **minimal**: `"Answer the following question.\n\n{question}"`

2. **medium**: `"Think carefully and give a thorough, well-reasoned answer to the following question. Take your time and consider multiple angles before responding.\n\n{question}"`

3. **detailed**: Long prompt describing what a great answer looks like (clear position, mechanisms, concrete examples, counterarguments, fresh angles, epistemic honesty). See `run_sweep_v2.py` lines 62-74 for full text.

4. **think_hard**: Instructs the model to plan, think step by step, try multiple approaches, backtrack, self-verify, self-criticize, consider critics, distinguish knowledge levels. See `run_sweep_v2.py` lines 76-90 for full text.

5. **think_hard_rubric**: Same as think_hard, but appends: `"Your answer will be evaluated on the following rubric. Study it carefully and optimize your response accordingly:"` followed by the dataset-specific rubric (philosophy or weird questions). The `build_prompt()` function at line 102 handles this — it selects the correct rubric based on the dataset.

### Known issue: think_hard_rubric breaks some models
When given the rubric in the prompt, GPT-4o and Qwen models tend to output rubric JSON (self-grading) instead of actually answering the question. GPT-4o scored 0 on 518/560 samples with think_hard_rubric. Claude models handle it fine. This is an interesting finding, not a bug to fix.

### Output
- `results_06_02/sweep_v2_scores.parquet` — all results (19,600 rows = 7 models × 5 prompts × 56 questions × 10 samples)
- Bar charts of mean score by model and prompt style (per dataset and combined)
- Scatter plots of score vs reasoning tokens for models that have them (Opus models)

### Previous results (now lost, for reference)
- 19,600 rows total, 19,539 valid scores
- Claude Opus 4.6 at "high" with think_hard prompt scored highest overall
- think_hard_rubric helped Claude models but destroyed GPT-4o and Qwen performance
- Prompting strategy matters more for some models than others

### How to run
```bash
mkdir -p /workspace/results_06_02
nohup python3.13 /workspace/run_sweep_v2.py > /workspace/results_06_02/sweep_nohup.log 2>&1 &
```

The sweep takes approximately 1-2 hours. The script has **resume capability** via `sweep_v2_partial.parquet` — it checks which (model, prompt_level) conditions are complete and skips them.

Runtime estimate per condition: ~2-3 minutes for standard models, ~5-8 minutes for Opus models (slower due to reasoning). Total: ~35 conditions × ~4 min avg = ~2+ hours.

---

## Grading Pipeline

Both scripts use the same grading approach:

1. **Generate completions**: Send the prompt to the model N=10 times at temperature > 0 to get diverse samples
2. **Judge each completion**: Send to GPT-5.2 at temperature=0 with the grading prompt:
   ```
   {system message about what you're grading}

   ## Question
   {the original question}

   ## Answer to Grade
   {the model's response}

   ## Grading Rubric
   {the full rubric markdown}
   ```
3. **Parse scores**: Extract JSON from judge response, compute total from 6 sub-scores
4. **Checkpoint**: Save partial results as parquet after each condition

The `parse_judge_score()` function handles various JSON formats (with/without code fences, text before/after). Parse rate is ~99%+ with GPT-5.2.

---

## Post-Experiment Analysis

After running, you should produce:

### Plots
1. **Bar chart**: Model (x-axis) × Prompt Style (grouped bars), Mean Score (y-axis), with standard error bars. One plot per dataset + one combined.
2. **Scatter plot**: Reasoning tokens (x-axis) vs Score (y-axis), colored by prompt style. One subplot per model (only models with reasoning tokens). No trend lines.

### Data artifacts
1. **Examples file**: Extract all responses from a particular model (e.g., Opus 4.6) into a separate parquet + JSONL for easy browsing.
2. **Jupyter notebook**: Create an interactive analysis notebook with:
   - Pivot tables of mean scores by prompt × dataset
   - Token usage stats
   - Helper functions: `show_response(row)`, `show_responses(df)`, `compare_question(question_id)`
   - Best/worst response browsing
   - Filtering by criteria (highest-scoring minimal prompts, biggest prompt uplift, etc.)
   - Score vs token scatter plots

---

## Gotchas and Lessons Learned

1. **Process persistence**: Use `nohup` for long-running experiments. Background processes (`&`) can get killed by other sessions or workspace restructuring.

2. **Workspace restructuring**: Other Claude Code instances may delete your results directory. Consider saving results to a path outside `/workspace/` or backing up parquet files to `/home/claude-user/`.

3. **Python version**: The workspace may have Python 3.8 as default `python`. Use `python3.13` explicitly.

4. **Opus temperature**: Claude Opus 4.5 and 4.6 require `temperature=1.0` for thinking/extended output. Other temperatures may cause errors.

5. **Rate limiting**: OpenRouter rate limits vary by model. The retry logic in the scripts handles this with exponential backoff (2s, 5s, 15s, 30s, 60s, 120s, 120s). If you see persistent rate limits on Opus, reduce `semaphore_size` from 12 to 8.

6. **Checkpointing**: Always checkpoint after each condition. The experiments are expensive (~$50-100 in API credits for the full sweep) and crashes/kills are common.

7. **Result validation**: After running, always check:
   - Total row count matches expected (models × prompts × questions × samples)
   - Valid score count (should be 99%+)
   - No model has all-zero scores (indicates API or parsing failure)
   - Reasoning tokens are non-zero for Opus models (indicates reasoning is actually enabled)
