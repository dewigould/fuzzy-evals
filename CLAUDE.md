# Fuzzy Evals — Agent Guide


### Headlines
This is a repo designed to study distillation. The goal is to distill reasoning traces into base models, and study intra- and cross-domain performance. We distill math and reasoning traces into base models, and look for cases where math improves math performance, and code improves code performance.

We are using the tinker API and cookbook to run distillation, and evaluate across a suite of evals (in eval_tools).

This is an exploratory project, we experiment with parameters, datasets, reasoning traces, base models to try to find combinations which lead to strong intra-domain performance, and good cross-domain performance.


## Quick Start

```bash
cd /workspace/fuzzy-evals

# Run a single training experiment
python run.py single --name math_sonnet --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --task math --traces sonnet --max-length 4096

# Evaluate a base model (no training)
python run.py base-eval --name qwen30b_base \
    --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Run a batch of experiments from a Python file
python run.py batch my_overnight.py --max-parallel 4

# Evaluate an existing checkpoint (skip training)
python run.py eval --sampler-path "tinker://..." \
    --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507" --task math

# Compare results across experiments
python run.py compare logs/2026-02-26_*
```

## Architecture

```
fuzzy-evals/
├── config.py           # Constants, paths, frozen dataclass configs
├── infer.py            # Inference (renderer, tokenizer, think prefix seeding)
├── utils.py            # Shared utilities (cache, parsing, OpenRouter calls)
├── train.py            # LoRA SFT training (data filtering + tinker SDK)
├── evaluate.py         # Eval orchestrator (runs eval_tools/ modules)
├── analyze.py          # Plots (checkpoint curve, training loss, comparison)
├── runner.py           # Pipeline orchestrator (train → eval → select → full eval)
├── run.py              # CLI entry point (single, base-eval, batch, eval, compare)
├── experiments/        # Experiment launch scripts (.sh)
├── eval_tools/         # Per-dataset evaluation modules (11 datasets)
├── fuzzy_data/         # Fuzzy eval data (questions, rubrics, graders)
│   ├── questions/      # Question JSONs (philosophy, weird, futuristic tech)
│   ├── rubrics/        # LLM judge rubrics (6 criteria, 0-8 each, max 48)
│   └── graders/        # Grading prompt builders per fuzzy dataset
├── traces/             # Symlinks to reasoning trace data
├── filtered_data/      # [gitignored] Token-filtered trace cache
├── eval_cache/         # [gitignored] Cached eval datasets
└── logs/               # [gitignored] Experiment output directories
```

### Pipeline Flow

**Training experiment** (`run_experiment`):
1. **Filter data** — tokenize traces, keep examples ≤ max_length, cache in `filtered_data/`
2. **Train** — LoRA SFT via tinker SDK
3. **Checkpoint eval** — fast eval on small subset (default: math_500 + mbppplus, 100 problems each)
4. **Select best** — pick checkpoint with highest primary metric accuracy
5. **Full eval** — comprehensive benchmark suite (all 11 datasets)
6. **Generate plots + summary** — checkpoint curve, training loss, summary.json

**Base model eval** (`run_base_eval`):
1. Full eval on specified datasets
2. Generate summary.json

**Comparison** (`compare`):
1. Load summary.json from multiple log dirs
2. Generate grouped bar chart + CSV table

## Config Reference

### `TrainingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_model` | str | required | HuggingFace model ID |
| `task` | str | required | `"math"` or `"code"` |
| `trace_source` | str | required | `"sonnet"`, `"qwen"`, or `"kimi"` |
| `data_path` | str\|None | None | Override trace JSONL path |
| `max_length` | int | 4096 | Max sequence length in tokens |
| `lora_rank` | int | 32 | LoRA rank |
| `learning_rate` | float | 2e-4 | Learning rate |
| `lr_schedule` | str | "cosine" | LR schedule |
| `batch_size` | int | 50 | Batch size |
| `max_samples` | int | 25000 | Max training samples |
| `save_every` | int | 100 | Save checkpoint every N steps |
| `eval_every` | int | 100 | Evaluate every N steps |

### `CheckpointEvalConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `datasets` | tuple | ("math_500", "mbppplus") | Datasets for checkpoint eval |
| `max_problems` | int | 100 | Problems per dataset |
| `max_tokens` | int\|None | None | None = inherit from training.max_length |

### `FullEvalConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `datasets` | tuple | (all 6 hard benchmarks) | Datasets for full eval |
| `max_problems` | int\|None | None | Limit problems (None = full) |
| `max_tokens` | int\|None | None | None = inherit from training.max_length (8192 for base evals) |

### `Experiment`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Experiment name |
| `training` | TrainingConfig | required | Training configuration |
| `checkpoint_eval` | CheckpointEvalConfig | default | Checkpoint eval config |
| `full_eval` | FullEvalConfig | default | Full eval config |
| `skip_training` | bool | False | Skip training, use sampler_path |
| `sampler_path` | str\|None | None | Pre-trained checkpoint URI |

### `BaseModelEval`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Eval name |
| `base_model` | str | required | HuggingFace model ID |
| `eval` | FullEvalConfig | default | Eval configuration |

## Batch Files

Define `EXPERIMENTS: list[Experiment | BaseModelEval]` in a Python file:

```python
from config import Experiment, BaseModelEval, TrainingConfig, FullEvalConfig

BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"

EXPERIMENTS = [
    BaseModelEval(name="qwen30b_base", base_model=BASE),
    Experiment(
        name="math_sonnet_4k",
        training=TrainingConfig(
            base_model=BASE, task="math", trace_source="sonnet", max_length=4096
        ),
    ),
    Experiment(
        name="code_sonnet_4k",
        training=TrainingConfig(
            base_model=BASE, task="code", trace_source="sonnet", max_length=4096
        ),
    ),
]
```

Run with: `python run.py batch my_overnight.py --max-parallel 4`

## Evaluation Datasets (11 total)

| Dataset | Type | Metric | Grading |
|---------|------|--------|---------|
| math_500 | Math | accuracy | `extract_boxed` + `grade_answer` (sympy) |
| aime | Math | accuracy | same |
| omni_math | Math | accuracy | Two-stage: math grading → GPT-5.2 judge fallback |
| kodcode_500 | Code | accuracy | Subprocess pytest-style tests |
| codeforces_500 | Code | accuracy | stdin/stdout test execution |
| livecodebench_v5 | Code | accuracy | Subprocess execution |
| humanevalplus | Code | accuracy | Subprocess execution |
| mbppplus | Code | accuracy | Subprocess execution |
| fuzzy_philosophy | Fuzzy | mean_score | GPT-5.2 judge with rubric (max 48) |
| fuzzy_weird_qs | Fuzzy | mean_score | GPT-5.2 judge with rubric |
| fuzzy_futuristic_tech | Fuzzy | mean_score | GPT-5.2 judge with rubric |

### Result Parquet Columns

**Math datasets**: `problem`, `expected_answer`, `model_answer`, `full_response`, `correct` (bool)
**Code datasets**: `problem`, `full_response`, `extracted_code`, `passed` (bool) — note: KodCode uses `passed`, not `correct`
**Fuzzy datasets**: `question`, `full_response`, `score` (0-48), `judge_response`

## Logs Structure

```
logs/
└── 2026-02-26_2130_math_sonnet_4k/
    ├── config.json                          # Experiment config (frozen)
    ├── training/
    │   ├── metrics.jsonl                    # Per-step training metrics
    │   └── checkpoints.jsonl                # {batch, sampler_path} per checkpoint
    ├── checkpoint_eval/
    │   └── step100/                         # Per-checkpoint eval results
    │       └── results_*.parquet
    ├── selected_checkpoint.json             # {step, sampler_path, reason}
    ├── full_eval/
    │   ├── results_*.parquet                # Per-dataset result files
    │   └── eval_summary.json                # {dataset: {accuracy: ...}}
    ├── plots/
    │   ├── checkpoint_curve.png
    │   └── training_loss.png
    └── summary.json                         # Final summary with all results
```

## Data

### Trace Sources

| Source | Description | Available Tasks |
|--------|-------------|----------------|
| sonnet | Claude Sonnet 4.5 reasoning traces | math, code |
| qwen | Qwen3-235B thinking traces | math, code |
| kimi | Kimi-K2.5 reasoning traces | math, code |

Trace data is stored on HuggingFace: [`dewigould/fuzzy-evals-traces`](https://huggingface.co/datasets/dewigould/fuzzy-evals-traces) (private). Download into `traces/`:

```bash
huggingface-cli download dewigould/fuzzy-evals-traces --repo-type dataset --local-dir traces/
```

### Filtered Data Cache

Token-filtered traces are cached in `filtered_data/` with key: `{model_short}_{task}_{traces}_{max_length}.jsonl`. Computed once and shared across experiments. Delete a cache file to recompute.

## Base Models

| Short Name | HuggingFace ID |
|-----------|----------------|
| llama8b_base | meta-llama/Llama-3.1-8B |
| llama8b_it | meta-llama/Llama-3.1-8B-Instruct |
| llama70b_it | meta-llama/Llama-3.3-70B-Instruct |
| qwen8b_base | Qwen/Qwen3-8B-Base |
| qwen30b_base | Qwen/Qwen3-30B-A3B-Base |
| qwen30b_it | Qwen/Qwen3-30B-A3B-Instruct-2507 |

## Think Prefix Logic

Distilled models learned `<think>...</think>` reasoning from teacher traces. At eval time:

- **In-domain** (e.g., math-distilled on math datasets): `think_prefix=False` — model generates `<think>` natively
- **Cross-domain** (e.g., math-distilled on code datasets): `think_prefix=True` — seed `<think>\n` via renderer's `prefill` parameter
- **Base model**: `think_prefix=False` always

**Critical**: Think prefix is a generation biasing technique, not a post-processor. Never modify model output after sampling. See `infer.py:generate()`.

## ContentPart TypedDicts

The Qwen3 renderer's `parse_response()` returns content as either:
- `str` — when no `<think>` tags
- `list[ContentPart]` — TypedDicts (plain dicts), NOT objects with attributes

ContentPart types: `{"type": "thinking", "thinking": str}` and `{"type": "text", "text": str}`.
Use `part.get("type")` for discrimination, **never** `hasattr()`.

## Common Pitfalls

1. **`hasattr()` on ContentParts**: TypedDicts are plain dicts — `hasattr(part, "thinking")` is always False. Use `part.get("type") == "thinking"`.
2. **Post-hoc `<think>` prepend**: Never prepend `<think>\n` to model output. Use `renderer.build_generation_prompt(messages, prefill="<think>\n")`.
3. **max_tokens vs context window**: Model context is 32,768 tokens. Leave headroom for prompts (28,672 default).
4. **Truncation kills accuracy**: Distilled models generate very long CoT. If output is truncated before `\boxed{}` or code block, the answer is lost.
5. **Code eval OOM**: Code datasets run subprocess tests that can exhaust memory. `serialize_code_eval=True` (default) limits concurrency via `MAX_CONCURRENT_CODE_EVALS` (default 2) across all threads.
6. **KodCode column name**: KodCode uses `passed` (not `correct`) as the boolean column in parquet results.
7. **Python output buffering**: Use `PYTHONUNBUFFERED=1` when running evals in the background.

## Adding a New Eval Dataset

1. Create `eval_tools/my_dataset.py` with:
   ```python
   async def run(sampling_client, renderer, tokenizer, results_dir, model_name,
                 think_prefix, max_tokens, max_problems=None, **kwargs) -> dict:
       # Return {"accuracy": float, "n_correct": int, "total": int}
       # Or: {"mean_score": float}
   ```
2. Register in `evaluate.py` `DATASET_MODULES`: `"my_dataset": "eval_tools.my_dataset"`
3. Add to appropriate dataset set in `config.py` (`MATH_DATASETS`, `CODE_DATASETS`, or `FUZZY_DATASETS`)
4. Add to default `FullEvalConfig.datasets` tuple if it should run by default

Follow `eval_tools/math_500.py` (accuracy-based) or `eval_tools/fuzzy_philosophy.py` (judge-based).

## Environment

- **Python**: 3.13 (venv at `fuzzy-evals-env/`)
- **API keys**: `/workspace/.env` (OPENROUTER_API_KEY, TINKERY_API_KEY, HF_TOKEN)
- **Judge model**: GPT-5.2 via OpenRouter
- **Training infra**: tinker SDK + tinker_cookbook (see `/workspace/tinker-cookbook/CLAUDE.md`)
- **Key dependencies**: tinker, tinker_cookbook, datasets, pandas, aiohttp, sympy, pylatexenc, matplotlib

## File Quick Reference

| What | Where |
|------|-------|
| Constants, configs, dataclasses | `config.py` |
| Inference logic, think prefix | `infer.py` |
| Shared utilities, OpenRouter | `utils.py` |
| Training (filter + LoRA SFT) | `train.py` |
| Eval orchestrator | `evaluate.py` |
| Plots, comparison, summaries | `analyze.py` |
| Pipeline orchestrator | `runner.py` |
| CLI entry point | `run.py` |
| Per-dataset eval modules | `eval_tools/` |
| Experiment launch scripts | `experiments/` |
| Fuzzy questions, rubrics, graders | `fuzzy_data/` |
| Math grading (boxed, sympy) | `/workspace/tinker-cookbook/.../math_grading.py` |

## Running Experiments

Always write a launch script in `experiments/` so every run is recorded. Two patterns:

**Single experiment** — `.sh` with inline Python:
```bash
bash experiments/smoke_llama8b_math_kimi.sh
```

**Sweep / batch** — `.sh` wrapping a `.py` batch file:
```bash
# experiments/my_sweep.py defines EXPERIMENTS list with loops
# experiments/my_sweep.sh runs it:
bash experiments/my_sweep.sh
```

Run in background: `nohup bash experiments/my_sweep.sh > experiments/my_sweep.log 2>&1 &`

Each experiment writes a `run.log` in its log dir with orchestration output. Per-experiment results are always in `logs/{timestamp}_{name}/summary.json`.
