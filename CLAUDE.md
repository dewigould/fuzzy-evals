# Fuzzy Evals — Agent Guide

This repo contains evaluation and training infrastructure for studying reasoning distillation and LLM-as-judge evaluation on "fuzzy" (open-ended) and "hard" (math/code) benchmarks.

## Quick Orientation

```
fuzzy-evals/
├── dataset_jsons/          # 56 fuzzy eval questions (philosophy, weird, futuristic tech)
├── rubrics/                # LLM judge rubrics (6 criteria, 0-8 each, max 48)
├── graders/                # Rubric loaders + grading prompt builders
├── scripts/                # Standalone experiment scripts (effort sweep, model sweep, training)
├── results/                # Legacy experiment outputs (Opus sweep)
├── distillation/           # ** Main active experiment ** (training, eval, analysis)
│   ├── config.py           # Shared constants, paths, system prompts
│   ├── infer.py            # Central inference (renderer, tokenizer, think prefix seeding)
│   ├── evaluate.py         # Main eval pipeline (reads eval_config.yaml)
│   ├── eval_config.yaml    # Which models × which datasets to evaluate
│   ├── eval_tools/         # Per-dataset evaluation modules (8 datasets)
│   ├── train_math.py       # Math SFT training entry point
│   ├── train_code.py       # Code SFT training entry point
│   ├── sweep.py            # Training sweep orchestrator
│   ├── analysis/           # Plotting and analysis scripts
│   └── training_runs/      # Saved configs, metrics, checkpoints
└── schema.py               # FuzzyQuestion dataclass
```

## Key Concepts

### Model
Base model is **Qwen/Qwen3-30B-A3B-Instruct-2507** (MoE, 3B active params). Distilled variants are LoRA rank-32 fine-tunes trained via the [tinker](file:///workspace/tinker-cookbook/CLAUDE.md) framework.

### Think Prefix Seeding
Distilled models learned `<think>...</think>` reasoning from teacher traces. At inference:
- **In-domain** (e.g., math_distill on math): `think_prefix=False` — model generates `<think>` natively
- **Cross-domain** (e.g., math_distill on code): `think_prefix=True` — seed `<think>\n` via renderer's `prefill` parameter
- **Base model**: `think_prefix=False` always

**Critical**: Think prefix is a generation biasing technique, not a post-processor. The model's output must never be modified after sampling. See `infer.py` for the correct implementation.

### ContentPart TypedDicts
The Qwen3 renderer's `parse_response()` returns content as either:
- `str` — when no `<think>` tags in output
- `list[ContentPart]` — TypedDicts (plain dicts), NOT objects with attributes

ContentPart types: `{"type": "thinking", "thinking": str}` and `{"type": "text", "text": str}`. Use `part.get("type")` for discrimination, never `hasattr()`.

### Evaluation Datasets (8 total)
| Dataset | Type | Size | Metric | Grading |
|---------|------|------|--------|---------|
| math_500 | Math | 500 | accuracy | `extract_boxed` + `grade_answer` (sympy) |
| aime | Math | 90 | accuracy | same |
| omni_math | Math | 4,428 | accuracy | Two-stage: math grading → GPT-5.2 judge fallback |
| kodcode_500 | Code | 5,764 | accuracy | Subprocess execution of pytest-style tests |
| codeforces_500 | Code | ~408 | accuracy | stdin/stdout test execution |
| fuzzy_philosophy | Fuzzy | 10 | mean_score | GPT-5.2 judge with rubric (max 48) |
| fuzzy_weird_qs | Fuzzy | 46 | mean_score | GPT-5.2 judge with rubric |
| fuzzy_futuristic_tech | Fuzzy | 10 | mean_score | GPT-5.2 judge with rubric |

### Training Configs
Four sweep configs per task (A_fast → D_long), varying `max_prompts`, `learning_rate`, `lr_schedule`, `num_epochs`. All use batch_size=50, LoRA rank=32, max_length=8192. Checkpoints saved every 100 steps (= 5,000 examples). Winner selected by lowest test NLL.

## Running Evaluations

```bash
cd /workspace/fuzzy-evals/distillation

# Full eval (all models × all datasets from config)
PYTHONUNBUFFERED=1 python3.13 evaluate.py

# Single model
PYTHONUNBUFFERED=1 python3.13 evaluate.py --model math_distill

# Quick sanity check
PYTHONUNBUFFERED=1 python3.13 evaluate.py --model base --max-problems 10 --fuzzy-samples 1

# Specific datasets
PYTHONUNBUFFERED=1 python3.13 evaluate.py --model base --datasets math_500,aime
```

Results go to `distillation/results/{model_name}/results_{dataset}.parquet`.

All datasets within a model run **concurrently** via `asyncio.gather`. Use `PYTHONUNBUFFERED=1` to see output in real time.

## Running Training

```bash
cd /workspace/fuzzy-evals/distillation

# Single config
python3.13 train_math.py D_long

# Full sweep (runs A/B/C/D sequentially, picks winner)
python3.13 sweep.py --task math
python3.13 sweep.py --task code
```

## Running Analysis

```bash
cd /workspace/fuzzy-evals/distillation

# Generate all plots
python3.13 analysis/generate_all.py

# Checkpoint sweep (evaluates multiple training checkpoints)
PYTHONUNBUFFERED=1 python3.13 analysis/checkpoint_sweep.py
```

Plots saved to `distillation/analysis/plots/`.

## Adding a New Eval Dataset

1. Create `eval_tools/my_dataset.py` with an `async def run(sampling_client, renderer, tokenizer, results_dir, model_name, think_prefix, max_tokens, max_problems=None, **kwargs) -> dict` function
2. Register in `evaluate.py` DATASET_MODULES: `"my_dataset": "eval_tools.my_dataset"`
3. Add to `eval_config.yaml` datasets list
4. If distilled models should use native thinking (no prefill), add to the model's `no_think_prefix_datasets`

Follow the pattern in `math_500.py` (accuracy-based) or `fuzzy_philosophy.py` (judge-based).

## Environment

- **Python**: 3.13 (venv at `fuzzy-evals-env/`)
- **API keys**: `.env` file (OPENROUTER_API_KEY, TINKERY_API_KEY, HF_TOKEN)
- **Judge model**: GPT-5.2 via OpenRouter
- **Training infra**: tinker SDK + tinker_cookbook (see `/workspace/tinker-cookbook/CLAUDE.md`)
- **Key dependencies**: tinker, tinker_cookbook, datasets, pandas, aiohttp, sympy, pylatexenc

## Common Pitfalls

1. **`hasattr()` on ContentParts**: TypedDicts are plain dicts — `hasattr(part, "thinking")` is always False. Use `part.get("type") == "thinking"` instead.
2. **Post-hoc `<think>` prepend**: Never prepend `<think>\n` to model output. Use the renderer's `prefill` parameter in `build_generation_prompt()`.
3. **max_tokens vs context window**: Model context is 32,768 tokens. Leave headroom for prompts (28,672 works).
4. **Python output buffering**: Always use `PYTHONUNBUFFERED=1` when running evals in the background.
5. **Base model think_prefix**: The base Qwen3 model should use `think_prefix=False` on all datasets.
6. **Truncation kills accuracy**: Distilled models generate very long CoT. If output is truncated before `\boxed{}` or the code block, the answer is lost. This is the primary failure mode.
7. **KodCode column name**: KodCode uses `passed` (not `correct`) as the boolean column in parquet results.

## File Quick Reference

| What | Where |
|------|-------|
| Model name, system prompts | `distillation/config.py` |
| Inference logic, think prefix | `distillation/infer.py` |
| Eval pipeline entry point | `distillation/evaluate.py` |
| Model/dataset selection | `distillation/eval_config.yaml` |
| Math grading (boxed, sympy) | `/workspace/tinker-cookbook/.../math_grading.py` |
| Training checkpoints | `distillation/training_runs/{math,code}/D_long/checkpoints.jsonl` |
| Eval results (parquet) | `distillation/results/{model_name}/` |
| Fuzzy questions JSON | `dataset_jsons/` |
| Grading rubrics | `rubrics/` |
| Analysis plots | `distillation/analysis/plots/` |
