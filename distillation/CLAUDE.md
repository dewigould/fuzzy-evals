# Distillation Experiment — Agent Guide

SFT distillation of reasoning traces into Qwen3-30B-A3B-Instruct-2507 (3B active MoE), evaluated across 8 benchmarks.

## Architecture

```
distillation/
├── config.py              # Constants: MODEL_NAME, paths, system prompts, API keys
├── infer.py               # generate() — single entry point for all inference
├── evaluate.py            # CLI: orchestrates eval across models × datasets
├── eval_config.yaml       # Declares models (with checkpoints) and dataset list
├── eval_tools/            # One module per dataset, each exports async run()
├── utils.py               # parse_think_tags, extract_code, parquet I/O, OpenRouter client
├── train_math.py          # Math SFT entry point (takes config name as arg)
├── train_code.py          # Code SFT entry point
├── sweep.py               # Runs all 4 configs (A/B/C/D), picks winner by test NLL
├── sweep_configs.py       # Hyperparameter definitions for A_fast..D_long
├── analysis/              # Post-hoc plotting and checkpoint sweep scripts
├── training_runs/         # Saved per-run: config.json, metrics.jsonl, checkpoints.jsonl
└── results/               # Eval outputs: {model_name}/results_{dataset}.parquet
```

## Data Flow

```
eval_config.yaml
  → evaluate.py reads model configs (checkpoint paths, think_prefix rules, max_tokens)
    → for each model: creates sampling_client via infer.py
      → for each dataset (all concurrently via asyncio.gather):
        → eval_tools/{dataset}.py::run() generates responses + grades them
          → saves results_{dataset}.parquet to results/{model_name}/
```

## Inference Pipeline (infer.py)

```python
# Setup (once per session)
renderer, tokenizer = setup_renderer_and_tokenizer()
client = create_checkpoint_client("tinker://...")  # or create_base_client()

# Generate (called by all eval modules)
completions = await generate(
    client, renderer, tokenizer,
    messages=[{"role": "user", "content": "..."}],
    max_tokens=28672,
    think_prefix=True,   # seeds <think>\n via renderer prefill
    temperature=0.0,
    num_samples=1,
)
```

Key internals:
- `prefill="<think>\n"` passed to `renderer.build_generation_prompt()` when `think_prefix=True`
- `_content_to_string()` converts TypedDict ContentParts back to a flat string with `<think>...</think>` tags
- Output is NEVER modified post-hoc

## eval_config.yaml Structure

```yaml
models:
  model_name:
    type: checkpoint          # or "base_model"
    sampler_path: "tinker://..." # checkpoint URI
    max_tokens: 28672         # override per model (default: inference.max_tokens)
    no_think_prefix_datasets: # datasets where model generates <think> natively
      - math_500
      - aime

datasets:
  - math_500
  - aime
  - kodcode_500
  # ...

inference:
  max_tokens: 8192            # default for models without override
```

## Eval Module Interface

Every module in `eval_tools/` exports:

```python
async def run(
    sampling_client,            # tinker sampling client
    renderer,                   # Qwen3InstructRenderer
    tokenizer,                  # tokenizer
    results_dir: Path,          # where to save parquet
    model_name: str,            # for labeling results
    think_prefix: bool = True,  # whether to seed <think>
    max_tokens: int = 8192,     # generation budget
    max_problems: int | None = None,  # limit dataset size
    **kwargs,                   # fuzzy_samples for fuzzy evals
) -> dict:                      # {"dataset": ..., "accuracy": ...} or {"mean_score": ...}
```

## Training Data

| Task | Dataset | Source | Size |
|------|---------|--------|------|
| Math | OpenR1-Math-220k | open-r1/OpenR1-Math-220k | 220K |
| Code | OpenCodeReasoning | nvidia/OpenCodeReasoning | 736K |

Both contain `<think>...</think>` reasoning traces from strong teacher models. Training truncates to `max_length=8192` tokens.

## Checkpoints

Saved every 100 steps = 5,000 examples. Stored as tinker:// URIs in `training_runs/{task}/D_long/checkpoints.jsonl`.

| Task | Tinker Run ID | Best Step | Examples |
|------|--------------|-----------|----------|
| Math | 20cab21e-... | 1800 | 90,000 |
| Code | f136a2a0-... | 1900 | 95,000 |

## Known Issues

**Distilled models underperform base on in-domain tasks.** Root cause: verbose CoT from teacher traces causes:
1. **Truncation** — model hits max_tokens mid-reasoning, never produces `\boxed{}` or final code block
2. **Reasoning loops** — model double-checks repeatedly without committing to an answer
3. Of math_distill's 17 errors on MATH-500, 11 were `no_boxed` (no answer extracted), and 10/11 had raw output >50K chars

When the model DOES produce a boxed answer, accuracy matches or exceeds base (93% vs 94%).

## Results Column Reference

**Math datasets** (math_500, aime, omni_math): `correct` (bool), `predicted_answer`, `ground_truth`, `cot`, `user_output`, `raw_output`

**Code datasets** (kodcode): `passed` (bool — NOT `correct`), `detail`, `cot`, `user_output`, `raw_output`

**Fuzzy datasets**: `total_score` (float), per-rubric scores (e.g., `thesis_clarity`), `judge_raw`, `cot`, `user_output`, `raw_output`

**Omni-MATH** additionally has: `grading_method` ("stage1", "llm_judge", "stage1_fail"), `difficulty`, `source`
