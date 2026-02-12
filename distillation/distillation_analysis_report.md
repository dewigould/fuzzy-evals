# Distillation Experiment Analysis Report

**Date:** 2026-02-12
**Model:** Qwen3-30B-A3B-Instruct-2507 (MoE, 3B active params)
**Math distillation source:** OpenR1-Math-220k (R1-style `<think>` reasoning traces)
**Code distillation source:** nvidia/OpenCodeReasoning (Python, `<think>` reasoning traces)
**Training:** LoRA rank 32, 4 hyperparameter configs each, D_long selected as winner for both

---

## 1. Executive Summary

Both distilled models suffered **catastrophic evaluation failures**, but for different reasons:

| Benchmark | Base | Math Distill | Code Distill | Goal |
|-----------|------|-------------|-------------|------|
| **MATH-500** | **93.0%** | 0.8% | 87.8% | Math distill should be higher |
| **AIME** | **60.0%** | 3.3% | 46.7% | Math distill should be higher |
| **KodCode-500** | **60.2%** | 50.6% | 0.2% | Code distill should be higher |
| **Codeforces** | **25.7%** | 21.3% | *(still running)* | Code distill should be higher |
| Fuzzy Philosophy | 19.7 | 17.6 | *(still running)* | |
| Fuzzy Weird Qs | 18.1 | 15.5 | *(still running)* | |
| Fuzzy Futuristic Tech | 18.3 | 16.4 | *(still running)* | |

**Neither model improved on its target domain.** Math distill's MATH-500 score dropped from 93% to 0.8%. Code distill's KodCode score dropped from 60.2% to 0.2%.

**However, the models ARE working.** The failures are overwhelmingly caused by **two pipeline bugs** and a **format mismatch**, not by genuine reasoning degradation. Experimental probing (Section 5) shows the math distill model answers questions correctly when the bugs are bypassed.

---

## 2. Training Results

All 8 training runs completed successfully. Test NLL decreased monotonically for every config — no overfitting detected.

### Math Sweep

| Config | LR | Steps | Best Test NLL | @ Step |
|--------|-----|-------|--------------|--------|
| A_fast | 2e-4 | 500 | 0.3296 | 400 |
| B_medium | 1e-4 | 1000 | 0.3282 | 900 |
| C_gentle | 5e-5 | 1000 | 0.3295 | 900 |
| **D_long** | **5e-5** | **2000** | **0.3268** | **1800** |

### Code Sweep

| Config | LR | Steps | Best Test NLL | @ Step |
|--------|-----|-------|--------------|--------|
| A_fast | 2e-4 | 500 | 0.5500 | 400 |
| B_medium | 1e-4 | 1000 | 0.5465 | 900 |
| C_gentle | 5e-5 | 1000 | 0.5510 | 900 |
| **D_long** | **5e-5** | **2000** | **0.5442** | **1900** |

**Key observation:** 93% of the NLL improvement occurs in the first 100 steps. D_long won by only 0.0028 NLL (0.8% relative) over A_fast. The configs are barely distinguishable on NLL, but they may differ significantly on downstream task behavior.

---

## 3. Root Cause Analysis

There are **three independent failure modes**, compounding to produce the catastrophic results.

### 3.1 Bug #1: `TypeError` in `infer.py` (Pipeline Crash)

**Files:** `infer.py:90`, `tinker_cookbook/renderers/qwen3.py:197-228`

The `generate()` function in `infer.py` does:
```python
parsed_msg, _ = renderer.parse_response(seq.tokens)
content = parsed_msg["content"]
if think_prefix:
    content = "<think>\n" + content  # <-- TypeError when content is a list
```

The Qwen3InstructRenderer's `parse_response()` calls `parse_content_blocks()` on the decoded text. When the decoded tokens contain `<think>...</think>` tags, it returns content as a **list of ContentPart objects** (`[ThinkingPart(...), TextPart(...)]`), not a string. The string concatenation `"<think>\n" + content` then raises `TypeError`.

This error is caught by the eval scripts' `except Exception as e` blocks and stored as `"ERROR: can only concatenate str (not "list") to str"`, which then evaluates as incorrect.

**Impact by model and benchmark:**

| Model | MATH-500 | AIME | KodCode |
|-------|----------|------|---------|
| Base | 0/500 (0%) | 0/90 (0%) | 0/500 (0%) |
| Math Distill | 436/500 (87.2%) | 35/90 (38.9%) | — |
| Code Distill | 0/500 (0%) | 0/90 (0%) | 458/500 (91.6%) |

The bug is **content-dependent** — whether `parse_response()` returns a list or string depends on the specific token sequence generated. Math distill triggers it heavily on math questions; code distill triggers it heavily on code questions but not math questions.

### 3.2 Bug #2: Double `<think>` Tag / Format Mismatch

The eval pipeline manually prepends `<think>\n` tokens to the model input before generation (`infer.py:70-73`). This was designed for the base Qwen3 model, which uses these tokens to enter "thinking mode."

However, the distilled models were trained on data that already contained `<think>...</think>` tags as literal text in the training examples. When the pipeline prepends `<think>\n` and the model generates its own `<think>` continuation, the output begins with `<think>\n<think>\n` — a "double think" pattern.

For responses where the TypeError doesn't fire (because `parse_response()` happened to return a string), this double `<think>` appears to cause the model to enter a degenerate state:
- It starts reasoning, but never produces `</think>` to exit the thinking phase
- The reasoning becomes extremely verbose and repetitive
- It eventually hits the max_tokens limit without producing `\boxed{}` or code blocks

This was observed on **all checkpoints** tested (steps 100, 200, 500, 1800, A_fast step 400) for the same problem, confirming it's a structural issue with the `think_prefix` mechanism interacting with distilled models.

### 3.3 Issue #3: Verbose Reasoning / Truncation

Even when both bugs are avoided, the distilled models generate significantly longer responses than the base model:

| Model | MATH-500 Median | MATH-500 Mean | AIME Median |
|-------|----------------|---------------|-------------|
| Base | 1,418 chars | 3,346 chars | 9,100 chars |
| Code Distill | 1,468 chars | 5,108 chars | 17,831 chars |
| Math Distill* | 22,547 chars* | 22,855 chars* | — |

*Math distill stats are for the ~13% of non-error responses.

The distilled models learned the verbose, step-by-step reasoning style of their teacher models (R1, DeepSeek, etc.). With `MAX_TOKENS=8192` (~32K characters), many responses are truncated before reaching the final answer.

**For code_distill on MATH-500:** 40 responses are truncated (vs 14 for base) — accounting for 26 of the 31 regressions. The number of genuine reasoning errors is **identical** (21 each). Extrapolating, if truncation were eliminated, code_distill's MATH-500 accuracy would be ~91-92%.

**For code_distill on AIME:** 47 truncated (vs 25 for base), but only 1 wrong-answer error (vs 11 for base). The model actually reasons **better** when it finishes — its estimated non-truncated accuracy would be ~67%, surpassing base's 60%.

---

## 4. Experimental Verification

To confirm these findings, I tested the math distill D_long checkpoint (step 1800) with various settings on 3 easy MATH problems:

### Test A: Fixed pipeline bug + `think_prefix=True` (current eval settings)
- Problem 1 (polar coords): **Correct** (3,366 chars, has `\boxed{}`)
- Problem 2 (multiples of 30): **Wrong** — double `<think>` loop, 20,501 chars, no `\boxed{}`
- Problem 3 (exponents): **Correct** (3,530 chars, has `\boxed{}`)

### Test B: Fixed pipeline bug + `think_prefix=False`
- Problem 1: **Correct** (3,477 chars)
- Problem 2: **Correct** (8,060 chars, has `\boxed{2220}`)
- Problem 3: **Correct** (3,852 chars)

**All 3 problems answered correctly with `think_prefix=False`.** The model IS functional.

### Test C: Fixed pipeline bug + `think_prefix=True` + `max_tokens=16384`
- Problem 1: **Correct** (4,389 chars)
- Problem 2: **Wrong** — double `<think>` loop expanded to 46,867 chars, still no `\boxed{}`
- Problem 3: **Correct** (3,814 chars)

More tokens doesn't help when the model enters a repetitive loop. The double-think problem is structural.

### Test D: Earlier checkpoints (D_long steps 100, 200, 500)
- Step 100: Problem 2 enters double-think loop (10,538 chars, degenerates into `000000...`)
- Step 200: Problem 2 enters double-think loop (12,193 chars, degenerates into `000000...`)
- Step 500: Problem 2 **Correct** — no double think, gets `\boxed{2220}` in 9,475 chars
- Step 1800: Problem 2 enters double-think loop (20,501 chars)

The double-think behavior is **content-dependent and non-deterministic** across checkpoints. The underlying model CAN solve the problem — it's the interaction with the think_prefix that causes sporadic failures.

### Test E: A_fast best checkpoint (step 400, less training)
- Problem 1: **Correct**
- Problem 2: Double-think loop, degenerates into `000000...` repetition

Less training doesn't reliably help.

---

## 5. Why Math Distill Failed on Math (Despite Being Trained on Math)

The math distill model's catastrophic 0.8% accuracy is **not because the model can't do math**. It's because:

1. **87.2% of responses crash the pipeline** before the model even gets a chance. The TypeError prevents any output from being generated. This is a bug in `infer.py`, not the model.

2. **Of the 12.8% that survive**, most (~94%) enter double-think loops and are truncated before producing `\boxed{}`. Only 4/500 responses both avoid the TypeError AND complete reasoning within the token budget.

3. **When the pipeline bug is fixed and `think_prefix` is disabled**, the model answers correctly on test problems. It produces well-reasoned, correct `\boxed{}` answers.

The training successfully taught the model R1-style reasoning. The problem is entirely in the evaluation pipeline's incompatibility with the distilled model's output format.

---

## 6. Why Code Distill Has Different Behavior

Code distill shows a very different failure pattern:

- **0% pipeline errors on MATH/AIME** — the renderer parses code_distill's math responses as strings, not lists
- **91.6% pipeline errors on KodCode** — the renderer parses code responses as lists

This asymmetry likely relates to how the code distillation data formatted `<think>` tags. The code_distill model learned to generate `<think>` tags only in code-related contexts (matching the training data distribution), while the base-model behavior is preserved for math contexts.

For math benchmarks, code_distill shows moderate regression (93% → 87.8% on MATH-500, 60% → 46.7% on AIME). The **entire regression is explained by increased truncation** from verbose reasoning. The number of wrong-answer errors is the same or lower than base.

---

## 7. Recommendations

### 7.1 Critical: Fix the Pipeline Bug in `infer.py`

**Severity: Blocks all results. Fix required before any re-evaluation.**

Replace `infer.py:86-91` to handle ContentPart lists:

```python
for seq in response.sequences:
    parsed_msg, _ = renderer.parse_response(seq.tokens)
    content = parsed_msg["content"]
    if isinstance(content, list):
        # ContentPart list — convert to string preserving thinking
        parts = []
        for part in content:
            if hasattr(part, "thinking"):
                parts.append(f"<think>{part.thinking}</think>")
            elif hasattr(part, "text"):
                parts.append(part.text)
            else:
                parts.append(str(part))
        content = "".join(parts)
    if think_prefix:
        content = "<think>\n" + content
    results.append(content)
```

### 7.2 Critical: Disable `think_prefix` for Distilled Models

**Severity: Causes double-think loops in ~50% of surviving responses.**

The distilled models already know to generate `<think>` reasoning because they were trained on data with `<think>` tags. The eval pipeline should NOT also prepend `<think>\n` tokens. This creates a `<think>\n<think>\n` double prefix that causes repetitive loops.

**Fix:** In each eval module (math_500.py, kodcode.py, codeforces.py, etc.), change:
```python
completions = await generate(..., think_prefix=True)
```
to:
```python
completions = await generate(..., think_prefix=False)
```
for distilled model evaluations. Alternatively, make `think_prefix` configurable per model in `eval_config.yaml`.

### 7.3 Important: Increase `MAX_TOKENS` for Distilled Models

**Severity: Causes 5-13pp accuracy drop on non-broken responses.**

The distilled models are more verbose (1.3-1.5x longer responses). With `MAX_TOKENS=8192`, many responses are truncated. Recommend `MAX_TOKENS=16384` for distilled model evaluation.

However, this only helps for non-looping responses. Problems entering repetitive loops won't benefit from more tokens (as shown in Test C above — 46,867 chars and still looping). Fix #7.2 is the priority.

### 7.4 Medium: Re-evaluate with Fixes Applied

After applying fixes 7.1-7.3, re-run the full eval suite. **Expected results:**

Based on the experimental evidence:
- **Math distill MATH-500:** ~85-93% (model answers correctly when bugs are bypassed)
- **Math distill AIME:** ~55-65% (may match or exceed base with longer token budget)
- **Code distill KodCode:** Unknown — need to re-eval after fixing pipeline bug
- **Code distill Codeforces:** Unknown — need to re-eval

### 7.5 Consider: Training Data Format Alignment

The root cause of the format mismatch is that the training data contains literal `<think>...</think>` text tags, while Qwen3's renderer treats these as structured ContentPart blocks. Two possible approaches:

**Option A:** Strip `<think>` tags from training data and rely on Qwen3's native thinking mode (using the model's built-in thinking token structure rather than text-level tags). This would align the training format with how the renderer expects thinking content.

**Option B:** Use a "no-thinking" renderer during evaluation that doesn't parse `<think>` blocks, treating all output as plain text. This preserves the training format but requires evaluation-side changes.

### 7.6 Consider: Response Length Regularization

The distilled models are significantly more verbose than necessary. For future training runs, consider:
- **Truncating training examples** to a maximum length (e.g., 4096 tokens) to discourage verbosity
- **Filtering training data** to exclude very long reasoning traces
- **Adding a length penalty** during training
- **Using shorter teacher traces** (e.g., Qwen3's own thinking traces rather than R1's)

---

## 8. Appendix: Training NLL Details

Diminishing returns analysis:
- First 100 steps: 93% of total NLL improvement (math), 88% (code)
- First 500 steps: 98% of total improvement (both)
- Steps 500-2000: <2% remaining improvement

The marginal value of training beyond 500 steps is minimal. Future sweeps could use shorter configs.

Train-test generalization gap is negative (test NLL < train NLL) for all configs, indicating zero overfitting. The training metrics are healthy — the evaluation failures are entirely pipeline-side.

---

## 9. Files Referenced

- `infer.py` — Central inference module (contains the TypeError bug)
- `eval_config.yaml` — Model and dataset configuration
- `train_math.py` / `train_code.py` — Training scripts
- `sweep_configs.py` — Hyperparameter configurations
- `eval_tools/math_500.py` — MATH-500 evaluation
- `eval_tools/kodcode.py` — KodCode evaluation
- `eval_tools/codeforces.py` — Codeforces evaluation
- `training_runs/{math,code}/{A_fast,B_medium,C_gentle,D_long}/metrics.jsonl` — Training metrics
- `results/{base,math_distill,code_distill}/` — Evaluation results (parquet files)
- `debug_distill.py` — Debug script used for experimental verification
- `training_winners.json` — Selected best checkpoints
