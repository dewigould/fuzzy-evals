# Experiment Analysis: Llama-8B Distillation Overnight Sweep

**Sweep**: `2026-02-27_1920_overnight_sweep`
**Base model for all training**: Llama-3.1-8B-Instruct
**Teachers**: Kimi-K2.5, Qwen3-235B (note: Sonnet was **not** included in this sweep's training runs)
**Tasks**: math, code
**Context lengths**: 4096 (4k), 16384 (16k)
**LoRA ranks**: 32, 64, 128
**Total experiments**: 19 training runs + 8 base model evals

> **Important caveat**: The sweep only trained with Kimi and Qwen traces. Sonnet traces exist in `traces/` but no Sonnet distillation experiments were run. All Sonnet cells in the tables below are marked N/A.

---

## Question 1: Benchmark Scores at Best Checkpoint

### Base Model Scores (no training)

| Model | max_tokens | MATH-500 | AIME | MBPP+ | HumanEval+ | LiveCodeBench |
|-------|-----------|----------|------|-------|------------|---------------|
| Llama-3.1-8B-Instruct | 4k | 50.8% | 4.4% (4/90) | 57.9% | 62.2% | 7.2% |
| Llama-3.1-8B-Instruct | 16k | 50.2% | 4.4% (4/90) | 56.9% | 63.4% | 6.6% |
| Llama-3.3-70B-Instruct | 4k | 75.8% | 25.6% (23/90) | 75.9% | 78.0% | 22.8% |
| Llama-3.3-70B-Instruct | 16k | 74.2% | 25.6% (23/90) | 75.7% | 78.7% | 23.4% |
| Qwen3-8B-Base | 4k | 81.2% | 14.4% (13/90) | 74.1% | 76.8% | 13.8% |
| Qwen3-8B-Base | 16k | 81.0% | 15.6% (14/90) | 73.8% | 78.7% | 13.2% |
| Qwen3-30B-A3B-Instruct | 4k | 90.2% | 35.6% (32/90) | 76.5% | 92.1% | 29.3% |
| Qwen3-30B-A3B-Instruct | 16k | 97.0% | 71.1% (64/90) | 77.0% | 90.9% | 31.7% |

### Math-Distilled Models (Kimi teacher)

| max_tokens | LoRA rank | MATH-500 | AIME | OmniMath | MBPP+ | HumanEval+ | LiveCodeBench | Best step |
|-----------|-----------|----------|------|----------|-------|------------|---------------|-----------|
| 4k | 32 | 50.4% | 1.1% (1) | 11.0% | 54.8% | 50.0% | 4.8% | 279 |
| 4k | 64 | **54.0%** | 1.1% (1) | 8.2% | 54.8% | 48.2% | 3.0% | 240 |
| 4k | 128 | 52.8% | 0.0% (0) | 11.6% | 55.8% | 50.0% | 6.0% | 200 |
| 16k | 32 | 47.8% | 3.3% (3) | 9.6% | 51.6% | 46.3% | 3.6% | 400 |
| 16k | 64 | 47.6% | 1.1% (1) | 9.4% | 46.6% | 48.8% | 4.2% | 160 |
| 16k | 128 | 46.2% | 3.3% (3) | 9.0% | 51.9% | 45.1% | 6.6% | 200 |

### Math-Distilled Models (Qwen teacher)

| max_tokens | LoRA rank | MATH-500 | AIME | OmniMath | MBPP+ | HumanEval+ | LiveCodeBench | Best step |
|-----------|-----------|----------|------|----------|-------|------------|---------------|-----------|
| 4k | 32 | 44.2% | 3.3% (3) | 9.4% | 52.4% | 50.0% | 4.2% | 190 |
| 4k | 64 | 44.0% | 2.2% (2) | 10.2% | 55.0% | 48.2% | 3.6% | 80 |
| 4k | 128 | 47.6% | 3.3% (3) | 8.2% | 52.4% | 52.4% | 5.4% | 160 |
| 16k | 32 | 42.0% | 1.1% (1) | 7.6% | 49.7% | 45.1% | 3.0% | 240 |
| 16k | 64 | 41.8% | 1.1% (1) | 8.0% | 51.6% | 42.1% | 3.6% | 320 |
| 16k | 128 | 43.0% | 2.2% (2) | 8.6% | 50.5% | 48.2% | 3.6% | 400 |

### Code-Distilled Models (Kimi teacher)

| max_tokens | LoRA rank | MATH-500 | AIME | OmniMath | MBPP+ | HumanEval+ | LiveCodeBench | Best step |
|-----------|-----------|----------|------|----------|-------|------------|---------------|-----------|
| 4k | 32 | 51.0% | 2.2% (2) | 11.0% | **60.1%** | 54.3% | **9.0%** | 80 |
| 4k | 64 | 49.4% | 4.4% (4) | 12.4% | 59.0% | 53.7% | 4.2% | 40 |
| 4k | 128 | 49.6% | 4.4% (4) | 10.6% | 58.7% | 54.3% | 4.8% | 160 |
| 16k | 32 | 44.4% | 3.3% (3) | 7.2% | 54.8% | 48.2% | 4.2% | 80 |

### Code-Distilled Models (Qwen teacher)

| max_tokens | LoRA rank | MATH-500 | AIME | OmniMath | MBPP+ | HumanEval+ | LiveCodeBench | Best step |
|-----------|-----------|----------|------|----------|-------|------------|---------------|-----------|
| 4k | 32 | 44.2% | 3.3% (3) | 7.6% | 45.2% | 38.4% | 4.2% | 240 |
| 4k | 64 | 44.6% | 2.2% (2) | 8.8% | 46.6% | 36.0% | 4.8% | 40 |
| 4k | 128 | 43.6% | 3.3% (3) | 8.8% | 51.1% | 34.2% | 6.0% | 40 |

### Key Takeaways

- **Best math-distilled model**: Kimi 4k r64 at 54.0% MATH-500 (+3.2pp over base Llama-8B).
- **Best code-distilled model**: Kimi 4k r32 at 60.1% MBPP+ (+2.2pp over base) and 9.0% LiveCodeBench (+1.8pp).
- **No distilled model improves AIME** over the base model's 4.4%. Most score 0-3/90.
- **Kimi traces consistently outperform Qwen traces** across all settings.
- **Sonnet was not trained** in this sweep, so no comparison is possible.

---

## Question 2: Training Examples Surviving Filtering at 16k vs 4k

Survival counts for Llama-8B tokenizer:

| Teacher | Task | Raw traces | Survive @ 4k | Survive @ 16k | 4k survival rate | 16k survival rate |
|---------|------|-----------|-------------|--------------|-----------------|------------------|
| Kimi | math | 35,771 | 14,240 | 32,848 | 39.8% | 91.8% |
| Kimi | code | 29,206 | 21,816 | — | 74.7% | (not run at 16k) |
| Qwen | math | 34,969 | 9,858 | 29,281 | 28.2% | 83.7% |
| Qwen | code | 28,354 | 14,146 | — | 49.9% | (not run at 16k) |
| Sonnet | math | 28,939 | 28,899 | 28,939 | 99.9% | 100.0% |
| Sonnet | code | 28,270 | 28,242 | 28,270 | 99.9% | 100.0% |

**Key findings**:
- Moving from 4k to 16k **more than doubles** the available Kimi math data (14,240 → 32,848) and **triples** Qwen math data (9,858 → 29,281).
- Sonnet traces are so compact that filtering has virtually zero effect at any length — 99.9% fit within 4k tokens.
- Kimi code traces are more compact than math traces (74.7% survival at 4k vs 39.8%), suggesting math reasoning traces are longer than code traces for Kimi.
- Qwen is the most verbose teacher: only 28.2% of its math traces fit in 4k, and even at 16k, 16.3% are still too long.
- The `max_samples` config was set to match available data: Qwen 4k math capped at 9,500 (of 9,858 available), Kimi 4k math at 14,000 (of 14,240).

---

## Question 3: Trace Length Distribution and Truncation

Character-level statistics for response content (first 1,000 examples per file, Llama-8B tokenizer):

| Teacher | Task | max_len | Median chars | Avg chars | P10 | P90 | P95 |
|---------|------|---------|-------------|-----------|-----|-----|-----|
| Kimi | math | 4k | 5,519 | 6,268 | 1,697 | 11,573 | 13,394 |
| Kimi | math | 16k | 11,824 | 15,127 | 2,038 | 32,849 | 41,327 |
| Qwen | math | 4k | 4,970 | 5,584 | 1,582 | 10,413 | 12,237 |
| Qwen | math | 16k | 16,795 | 21,254 | 2,482 | 46,853 | 57,428 |
| Sonnet | math | 4k | 2,356 | 2,855 | 757 | 5,352 | 6,515 |
| Sonnet | code | 4k | 2,019 | 2,417 | 726 | 4,429 | 5,583 |
| Kimi | code | 4k | 5,167 | 5,738 | 1,575 | 10,698 | 12,572 |
| Qwen | code | 4k | 7,159 | 8,029 | 1,953 | 15,448 | 18,773 |

**Truncation rates (fraction of traces exceeding max_length)**:

| Teacher | Task | 4k truncated | 16k truncated |
|---------|------|-------------|--------------|
| Kimi | math | 60.2% | 8.2% |
| Qwen | math | 71.8% | 16.3% |
| Sonnet | math | 0.1% | 0.0% |
| Kimi | code | 25.3% | — |
| Qwen | code | 50.1% | — |
| Sonnet | code | 0.1% | 0.0% |

At 4k, the **majority** of Kimi and Qwen math traces are lost to truncation. The 16k setting recovers most of them, but Qwen still loses 16.3%. Sonnet traces are essentially never truncated — they are 2-3x shorter than Kimi/Qwen traces.

The median Qwen 16k math trace (16,795 chars) is ~7x longer than the median Sonnet 4k math trace (2,356 chars), illustrating the dramatic verbosity difference between teachers.

---

## Question 4: Peak Performance Step for Best 16k Run

The best-performing 16k training experiment is **`llama8b_math_kimi_16k_r32`** (selected at step 400, the final checkpoint). Checkpoint eval trajectory on math_500 and mbppplus (100 problems each):

| Step | math_500 | mbppplus |
|------|----------|----------|
| 40 | 0.33 | 0.41 |
| 80 | 0.38 | 0.49 |
| 120 | 0.39 | 0.48 |
| 160 | 0.38 | 0.48 |
| 200 | 0.41 | 0.47 |
| 240 | 0.39 | 0.46 |
| 280 | 0.44 | 0.50 |
| 320 | 0.45 | 0.51 |
| 360 | 0.44 | 0.48 |
| 400 | 0.47 | 0.48 |

**Analysis**: math_500 shows a **clear upward trend through the end of training** (0.33 → 0.47), with the final checkpoint being the best. The model was still improving and had likely not yet plateaued. mbppplus (cross-domain, code) fluctuated without a clear trend (0.41-0.51), suggesting the math distillation does not progressively help or hurt code performance.

For comparison, `llama8b_math_qwen_16k_r128` also peaked at step 400 (0.42 math_500), showing a steadily climbing trajectory from 0.23 at step 40. This run was most clearly still improving, suggesting **longer training (more steps) could benefit the 16k runs**.

---

## Question 5: 4k vs 16k Delta — Does Longer Context Help?

Comparing same teacher × same LoRA rank, 4k vs 16k (delta = 16k minus 4k):

### Math distillation with Kimi

| Rank | MATH-500 Δ | AIME Δ | OmniMath Δ | MBPP+ Δ | HumanEval+ Δ | LCB Δ |
|------|-----------|--------|-----------|---------|-------------|-------|
| r32 | -2.6pp | +2.2pp | -1.4pp | -3.2pp | -3.7pp | -1.2pp |
| r64 | -6.4pp | 0.0pp | +1.2pp | -8.2pp | +0.6pp | +1.2pp |
| r128 | -6.6pp | +3.3pp | -2.6pp | -3.9pp | -4.9pp | +0.6pp |
| **Avg** | **-5.2pp** | **+1.8pp** | **-0.9pp** | **-5.1pp** | **-2.7pp** | **+0.2pp** |

### Math distillation with Qwen

| Rank | MATH-500 Δ | AIME Δ | OmniMath Δ | MBPP+ Δ | HumanEval+ Δ | LCB Δ |
|------|-----------|--------|-----------|---------|-------------|-------|
| r32 | -2.2pp | -2.2pp | -1.8pp | -2.7pp | -4.9pp | -1.2pp |
| r64 | -2.2pp | -1.1pp | -2.2pp | -3.4pp | -6.1pp | 0.0pp |
| r128 | -4.6pp | -1.1pp | +0.4pp | -1.9pp | -4.2pp | -1.8pp |
| **Avg** | **-3.0pp** | **-1.5pp** | **-1.2pp** | **-2.7pp** | **-5.1pp** | **-1.0pp** |

### Summary

**16k is worse than 4k across nearly every benchmark for Llama-8B distillation.** The average delta is negative on MATH-500 (-5.2pp Kimi, -3.0pp Qwen), MBPP+ (-5.1pp, -2.7pp), and HumanEval+ (-2.7pp, -5.1pp).

This is **counterintuitive** — more data and longer traces should help. Possible explanations:

1. **The model can't absorb long traces**: Llama-8B may lack the capacity to learn from 16k-length reasoning patterns. The signal-to-noise ratio in very long traces may exceed what an 8B model can extract with LoRA.
2. **Training convergence**: 16k runs train for 400 steps on more data, but the per-token learning may be slower. The checkpoint curves show 16k models are still improving at step 400, suggesting they need more training time.
3. **The 4k filter is beneficial**: By keeping only the shortest (and likely easiest/cleanest) traces, the 4k filter may create a higher-quality training set. The truncated-away long traces may be noisier or harder to learn from.

The degradation is **consistent across both Kimi and Qwen**, so it is not specific to verbose teachers. If anything, the more verbose teacher (Qwen) shows a slightly smaller 4k→16k degradation on MATH-500, but larger on code benchmarks.

---

## Question 6: Effect of LoRA Rank (32 vs 64 vs 128)

### Math distillation — MATH-500 scores

| Teacher | max_len | r32 | r64 | r128 | Best rank |
|---------|---------|-----|-----|------|-----------|
| Kimi | 4k | 50.4% | **54.0%** | 52.8% | r64 |
| Kimi | 16k | **47.8%** | 47.6% | 46.2% | r32 |
| Qwen | 4k | 44.2% | 44.0% | **47.6%** | r128 |
| Qwen | 16k | **42.0%** | 41.8% | **43.0%** | r128 |

### Code distillation — MBPP+ scores

| Teacher | max_len | r32 | r64 | r128 | Best rank |
|---------|---------|-----|-----|------|-----------|
| Kimi | 4k | **60.1%** | 59.0% | 58.7% | r32 |
| Qwen | 4k | 45.2% | 46.6% | **51.1%** | r128 |

### Training loss comparison

Training loss differences across ranks are negligible (< 0.002 NLL difference at final step). For example, Kimi math 4k: r32=0.3376, r64=0.3372, r128=0.3372. The optimization landscape is essentially identical — LoRA rank does not meaningfully affect training loss.

### Analysis

**There is no clear, consistent trend with LoRA rank.** Results across ranks are noisy and fall within a ~4pp range:

- For Kimi math, r64 is best at 4k but r32 is best at 16k.
- For Qwen math, r128 is best at both 4k and 16k, but only by 1-3pp.
- For Kimi code, r32 is actually best (60.1%).
- Training loss is virtually identical across ranks.

**Higher rank does NOT help more at 16k than at 4k.** The rank-performance relationship is inconsistent at both context lengths. This suggests that the rank-32 LoRA already has sufficient capacity — the bottleneck is elsewhere (likely the base model's 8B parameter count, not the adapter's expressiveness).

---

## Question 7: Base Llama-8B Performance — Formatting vs. Capability

| Benchmark | Llama-8B-Instruct (4k) | Score |
|-----------|------------------------|-------|
| MATH-500 | 50.8% (254/500) | Moderate |
| AIME | 4.4% (4/90) | Near-zero |
| OmniMath | 14.4% (72/500) | Low |
| MBPP+ | 57.9% (219/378) | Decent |
| HumanEval+ | 62.2% (102/164) | Decent |
| LiveCodeBench | 7.2% (12/167) | Low |

**Note**: These are the Llama-3.1-8B-**Instruct** model scores (not the raw base model). The model was evaluated with its chat template. The 4k and 16k base evals produce essentially identical scores (within 1-2pp), confirming that for the base model, extra generation budget doesn't help.

**Is the near-zero AIME performance due to formatting or capability?**

Examining the base model's AIME outputs:

1. **Problem 0** (gt=24): Model predicted 7. Output was only 1,607 characters — a structured "Step 1, Step 2..." attempt that reached the wrong answer. The model **does produce `\boxed{}` formatted answers** and follows the expected format. This is a genuine reasoning failure, not a formatting issue.

2. **Problem 1** (gt=550): Model predicted 1840. Output was 2,509 chars — too short for this complex geometry problem. The model attempted a solution but lacked the reasoning depth to get to the correct answer.

3. **Problem 2** (gt=49): Model predicted "no_boxed" — 13,327 chars of output at 4k but never converged on a final answer. At 16k, the same problem generated 54,592 chars but still no `\boxed{}` answer. **This is both a capability and formatting failure**: the model rambles without converging, and more tokens make it worse.

**Conclusion**: The low AIME score is primarily a **genuine capability limitation**, not a formatting issue. The model can produce `\boxed{}` answers (it does so for MATH-500 at 50.8% accuracy). On AIME's harder problems, it either:
- Gets the math wrong (most common)
- Fails to converge on an answer within the token budget (especially with longer context, where it generates verbose non-convergent reasoning)

---

## Question 8: Training Data Overlap with AIME Eval Problems

### Overlap measurement

| Teacher traces | AIME problems in training | Total training problems | Overlap rate |
|---------------|--------------------------|----------------------|-------------|
| Kimi math | 19/90 | 35,771 | 0.05% of training data |
| Qwen math | 19/90 | 34,969 | 0.05% of training data |
| Sonnet math | 15/90 | 28,939 | 0.05% of training data |

**~19 of 90 AIME eval problems (~21%) appear in the Kimi/Qwen training traces.** The 15 overlapping problems from Sonnet are a subset of the 19 from Kimi/Qwen, suggesting a common upstream math dataset was used to generate all teacher traces.

### Does the best distilled model exploit this overlap?

The best distilled Llama-8B models on AIME score only **0-4/90**. This is actually **equal to or worse than the base model** (4/90). The distilled models that do get problems right frequently solve the same easy indices (problems 10, 64, 81 appear most often across experiments).

Given that ~19 AIME problems were in training data but the model can only solve 0-4, the distilled model is **not meaningfully memorizing or exploiting the overlap**. It appears the AIME problems are simply too hard for Llama-8B to learn from a single training pass, regardless of having seen similar problems. The 4/90 correct answers likely reflect problems easy enough that even the base model can solve them, rather than training-set memorization.

### Are the correct problems novel or from training?

Without per-problem overlap labels, we can't definitively say which of the 4 correct problems were in training. However, since the base model (with no distillation) also gets 4/90 on the same test — and some experiments get **fewer** correct than the base model — the distillation has not added AIME capability from the training data.

---

## Question 9: Training Loss Curves — 4k vs 16k

### Math distillation with Kimi traces

| Setting | Steps | Initial loss | Final loss | Total drop |
|---------|-------|-------------|------------|-----------|
| 4k (r32) | 279 | 0.5654 | 0.3376 | -0.228 |
| 4k (r64) | 279 | 0.5654 | 0.3372 | -0.228 |
| 4k (r128) | 279 | 0.5654 | 0.3372 | -0.228 |
| 16k (r32) | 400 | 0.6809 | 0.4238 | -0.257 |
| 16k (r64) | 400 | 0.6809 | 0.4228 | -0.258 |
| 16k (r128) | 400 | 0.6809 | 0.4226 | -0.258 |

### Math distillation with Qwen traces

| Setting | Steps | Initial loss | Final loss | Total drop |
|---------|-------|-------------|------------|-----------|
| 4k (r32) | 190 | 0.6717 | 0.3450 | -0.327 |
| 4k (r64) | 190 | 0.6717 | 0.3445 | -0.327 |
| 4k (r128) | 190 | 0.6717 | 0.3443 | -0.327 |
| 16k (r32) | 400 | 0.8106 | 0.4995 | -0.311 |
| 16k (r64) | 400 | 0.8106 | 0.4984 | -0.312 |
| 16k (r128) | 400 | 0.8106 | 0.4972 | -0.313 |

### Key observations

1. **16k runs start at higher loss** (Kimi: 0.6809 vs 0.5654; Qwen: 0.8106 vs 0.6717). This makes sense — longer sequences are harder to predict on average.

2. **16k runs plateau at higher final loss** (Kimi: 0.4238 vs 0.3376; Qwen: 0.4995 vs 0.3450). The gap is substantial: 0.09 NLL for Kimi and 0.15 NLL for Qwen.

3. **The 16k loss does NOT converge to the 4k level with more steps.** Even with 400 steps (vs 279 for 4k Kimi, 190 for 4k Qwen), the 16k loss remains 0.09-0.15 NLL higher. This is not a convergence issue — it's a fundamentally harder optimization target.

4. **LoRA rank has no effect on loss** (< 0.002 difference across r32/r64/r128 in all settings).

5. **Total drop is comparable** (4k drops ~0.23-0.33; 16k drops ~0.26-0.31), suggesting the models learn at a similar rate but start and end at different absolute levels.

6. **Test loss tracks training loss** (only measured at steps 0 and 200, no overfitting detected).

The higher final loss at 16k likely explains the worse benchmark performance: the model is less confident in its predictions on longer sequences, and this manifests as lower accuracy.

---

## Question 10: Qualitative Analysis of Model Outputs on AIME

Since no distilled Llama-8B model achieves notable AIME performance, we examine the **best overall model**: Qwen3-30B-A3B-Instruct at 16k (64/90 correct).

### Reasoning pattern frequency in 5 examined problems

| Pattern | Problem 0 (correct) | Problem 1 (correct) | Problem 2 (correct) | Problem 11 (wrong) | Problem 16 (wrong) |
|---------|-------------------|-------------------|--------------------|--------------------|---------------------|
| "Wait" | 2 | 2 | 11 | 0 | 7 |
| "Alternatively" | 3 | 1 | 1 | 0 | 0 |
| "Actually" | — | — | — | 0 | 1 |
| "mistake" | — | — | — | 0 | 1 |
| "I think" | — | — | — | 0 | 1 |
| Output length | 10,791 | 8,466 | 18,500 | 4,132 | 26,005 |

### Does the model produce Long CoT with backtracking?

**Yes, extensively.** The Qwen-30B model produces classic extended reasoning with frequent self-correction:

- **Problem 2** (correct, gt=49): 18,500 chars with **11 "Wait" moments**. The model repeatedly rechecks its combinatorial counting, catching errors and revising. This is classic Long CoT with backtracking — not a clean linear trace.

- **Problem 16** (incorrect, gt=371): 26,005 chars with **7 "Wait" moments**, 1 "mistake", 1 "wrong". The model detected its own error and attempted self-correction, but the correction was itself wrong (predicted 365 instead of 371). This shows that self-correction is present but imperfect.

- **Problem 11** (incorrect, gt=81): Only 4,132 chars with **zero reasoning hedging patterns**. The model jumped to a quick, confident (but wrong) answer without exploring alternatives. This is a failure mode where the model's brevity indicates overconfidence.

### Output length vs. problem difficulty

There is a clear correlation between output length and problem difficulty, but **not a reliable one**:
- Correct solutions range from 8,466 to 18,500 chars
- The shortest output (4,132 chars) was incorrect — suggesting the model didn't think hard enough
- The longest output (26,005 chars) was also incorrect — suggesting that more tokens don't guarantee correctness

### How does this compare to Sonnet-style traces?

Sonnet traces (examined from the training data) are dramatically shorter — median 2,356 chars vs Kimi/Qwen's 5,000-17,000 chars. Sonnet produces clean, linear reasoning without backtracking patterns. The Qwen-30B model's outputs are much more similar to the Kimi/Qwen teacher style: verbose, exploratory, and self-correcting. This suggests the Qwen-30B model natively uses extended thinking, while Sonnet's training traces represent a more compressed reasoning style.

### Distilled Llama-8B outputs on AIME

For comparison, the base Llama-8B-Instruct model produces short, structured outputs (1,500-2,500 chars) that follow a "Step 1, Step 2" format. These lack any backtracking or self-correction. With 16k tokens, the model sometimes generates very long outputs (up to 54,592 chars) but these are verbose rambling, not productive extended reasoning. The distilled models don't show markedly different output patterns from the base model on AIME, suggesting the distillation didn't successfully transfer the extended reasoning behavior for problems at this difficulty level.

---

## Summary of Findings

1. **Distillation provides marginal gains on in-domain benchmarks** (+3.2pp MATH-500, +2.2pp MBPP+) but **degrades or doesn't help on hard benchmarks** (AIME, OmniMath).

2. **16k context is counterproductively worse than 4k** for Llama-8B distillation, likely due to insufficient model capacity and training convergence.

3. **Kimi traces consistently outperform Qwen traces**, possibly due to better signal-to-noise ratio (Kimi is intermediate in verbosity).

4. **LoRA rank (32/64/128) has no consistent effect** — training loss is identical, and benchmark differences are noisy.

5. **The base model capability is the dominant factor**: Qwen-30B (71.1% AIME) dwarfs anything achievable by distilling into Llama-8B (0-4.4% AIME).

6. **Sonnet traces were not included in training**, making it impossible to compare teacher verbosity effects. This is a gap worth filling in future sweeps.
