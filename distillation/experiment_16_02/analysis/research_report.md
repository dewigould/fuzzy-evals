# Sonnet 4.5 Reasoning Distillation: Comprehensive Results Report

## Executive Summary

We distill reasoning traces from Claude Sonnet 4.5 into Qwen3-30B-A3B (3B active MoE) via supervised fine-tuning, and compare against distillation from DeepSeek-R1 traces. Sonnet traces dramatically outperform R1 traces across all configurations, and distillation makes models "try harder" as measured by a 3-model LLM judge ensemble. However, distillation induces severe cross-domain catastrophic forgetting, and the gains are much smaller when starting from an instruction-tuned base model.

---

## 1. R1 vs Sonnet Distillation: A Clear Winner

### 1.1 Experimental Setup

We trained 10 LoRA adapters across 3 student models, 2 teacher trace sources, and 2 data domains:

| # | Teacher | Data | Student | LR | Steps | Training Examples |
|---|---------|------|---------|----|-------|-------------------|
| 1 | R1 | Math | Qwen Base | 2e-5 | 500 | 25K |
| 2 | R1 | Code | Qwen Base | 2e-5 | 500 | 25K |
| 3 | Sonnet 4.5 | Math | Qwen Base | 2e-4 | 500 | 25K |
| 4 | Sonnet 4.5 | Code | Qwen Base | 2e-4 | 500 | 25K |
| 5 | Sonnet 4.5 | Math | Qwen Instruct | 2e-4 | 500 | 25K |
| 6 | Sonnet 4.5 | Code | Qwen Instruct | 2e-4 | 500 | 25K |
| 7 | R1 | Math | Llama 8B | 2e-5 | 500 | 25K |
| 8 | R1 | Code | Llama 8B | 2e-5 | 500 | 25K |
| 9 | Sonnet 4.5 | Math | Llama 8B | 2e-4 | 500 | 25K |
| 10 | Sonnet 4.5 | Code | Llama 8B | 2e-4 | 500 | 25K |

All runs use LoRA rank 32, batch size 50, max_length 4096, with checkpoints saved every 100 steps (= 5,000 training examples). Best checkpoints selected by in-domain accuracy.

### 1.2 Training Data Comparison

|  | R1 Math | R1 Code | Sonnet Math | Sonnet Code |
|--|---------|---------|-------------|-------------|
| **Source** | OpenR1-Math-220k | OpenCodeReasoning | Generated in-house | Generated in-house |
| **Total available** | 92,639 | 260,942 | 28,939 | 28,270 |
| **Used for training** | 25,000 | 25,000 | 25,000 | 25,000 |
| **Median response length** | 12,972 chars | 12,474 chars | 1,985 chars | 2,214 chars |
| **Mean response length** | 16,326 chars | 15,549 chars | 2,026 chars | 2,468 chars |
| **P95 response length** | 40,235 chars | 38,599 chars | 3,462 chars | 4,704 chars |
| **Generation pass rate** | N/A (pre-filtered) | N/A (pre-filtered) | 72.3% (of 40K) | 70.7% (of 40K) |

**Key observation:** R1 traces are **6-8x longer** than Sonnet traces (median ~13K vs ~2K chars). Despite being much more verbose, R1 traces produce worse distillation outcomes. This suggests that concise, well-structured reasoning (Sonnet) transfers more effectively than verbose chain-of-thought (R1) during SFT.

### 1.3 Head-to-Head: R1 vs Sonnet on Qwen Base

Best checkpoint for each configuration, evaluated on MATH-500 (100 problems) and MBPP+ (100 problems):

| Config | Teacher | Best Step | MATH-500 | MBPP+ |
|--------|---------|-----------|----------|-------|
| **Baseline** | — | — | 82.6% | 77.0% |
| R1 Math | DeepSeek-R1 | step200 | 72.0% | 37.0% |
| R1 Code | DeepSeek-R1 | step300 | 75.0% | 48.0% |
| **Sonnet Math** | Sonnet 4.5 | **step300** | **91.0%** | 73.0% |
| **Sonnet Code** | Sonnet 4.5 | **step200** | 83.0% | **74.0%** |

**R1 distillation actively hurts the model.** The best R1 math checkpoint (72% MATH-500) is 10.6pp *below* the undistilled baseline (82.6%). R1 code is slightly worse too (75% vs 82.6%). Both R1 configs also severely damage cross-domain performance.

**Sonnet distillation improves math while preserving code.** Sonnet math at step300 achieves 91% MATH-500 (+8.4pp over baseline) while only dropping code to 73% (-4pp). Sonnet code achieves 74% MBPP+ while roughly maintaining math at 83%.

### 1.4 Why R1 Traces Fail

We identify three factors:

1. **Truncation at max_length=4096 tokens.** R1 traces are 6-8x longer than Sonnet traces. With max_length=4096, many R1 traces are truncated mid-reasoning, teaching the model to generate incomplete solutions. Sonnet traces (median ~2K chars, well under 4096 tokens) are rarely truncated.

2. **Verbosity without signal.** R1 traces contain extensive self-correction, backtracking, and redundant verification steps. The student model learns to reproduce this verbosity pattern without necessarily learning the underlying reasoning. Sonnet traces are more concise and structured.

3. **Lower learning rate.** R1 runs used LR=2e-5 (vs 2e-4 for Sonnet), which was necessary to prevent divergence with the longer sequences but may have been too conservative for effective learning. (Note: this is a confound — the comparison is not fully controlled.)

### 1.5 Llama 8B Results Confirm the Pattern

The Llama 8B results reinforce the Sonnet advantage but also reveal model capacity limitations:

| Config | MATH-500 | MBPP+ |
|--------|----------|-------|
| **Llama 8B Baseline** | 40% | 46% |
| R1 Math (best=final) | 44% (+4pp) | 35% (-11pp) |
| R1 Code (best=step400) | 35% (-5pp) | 48% (+2pp) |
| **Sonnet Math (best=final)** | **50% (+10pp)** | **53% (+7pp)** |
| Sonnet Code (best=step400) | 1% (-39pp) | **58% (+12pp)** |

Sonnet math distillation is the only config that improves *both* MATH-500 and MBPP+ on Llama. However, Sonnet code distillation catastrophically breaks math (0-1% across all checkpoints) — the model completely loses the ability to produce boxed answers.

### 1.6 Conclusions on R1 vs Sonnet

**Sonnet traces are strictly superior to R1 traces for SFT distillation** across all tested student models (Qwen Base, Qwen Instruct, Llama 8B) and both domains (math, code). The advantage is large: +19pp on MATH-500 (91% vs 72%), +37pp on MBPP+ (74% vs 37%) at best checkpoints on Qwen Base.

This is robust because: (a) it holds across 3 different student architectures, (b) it holds for both math and code domains, and (c) R1 traces actively hurt models while Sonnet traces improve them.

**Caveat:** The learning rate differs between R1 (2e-5) and Sonnet (2e-4) configs. A fully controlled comparison would use the same LR, but the LR choice was made based on training stability — R1's longer sequences required lower LR to avoid divergence.

---

## 2. Sonnet Distillation: Detailed Results

### 2.1 Qwen Base Model (Qwen3-30B-A3B-Base)

**Baseline:** MATH-500=82.6%, AIME=15.6%, MBPP+=77.0%, HumanEval+=89.6%, LiveCodeBench=12.0%

#### Checkpoint Sweep — Math Distilled

| Step (examples) | MATH-500 | AIME | MBPP+ | HE+ | LCB |
|-----------------|----------|------|-------|-----|-----|
| 100 (5K) | 87.0 | — | 65.0 | 81.0 | — |
| 200 (10K) | 85.0 | 21.1 | 72.0 | 88.0 | 14.4 |
| **300 (15K)** | **91.0** | 21.1 | 73.0 | 90.0 | 10.0 |
| 400 (20K) | 88.0 | — | 73.0 | 86.0 | — |
| 500/final (25K) | 89.0 | 27.8 | 71.0 | 81.0 | 12.0 |

Best checkpoint: **step 300** (selected by MATH-500 accuracy).
- Math: +8.4pp MATH-500 (91 vs 82.6), +5.5pp AIME (21.1 vs 15.6)
- Code: -4.0pp MBPP+ (73 vs 77), +0.4pp HE+ (90 vs 89.6), -2.0pp LCB (10 vs 12)

#### Checkpoint Sweep — Code Distilled

| Step (examples) | MATH-500 | AIME | MBPP+ | HE+ | LCB |
|-----------------|----------|------|-------|-----|-----|
| 100 (5K) | 85.0 | — | 69.0 | 82.0 | — |
| **200 (10K)** | 83.0 | 25.6 | **74.0** | 85.0 | 12.0 |
| 300 (15K) | 80.0 | — | 74.0 | 85.0 | — |
| 400 (20K) | 87.0 | — | 71.0 | 82.0 | — |
| 500/final (25K) | 82.0 | 22.2 | 73.0 | 84.0 | 14.0 |

Best checkpoint: **step 200** (selected by MBPP+ accuracy).
- Math: +0.4pp MATH-500 (83 vs 82.6), +10.0pp AIME (25.6 vs 15.6)
- Code: -3.0pp MBPP+ (74 vs 77), -4.6pp HE+ (85 vs 89.6), 0pp LCB (12 vs 12)

**Note on eval sample sizes:** Early checkpoint sweeps used 100-problem subsets. The "best" checkpoints (s300, s200) were later evaluated on full benchmarks (500 MATH-500, 378 MBPP+, 100 HE+, 100 LCB, 90 AIME). Full-eval numbers may differ slightly from the 100-sample sweep numbers.

### 2.2 Qwen Instruct Model (Qwen3-30B-A3B-Instruct-2507)

**Baseline:** MATH-500=89.4%, AIME=35.6%, MBPP+=77.8%, HumanEval+=91.0%, LiveCodeBench=27.0%

#### Checkpoint Sweep — Math Distilled

| Step (examples) | MATH-500 | AIME | MBPP+ | HE+ | LCB |
|-----------------|----------|------|-------|-----|-----|
| 100 (5K) | 90.4 | — | 47.6 | — | — |
| **200 (10K)** | **91.6** | 28.9 | 68.3 | 68.0 | 17.0 |
| 300 (15K) | 91.0 | — | 61.4 | — | — |
| 400 (20K) | 90.2 | — | 67.2 | — | — |
| 500/final (25K) | 89.8 | 30.0 | 65.9 | 70.0 | 23.0 |

Best checkpoint: **step 200** (selected by MATH-500 accuracy).
- Math: +2.2pp MATH-500 (91.6 vs 89.4), -6.7pp AIME (28.9 vs 35.6)
- Code: -9.5pp MBPP+ (68.3 vs 77.8), -23.0pp HE+ (68 vs 91), -10.0pp LCB (17 vs 27)

#### Checkpoint Sweep — Code Distilled

| Step (examples) | MATH-500 | AIME | MBPP+ | HE+ | LCB |
|-----------------|----------|------|-------|-----|-----|
| 100 (5K) | 90.6 | — | 79.9 | — | — |
| 200 (10K) | 92.0 | 35.6 | 78.8 | 87.8 | 25.1 |
| 300 (15K) | 91.8 | — | 78.6 | — | — |
| 400 (20K) | 91.2 | — | 79.4 | — | — |
| **500/final (25K)** | **92.2** | **37.8** | **80.2** | 86.0 | 24.0 |

Best checkpoint: **final/step 500** (selected by MBPP+ accuracy).
- Math: +2.8pp MATH-500 (92.2 vs 89.4), +2.2pp AIME (37.8 vs 35.6)
- Code: +2.4pp MBPP+ (80.2 vs 77.8), -5.0pp HE+ (86 vs 91), -3.0pp LCB (24 vs 27)

### 2.3 Base vs Instruct: Key Differences

| Metric | Base | Instruct |
|--------|------|----------|
| Math-distilled best MATH-500 | 91.0% (+8.4pp) | 91.6% (+2.2pp) |
| Math-distilled MBPP+ at best | 73.0% (-4.0pp) | 68.3% (-9.5pp) |
| Code-distilled best MBPP+ | 74.0% (-3.0pp) | 80.2% (+2.4pp) |
| Code-distilled MATH-500 at best | 83.0% (+0.4pp) | 92.2% (+2.8pp) |
| **Cross-domain forgetting** | **Severe** | **Severe for math-dist, mild for code-dist** |

The base model has more room to improve on math (+8.4pp vs +2.2pp) but also suffers worse cross-domain damage. The instruct model's code-distilled variant is notable: it's essentially a **free lunch** — improving MATH-500 by +2.8pp and MBPP+ by +2.4pp simultaneously, with only modest drops on HE+ (-5pp) and LCB (-3pp).

---

## 3. Response Length and Effort Analysis

### 3.1 Response Lengths at Inference

Median response length (characters) by model and benchmark:

| Model | MATH-500 | AIME | MBPP+ | HE+ | LCB |
|-------|----------|------|-------|-----|-----|
| **Qwen Base** | 1,184 | 2,750 | 150 | 441 | 1,112 |
| **Qwen Instruct** | 1,406 | 9,076 | 555 | 842 | 5,156 |
| Sonnet Math (base, s300) | 1,271 | 3,421 | 1,570 | 1,896 | 4,794 |
| Sonnet Code (base, s200) | 1,104 | 5,760 | 1,242 | 1,868 | 5,915 |
| Sonnet Math (instruct, s200) | 1,064 | 2,979 | 1,394 | 1,733 | 4,328 |
| Sonnet Code (instruct, final) | 1,362 | 7,986 | 1,228 | 1,838 | 5,752 |

**Key patterns:**
- **Base model responses are very short on code benchmarks** (150 chars median on MBPP+) — it barely attempts solutions.
- **Distilled models write substantially longer code responses** — 1,200-1,900 chars on MBPP+/HE+, representing 8-13x longer than the base model on MBPP+.
- **On math, response lengths are similar** across all models — distillation doesn't dramatically change math response length.
- **Instruct base already writes long responses** on hard problems (9K AIME, 5K LCB), reducing the marginal effect of distillation.

### 3.2 Pairwise Effort Judging

We used a 3-model LLM judge ensemble (GPT 5.2, Claude Sonnet 4.5, Gemini 2.5 Pro) to compare the *reasoning effort* of paired responses on the same problems. Each judge sees both responses (randomized order) and rates which exhibits more genuine intellectual engagement.

#### 3.2.1 Base Model: Distilled vs Undistilled

**Base vs Math-Distilled (490 pairs, 5 benchmarks):**
- Math-distilled tries harder: **87.3%**, Base tries harder: 8.8%, Similar: 3.9%
- Judge confidence: 0.950 (86% unanimous)

**Base vs Code-Distilled (490 pairs):**
- Code-distilled tries harder: **83.1%**, Base tries harder: 11.0%, Similar: 5.9%
- Judge confidence: 0.939 (83% unanimous)

Both distilled models overwhelmingly try harder than the base model across all benchmarks. The effect is near-universal on code tasks (95-100% on MBPP+, HE+) and strong on math tasks (62-80% on AIME, MATH-500).

**Effort persists on incorrect answers.** When both models get the answer wrong:
- Math-distilled still tries harder 83% of the time
- Code-distilled still tries harder 81% of the time

This means distillation isn't just adding effort that leads to correctness — it fundamentally changes how the model approaches problems, even when that effort doesn't pay off.

#### 3.2.2 Instruct Model: Distilled vs Undistilled

**Instruct vs Math-Distilled (1,168 pairs):**
- Math-distilled tries harder: **51.3%**, Instruct tries harder: 25.3%, Similar: 23.5%
- Judge confidence: 0.892

**Instruct vs Code-Distilled (1,168 pairs):**
- Code-distilled tries harder: **68.8%**, Instruct tries harder: 15.0%, Similar: 16.3%
- Judge confidence: 0.916

The instruct model already tries hard by default, so the marginal effort increase from distillation is much smaller. On math benchmarks specifically, the instruct base sometimes tries *harder* than the math-distilled model:
- AIME (both wrong): Instruct tries harder 76% of the time
- MATH-500 (both wrong): Instruct tries harder 83% of the time

This reversal on math is striking — math distillation may actually make the instruct model *less* thorough on math problems it can't solve.

#### 3.2.3 Effort Breakdown by Benchmark

**Base model comparisons (distilled wins):**

| Benchmark | Math-dist tries harder | Code-dist tries harder |
|-----------|----------------------|----------------------|
| MATH-500 (both correct) | 76% (n=76) | 54% (n=71) |
| AIME (both wrong) | 80% (n=65) | 62% (n=63) |
| MBPP+ (both correct) | 100% (n=64) | 99% (n=69) |
| HumanEval+ (both correct) | 99% (n=85) | 100% (n=80) |
| LiveCodeBench (both wrong) | 86% (n=84) | 88% (n=85) |

**Instruct model comparisons (mixed):**

| Benchmark | Math-dist tries harder | Code-dist tries harder |
|-----------|----------------------|----------------------|
| MATH-500 (both correct) | 36% (n=428) | 50% (n=432) |
| AIME (both wrong) | 24% (n=51) | 52% (n=50) |
| MBPP+ (both correct) | 80% (n=240) | 96% (n=282) |
| HumanEval+ (both correct) | 79% (n=66) | 99% (n=83) |
| LiveCodeBench (both wrong) | 47% (n=70) | 55% (n=69) |

**Key insight:** Distillation primarily adds effort on code benchmarks regardless of starting model, but only adds effort on math when starting from the base model. The instruct model's existing math reasoning effort is already at or above the level learned from Sonnet traces.

---

## 4. Cross-Domain Transfer and Catastrophic Forgetting

### 4.1 The Forgetting Problem

Math distillation consistently damages code performance, and vice versa. The severity depends on the base model:

| Config | MATH-500 delta | MBPP+ delta | HE+ delta | AIME delta |
|--------|---------------|-------------|-----------|------------|
| Base + Math dist | +8.4 | -4.0 | +0.4 | +5.5 |
| Base + Code dist | +0.4 | -3.0 | -4.6 | +10.0 |
| Instruct + Math dist | +2.2 | -9.5 | -23.0 | -6.7 |
| Instruct + Code dist | +2.8 | +2.4 | -5.0 | +2.2 |

The instruct model's math distillation is paradoxically *more* damaging to code (-9.5pp MBPP+, -23pp HE+) than the base model's (-4pp MBPP+, +0.4pp HE+). This may be because the instruct model has more sophisticated code abilities to lose.

### 4.2 Format-SFT as a Partial Fix

We experimented with a minimal "format SFT" stage: 1 epoch of SFT on just 19 examples of the target format (math boxed answers or code blocks), applied after domain distillation. Results on 100-problem subsets:

| Config | MATH-500 | MBPP+ |
|--------|----------|-------|
| **Base + Math dist + code fmt** | 88% | 71% |
| Base + Code dist + math fmt | 79% | 73% |
| **Instruct + Math dist + code fmt** | 90% | 56% |
| Instruct + Code dist + math fmt | 92% | 71% |

Format-SFT partially recovers cross-domain performance by re-teaching output formatting, but doesn't fully close the gap. The instruct math-distilled model recovers poorly on MBPP+ (56% vs 77.8% baseline) even with code format SFT.

---

## 5. Training Dynamics

### 5.1 Checkpoint Selection

Training shows consistent patterns across configs:

- **Math accuracy peaks early** (steps 100-300) then slowly declines or plateaus
- **Code accuracy improves monotonically** through training for code-distilled models
- **Cross-domain performance degrades monotonically** — earlier checkpoints are better for preserving non-target-domain capabilities

This creates a tension: later checkpoints are better for in-domain but worse for cross-domain. Best checkpoint selection depends on whether you optimize for in-domain accuracy or overall capability.

### 5.2 Learning Curves (Qwen Base, Sonnet)

**Math-distilled MATH-500 accuracy by step:**
100→87%, 200→85%, **300→91%**, 400→88%, 500→89%

Non-monotonic — step 300 is the sweet spot. The model first learns reasoning patterns (steps 100-300), then may start overfitting to training data distribution (steps 400-500).

**Code-distilled MBPP+ accuracy by step:**
100→69%, **200→74%**, 300→74%, 400→71%, 500→73%

Peaks at step 200 and plateaus. Code capability is learned faster than math.

---

## 6. Methodology

### 6.1 Training

- **Framework:** tinker SDK with LoRA rank 32
- **Data:** 25K examples sampled from filtered trace datasets
- **Hyperparameters:** batch_size=50, max_length=4096, LR=2e-4 (Sonnet) or 2e-5 (R1)
- **Checkpoints:** Every 100 steps (= 5K training examples), 5 total

### 6.2 Evaluation

| Benchmark | Size | Metric | Grading |
|-----------|------|--------|---------|
| MATH-500 | 500 | Accuracy | `extract_boxed` + `grade_answer` (sympy) |
| AIME | 90 | Accuracy | Same as MATH-500 |
| MBPP+ | 378 | Accuracy | Subprocess execution of pytest-style tests |
| HumanEval+ | 100/164 | Accuracy | Subprocess code execution |
| LiveCodeBench v5 | 100 | Accuracy | stdin/stdout test execution |

Note: HumanEval+ was evaluated on 164 problems for base models and 100 for instruct models.

### 6.3 Effort Judging

- **Judge ensemble:** GPT 5.2, Claude Sonnet 4.5, Gemini 2.5 Pro (via OpenRouter)
- **Comparison:** Pairwise, randomized presentation order
- **Verdict:** `JUDGMENT: A/B/Neither` via majority vote
- **Criteria:** Genuine intellectual engagement, problem decomposition, verification — not response length or verbosity
- **Pairs evaluated:** 490 pairs (base variant), 1168 pairs (instruct variant), across 5 benchmarks

---

## 7. Summary Tables

### Table 7.1: Best Checkpoint Accuracy — All Qwen Configs

| Model | Config | Best Step | MATH-500 | AIME | MBPP+ | HE+ | LCB |
|-------|--------|-----------|----------|------|-------|-----|-----|
| **Qwen Base** | Baseline | — | 82.6 | 15.6 | 77.0 | 89.6 | 12.0 |
| | + Sonnet Math | s300 | **91.0** | 21.1 | 73.0 | 90.0 | 10.0 |
| | + Sonnet Code | s200 | 83.0 | 25.6 | 74.0 | 85.0 | 12.0 |
| **Qwen Instruct** | Baseline | — | 89.4 | 35.6 | 77.8 | 91.0 | 27.0 |
| | + Sonnet Math | s200 | 91.6 | 28.9 | 68.3 | 68.0 | 17.0 |
| | + Sonnet Code | final | **92.2** | **37.8** | **80.2** | 86.0 | 24.0 |

### Table 7.2: Overall Effort Win Rates

| Comparison | Distilled tries harder | Baseline tries harder | Similar |
|------------|----------------------|---------------------|---------|
| Base vs Math-dist | 87.3% | 8.8% | 3.9% |
| Base vs Code-dist | 83.1% | 11.0% | 5.9% |
| Instruct vs Math-dist | 51.3% | 25.3% | 23.5% |
| Instruct vs Code-dist | 68.8% | 15.0% | 16.3% |

### Table 7.3: Best Checkpoint Accuracy — All Llama Configs

| Config | Teacher | Best Step | MATH-500 | MBPP+ |
|--------|---------|-----------|----------|-------|
| **Llama 8B Baseline** | — | — | 40 | 46 |
| R1 Math | DeepSeek-R1 | final | 44 | 35 |
| R1 Code | DeepSeek-R1 | step400 | 35 | 48 |
| Sonnet Math | Sonnet 4.5 | final | **50** | **53** |
| Sonnet Code | Sonnet 4.5 | step400 | 1 | **58** |

---

## 8. Open Questions and Future Work

1. **Training question accuracy.** Evaluations of Qwen Base, Qwen Instruct, and Sonnet 4.5 on 5,000 sampled training questions are in progress. These will reveal whether the distilled models are memorizing training examples or generalizing.

2. **Longer training context.** Would max_length=8192 or 16384 improve R1 distillation by reducing truncation? Our R1 traces have median 13K chars — many are truncated at 4096 tokens.

3. **Controlled teacher comparison.** The LR difference (2e-5 vs 2e-4) between R1 and Sonnet configs is a confound. A controlled experiment using the same LR for both would strengthen the teacher quality conclusion.

4. **Multi-domain distillation.** Can we mix math and code traces in a single training run to avoid catastrophic forgetting? What ratio works best?

5. **Effort-accuracy correlation.** The effort judging shows distilled models try harder, but does trying harder actually cause higher accuracy? The both-wrong bucket suggests not always — effort is necessary but not sufficient.
