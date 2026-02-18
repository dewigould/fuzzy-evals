# Qwen3-30B-A3B-Base — Complete Distillation Results

## Model

**Qwen/Qwen3-30B-A3B-Base** — 30B parameter Mixture-of-Experts model with 3B active parameters per token. This is the base (pre-trained, non-instruct) variant.

**Base model accuracy**: MATH-500 **80.6%**, MBPP+ **75.7%**

## Experimental Setup

All training runs share the same configuration:
- **Method**: LoRA SFT (rank 32)
- **Training samples**: 25,000 (from larger pools of correct-only traces)
- **Batch size**: 50
- **LR schedule**: Cosine
- **Max sequence length**: 4,096 tokens
- **Epochs**: 1
- **Checkpoints**: Every 100 steps (= 5,000 samples), 5 total (steps 100-500)
- **Evaluation**: MATH-500 and MBPP+ at every checkpoint, temperature 0.0

### Variable: Learning Rate

| Run | Learning Rate |
|-----|--------------|
| R1 math / R1 code | 2e-5 |
| Sonnet math / Sonnet code | 2e-4 |

### Variable: Training Data

| Run | Teacher | Task | Pool Size | Correct | Pass Rate |
|-----|---------|------|-----------|---------|-----------|
| R1 math | DeepSeek-R1 (open-source traces) | Math | — | 220K available | — |
| R1 code | DeepSeek-R1 (open-source traces) | Code | — | 736K available | — |
| Sonnet math | Claude Sonnet 4.5 (via OpenRouter) | Math | 40,000 | 28,939 | 72.8% |
| Sonnet code | Claude Sonnet 4.5 (via OpenRouter) | Code | 40,000 | 28,270 | 70.7% |

Key difference: R1 traces average ~3K tokens with verbose self-correction; Sonnet traces average ~700 tokens and are concise.

---

## 1. R1 Math-Distilled

**Data**: OpenR1-Math-220k reasoning traces (DeepSeek-R1)
**Tinker run**: `fffe39fc-f476-5ea3-b1c5-966c5e0e10c6`

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 58%      | 10%   |
| 200  | 10K     | **72%**  | 31%   |
| 300  | 15K     | 65%      | 41%   |
| 400  | 20K     | 65%      | 48%   |
| 500  | 25K     | 67%      | 44%   |
| Base | —       | 80.6%    | 75.7% |

- MATH-500 peaks at step 200 (72%) then degrades — **9pp below base**
- MBPP+ collapses to 10% at step 100, partially recovers to 48% but never approaches base (75.7%)
- Best checkpoint (step 200): 72% MATH, 31% MBPP+

---

## 2. R1 Code-Distilled

**Data**: OpenCodeReasoning traces (DeepSeek-R1)
**Tinker run**: `7ea1d446-fe88-59f9-94d8-686a9311eba9`

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 66%      | 15%   |
| 200  | 10K     | 73%      | 43%   |
| 300  | 15K     | **75%**  | 48%   |
| 400  | 20K     | 74%      | **56%** |
| 500  | 25K     | 75%      | 49%   |
| Base | —       | 80.6%    | 75.7% |

- MATH-500 reaches 75% (step 300) — closer to base than R1 math, but still 6pp below
- MBPP+ peaks at 56% (step 400) — 20pp below base despite code-specific training
- R1 code traces perform surprisingly well on math, possibly because structured reasoning transfers
- Best MBPP+ checkpoint (step 400): 56% MBPP+, 74% MATH

---

## 3. Sonnet Math-Distilled

**Data**: 28,939 correct Sonnet 4.5 math traces (from 40K generated, 72.8% pass rate)
**Tinker run**: `f889cb10-1497-5573-951c-c538b60bb629`

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 85%      | 65%   |
| 200  | 10K     | 82%      | 62%   |
| 300  | 15K     | **88%**  | 57%   |
| 400  | 20K     | 85%      | 63%   |
| 500  | 25K     | 86%      | 62%   |
| Base | —       | 80.6%    | 75.7% |

- MATH-500 peaks at **88%** (step 300) — **+7.4pp over base**, **+16pp over best R1 math**
- MBPP+ degrades to 57-65% range (expected: math-only training), but far less than R1 math (10-48%)
- Performance is strong from the very first checkpoint (85% MATH at step 100 / 5K samples)
- Best checkpoint (step 300): 88% MATH, 57% MBPP+

---

## 4. Sonnet Code-Distilled

**Data**: 28,270 correct Sonnet 4.5 code traces (from 40K generated, 70.7% pass rate)
**Tinker run**: `f0452a6d-df68-5a5d-b3ea-c4c486173024`

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 82%      | 69%   |
| 200  | 10K     | 79%      | **74%** |
| 300  | 15K     | 77%      | **74%** |
| 400  | 20K     | 84%      | 71%   |
| 500  | 25K     | 79%      | 73%   |
| Base | —       | 80.6%    | 75.7% |

- MBPP+ peaks at **74%** (steps 200-300) — only **1.7pp below base**, essentially closing the gap
- MATH-500 stays strong throughout (77-84%), never dropping more than 4pp below base
- This is the best overall configuration: strong on both benchmarks simultaneously
- Best MBPP+ checkpoint (step 200): 74% MBPP+, 79% MATH
- Best MATH checkpoint (step 400): 84% MATH, 71% MBPP+

---

## Summary: All Qwen Results

### Best In-Domain Performance

| Training Data | Best MATH-500 | Step | Best MBPP+ | Step |
|---------------|---------------|------|------------|------|
| R1 math       | 72%           | 200  | 48%        | 400  |
| R1 code       | 75%           | 300  | 56%        | 400  |
| Sonnet math   | **88%**       | 300  | 65%        | 100  |
| Sonnet code   | 84%           | 400  | **74%**    | 200  |
| Base (no FT)  | 80.6%         | —    | 75.7%      | —    |

### Cross-Domain Transfer (at best in-domain checkpoint)

| Training Data | Best In-Domain | In-Domain Score | Cross-Domain Score |
|---------------|----------------|-----------------|---------------------|
| R1 math       | MATH @ s200    | 72%             | MBPP+ 31%          |
| R1 code       | MBPP+ @ s400   | 56%             | MATH 74%           |
| Sonnet math   | MATH @ s300    | **88%**         | MBPP+ 57%          |
| Sonnet code   | MBPP+ @ s200   | **74%**         | MATH 79%           |

### Delta from Base Model

| Training Data | Best MATH-500 vs Base | Best MBPP+ vs Base |
|---------------|----------------------|---------------------|
| R1 math       | -8.6pp               | -27.7pp             |
| R1 code       | -5.6pp               | -19.7pp             |
| Sonnet math   | **+7.4pp**           | -10.7pp             |
| Sonnet code   | +3.4pp               | **-1.7pp**          |

### Sonnet vs R1 Head-to-Head

| Metric | R1 Math | Sonnet Math | Sonnet Gain |
|--------|---------|-------------|-------------|
| Best MATH-500 | 72% | **88%** | **+16pp** |
| MBPP+ at best MATH | 31% | 57% | **+26pp** |

| Metric | R1 Code | Sonnet Code | Sonnet Gain |
|--------|---------|-------------|-------------|
| Best MBPP+ | 56% | **74%** | **+18pp** |
| MATH at best MBPP+ | 74% | 79% | **+5pp** |

---

## Training Dynamics

### Convergence Speed

- **R1 traces**: Performance is noisy and unstable. MATH-500 can drop 7pp between checkpoints (e.g., R1 math s200→s300: 72→65%). MBPP+ starts near-zero and slowly recovers.
- **Sonnet traces**: Performance is strong from step 100 (5K samples). The model reaches 85% MATH and 65% MBPP+ with just 5K Sonnet math examples, vs 58% MATH and 10% MBPP+ with 5K R1 examples.

### Optimal Checkpoint

| Config | Best MATH Step | Best MBPP+ Step | Recommended |
|--------|---------------|-----------------|-------------|
| R1 math | 200 (72%) | 400 (48%) | Step 200 (best math, but MBPP+ is only 31%) |
| R1 code | 300 (75%) | 400 (56%) | Step 400 (best overall balance) |
| Sonnet math | 300 (88%) | 100 (65%) | Step 300 (peak math, acceptable MBPP+ at 57%) |
| Sonnet code | 400 (84%) | 200 (74%) | Step 200 (best overall balance: 79% MATH + 74% MBPP+) |

### Why Sonnet Traces Outperform R1

1. **Conciseness**: Sonnet traces average ~700 tokens vs R1's ~3K tokens. With max_length=4096, R1 traces are frequently truncated, losing the final answer and creating noisy training signal.
2. **No self-correction loops**: R1 traces contain "Wait, let me reconsider..." patterns that the student model learns to imitate, wasting token budget on hesitation rather than reasoning.
3. **Clean format compliance**: Sonnet traces consistently follow the `<think>...</think>` + answer format, while R1 traces have more format variation.
4. **Higher training LR**: Sonnet runs used 2e-4 vs R1's 2e-5 (10x higher). The concise, consistent Sonnet data may tolerate higher learning rates without destabilizing, enabling faster and more complete adaptation.

---

## Notable Findings

### 1. Sonnet math distillation beats the base model on MATH

This is the standout result. The base Qwen3-30B-A3B-Base model scores 80.6% on MATH-500. After training on just 15K Sonnet 4.5 math traces, it reaches **88%** — a +7.4pp improvement. This is remarkable because:
- The model is learning to reason in a structured `<think>...</think>` format it wasn't pre-trained on
- R1 traces on the same setup actually **degraded** math performance below base (72% vs 80.6%)
- The improvement comes at the cost of only 19pp on MBPP+ (75.7 → 57%), far less than R1's 45pp degradation

### 2. Sonnet code distillation nearly preserves MBPP+

Sonnet code traces achieve 74% MBPP+ — within 1.7pp of the base model's 75.7%. Meanwhile, R1 code traces peak at 56%, a 20pp degradation. This suggests Sonnet's concise traces create a much cleaner code training signal.

### 3. Math knowledge is remarkably robust to code training

Even after code-only training, Qwen maintains 77-84% MATH-500 (base: 80.6%). This stands in stark contrast to Llama-8B, which drops to 0% MATH after code training. The MoE architecture likely enables better compartmentalization of skills.

### 4. The Pareto frontier is Sonnet-dominated

Every point on the MATH-MBPP+ Pareto frontier comes from a Sonnet-trained checkpoint:
- Highest MATH: Sonnet math s300 (88% MATH, 57% MBPP+)
- Best balance: Sonnet code s200 (79% MATH, 74% MBPP+)
- No R1 checkpoint is Pareto-optimal

---

## Appendix: Training Run Details

### Tinker Run IDs

| Config | Tinker Run UUID |
|--------|----------------|
| R1 math (4k_lr5) | `fffe39fc-f476-5ea3-b1c5-966c5e0e10c6` |
| R1 code (4k_lr5) | `7ea1d446-fe88-59f9-94d8-686a9311eba9` |
| Sonnet math | `f889cb10-1497-5573-951c-c538b60bb629` |
| Sonnet code | `f0452a6d-df68-5a5d-b3ea-c4c486173024` |

### Training Data Sources

| Config | Data Source | File |
|--------|-----------|------|
| R1 math | open-r1/OpenR1-Math-220k | HuggingFace dataset |
| R1 code | nvidia/OpenCodeReasoning | HuggingFace dataset |
| Sonnet math | Generated via OpenRouter (Sonnet 4.5) | `generate_reasoning_traces/data/correct_only.jsonl` |
| Sonnet code | Generated via OpenRouter (Sonnet 4.5) | `generate_reasoning_traces/data_code/correct_only.jsonl` |

### Hyperparameters

| Parameter | R1 Runs | Sonnet Runs |
|-----------|---------|-------------|
| Learning rate | 2e-5 | 2e-4 |
| LR schedule | Cosine | Cosine |
| LoRA rank | 32 | 32 |
| Batch size | 50 | 50 |
| Max length | 4096 | 4096 |
| Max prompts | 25,000 | 25,000 |
| Epochs | 1 | 1 |
| Optimizer | AdamW (β1=0.9, β2=0.95) | AdamW (β1=0.9, β2=0.95) |
| Filter max length | 4096 | 4096 (math), None (code) |

### Full Checkpoint Results Table

| Config | Step | Samples | MATH-500 | MBPP+ |
|--------|------|---------|----------|-------|
| R1 math | 100 | 5K | 58% | 10% |
| R1 math | 200 | 10K | 72% | 31% |
| R1 math | 300 | 15K | 65% | 41% |
| R1 math | 400 | 20K | 65% | 48% |
| R1 math | 500 | 25K | 67% | 44% |
| R1 code | 100 | 5K | 66% | 15% |
| R1 code | 200 | 10K | 73% | 43% |
| R1 code | 300 | 15K | 75% | 48% |
| R1 code | 400 | 20K | 74% | 56% |
| R1 code | 500 | 25K | 75% | 49% |
| Sonnet math | 100 | 5K | 85% | 65% |
| Sonnet math | 200 | 10K | 82% | 62% |
| Sonnet math | 300 | 15K | 88% | 57% |
| Sonnet math | 400 | 20K | 85% | 63% |
| Sonnet math | 500 | 25K | 86% | 62% |
| Sonnet code | 100 | 5K | 82% | 69% |
| Sonnet code | 200 | 10K | 79% | 74% |
| Sonnet code | 300 | 15K | 77% | 74% |
| Sonnet code | 400 | 20K | 84% | 71% |
| Sonnet code | 500 | 25K | 79% | 73% |
| **Base** | — | — | **80.6%** | **75.7%** |
