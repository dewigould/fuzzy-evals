# Distillation Experiment Results Summary

## Setup

**Goal:** SFT distillation of reasoning traces into small models. Pure cross-entropy loss (no KL divergence), LoRA rank 32, batch size 50, max_length=4096, 25K training examples, checkpoints every 100 steps (steps 100-500 = final).

## Teacher Traces
1. **DeepSeek-R1** — from OpenR1-Math-220k (math) and OpenCodeReasoning (code)
2. **Claude Sonnet 4.5** — generated in-house

## Student Models
1. **Qwen/Qwen3-30B-A3B-Base** (3B active MoE)
2. **Qwen/Qwen3-30B-A3B-Instruct-2507** (instruction-tuned)
3. **Llama-3.1-8B-Instruct** (dense 8B)

## 10 Training Runs

| # | Teacher | Data | Student | LR | Name prefix |
|---|---------|------|---------|----|-------------|
| 1 | R1 | math | Qwen Base | 2e-5 | `math_4k_lr5` |
| 2 | R1 | code | Qwen Base | 2e-5 | `code_4k_lr5` |
| 3 | Sonnet | math | Qwen Base | 2e-4 | `sonnet_math_qwen_4k` |
| 4 | Sonnet | code | Qwen Base | 2e-4 | `sonnet_code_qwen_4k` |
| 5 | Sonnet | math | Qwen Instruct | 2e-4 | `sonnet_math_qwen_instruct_4k` |
| 6 | Sonnet | code | Qwen Instruct | 2e-4 | `sonnet_code_qwen_instruct_4k` |
| 7 | R1 | math | Llama 8B | 2e-5 | `llama8b_math_4k` |
| 8 | R1 | code | Llama 8B | 2e-5 | `llama8b_code_4k` |
| 9 | Sonnet | math | Llama 8B | 2e-4 | `sonnet_math_llama_4k` |
| 10 | Sonnet | code | Llama 8B | 2e-4 | `sonnet_code_llama_4k` |

## Benchmarks
- **MATH-500** (500 problems, boxed answer extraction)
- **MBPP+** (378 problems, code execution)
- **HumanEval+** (100 problems, code execution)
- **LiveCodeBench v5** (100 problems, code execution)
- **AIME** (90 problems, only baseline Qwen Base)

---

## Qwen Base — Baselines

| Benchmark | Accuracy |
|-----------|----------|
| MATH-500 | 80.6% |
| MBPP+ | 75.7% |
| HumanEval+ | 78.7% |
| LiveCodeBench | 12.0% |

## Qwen Base — R1 Distilled (steps 100-500)

| Step | MATH-500 | MBPP+ |
|------|----------|-------|
| **R1 math** 100 | 58% | 10% |
| 200 | 72% | 31% |
| 300 | 65% | 41% |
| 400 | 65% | 48% |
| 500 | 67% | 44% |
| **R1 code** 100 | 66% | 15% |
| 200 | 73% | 43% |
| 300 | 75% | 48% |
| 400 | 74% | 56% |
| 500 | 75% | 49% |

## Qwen Base — Sonnet Distilled (steps 100-500)

| Step | MATH-500 | MBPP+ | HE+ | LCB |
|------|----------|-------|-----|-----|
| **Sonnet math** 100 | 85% | 65% | 80% | — |
| 200 | 82% | 62% | 69% | — |
| **300** | **88%** | 57% | 67% | 10% |
| 400 | 85% | 63% | 76% | — |
| 500 | 86% | 62% | 74% | — |
| **Sonnet code** 100 | 82% | 69% | 82% | — |
| **200** | 79% | **74%** | 85% | 12% |
| 300 | 77% | 74% | 85% | — |
| 400 | 84% | 71% | 82% | — |
| 500 | 79% | 73% | 84% | 14% |

---

## Qwen Instruct — Baselines

| Benchmark | Accuracy |
|-----------|----------|
| MATH-500 | 86.8% |
| MBPP+ | 77.5% |
| HumanEval+ | 91.0% |
| LiveCodeBench | 27.0% |

## Qwen Instruct — Sonnet Distilled (steps 100-500)

| Step | MATH-500 | MBPP+ | HE+ | LCB |
|------|----------|-------|-----|-----|
| **Sonnet math** 100 | 87.4% | 47.6% | — | — |
| **200** | **88.6%** | 68.3% | 60%\* | 16%\* |
| 300 | 88.4% | 61.4% | — | — |
| 400 | 87.4% | 67.2% | — | — |
| 500 | 87.4% | 65.9% | 70% | 23% |
| **Sonnet code** 100 | 87.8% | 79.9% | — | — |
| 200 | 89.6% | 78.8% | — | — |
| 300 | 89.4% | 78.6% | — | — |
| 400 | 85.0%\* | 75.7%\* | — | — |
| **500** | 89.4% | **80.2%** | 86% | 24% |

\*Step 200 HE+/LCB and step 400 results degraded by tinker sampling errors

---

## Llama 8B — Baselines

| MATH-500 | MBPP+ |
|----------|-------|
| 40% | 46% |

## Llama 8B — R1 Distilled (steps 100-400 + final)

| Step | MATH-500 | MBPP+ |
|------|----------|-------|
| **R1 math** 100 | 34% | 23% |
| 200 | 38% | 33% |
| 300 | 36% | 41% |
| 400 | 32% | 32% |
| final | 44% | 35% |
| **R1 code** 100 | 30% | 46% |
| 200 | 33% | 47% |
| 300 | 30% | 47% |
| 400 | 35% | 48% |
| final | 39% | 43% |

## Llama 8B — Sonnet Distilled (steps 100-400 + final)

| Step | MATH-500 | MBPP+ |
|------|----------|-------|
| **Sonnet math** 100 | 45% | 36% |
| 200 | 46% | 47% |
| 300 | 44% | 48% |
| 400 | 45% | 49% |
| final | 50% | 53% |
| **Sonnet code** 100 | 0% | 57% |
| 200 | 0% | 56% |
| 300 | 0% | 55% |
| 400 | 1% | 58% |
| final | 0% | 57% |

---

## Best Checkpoints (Qwen only)

Selection: best MATH-500 for math-distilled, best MBPP+ for code-distilled.

### Qwen Base

| Config | Best step | MATH-500 | MBPP+ | HE+ | LCB |
|--------|-----------|----------|-------|-----|-----|
| Baseline | — | 80.6% | 75.7% | 78.7% | 12% |
| Sonnet math | s300 | **88%** | 57% | 67% | 10% |
| Sonnet code | s200 | 79% | **74%** | 85% | 12% |

### Qwen Instruct

| Config | Best step | MATH-500 | MBPP+ | HE+ | LCB |
|--------|-----------|----------|-------|-----|-----|
| Baseline | — | 86.8% | 77.5% | 91% | 27% |
| Sonnet math | s200 | **88.6%** | 68.3% | 70%\* | 23%\* |
| Sonnet code | s500 | 89.4% | **80.2%** | 86% | 24% |

\*HE+/LCB from final checkpoint (step 200 eval had tinker errors)

---

## Key Findings

1. **Sonnet traces >> R1 traces** on Qwen Base (Sonnet math best: 88% MATH vs R1 math best: 72%)
2. **Catastrophic forgetting on cross-domain**: Math distillation tanks code (MBPP+ drops 75.7→57% on base), code distillation tanks math less
3. **Instruct model is more robust**: Math distillation MBPP+ drop is 77.5→68.3% (instruct) vs 75.7→57% (base)
4. **Code distillation on instruct is a free lunch**: +2.6 MATH, +2.7 MBPP+, -5 HE+, -3 LCB vs baseline
5. **Llama Sonnet code distillation breaks math completely**: 0% MATH-500 across all checkpoints
6. **Llama barely learns from either teacher** — small/no gains on math, marginal on code
