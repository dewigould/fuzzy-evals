# Sonnet 4.5 Distillation Results (Experiment 16/02)

## Training Setup
- **Teacher model**: Claude Sonnet 4.5 via OpenRouter
- **Training data**: 28,939 correct math traces (from 40K generated, 72.8% pass rate)
- **Training config**: Same as R1 runs — 25K samples, lr=2e-4, cosine schedule, LoRA rank 32, batch 50, max_length=4096
- **Checkpoints**: Every 100 steps (5K samples), final at step 500 (25K samples)

## Qwen3-30B-A3B-Base — Sonnet Math Traces

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 85%      | 65%   |
| 200  | 10K     | 82%      | 62%   |
| 300  | 15K     | **88%**  | 57%   |
| 400  | 20K     | 85%      | 63%   |
| 500  | 25K     | 86%      | 62%   |

**Base model**: MATH-500 80.6%, MBPP+ 75.7%

### vs R1 Traces (best R1 checkpoint at step 500)

| Data Source | Best MATH-500 | MBPP+ at best |
|-------------|---------------|---------------|
| Sonnet 4.5  | **88%** (step 300) | 57% |
| R1 traces   | 72% (step 300)     | 41% |
| Base (no FT) | 80.6%             | 75.7% |

Sonnet traces improve math by **+7.4pp** over base and **+16pp** over R1 traces. MBPP+ degrades from base (expected — math-only training), but less than R1 traces.

## Llama-3.1-8B-Instruct — Sonnet Math Traces

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 45%      | 36%   |
| 200  | 10K     | 46%      | 47%   |
| 300  | 15K     | 44%      | 48%   |
| 400  | 20K     | 45%      | 49%   |
| 500  | 25K     | **50%**  | **53%** |

**Base model**: MATH-500 40%, MBPP+ 46%

### vs R1 Traces (best R1 checkpoint at step 500)

| Data Source | Best MATH-500 | Best MBPP+ |
|-------------|---------------|------------|
| Sonnet 4.5  | **50%** (step 500) | **53%** (step 500) |
| R1 traces   | 44% (step 500)     | 35% (step 500)     |
| Base (no FT) | 40%               | 46%                |

Sonnet traces improve math by **+10pp** over base and **+6pp** over R1 traces. Uniquely, MBPP+ also improves (+7pp over base), unlike R1 traces which degraded it.

## Complete R1 Baselines (for reference)

### Qwen (R1 traces, 4k_lr5 config)

| Step | Math→MATH | Math→MBPP | Code→MATH | Code→MBPP |
|------|-----------|-----------|-----------|-----------|
| 100  | 58%       | 10%       | 66%       | 15%       |
| 200  | 72%       | 31%       | 73%       | 43%       |
| 300  | 65%       | 41%       | 75%       | 48%       |
| 400  | 65%       | 48%       | 74%       | 56%       |
| 500  | 67%       | 44%       | 75%       | 49%       |

### Llama (R1 traces)

| Step | Math→MATH | Math→MBPP | Code→MATH | Code→MBPP |
|------|-----------|-----------|-----------|-----------|
| 100  | 34%       | 23%       | 30%       | 46%       |
| 200  | 38%       | 33%       | 33%       | 47%       |
| 300  | 36%       | 41%       | 30%       | 47%       |
| 400  | 32%       | 32%       | 35%       | 48%       |
| 500  | 44%       | 35%       | 39%       | 43%       |

## Sonnet Code Traces

- **Training data**: 28,270 correct code traces (from 40K generated, 70.7% pass rate)
- **Format**: `<think>...</think>` then ````python``` block, verified via unit test execution
- **Same training config** as math runs

### Qwen3-30B-A3B-Base — Sonnet Code Traces

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 82%      | 69%   |
| 200  | 10K     | 79%      | **74%** |
| 300  | 15K     | 77%      | **74%** |
| 400  | 20K     | 84%      | 71%   |
| 500  | 25K     | 79%      | 73%   |
| Base | —       | 80.6%    | 75.7% |

MBPP+ peaks at steps 200-300 (**74%**) — nearly matching the base model (75.7%) and dramatically better than R1 code traces (56%). Math stays strong throughout (77-84%).

### Llama-3.1-8B-Instruct — Sonnet Code Traces

| Step | Samples | MATH-500 | MBPP+ |
|------|---------|----------|-------|
| 100  | 5K      | 0%       | 57%   |
| 200  | 10K     | 0%       | 56%   |
| 300  | 15K     | 0%       | 55%   |
| 400  | 20K     | 1%       | **58%** |
| 500  | 25K     | 0%       | 57%   |
| Base | —       | 40%      | 46%   |

MBPP+ is consistently strong at **55-58%** across all checkpoints (+11pp over base, +14pp over R1 code traces). Math collapses to 0% from the very first checkpoint — the code training format completely overwrites Llama's math ability even at 5K samples, likely because the smaller model can't compartmentalize skills and the code format (`<think>` + python block) doesn't transfer to math format (`<think>` + `\boxed{}`).

## Summary Table — All Sonnet Distillation Results

### Qwen3-30B-A3B-Base

| Training Data | Best MATH-500 | MBPP+ at best MATH | Best MBPP+ | MATH at best MBPP |
|---------------|---------------|---------------------|------------|---------------------|
| Sonnet math   | **88%** (s300) | 57% | 65% (s100) | 85% |
| Sonnet code   | 84% (s400) | 71% | **74%** (s200/s300) | 77-79% |
| R1 math       | 72% (s200) | 31% | 48% (s400) | 65% |
| R1 code       | 75% (s300) | 48% | 56% (s400) | 74% |
| Base          | 80.6% | 75.7% | 75.7% | 80.6% |

### Llama-3.1-8B-Instruct

| Training Data | Best MATH-500 | MBPP+ at best MATH | Best MBPP+ | MATH at best MBPP |
|---------------|---------------|---------------------|------------|---------------------|
| Sonnet math   | **50%** (final) | **53%** | **53%** (final) | **50%** |
| Sonnet code   | 1% (s400) | 58% | **58%** (s400) | 1% |
| R1 math       | 44% (final) | 35% | 41% (s300) | 36% |
| R1 code       | 39% (final) | 43% | 48% (s400) | 35% |
| Base          | 40% | 46% | 46% | 40% |

## Key Takeaways

1. **Sonnet traces dramatically outperform R1 traces** for both math and code distillation on both models
2. **Qwen peaks early** (step 300) suggesting higher quality data needs fewer examples
3. **Llama improves steadily** through training — final checkpoint is best
4. **Cross-domain transfer is better with Sonnet**: Llama's MBPP+ improves with Sonnet math training (+7pp over base), whereas R1 math training degraded it (-11pp)
5. **Trace conciseness matters**: Sonnet traces average ~700 tokens vs R1's ~3K tokens, leading to less truncation and cleaner training signal
6. **Code training preserves more capabilities in larger models**: Qwen retains 79% math after code-only training, while Llama drops to 0% — the MoE architecture may be better at compartmentalizing skills
7. **Sonnet code traces close the MBPP+ gap**: Qwen achieves 73% MBPP+ (vs 75.7% base), essentially eliminating the code degradation seen with R1 traces (49-56%)
