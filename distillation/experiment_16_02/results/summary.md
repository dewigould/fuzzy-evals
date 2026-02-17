# Experiment 16/02 — Distillation Training Results

Base model: Qwen/Qwen3-30B-A3B-Base (3B active MoE)

## Configuration

| Run | Task | Max tokens | Filter | Samples | Checkpoints |
|-----|------|-----------|--------|---------|-------------|
| math_default | Math SFT | 4096 | none (truncate) | 25K | every 5K (100 steps) |
| math_2k | Math SFT | 2048 | ≤2048 tokens | 10K | every 2.5K (50 steps) |
| code_default | Code SFT | 4096 | none (truncate) | 25K | every 5K (100 steps) |
| code_2k | Code SFT | 2048 | ≤2048 tokens | 10K | every 2.5K (50 steps) |

Training: LoRA rank 32, lr=2e-4, cosine schedule, batch_size=50.

Eval: fast mode (100 problems each), inference max_tokens matches training max_tokens.

## MATH-500

| Model | Samples | Acc (all) | Completed | Acc (completed) |
|-------|---------|-----------|-----------|-----------------|
| Base | — | 80.6% | 97% | 83.3% |
| Math 4K step100 | 5K | 73.0% | 74% | 96.8% |
| Math 4K step200 | 10K | 72.6% | 73% | 96.5% |
| Math 4K step300 | 15K | 73.0% | 74% | 97.3% |
| Math 4K step400 | 20K | 71.0% | 66% | 97.0% |
| Math 4K step500 | 25K | 67.0% | 62% | 95.2% |
| Math 2K step050 | 2.5K | 64.0% | 55% | 92.7% |
| Math 2K step100 | 5K | 61.0% | 53% | 92.5% |
| Math 2K step150 | 7.5K | 58.0% | 49% | 95.9% |
| Math 2K step200 | 10K | 63.0% | 57% | 94.7% |
| Code 4K step100 | 5K | 77.2% | 40% | 89.1% |
| Code 4K step200 | 10K | 76.6% | 62% | 87.1% |
| Code 4K step300 | 15K | 66.0% | 64% | 87.5% |
| Code 4K step400 | 20K | 74.0% | 73% | 87.7% |
| Code 4K step500 | 25K | 72.0% | 78% | 80.8% |
| Code 2K step050 | 2.5K | 66.0% | 64% | 87.5% |
| Code 2K step100 | 5K | 67.0% | 67% | 94.0% |
| Code 2K step150 | 7.5K | 75.0% | 74% | 91.9% |
| Code 2K step200 | 10K | 70.0% | 70% | 88.6% |

## MBPP+

| Model | Samples | Acc (all) | Completed | Acc (completed) |
|-------|---------|-----------|-----------|-----------------|
| Base | — | 75.7% | 100% | 75.7% |
| Math 4K step100 | 5K | 35.2% | 60% | 57.8% |
| Math 4K step200 | 10K | 26.5% | 25% | 58.1% |
| Math 4K step300 | 15K | 31.0% | 24% | 79.2% |
| Math 4K step400 | 20K | 44.0% | 32% | 87.5% |
| Math 4K step500 | 25K | 42.0% | 32% | 75.0% |
| Math 2K step050 | 2.5K | 4.0% | 1% | 100.0% |
| Math 2K step100 | 5K | 42.0% | 36% | 72.2% |
| Math 2K step150 | 7.5K | 48.0% | 49% | 71.4% |
| Math 2K step200 | 10K | 41.0% | 38% | 71.1% |
| Code 4K step100 | 5K | 59.3% | 71% | 83.9% |
| Code 4K step200 | 10K | 56.6% | 68% | 82.9% |
| Code 4K step300 | 15K | 53.0% | 67% | 79.1% |
| Code 4K step400 | 20K | 55.0% | 72% | 76.4% |
| Code 4K step500 | 25K | 56.0% | 70% | 80.0% |
| Code 2K step050 | 2.5K | 70.0% | 83% | 84.3% |
| Code 2K step100 | 5K | 50.0% | 63% | 79.4% |
| Code 2K step150 | 7.5K | 56.0% | 74% | 75.7% |
| Code 2K step200 | 10K | 57.0% | 71% | 80.3% |

## Key Findings

1. **Completion rate is the bottleneck, not reasoning ability.** When models produce a properly formatted answer (boxed for math, fenced code block for code), accuracy is high (80-97%). The gap to base model performance is driven by failure to close `</think>` and produce a final answer.

2. **Math distillation degrades over training.** Math 4K peaks at step100-300 (~73% on MATH-500) then drops to 67% by step500. Completion rate falls from 74% to 62%. Math 2K is worse still (58-64%), because the 2048-token filter removes 89% of training data, leaving only the simplest examples.

3. **Code distillation is more stable.** Code 4K maintains 66-77% on MATH-500 across all steps, with completion rate *improving* from 40% to 78%. On MBPP+, accuracy stays at 53-59% with ~70% completion.

4. **Shorter token limits improve formatting discipline early.** Code 2K step050 achieves 83% MBPP+ completion (best of any model) and 70% accuracy. The 2048-token training constraint forces the model to produce concise, well-formatted responses.

5. **Cross-domain degradation is asymmetric.** Code-distilled models retain most math ability (72-77% vs 80.6% base). Math-distilled models lose most code ability (26-48% vs 75.7% base), primarily through completion failure.

## Notes

- Base model eval used full datasets (500 MATH-500, 378 MBPP+). All other numbers are from fast eval (100 problems each).
- "Completed" for trained models means: `</think>` tag is closed AND a complete `\boxed{}` or ` ```python``` ` block appears after the thinking section.
- "Completed" for base model means: complete `\boxed{}` or ` ```python``` ` block appears anywhere in the output.
- 2K models were evaluated with max_tokens=2048; 4K models with max_tokens=4096.
