# Pipeline Audit: Why Is Distillation Giving Near-Zero Improvement?

## Executive Summary

After auditing the full pipeline — training data, loss masking, chat templates, evaluation, answer extraction, LoRA config, and generation parameters — the pipeline is **largely correct**. There is no single catastrophic bug like broken loss masking or unloaded adapters.

The central finding is that **distillation teaches the student model the form of extended reasoning but not the convergence mechanism**. When the distilled model cannot solve a problem, instead of giving up or guessing, it enters degenerate reasoning loops — repeating the same calculations and phrases indefinitely until it hits the token limit. This accounts for ~76% of all incorrect answers.

| Issue | Severity | Impact |
|-------|----------|--------|
| **1. Degenerate reasoning loops** | **CRITICAL** | 76% of incorrect outputs are repetitive loops, not coherent reasoning. The model learned to produce long CoT but not to converge on answers. More tokens doesn't help — 16k models loop 4x longer but don't solve more problems. |
| **2. Greedy decoding (temp=0) with no sampling** | **HIGH** | Li et al. use temp=0.6, top_p=0.95, N=64 samples for pass@1. We use temp=0.0, N=1. Greedy decoding may make loops worse by committing to the highest-probability continuation at every step. |

### Secondary Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| 3. No Sonnet training runs in the 8B sweep | MEDIUM | Can't compare the most compact teacher (shorter CoT, may loop less) |
| 4. Qwen models trained at wrong LR | MEDIUM | Qwen models get 2e-4 instead of recommended ~5e-4 (2.5x too low) |
| 5. AIME contamination (21%) | LOW | 19/90 AIME problems appear in training data, but moot since distilled models score 0-4/90 anyway |
| 6. No warmup, no gradient clipping | LOW | Standard for LoRA SFT but non-ideal |

---

## Issue 1 (CRITICAL): Degenerate Reasoning Loops

### The problem

Distilled models learned to produce extended `<think>...</think>` reasoning from teacher traces. But when they encounter problems they can't solve, they don't converge — they enter repetitive loops, cycling through the same reasoning steps until they exhaust the token budget. The outputs never close `</think>` and never produce `\boxed{}`.

This is **not a truncation problem**. Giving the model more tokens just lets the loop run longer.

### The evidence

Classification of 1,430 incorrect outputs across 8 experiment/dataset combinations:

| Category | Count | % | Description |
|----------|-------|---|-------------|
| **Severe loop** | 594 | 41.5% | Same phrase/sentence repeated 10+ times verbatim |
| **Circling (no progress)** | 297 | 20.8% | Varied wording but stuck in `<think>`, never converges |
| **Moderate loop** | 155 | 10.8% | 5-9 exact phrase repetitions |
| **Circular markers** | 40 | 2.8% | Dense "wait", "actually", "let me reconsider" with no forward progress |
| Wrong answer (normal) | 280 | 19.6% | Model reached `\boxed{}` but gave incorrect answer |
| Truncated coherent | 64 | **4.5%** | Genuinely cut short mid-reasoning |

**~76% of failures are degenerate loops. Only 4.5% are plausibly coherent reasoning that got cut off.**

Concrete examples of looping:
- One output repeats "We need to find the number of points where exactly 2 lines intersect" **1,085 times** across 87K characters
- Another repeats "We change direction exactly 4 times" **55 times** at 4k
- Some degenerate into within-line repetition: `$u \equiv 3 \pmod{50}$? No, $20 = 21 - 1$, so` repeated dozens of times
- Some generate endlessly expanding tables that never converge

### The 16k results confirm this is not truncation

If truncation were the real issue, 16k models should perform much better than 4k (4x more token budget). They don't:

| Metric | 4k models | 16k models |
|--------|-----------|------------|
| Mean incorrect output length | 8,286 chars | 32,328 chars |
| Severe looping rate (incorrect) | 24.7% | **90.8%** |
| Problems solved that the other missed | 48 | 61 |

The 16k models produce **4x longer incorrect outputs** but solve essentially the same number of problems. The extra tokens just let the loop run longer. Among incorrect AIME outputs at 16k, 90.8% are severe loops.

For the 200 problems wrong in both 4k and 16k, the 16k output is 4.0x longer but equally wrong.

### Correct vs incorrect outputs are strikingly different

| Metric | Correct outputs | Incorrect outputs |
|--------|----------------|-------------------|
| Mean length (4k kimi, MATH-500) | 3,119 chars | 8,286 chars |
| Mean length (16k kimi, MATH-500) | 3,768 chars | 32,328 chars |
| Has `</think>` + `\boxed{}` | Always | Rarely (19.6%) |
| "wait" per output | 0.8 | 2.8 |
| "let me" per output | 3.5 | 9.1 |
| "reconsider" per output | 0.1 | 0.7 |

When the model *can* solve a problem, it does so efficiently — closing `<think>`, giving `\boxed{}`, at 2-8x shorter length. When it *can't*, it spirals.

### What this means

The student model learned the **form** of extended reasoning (long `<think>` blocks with backtracking and reconsideration) but not the **convergence mechanism** (knowing when to stop exploring and commit to an answer). This is a fundamental limitation of the distillation — the student is imitating the teacher's surface behavior without acquiring the underlying capability.

This also explains why distillation doesn't improve results: the model was already capable of solving the easy problems (it gets them right with shorter reasoning), and for the hard problems, it just loops. The distillation added verbosity to the easy problems and loops to the hard ones, netting approximately zero improvement.

### Implications for fixes

- **More tokens won't help** — confirmed by 16k results
- **Sampling (temp>0) might help** — could break out of deterministic loops, but the underlying loop tendency would remain
- **Shorter teacher traces (Sonnet)** — might reduce loop tendency since the model learns more concise reasoning patterns
- **Harder training data** — the easy training problems may not teach the model when to give up or try a fundamentally different approach
- **Repetition penalty at eval time** — could suppress loops but is a band-aid
- **Training with a length penalty or EOS reward** — more fundamental fix, but changes the training paradigm

---

## Issue 2 (HIGH): Greedy Decoding vs. Sampling

### The problem

All evals use temperature=0.0 (greedy decoding), N=1 sample:

```python
# infer.py default
async def generate(..., temperature: float = 0.0, ...)

# aime.py hardcodes it
completions = await generate(..., temperature=0.0, ...)
```

Li et al. (2025) use temperature=0.6, top_p=0.95, and generate N=64 samples per problem, taking the majority vote (pass@1 with consensus). This is a standard evaluation protocol for reasoning models.

### Why this matters

Greedy decoding commits to the single highest-probability token at every step. This may actively worsen the looping problem — once the model starts repeating a phrase, greedy decoding will keep selecting the highest-probability continuation, which is the same phrase again. Sampling with temperature>0 could break out of these loops by introducing randomness.

The 64-sample majority vote is also important: if a model solves a problem 30% of the time with sampling, greedy decoding might score 0% (if the greedy path enters a loop), while pass@1 with 64 samples and majority vote would score ~30%.

### The fix

For a fair comparison to published results, eval should use:
- `temperature=0.6`
- `top_p=0.95`
- Multiple samples with majority vote (even N=8 would help; N=64 is ideal but expensive)

For comparing models against each other (internal benchmarking), greedy decoding is fine as long as it's consistent. But it may systematically underestimate distilled models if greedy decoding is more prone to loops than sampling.

---

## What Was Verified Correct

### Filtering token count: CORRECT (conservative)

A concern was raised that the filtering method (raw text concatenation + 30 token overhead) might underestimate the true rendered token count, allowing examples that are too long to slip through and get truncated during training (losing `\boxed{}` answers). **This was tested empirically and is NOT a problem.**

Across 3,600 examples (200 per model/teacher/task combination, covering all renderers: llama3, qwen3_instruct, role_colon):

| Renderer | Token gap (render - filter) | Direction | Truncation risk |
|----------|----------------------------|-----------|-----------------|
| llama3 | -15 to -16 | Filter overestimates | Zero |
| qwen3_instruct | -16 to -17 | Filter overestimates | Zero |
| role_colon | -20 to -22 | Filter overestimates | Zero |

The filter is **conservative** — it rejects ~15-22 examples that would actually fit, but never lets through examples that would be truncated. Zero cases of "filter says ≤4096 but render >4096" were found. The TEMPLATE_OVERHEAD of 30 tokens is more than sufficient for all renderers tested.

### Data integrity: CLEAN

All 185,509 raw trace rows across 6 files (3 teachers × 2 tasks) have:
- Non-empty `messages` array with user + assistant turns
- Non-null, non-empty `generation` field
- Valid `problem` or `question` key

Zero duplicates in the first 2,000 filtered examples sampled. The `map_fn` fallback paths (for rows without `messages`) are effectively dead code — all rows have `messages`.

### Loss masking: CORRECT

The loss is computed on assistant content tokens only. System prompt, user message, and all chat template headers are masked (weight=0). The `<think>` tags and `\boxed{}` answer are both included in the training target (weight=1).

Verified by inspecting:
- `train.py` line 238: `train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES`
- `base.py` lines 1015-1039: headers get weight=0, assistant output gets weight=1
- Token-by-token inspection: mask transitions from 0 to 1 at exactly the first content token after the assistant header

Token-level mask metrics (50 examples per teacher, Llama-8B, math 4k):

| Teacher | Avg total tokens | Avg trained tokens | Trained fraction |
|---------|-----------------|-------------------|-----------------|
| Sonnet | 987 | 879 | 89.1% |
| Kimi | 2,507 | 2,394 | 95.5% |
| Qwen | 2,348 | 2,240 | 95.4% |

The trained fraction is high and healthy. The prompt overhead (system + user + headers) is ~100-150 tokens, which is small relative to the response.

### Adapter loading: CORRECT

Trained LoRA adapters are correctly loaded during evaluation:

- Training saves `sampler_path` (a `tinker://` URI) to `checkpoints.jsonl`
- Runner reads `sampler_path`, threads it through checkpoint selection
- Eval creates `create_sampling_client(model_path=sampler_path)` — distinct from the base model path `create_sampling_client(base_model=model_name)`
- No silent fallback to base model — missing sampler_path raises an exception
- The two eval functions (`evaluate_checkpoint` vs `evaluate_base_model`) use entirely separate code paths

### System prompts: CONSISTENT

Training and evaluation use the same system prompts from `config.py`:
- Math: `"You are a helpful assistant. Answer the following question, putting your final answer inside \boxed{}"`
- Code: `"You are a helpful assistant. Answer the following question. Enclose your code within delimiters as follows: ```python \n #YOUR CODE HERE \n ``` \n\n"`

The system prompt is injected by `train.py` at training time (line 255) and by each eval module at eval time. Both use the same `MATH_SYSTEM_PROMPT` / `CODE_SYSTEM_PROMPT` constants.

### Chat template: CORRECT

Each model uses its recommended renderer:
- Llama-3.1-8B-Instruct → `llama3` (Llama3Renderer)
- Llama-3.3-70B-Instruct → `llama3`
- Qwen3-30B-Instruct → `qwen3_instruct` (Qwen3InstructRenderer)
- Qwen3-8B-Base → `role_colon` (RoleColonRenderer)

The renderer is used identically for training and evaluation. No template mismatch.

### Think prefix logic: CORRECT

| Scenario | think_prefix | Behavior |
|----------|-------------|----------|
| Math-distilled model on math evals | False | Model generates `<think>` natively (learned from training) |
| Math-distilled model on code evals | True | `<think>\n` seeded via prefill |
| Code-distilled model on code evals | False | Model generates `<think>` natively |
| Code-distilled model on math evals | True | `<think>\n` seeded via prefill |
| Base model on all evals | False | No thinking (model doesn't know `<think>`) |

The think_prefix is implemented as prefill (appended to the prompt), not post-hoc modification. This is correct.

### Truncation direction: CORRECT

The pre-filtering step (`filtered_data/`) removes examples exceeding `max_length`. Examples in the filtered data fit within the limit, so `datum_from_model_input_weights` does not need to truncate. Verified: a near-boundary example (4048 tokens) preserved its `\boxed{}` answer intact at the end.

### Data format: CORRECT

All trace files (sonnet, kimi, qwen) use the same JSONL format with `messages` containing user+assistant turns. The system prompt is not in the traces — it's injected at training time. `<think>...</think>` tags are present in all teachers' assistant responses. The filtered data is a pure subset (no transformation).

### Answer extraction: MOSTLY CORRECT

- `extract_boxed()` takes the **last** `\boxed{}` in the full output (correct for multi-step reasoning)
- Two-stage grading: sympy normalization → GPT-5.2 judge fallback (robust)
- Code extraction strips `<think>` tags before extracting code blocks (correct)

One minor issue: `extract_boxed` runs on the full output including `<think>` content. If the model produces `\boxed{wrong}` inside `<think>` and then `\boxed{right}` after `</think>`, the extraction is correct (takes the last one). But if the model produces `\boxed{right}` inside `<think>` and never produces another `\boxed{}` outside, it would still be extracted correctly.

### LoRA configuration: CORRECT

- LoRA applies to **all linear layers**: attention, MLP/MoE, and unembedding
- `lora_alpha` = 32 (fixed, set server-side)
- Scaling factor = `alpha/rank` = 32/rank. At rank 32 this is 1.0; at rank 128 this is 0.25
- No LoRA dropout (standard)
- This is a reasonable configuration

### Data contamination: MATH-500 CLEAN, AIME CONTAMINATED

| Eval set | Contamination |
|----------|--------------|
| MATH-500 | **0%** — zero overlap with any training traces |
| AIME | **21%** — 19/90 problems appear in kimi/qwen traces, 15/90 in sonnet |

MATH-500 is the clean benchmark. AIME results should be interpreted with the contamination caveat, though it's currently moot (distilled models score 0-4/90 due to loops).

---

## Recommended Action Plan

### Immediate

1. **Test sampling eval**: Run the best existing checkpoint with `temperature=0.6` on a small subset (50 MATH-500 problems). Check whether sampling breaks the loops. This is the cheapest diagnostic — if loops persist at temp=0.6, sampling is not the answer.

2. **Examine training trace quality**: Check whether the teacher traces themselves contain any repetitive/circular reasoning patterns that the student might be imitating. If the training data has loops, the student learned to loop.

3. **Test repetition penalty**: Run eval with a repetition penalty (e.g., `frequency_penalty=0.3`) to see if suppressing repetition recovers meaningful accuracy. This is a band-aid but would quantify how much accuracy is "locked up" behind the loops.

### Short-term

4. **Run Sonnet distillation on Llama-8B**: Sonnet traces are 3-5x shorter than Kimi/Qwen and more structured. Shorter CoT patterns may be less prone to looping since the model doesn't learn the "extended exploration" style that degenerates.

5. **Harder training data**: The current AoPS 5+ filter is weak (only removed 10%). Li et al. used 17K hard-filtered problems. Harder training data may teach the model to reason more carefully rather than loop on problems beyond its capability.

6. **Add identity test**: After creating a checkpoint client, generate 5 canonical prompts and verify outputs differ from base model. Log this in the run.log.

### Medium-term

7. **Fix LR for Qwen models**: Use model-specific learning rates from tinker's `get_lr()` function instead of a fixed 2e-4.

8. **Investigate loop mechanism**: Profile *when* loops start in the output. If loops consistently begin after N tokens, this suggests a context window or attention pattern issue. If loops start at problem-dependent points, it's more about the model hitting the limits of its capability.

9. **Consider training modifications**: DPO/rejection sampling (train on correct completions only, penalize loops), length penalty, or EOS prediction training could address the loop problem more fundamentally.

---

## Comparison to Li et al. (2025) Setup

| Parameter | Li et al. | Our setup | Assessment |
|-----------|-----------|-----------|------------|
| Student | Qwen2.5-32B-Instruct | Llama-8B-Instruct | 4x smaller. May lack capacity for extended reasoning. |
| Teacher | DeepSeek-R1, QwQ-32B | Sonnet, Kimi, Qwen-235B | Different teachers, different trace styles |
| Training data | 17K hard-filtered (Olympiad) | 10-29K (AoPS 5+, mostly easy) | **Ours likely too easy** |
| Loss function | NTP on completions | NTP on completions | Same |
| LR | 1e-5 (full), 1e-4 (LoRA) | 2e-4 (LoRA) | Ours is 2x higher, but tinker recommends this |
| Batch size | 96 | 50 | Similar range |
| Warmup | 10% linear | None | **Ours missing warmup** |
| Max length | ~16-32K (inferred) | 4096 / 16384 | Our 4k may be too short for training |
| LoRA rank | 64, 256 | 32, 64, 128 | Similar range |
| Epochs | ~1-3 (inferred) | 1 | Possibly undertrained |
| Eval temperature | 0.6 | 0.0 | **Major difference — may worsen loops** |
| Eval sampling | N=64, majority vote | N=1, greedy | **Major difference** |

The most important gap may be **student model capacity**. Li et al. use a 32B student; we use 8B. An 8B model may not have the capacity to internalize extended reasoning patterns — it learns to produce long outputs (the form) without the underlying capability to make them productive (the substance). This would explain why the loops happen: the model enters "reasoning mode" but doesn't have the capacity to actually reason through hard problems, so it cycles.

The eval protocol differences (temperature, sampling) may also matter — sampling could break loops on problems where the model *can* solve them but the greedy path happens to loop. This is worth testing.

---

## Appendix: Detailed Token Mask Inspection

One Kimi math training example on Llama-3.1-8B-Instruct (2081 tokens total, 1936 trained):

```
MASKED (prompt, 145 tokens):
  [  0] <|begin_of_text|>
  [  1] <|start_header_id|>
  [  2] system
  [  3] <|end_header_id|>
  [  4] \n\n
  [  5] You are a helpful assistant...putting your final answer inside \boxed{}
  [ 32] <|eot_id|>
  [ 33] <|start_header_id|>
  [ 34] user
  [ 35] <|end_header_id|>
  [ 36] \n\n
  [ 37-139] [problem text]
  [140] <|eot_id|>
  [141] <|start_header_id|>
  [142] assistant
  [143] <|end_header_id|>
  [144] \n\n

TRAINED (response, 1936 tokens):
  [145] <th          ← training starts here
  [146] ink
  [147] >\n
  [148] This         ← start of reasoning
  ...
  [2077] ins
  [2078] .
  [2079] }            ← closing brace of \boxed{}
  [2080] <|eot_id|>   ← EOS token is trained
```

The mask boundary is exactly correct: everything up to and including the assistant header is masked; everything from the first content token onward is trained.
