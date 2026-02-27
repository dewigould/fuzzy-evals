"""Overnight sweep: 4 base evals + 48 training experiments.

Base models: llama-8b-it, llama-70b-it (evaluated at both 4K and 16K max_tokens)
Training: 2 models x 2 tasks x 2 traces x 2 lengths x 3 ranks = 48 runs
Save/eval every 2K examples (40 steps). Up to 20K examples (less at 4K if data limited).
Full eval capped at 500 problems per dataset for speed.

Usage:
    python run.py batch experiments/overnight_sweep.py --max-parallel 8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    Experiment,
    BaseModelEval,
    TrainingConfig,
    CheckpointEvalConfig,
    FullEvalConfig,
)

# ── Models ──────────────────────────────────────────────────────────────────

LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"

MODELS = {
    "llama8b": LLAMA_8B,
    "llama70b": LLAMA_70B,
}

# ── Sweep axes ──────────────────────────────────────────────────────────────

TASKS = ["math", "code"]
TRACES = ["kimi", "qwen"]
MAX_LENGTHS = [4096, 16384]
RANKS = [32, 64, 128]

# ── Data availability ───────────────────────────────────────────────────────
# At 4K tokens, some trace/task combos have <20K examples after filtering.
# Values below are conservative (rounded down, leaving room for 200 test set).
# Both llama8b and llama70b use identical filtered data.

_MAX_SAMPLES_4K = {
    ("math", "kimi"): 14_000,   # 14,240 available
    ("math", "qwen"): 9_500,    # 9,858 available
    ("code", "kimi"): 20_000,   # 21,816 available
    ("code", "qwen"): 14_000,   # 14,146 available
}
_MAX_SAMPLES_16K = 20_000       # All combos have >27K at 16384 tokens

BATCH_SIZE = 50
SAVE_EVERY_EXAMPLES = 2_000     # 2K examples = 40 steps


def _steps(examples):
    return examples // BATCH_SIZE


def _max_samples(task, traces, max_length):
    if max_length == 4096:
        return _MAX_SAMPLES_4K[(task, traces)]
    return _MAX_SAMPLES_16K


# ── Build experiment list ───────────────────────────────────────────────────

EXPERIMENTS = []

# ── Shared eval config ─────────────────────────────────────────────────────
# Cap full eval at 500 problems per dataset for speed.
FULL_EVAL_MAX_PROBLEMS = 500

# 1. Base model evaluations (run first, no training needed)
for name, model_id in MODELS.items():
    for max_tok, tok_label in [(4096, "4k"), (16_384, "16k")]:
        EXPERIMENTS.append(BaseModelEval(
            name=f"{name}_base_{tok_label}",
            base_model=model_id,
            eval=FullEvalConfig(max_tokens=max_tok, max_problems=FULL_EVAL_MAX_PROBLEMS),
        ))

# 2. Training experiments
for model_name, model_id in MODELS.items():
    for task in TASKS:
        for traces in TRACES:
            for max_length in MAX_LENGTHS:
                max_samples = _max_samples(task, traces, max_length)
                length_short = f"{max_length // 1024}k"

                for rank in RANKS:
                    exp_name = f"{model_name}_{task}_{traces}_{length_short}_r{rank}"

                    EXPERIMENTS.append(Experiment(
                        name=exp_name,
                        training=TrainingConfig(
                            base_model=model_id,
                            task=task,
                            trace_source=traces,
                            max_length=max_length,
                            lora_rank=rank,
                            max_samples=max_samples,
                            save_every=_steps(SAVE_EVERY_EXAMPLES),
                            eval_every=_steps(SAVE_EVERY_EXAMPLES),
                        ),
                        checkpoint_eval=CheckpointEvalConfig(
                            max_problems=100,
                        ),
                        full_eval=FullEvalConfig(
                            max_problems=FULL_EVAL_MAX_PROBLEMS,
                            # max_tokens: None = auto-follow from training.max_length
                        ),
                    ))


# ── Summary ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_base = sum(1 for e in EXPERIMENTS if isinstance(e, BaseModelEval))
    n_train = sum(1 for e in EXPERIMENTS if isinstance(e, Experiment))
    print(f"Total: {len(EXPERIMENTS)} items ({n_base} base evals, {n_train} training experiments)\n")

    for i, item in enumerate(EXPERIMENTS):
        if isinstance(item, BaseModelEval):
            print(f"  {i+1:3d}. [BASE] {item.name}  ({item.base_model})")
        else:
            tc = item.training
            print(f"  {i+1:3d}. [TRAIN] {item.name}  "
                  f"(samples={tc.max_samples}, steps={tc.steps}, "
                  f"rank={tc.lora_rank}, len={tc.max_length})")
