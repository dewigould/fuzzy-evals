#!/bin/bash
# Smoke test: llama8b_it math distillation with kimi traces, 1000 samples
# 20 training steps (1000 samples / batch_size 50), checkpoints at step 10 and 20
# Eval limited to 20 problems per dataset for speed

cd /workspace/fuzzy-evals

PYTHONUNBUFFERED=1 python3.13 -c "
from config import Experiment, TrainingConfig, CheckpointEvalConfig, FullEvalConfig
from runner import run_experiment

exp = Experiment(
    name='smoke_llama8b_math_kimi',
    training=TrainingConfig(
        base_model='meta-llama/Llama-3.1-8B-Instruct',
        task='math',
        trace_source='kimi',
        max_length=4096,
        max_samples=1000,
        save_every=10,
        eval_every=10,
    ),
    checkpoint_eval=CheckpointEvalConfig(max_problems=20, max_tokens=4096),
    full_eval=FullEvalConfig(max_problems=20, max_tokens=4096),
)
run_experiment(exp)
"
