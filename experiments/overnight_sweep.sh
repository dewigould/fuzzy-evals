#!/bin/bash
# Overnight sweep: 8 base evals + 96 training experiments
#
# Base models: llama-8b-it, llama-70b-it, qwen-8b-base, qwen-30b-it (4K + 16K eval)
# Training: 2 models × 2 tasks × 2 traces × 2 lengths × 3 ranks
# Save every 2K examples, up to 20K (or data limit at 4K)
# Full eval capped at 500 problems per dataset
#
# OOM safety: code evals limited to MAX_CONCURRENT_CODE_EVALS=2
# regardless of --max-parallel (controlled by semaphore in evaluate.py)

cd /workspace/fuzzy-evals

echo "=== Overnight Sweep ==="
python3.13 experiments/overnight_sweep.py
echo ""
echo "Starting batch run..."
echo ""

PYTHONUNBUFFERED=1 python3.13 run.py batch experiments/overnight_sweep.py --max-parallel 10
