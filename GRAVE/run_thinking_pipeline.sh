#!/usr/bin/env bash
# Full pipeline for thinking-model trace generation, training, and evaluation.
#
# Steps:
#   1. Generate thinking traces (math + code) via OpenRouter API
#   2. Grade traces → correct_only.jsonl
#   3. Train 4 models: {math,code} × {base,IT}
#   4. Evaluate checkpoints on math_500, aime, mbppplus, humanevalplus (100 samples)
#
# Usage:
#   ./run_thinking_pipeline.sh          # Run entire pipeline
#   ./run_thinking_pipeline.sh generate # Only generate traces
#   ./run_thinking_pipeline.sh grade    # Only grade traces
#   ./run_thinking_pipeline.sh train    # Only train (traces must exist)
#   ./run_thinking_pipeline.sh eval     # Only evaluate (training must be done)

set -euo pipefail
cd "$(dirname "$0")"

TRACE_DIR="generate_reasoning_traces"
PYTHON="python3.13"
STEP="${1:-all}"

log() { echo "$(date '+%H:%M:%S') [pipeline] $*"; }

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate thinking traces
# ═══════════════════════════════════════════════════════════════════════════════
generate_traces() {
    log "Starting trace generation..."

    # Math traces
    if [ -f "$TRACE_DIR/data_thinking/correct_only.jsonl" ]; then
        log "Math traces already graded, skipping generation"
    else
        log "Generating math thinking traces (30K questions)..."
        PYTHONUNBUFFERED=1 $PYTHON "$TRACE_DIR/generate.py" --thinking &
        MATH_PID=$!
    fi

    # Code traces
    if [ -f "$TRACE_DIR/data_code_thinking/correct_only.jsonl" ]; then
        log "Code traces already graded, skipping generation"
    else
        log "Generating code thinking traces (40K questions)..."
        PYTHONUNBUFFERED=1 $PYTHON "$TRACE_DIR/generate_code.py" --thinking &
        CODE_PID=$!
    fi

    # Wait for both
    if [ -n "${MATH_PID:-}" ]; then
        wait $MATH_PID
        log "Math trace generation complete"
    fi
    if [ -n "${CODE_PID:-}" ]; then
        wait $CODE_PID
        log "Code trace generation complete"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Grade traces
# ═══════════════════════════════════════════════════════════════════════════════
grade_traces() {
    log "Grading traces..."

    # Grade math traces
    if [ -f "$TRACE_DIR/data_thinking/correct_only.jsonl" ]; then
        log "Math traces already graded"
    else
        log "Grading math thinking traces..."
        PYTHONUNBUFFERED=1 $PYTHON "$TRACE_DIR/grade.py" --thinking
        log "Math grading complete"
    fi

    # Grade code traces
    if [ -f "$TRACE_DIR/data_code_thinking/correct_only.jsonl" ]; then
        log "Code traces already graded"
    else
        log "Grading code thinking traces..."
        PYTHONUNBUFFERED=1 $PYTHON "$TRACE_DIR/grade_code.py" --thinking
        log "Code grading complete"
    fi

    # Report stats
    log "Trace statistics:"
    if [ -f "$TRACE_DIR/data_thinking/correct_only.jsonl" ]; then
        MATH_N=$(wc -l < "$TRACE_DIR/data_thinking/correct_only.jsonl")
        log "  Math correct traces: $MATH_N"
    fi
    if [ -f "$TRACE_DIR/data_code_thinking/correct_only.jsonl" ]; then
        CODE_N=$(wc -l < "$TRACE_DIR/data_code_thinking/correct_only.jsonl")
        log "  Code correct traces: $CODE_N"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Train 4 models
# ═══════════════════════════════════════════════════════════════════════════════
train_models() {
    log "Starting training (4 runs)..."

    # Train math base
    log "Training: math + base model"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task math --model-type base \
        2>&1 | tee training_runs/thinking_math_base_train.log &
    MATH_BASE_PID=$!

    # Train math IT
    log "Training: math + IT model"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task math --model-type it \
        2>&1 | tee training_runs/thinking_math_it_train.log &
    MATH_IT_PID=$!

    # Train code base
    log "Training: code + base model"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task code --model-type base \
        2>&1 | tee training_runs/thinking_code_base_train.log &
    CODE_BASE_PID=$!

    # Train code IT
    log "Training: code + IT model"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task code --model-type it \
        2>&1 | tee training_runs/thinking_code_it_train.log &
    CODE_IT_PID=$!

    # Wait for all
    wait $MATH_BASE_PID
    log "Math base training complete"
    wait $MATH_IT_PID
    log "Math IT training complete"
    wait $CODE_BASE_PID
    log "Code base training complete"
    wait $CODE_IT_PID
    log "Code IT training complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Evaluate checkpoints
# ═══════════════════════════════════════════════════════════════════════════════
eval_checkpoints() {
    log "Evaluating checkpoints..."
    PYTHONUNBUFFERED=1 $PYTHON eval_thinking_checkpoints.py
    log "Evaluation complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
case "$STEP" in
    all)
        generate_traces
        grade_traces
        train_models
        eval_checkpoints
        ;;
    generate)
        generate_traces
        ;;
    grade)
        grade_traces
        ;;
    train)
        train_models
        ;;
    eval)
        eval_checkpoints
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: $0 [all|generate|grade|train|eval]"
        exit 1
        ;;
esac

log "Pipeline step '$STEP' complete!"
