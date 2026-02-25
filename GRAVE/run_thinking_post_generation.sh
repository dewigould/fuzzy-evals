#!/usr/bin/env bash
# Run after trace generation is complete: grade → train → eval.
#
# Usage:
#   ./run_thinking_post_generation.sh           # Full: grade + train + eval
#   ./run_thinking_post_generation.sh grade      # Grade only
#   ./run_thinking_post_generation.sh train      # Train only (traces must be graded)
#   ./run_thinking_post_generation.sh eval       # Eval only (training must be done)

set -euo pipefail
cd "$(dirname "$0")"

TRACE_DIR="generate_reasoning_traces"
PYTHON="python3.13"
STEP="${1:-all}"

log() { echo "$(date '+%H:%M:%S') [post-gen] $*"; }

# ═══════════════════════════════════════════════════════════════════════════════
# Grade traces
# ═══════════════════════════════════════════════════════════════════════════════
grade_traces() {
    log "=== GRADING TRACES ==="

    # Verify generation completed
    MATH_G=$(wc -l < "$TRACE_DIR/data_thinking/generations.jsonl" 2>/dev/null || echo 0)
    CODE_G=$(wc -l < "$TRACE_DIR/data_code_thinking/generations.jsonl" 2>/dev/null || echo 0)
    log "Math generations: $MATH_G | Code generations: $CODE_G"

    if [ "$MATH_G" -lt 100 ] || [ "$CODE_G" -lt 100 ]; then
        log "WARNING: Very few generations. Generation may not be complete."
        log "Continue anyway? (Ctrl+C to abort, Enter to continue)"
        read -r
    fi

    # Grade math
    if [ -f "$TRACE_DIR/data_thinking/correct_only.jsonl" ]; then
        log "Math already graded, skipping"
    else
        log "Grading math traces..."
        PYTHONUNBUFFERED=1 $PYTHON "$TRACE_DIR/grade.py" --thinking 2>&1 | tee grading_math_thinking.log
    fi

    # Grade code
    if [ -f "$TRACE_DIR/data_code_thinking/correct_only.jsonl" ]; then
        log "Code already graded, skipping"
    else
        log "Grading code traces..."
        PYTHONUNBUFFERED=1 $PYTHON "$TRACE_DIR/grade_code.py" --thinking 2>&1 | tee grading_code_thinking.log
    fi

    # Report
    log "Grading complete!"
    MATH_C=$(wc -l < "$TRACE_DIR/data_thinking/correct_only.jsonl" 2>/dev/null || echo 0)
    CODE_C=$(wc -l < "$TRACE_DIR/data_code_thinking/correct_only.jsonl" 2>/dev/null || echo 0)
    log "Correct traces: math=$MATH_C code=$CODE_C"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Train 4 models (all in parallel)
# ═══════════════════════════════════════════════════════════════════════════════
train_models() {
    log "=== TRAINING 4 MODELS ==="

    # Verify correct_only.jsonl exists
    if [ ! -f "$TRACE_DIR/data_thinking/correct_only.jsonl" ]; then
        log "ERROR: Math correct_only.jsonl not found. Run grading first."
        exit 1
    fi
    if [ ! -f "$TRACE_DIR/data_code_thinking/correct_only.jsonl" ]; then
        log "ERROR: Code correct_only.jsonl not found. Run grading first."
        exit 1
    fi

    mkdir -p training_runs

    PIDS=""

    log "Starting: math + base"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task math --model-type base \
        2>&1 | tee training_runs/thinking_math_base_train.log &
    PIDS="$PIDS $!"

    log "Starting: math + IT"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task math --model-type it \
        2>&1 | tee training_runs/thinking_math_it_train.log &
    PIDS="$PIDS $!"

    log "Starting: code + base"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task code --model-type base \
        2>&1 | tee training_runs/thinking_code_base_train.log &
    PIDS="$PIDS $!"

    log "Starting: code + IT"
    PYTHONUNBUFFERED=1 $PYTHON train_thinking.py --task code --model-type it \
        2>&1 | tee training_runs/thinking_code_it_train.log &
    PIDS="$PIDS $!"

    log "Waiting for all 4 training runs (PIDs: $PIDS)..."
    for pid in $PIDS; do
        wait $pid
        log "PID $pid complete"
    done
    log "All training complete!"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate checkpoints
# ═══════════════════════════════════════════════════════════════════════════════
eval_checkpoints() {
    log "=== EVALUATING CHECKPOINTS ==="
    PYTHONUNBUFFERED=1 $PYTHON eval_thinking_checkpoints.py --max-problems 100 \
        2>&1 | tee eval_thinking_checkpoints.log
    log "Evaluation complete!"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
case "$STEP" in
    all)
        grade_traces
        train_models
        eval_checkpoints
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
        echo "Usage: $0 [all|grade|train|eval]"
        exit 1
        ;;
esac

log "Done!"
