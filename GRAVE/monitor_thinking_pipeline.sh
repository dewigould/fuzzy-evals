#!/usr/bin/env bash
# Monitor thinking trace generation + pipeline progress.
# Usage: ./monitor_thinking_pipeline.sh [--watch]

set -euo pipefail
cd "$(dirname "$0")"

TRACE_DIR="generate_reasoning_traces"

echo "=== Thinking Trace Pipeline Status ($(date)) ==="
echo

# Generation progress
echo "── Trace Generation ──"
MATH_Q=$(wc -l < "$TRACE_DIR/data_thinking/questions.jsonl" 2>/dev/null || echo 0)
MATH_G=$(wc -l < "$TRACE_DIR/data_thinking/generations.jsonl" 2>/dev/null || echo 0)
CODE_Q=$(wc -l < "$TRACE_DIR/data_code_thinking/questions.jsonl" 2>/dev/null || echo 0)
CODE_G=$(wc -l < "$TRACE_DIR/data_code_thinking/generations.jsonl" 2>/dev/null || echo 0)

MATH_PCT=$(echo "scale=1; $MATH_G * 100 / $MATH_Q" | bc 2>/dev/null || echo "0")
CODE_PCT=$(echo "scale=1; $CODE_G * 100 / $CODE_Q" | bc 2>/dev/null || echo "0")

echo "  Math: $MATH_G / $MATH_Q ($MATH_PCT%)"
echo "  Code: $CODE_G / $CODE_Q ($CODE_PCT%)"

# Check for running processes
echo
echo "── Running Processes ──"
MATH_RUNNING=$(pgrep -f "generate.py.*--thinking" 2>/dev/null | wc -l || echo 0)
CODE_RUNNING=$(pgrep -f "generate_code.py.*--thinking" 2>/dev/null | wc -l || echo 0)
echo "  Math gen: $([ "$MATH_RUNNING" -gt 0 ] && echo 'RUNNING' || echo 'STOPPED')"
echo "  Code gen: $([ "$CODE_RUNNING" -gt 0 ] && echo 'RUNNING' || echo 'STOPPED')"

# Grading status
echo
echo "── Grading ──"
for task in math code; do
    dir="$TRACE_DIR/data_${task/math/}_thinking"
    [ "$task" = "math" ] && dir="$TRACE_DIR/data_thinking"
    [ "$task" = "code" ] && dir="$TRACE_DIR/data_code_thinking"

    if [ -f "$dir/correct_only.jsonl" ]; then
        N=$(wc -l < "$dir/correct_only.jsonl")
        echo "  $task: DONE ($N correct traces)"
    elif [ -f "$dir/graded.jsonl" ]; then
        N=$(wc -l < "$dir/graded.jsonl")
        echo "  $task: graded.jsonl exists ($N entries, needs correct_only extraction)"
    else
        echo "  $task: NOT YET GRADED"
    fi
done

# Training status
echo
echo "── Training Runs ──"
for run in thinking_math_base thinking_math_it thinking_code_base thinking_code_it; do
    run_dir="training_runs/$run/A_r32_high_lr"
    if [ -f "$run_dir/checkpoints.jsonl" ]; then
        N_CKPT=$(wc -l < "$run_dir/checkpoints.jsonl")
        LAST_STEP=$(tail -1 "$run_dir/checkpoints.jsonl" | python3.13 -c "import json,sys; print(json.load(sys.stdin).get('batch','?'))" 2>/dev/null || echo "?")
        echo "  $run: $N_CKPT checkpoints (last step: $LAST_STEP)"
    elif [ -f "$run_dir/metrics.jsonl" ]; then
        echo "  $run: training in progress"
    else
        echo "  $run: NOT STARTED"
    fi
done

# Eval status
echo
echo "── Evaluations ──"
if [ -f "results/thinking_eval_summary.json" ]; then
    echo "  Summary exists"
    python3.13 -c "
import json
with open('results/thinking_eval_summary.json') as f:
    d = json.load(f)
for model, results in d.items():
    for ds, r in results.items():
        acc = r.get('accuracy', r.get('mean_score', '?'))
        if isinstance(acc, float):
            acc = f'{acc*100:.1f}%'
        print(f'  {model}: {ds} = {acc}')
" 2>/dev/null
else
    echo "  NOT YET RUN"
fi

# Latest log entries
echo
echo "── Recent Log Entries ──"
for logfile in thinking_math_gen.log thinking_code_gen.log; do
    if [ -f "$logfile" ]; then
        echo "  $logfile: $(tail -1 "$logfile" 2>/dev/null)"
    fi
done

if [ "${1:-}" = "--watch" ]; then
    echo
    echo "(Refreshing every 60s, Ctrl+C to stop)"
    sleep 60
    exec "$0" --watch
fi
