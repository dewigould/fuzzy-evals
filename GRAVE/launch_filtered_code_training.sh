#!/bin/bash
# Launch filtered code training (8K + 4K) and code eval watcher.
# Waits for filtered_data_kodcode/clean.jsonl to exist before starting.

set -e
cd "$(dirname "$0")"

echo "Waiting for filtered_data_kodcode/clean.jsonl..."
while [ ! -f filtered_data_kodcode/clean.jsonl ]; do
    sleep 10
done

echo "File exists, waiting for filter to complete..."
while pgrep -f "filter_repetitive_kodcode.py" > /dev/null 2>&1; do
    sleep 5
done

echo "Filter complete. Launching training runs..."
LINES=$(wc -l < filtered_data_kodcode/clean.jsonl)
echo "  Clean examples: $LINES"

# Launch 8K training
echo "Launching 8K code training..."
PYTHONUNBUFFERED=1 python3.13 train_kodcode_base.py A_r32_high_lr --filtered \
    > training_runs/base_kodcode_filtered_8k_train.log 2>&1 &
PID_8K=$!
echo "  8K PID: $PID_8K"

# Launch 4K training
echo "Launching 4K code training..."
PYTHONUNBUFFERED=1 python3.13 train_kodcode_base_4k.py A_r32_high_lr --filtered \
    > training_runs/base_kodcode_filtered_4k_train.log 2>&1 &
PID_4K=$!
echo "  4K PID: $PID_4K"

# Wait a bit for first checkpoints, then launch eval watcher
sleep 30
echo "Launching code eval watcher..."
PYTHONUNBUFFERED=1 python3.13 eval_code_filtered.py &
PID_EVAL=$!
echo "  Eval PID: $PID_EVAL"

echo ""
echo "All processes launched:"
echo "  8K training:  PID $PID_8K  (log: training_runs/base_kodcode_filtered_8k_train.log)"
echo "  4K training:  PID $PID_4K  (log: training_runs/base_kodcode_filtered_4k_train.log)"
echo "  Eval watcher: PID $PID_EVAL"
echo ""
echo "Waiting for training to finish..."

wait $PID_8K
echo "8K training done (exit: $?)"
wait $PID_4K
echo "4K training done (exit: $?)"
wait $PID_EVAL
echo "Eval watcher done (exit: $?)"

echo "All done!"
