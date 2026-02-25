#!/bin/bash
# Launch math→code format fine-tune: 100ex and 500ex variants.
# Trains both, then evaluates all checkpoints on math_500, aime, kodcode_500, livecodebench_v5.

set -e
cd "$(dirname "$0")"

echo "=== Math→Code Format Fine-Tune ==="
echo "Starting 500-example run..."
PYTHONUNBUFFERED=1 python3.13 train_math_code_finetune.py --n-examples 500 \
    > training_runs/math_code_finetune_500ex_train.log 2>&1 &
PID_500=$!
echo "  500ex PID: $PID_500"

echo "Starting 100-example run..."
PYTHONUNBUFFERED=1 python3.13 train_math_code_finetune.py --n-examples 100 \
    > training_runs/math_code_finetune_100ex_train.log 2>&1 &
PID_100=$!
echo "  100ex PID: $PID_100"

echo ""
echo "Waiting for training to complete..."
echo "  Logs: training_runs/math_code_finetune_{100,500}ex_train.log"

wait $PID_500
echo "500ex training done (exit: $?)"
wait $PID_100
echo "100ex training done (exit: $?)"

echo ""
echo "=== Training complete. Starting evaluations ==="

# Base model name for renderer
BASE_MODEL="Qwen/Qwen3-30B-A3B-Base"
DATASETS="math_500,aime,kodcode_500,livecodebench_v5"
MAX_PROBLEMS=100
MAX_TOKENS=16384

# Evaluate all checkpoints from both runs
for task_dir in training_runs/math_code_finetune_100ex training_runs/math_code_finetune_500ex; do
    task_name=$(basename "$task_dir")
    ckpt_file="$task_dir/checkpoints.jsonl"

    if [ ! -f "$ckpt_file" ]; then
        echo "  No checkpoints found in $task_dir"
        continue
    fi

    echo "Evaluating checkpoints from $task_name..."

    while IFS= read -r line; do
        batch=$(echo "$line" | python3.13 -c "import sys,json; print(json.load(sys.stdin)['batch'])")
        sampler_path=$(echo "$line" | python3.13 -c "import sys,json; print(json.load(sys.stdin)['sampler_path'])")
        results_name="${task_name}_step$(printf '%03d' "$batch")"

        # Skip if already evaluated
        results_dir="results/$results_name"
        if [ -f "$results_dir/results_livecodebench_v5.parquet" ]; then
            echo "  $results_name: already complete, skipping"
            continue
        fi

        echo "  Evaluating $results_name..."
        PYTHONUNBUFFERED=1 python3.13 evaluate.py \
            --sampler-path "$sampler_path" \
            --base-model-name "$BASE_MODEL" \
            --results-name "$results_name" \
            --datasets "$DATASETS" \
            --max-problems "$MAX_PROBLEMS" \
            --max-tokens "$MAX_TOKENS" \
            > "$task_dir/eval_step$(printf '%03d' "$batch").log" 2>&1 &

        # Run max 2 evals concurrently
        running=$(jobs -r | wc -l)
        if [ "$running" -ge 2 ]; then
            wait -n
        fi
    done < "$ckpt_file"
done

echo "Waiting for all evaluations to finish..."
wait

echo ""
echo "=== All done. Printing results ==="
python3.13 -c "
import json, pandas as pd
from pathlib import Path

datasets_list = ['math_500', 'aime', 'kodcode_500', 'livecodebench_v5']

# Print header
print(f\"{'Checkpoint':<40} {'math_500':>10} {'aime':>10} {'kodcode':>10} {'lcb_v5':>10}\")
print('-' * 85)

# Reference: best math checkpoint
print(f\"{'best_math_4k_step200 (before)':<40} {'78.8%':>10} {'30.0%':>10} {'6.0%':>10} {'0.0%':>10}\")
print()

for task in ['math_code_finetune_100ex', 'math_code_finetune_500ex']:
    ckpt_file = Path(f'training_runs/{task}/checkpoints.jsonl')
    if not ckpt_file.exists():
        continue
    for line in open(ckpt_file):
        ckpt = json.loads(line.strip())
        step = ckpt['batch']
        results_name = f'{task}_step{step:03d}'
        results_dir = Path(f'results/{results_name}')

        scores = {}
        for ds in datasets_list:
            col = 'passed' if ds in ('kodcode_500', 'livecodebench_v5') else 'correct'
            pq = results_dir / f'results_{ds}.parquet'
            if pq.exists():
                df = pd.read_parquet(pq)
                if col in df.columns:
                    scores[ds] = f'{df[col].mean()*100:.1f}%'
                else:
                    scores[ds] = '?'
            else:
                scores[ds] = 'N/A'

        print(f'{results_name:<40} {scores.get(\"math_500\",\"?\"):>10} {scores.get(\"aime\",\"?\"):>10} {scores.get(\"kodcode_500\",\"?\"):>10} {scores.get(\"livecodebench_v5\",\"?\"):>10}')
"

echo ""
echo "Done!"
