"""Extract coding questions from filtered_data_kodcode/clean.jsonl.

Usage:
    python extract_questions_code.py [--num N] [--seed SEED] [--effortful]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CODE_SOURCE_DATA, CODE_NUM_QUESTIONS,
    CODE_DATA_DIR, CODE_QUESTIONS_PATH,
    EFFORTFUL_CODE_DATA_DIR, EFFORTFUL_CODE_QUESTIONS_PATH,
    THINKING_CODE_DATA_DIR, THINKING_CODE_QUESTIONS_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Extract coding questions from KodCode clean.jsonl")
    parser.add_argument("--num", type=int, default=CODE_NUM_QUESTIONS, help="Number of questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    parser.add_argument("--effortful", action="store_true",
                        help="Write to effortful data directory")
    parser.add_argument("--thinking", action="store_true",
                        help="Write to thinking model data directory")
    args = parser.parse_args()

    if args.thinking:
        data_dir = THINKING_CODE_DATA_DIR
        questions_path = THINKING_CODE_QUESTIONS_PATH
    elif args.effortful:
        data_dir = EFFORTFUL_CODE_DATA_DIR
        questions_path = EFFORTFUL_CODE_QUESTIONS_PATH
    else:
        data_dir = CODE_DATA_DIR
        questions_path = CODE_QUESTIONS_PATH
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load all records (only the fields we need)
    print(f"Loading questions from {CODE_SOURCE_DATA} ...")
    records = []
    with open(CODE_SOURCE_DATA) as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "test": row["test"],
                "solution": row.get("solution", ""),
            })

    print(f"  Loaded {len(records)} records")

    # Shuffle and take first N
    random.seed(args.seed)
    random.shuffle(records)
    selected = records[:args.num]

    # Write
    with open(questions_path, "w") as f:
        for rec in selected:
            f.write(json.dumps(rec) + "\n")

    print(f"  Wrote {len(selected)} questions to {questions_path}")


if __name__ == "__main__":
    main()
