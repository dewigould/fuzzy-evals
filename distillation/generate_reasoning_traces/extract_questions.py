"""Extract math questions from filtered_data/clean.jsonl.

Usage:
    python extract_questions.py [--num N] [--seed SEED] [--effortful]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SOURCE_DATA, NUM_QUESTIONS,
    DATA_DIR, QUESTIONS_PATH,
    EFFORTFUL_DATA_DIR, EFFORTFUL_QUESTIONS_PATH,
    THINKING_DATA_DIR, THINKING_QUESTIONS_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Extract questions from clean.jsonl")
    parser.add_argument("--num", type=int, default=NUM_QUESTIONS, help="Number of questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    parser.add_argument("--effortful", action="store_true",
                        help="Write to effortful data directory")
    parser.add_argument("--thinking", action="store_true",
                        help="Write to thinking model data directory")
    args = parser.parse_args()

    if args.thinking:
        data_dir = THINKING_DATA_DIR
        questions_path = THINKING_QUESTIONS_PATH
    elif args.effortful:
        data_dir = EFFORTFUL_DATA_DIR
        questions_path = EFFORTFUL_QUESTIONS_PATH
    else:
        data_dir = DATA_DIR
        questions_path = QUESTIONS_PATH
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load all records (only the fields we need)
    print(f"Loading questions from {SOURCE_DATA} ...")
    records = []
    with open(SOURCE_DATA) as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "uuid": row["uuid"],
                "problem": row["problem"],
                "answer": row["answer"],
                "source": row.get("source", ""),
                "problem_type": row.get("problem_type", ""),
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
