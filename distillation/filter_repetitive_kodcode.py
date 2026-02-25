"""Filter degenerate repetitive examples from KodCode-V1-SFT-R1.

Same approach as filter_repetitive.py but adapted for KodCode's format:
  conversations[1]["value"] contains the assistant response with <think>...</think>

Two detectors:
  1. 50-gram repetition: any 50-word sequence appearing 2+ times
  2. Truncation: <think> present but no </think> (model never finished)

Usage:
  python filter_repetitive_kodcode.py [OPTIONS]
  python filter_repetitive_kodcode.py --dry-run --limit 5000
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset


def extract_think_text(content: str) -> str | None:
    """Extract text between <think> and </think> tags."""
    m = re.search(r"<think>\s*", content)
    if not m:
        return None
    start = m.end()
    m2 = re.search(r"</think>", content[start:])
    if m2:
        return content[start : start + m2.start()]
    return content[start:]


def tokenize_words(text: str) -> list[str]:
    return text.lower().split()


def has_repeated_50gram(words: list[str]) -> bool:
    n = 50
    if len(words) < n:
        return False
    seen = set()
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def is_degenerate(content: str, words: list[str], filter_truncated: bool) -> tuple[bool, list[str]]:
    reasons = []
    if has_repeated_50gram(words):
        reasons.append("50gram")
    if filter_truncated:
        if "<think>" in content and "</think>" not in content:
            reasons.append("truncated")
    return len(reasons) > 0, reasons


def main():
    parser = argparse.ArgumentParser(description="Filter repetitive examples from KodCode-V1-SFT-R1")
    parser.add_argument("--dataset", default="KodCode/KodCode-V1-SFT-R1", help="HF dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output-dir", default="filtered_data_kodcode", help="Output directory")
    parser.add_argument("--no-filter-truncated", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    filter_truncated = not args.no_filter_truncated
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Detectors: 50-gram repetition + {'truncation' if filter_truncated else 'no truncation'}")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    print()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    clean_path = output_dir / "clean.jsonl"
    flagged_path = output_dir / "flagged.jsonl"

    clean_file = None
    flagged_file = None
    if not args.dry_run:
        clean_file = open(clean_path, "w")
        flagged_file = open(flagged_path, "w")

    total = 0
    flagged_count = 0
    no_think = 0
    t0 = time.time()
    reasons_count = {"50gram": 0, "truncated": 0}

    try:
        for example in ds:
            total += 1
            if args.limit and total > args.limit:
                total -= 1
                break

            # KodCode format: conversations[1]["value"] is the assistant response
            conversations = example.get("conversations", [])
            if len(conversations) < 2:
                if clean_file:
                    clean_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                continue

            content = conversations[1].get("value", "")
            think_text = extract_think_text(content)

            if think_text is None or len(think_text) < 50:
                no_think += 1
                if clean_file:
                    clean_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                continue

            words = tokenize_words(think_text)
            degenerate, reasons = is_degenerate(content, words, filter_truncated)

            if degenerate:
                flagged_count += 1
                for r in reasons:
                    reasons_count[r] += 1
                if flagged_file:
                    record = {
                        "question_id": example.get("question_id", ""),
                        "question": example.get("question", "")[:200],
                        "n_words": len(words),
                        "reasons": reasons,
                    }
                    flagged_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                if clean_file:
                    clean_file.write(json.dumps(example, ensure_ascii=False) + "\n")

            if total % 5000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed
                pct = flagged_count / total * 100
                print(
                    f"  [{total:>7,}] flagged: {flagged_count:,} ({pct:.1f}%) "
                    f"| {rate:.0f} ex/s | elapsed: {elapsed:.0f}s"
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\nInterrupted at {total} examples")
    finally:
        if clean_file:
            clean_file.close()
        if flagged_file:
            flagged_file.close()

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"FILTERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total examples processed: {total:,}")
    print(f"  Flagged (degenerate):     {flagged_count:,} ({flagged_count/max(total,1)*100:.1f}%)")
    print(f"  Clean (kept):             {total - flagged_count - no_think:,}")
    print(f"  No <think> tag:           {no_think:,}")
    print(f"  Time: {elapsed:.0f}s ({total/max(elapsed,1):.0f} examples/s)")
    print()
    print(f"Flagging reasons:")
    print(f"  50-gram repetition:        {reasons_count['50gram']:,}")
    print(f"  Truncated (no </think>):   {reasons_count['truncated']:,}")

    if not args.dry_run:
        print(f"\nOutput files:")
        print(f"  Clean:   {clean_path} ({clean_path.stat().st_size / 1e9:.2f} GB)")
        print(f"  Flagged: {flagged_path} ({flagged_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
