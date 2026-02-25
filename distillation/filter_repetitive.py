"""Filter degenerate repetitive examples from OpenR1-Math-220k.

Detects reasoning traces that contain repetition loops — a known problem
with R1-distilled datasets where the teacher model got stuck and repeated
sentences/phrases until hitting the token limit.

Two detectors:
  1. 50-gram repetition: any 50-word sequence appearing 2+ times
  2. Truncation: <think> present but no </think> (model never finished)

Usage:
  python filter_repetitive.py [OPTIONS]

  # Dry run — just print stats, don't save
  python filter_repetitive.py --dry-run

  # Limit for testing
  python filter_repetitive.py --limit 1000 --dry-run
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def extract_think_text(content: str) -> str | None:
    """Extract text between <think> and </think> tags.

    If </think> is missing (truncated), returns everything after <think>.
    """
    m = re.search(r"<think>\s*", content)
    if not m:
        return None
    start = m.end()
    m2 = re.search(r"</think>", content[start:])
    if m2:
        return content[start : start + m2.start()]
    # No closing tag — return everything after <think>
    return content[start:]


def tokenize_words(text: str) -> list[str]:
    """Simple whitespace tokenizer for n-gram analysis."""
    return text.lower().split()


def has_repeated_50gram(words: list[str]) -> bool:
    """Check if any 50-word sequence appears 2+ times."""
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


def count_repeated_50grams(words: list[str]) -> int:
    """Count how many unique 50-grams appear 2+ times."""
    n = 50
    if len(words) < n:
        return 0
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i : i + n]))
    counts = Counter(ngrams)
    return sum(1 for c in counts.values() if c > 1)


def is_degenerate(content: str, think_text: str, words: list[str], filter_truncated: bool) -> tuple[bool, list[str]]:
    """Check if an example is degenerate. Returns (is_bad, list_of_reasons)."""
    reasons = []

    # Check 50-gram repetition
    if has_repeated_50gram(words):
        reasons.append("50gram")

    # Check truncation
    if filter_truncated:
        if "<think>" in content and "</think>" not in content:
            reasons.append("truncated")

    return len(reasons) > 0, reasons


def main():
    parser = argparse.ArgumentParser(description="Filter repetitive examples from OpenR1-Math-220k")
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k", help="HF dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output-dir", default="filtered_data", help="Output directory")
    parser.add_argument("--no-filter-truncated", action="store_true", help="Don't flag examples missing </think>")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N examples")
    parser.add_argument("--dry-run", action="store_true", help="Just print stats, don't save files")
    args = parser.parse_args()

    filter_truncated = not args.no_filter_truncated

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Detectors: 50-gram repetition (any 50-word seq appearing 2+ times)")
    if filter_truncated:
        print(f"           + truncation (missing </think>)")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    print()

    # Stream the dataset
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

    # Track distribution of flagging reasons
    reasons_count = {"50gram": 0, "truncated": 0}

    try:
        for example in ds:
            total += 1

            if args.limit and total > args.limit:
                total -= 1
                break

            # Extract the assistant message with <think> tags
            messages = example.get("messages", [])
            if len(messages) < 2:
                # No assistant response — keep it as clean
                if clean_file:
                    clean_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                continue

            content = messages[1].get("content", "")
            think_text = extract_think_text(content)

            if think_text is None or len(think_text) < 50:
                no_think += 1
                if clean_file:
                    clean_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                continue

            words = tokenize_words(think_text)

            degenerate, reasons = is_degenerate(content, think_text, words, filter_truncated)

            if degenerate:
                flagged_count += 1
                for r in reasons:
                    reasons_count[r] += 1

                if flagged_file:
                    record = {
                        "uuid": example.get("uuid", ""),
                        "problem": example.get("problem", "")[:200],
                        "n_words": len(words),
                        "reasons": reasons,
                    }
                    if "50gram" in reasons:
                        record["n_repeated_50grams"] = count_repeated_50grams(words)
                    flagged_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                if clean_file:
                    clean_file.write(json.dumps(example, ensure_ascii=False) + "\n")

            # Progress
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

    # Summary
    print(f"\n{'=' * 60}")
    print(f"FILTERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total examples processed: {total:,}")
    print(f"  Flagged (degenerate):     {flagged_count:,} ({flagged_count/max(total,1)*100:.1f}%)")
    print(f"  Clean (kept):             {total - flagged_count - no_think:,}")
    print(f"  No <think> tag:           {no_think:,}")
    print(f"  Time: {elapsed:.0f}s ({total/max(elapsed,1):.0f} examples/s)")
    print()
    print(f"Flagging reasons (examples can trigger multiple):")
    print(f"  50-gram repetition:        {reasons_count['50gram']:,}")
    print(f"  Truncated (no </think>):   {reasons_count['truncated']:,}")

    if not args.dry_run:
        print(f"\nOutput files:")
        print(f"  Clean:   {clean_path} ({clean_path.stat().st_size / 1e9:.2f} GB)")
        print(f"  Flagged: {flagged_path} ({flagged_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
