"""Analyze pairwise effort judging results.

Usage:
    python analyze.py output/base_vs_math_s300.parquet
    python analyze.py output/*.parquet          # analyze multiple files
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _effort_row(sub: pd.DataFrame) -> dict:
    """Compute effort stats for a subset."""
    n = len(sub)
    if n == 0:
        return {"n": 0, "a_pct": 0, "b_pct": 0, "tie_pct": 0, "conf": 0}
    a = (sub["effort_judgment"] == "A").sum()
    b = (sub["effort_judgment"] == "B").sum()
    t = (sub["effort_judgment"] == "Neither").sum()
    return {
        "n": n,
        "a_pct": a / n * 100,
        "b_pct": b / n * 100,
        "tie_pct": t / n * 100,
        "conf": sub["judge_confidence"].mean(),
    }


def summarize_overall(df: pd.DataFrame):
    """Print overall effort comparison."""
    print_section("OVERALL")
    name_a = df["model_a"].iloc[0]
    name_b = df["model_b"].iloc[0]
    s = _effort_row(df)
    print(f"  Model A: {name_a}")
    print(f"  Model B: {name_b}")
    print(f"  Total pairs: {s['n']}")
    print(f"  A tries harder:  {s['a_pct']:5.1f}%")
    print(f"  B tries harder:  {s['b_pct']:5.1f}%")
    print(f"  Similar effort:  {s['tie_pct']:5.1f}%")
    print(f"  Mean confidence: {s['conf']:.3f}")


def summarize_by_benchmark(df: pd.DataFrame):
    """Print effort breakdown per benchmark."""
    print_section("BY BENCHMARK")
    print(f"  {'Benchmark':<15} {'N':>5}  {'A%':>6} {'B%':>6} {'Tie%':>6}  {'Conf':>5}")
    print(f"  {'-' * 50}")
    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        s = _effort_row(sub)
        print(f"  {bench:<15} {s['n']:>5}  {s['a_pct']:>5.1f} {s['b_pct']:>5.1f} {s['tie_pct']:>5.1f}  {s['conf']:>5.3f}")


def summarize_by_bucket(df: pd.DataFrame):
    """Print effort breakdown per correctness bucket."""
    print_section("BY CORRECTNESS BUCKET")
    name_a = df["model_a"].iloc[0]
    name_b = df["model_b"].iloc[0]

    bucket_labels = {
        "both_correct": "Both correct",
        "both_wrong": "Both wrong",
        "a_only_correct": f"Only A ({name_a}) correct",
        "b_only_correct": f"Only B ({name_b}) correct",
    }

    print(f"  {'Bucket':<35} {'N':>5}  {'A%':>6} {'B%':>6} {'Tie%':>6}")
    print(f"  {'-' * 60}")
    for bucket, label in bucket_labels.items():
        sub = df[df["correctness_bucket"] == bucket]
        s = _effort_row(sub)
        if s["n"] > 0:
            print(f"  {label:<35} {s['n']:>5}  {s['a_pct']:>5.1f} {s['b_pct']:>5.1f} {s['tie_pct']:>5.1f}")


def summarize_by_benchmark_and_bucket(df: pd.DataFrame):
    """Print cross-tabulation of benchmark x correctness bucket."""
    print_section("BY BENCHMARK x BUCKET")

    for bench in df["benchmark"].unique():
        bench_df = df[df["benchmark"] == bench]
        print(f"\n  {bench} (n={len(bench_df)}):")
        print(f"    {'Bucket':<20} {'N':>4}  {'A%':>6} {'B%':>6} {'Tie%':>6}")
        print(f"    {'-' * 46}")
        for bucket in ["both_correct", "both_wrong", "a_only_correct", "b_only_correct"]:
            sub = bench_df[bench_df["correctness_bucket"] == bucket]
            s = _effort_row(sub)
            if s["n"] > 0:
                print(f"    {bucket:<20} {s['n']:>4}  {s['a_pct']:>5.1f} {s['b_pct']:>5.1f} {s['tie_pct']:>5.1f}")


def summarize_judge_agreement(df: pd.DataFrame):
    """Print judge agreement statistics."""
    print_section("JUDGE AGREEMENT")

    # Find judge winner columns
    judge_cols = [c for c in df.columns if c.endswith("_winner") and c.startswith("judge_")]

    if len(judge_cols) < 2:
        print("  Not enough judge columns for agreement analysis.")
        return

    # Count agreement levels
    unanimous = 0
    split_2_1 = 0
    total = len(df)

    for _, row in df.iterrows():
        winners = [row[c] for c in judge_cols if pd.notna(row[c]) and row[c] is not None]
        if len(winners) >= 2:
            from collections import Counter
            counts = Counter(winners)
            max_count = counts.most_common(1)[0][1]
            if max_count == len(winners):
                unanimous += 1
            else:
                split_2_1 += 1

    print(f"  Unanimous (3/3):  {unanimous}/{total} ({unanimous/total*100:.1f}%)")
    print(f"  Split (2/1):      {split_2_1}/{total} ({split_2_1/total*100:.1f}%)")

    # Per-judge bias
    print(f"\n  Per-judge breakdown:")
    print(f"    {'Judge':<15} {'A':>6} {'B':>6} {'Tie':>6} {'Fail':>6}")
    print(f"    {'-' * 40}")
    for col in judge_cols:
        short = col.replace("judge_", "").replace("_winner", "")
        vals = df[col]
        a = (vals == "A").sum()
        b = (vals == "B").sum()
        t = (vals == "Neither").sum()
        fail = vals.isna().sum() + (vals == "").sum()
        print(f"    {short:<15} {a:>6} {b:>6} {t:>6} {fail:>6}")


def analyze_file(path: Path):
    """Run all analyses on a single result parquet."""
    print(f"\n{'#' * 60}")
    print(f"  Analyzing: {path.name}")
    print(f"{'#' * 60}")

    df = pd.read_parquet(path)

    summarize_overall(df)
    summarize_by_benchmark(df)
    summarize_by_bucket(df)
    summarize_by_benchmark_and_bucket(df)
    summarize_judge_agreement(df)


def main():
    parser = argparse.ArgumentParser(description="Analyze pairwise effort judging results")
    parser.add_argument("files", nargs="+", help="Output parquet file(s) to analyze")
    args = parser.parse_args()

    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        analyze_file(path)


if __name__ == "__main__":
    main()
