"""CLI: Load two model result parquets, join, run ensemble effort judging, save output.

Usage:
    python compare.py \
        --model-a qwen3_base \
        --model-b sonnet_math_qwen_4k_step300 \
        --benchmarks math_500,aime \
        --name-a "Qwen Base" --name-b "Sonnet Math s300" \
        [--output output/base_vs_math_s300.parquet] \
        [--max-problems 10]
"""

import argparse
import asyncio
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

from config import BENCHMARK_CONFIG, RESULTS_DIR, EVAL_CACHE_DIR, OUTPUT_DIR, DISTILLATION_DIR
from judge import run_pairwise_judging


def load_problem_text(benchmark: str) -> dict | None:
    """Load problem text lookup for benchmarks where the parquet ID isn't the problem.

    Returns {id_value: problem_text} or None if the ID column already IS the problem.
    """
    if benchmark in ("math_500", "aime"):
        return None  # question column is the full problem text

    if benchmark in ("mbppplus", "humanevalplus"):
        cache_path = EVAL_CACHE_DIR / f"{benchmark}_seed42.json.gz"
        with gzip.open(cache_path, "rt") as f:
            data = json.load(f)
        id_key = "task_id"
        text_key = "prompt"
        return {item[id_key]: item[text_key] for item in data}

    if benchmark == "livecodebench_v5":
        # Load directly from HF + eval cache to avoid importing eval_tools
        # (which pulls in distillation/config.py and conflicts with our config.py)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            'livecodebench/code_generation_lite', 'test5.jsonl', repo_type='dataset'
        )
        with open(path) as f:
            all_rows = [json.loads(line) for line in f]
        # Get the shuffled order from eval cache
        cache_path = EVAL_CACHE_DIR / "livecodebench_v5_seed42.json.gz"
        with gzip.open(cache_path, "rt") as f:
            order = json.load(f)
        return {
            all_rows[item["original_index"]]["question_title"]:
                all_rows[item["original_index"]]["question_content"]
            for item in order
        }

    return None


def _correctness_bucket(correct_a: bool, correct_b: bool) -> str:
    if correct_a and correct_b:
        return "both_correct"
    elif correct_a:
        return "a_only_correct"
    elif correct_b:
        return "b_only_correct"
    else:
        return "both_wrong"


def load_and_join(
    model_a_dir: Path,
    model_b_dir: Path,
    benchmark: str,
    name_a: str,
    name_b: str,
    problem_lookup: dict | None,
    max_problems: int | None = None,
) -> list[dict]:
    """Load both parquets and produce paired dicts for judging."""
    cfg = BENCHMARK_CONFIG[benchmark]
    id_col = cfg["id_col"]
    correct_col = cfg["correct_col"]

    path_a = model_a_dir / f"results_{benchmark}.parquet"
    path_b = model_b_dir / f"results_{benchmark}.parquet"

    if not path_a.exists():
        print(f"  WARNING: {path_a} not found, skipping")
        return []
    if not path_b.exists():
        print(f"  WARNING: {path_b} not found, skipping")
        return []

    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    # Select needed columns
    a_cols = [id_col, "raw_output", correct_col]
    b_cols = [id_col, "raw_output", correct_col]
    if "difficulty" in df_a.columns:
        a_cols.append("difficulty")

    merged = df_a[a_cols].merge(df_b[b_cols], on=id_col, suffixes=("_a", "_b"))

    if len(merged) < min(len(df_a), len(df_b)):
        print(f"  WARNING: merge dropped rows ({len(merged)} of {min(len(df_a), len(df_b))})")

    if max_problems is not None:
        merged = merged.head(max_problems)

    pairs = []
    for _, row in merged.iterrows():
        id_val = row[id_col]

        # Resolve problem text
        if problem_lookup is not None:
            problem_text = problem_lookup.get(id_val, str(id_val)[:500])
        else:
            problem_text = str(id_val)

        correct_a = bool(row[f"{correct_col}_a"])
        correct_b = bool(row[f"{correct_col}_b"])

        pairs.append({
            "problem_id": str(id_val)[:200],  # truncate long question text for ID
            "domain": cfg["domain"],
            "benchmark": cfg["display"],
            "difficulty": row.get("difficulty", None),
            "model_a": name_a,
            "model_b": name_b,
            "response_a": str(row["raw_output_a"] or ""),
            "response_b": str(row["raw_output_b"] or ""),
            "correct_a": correct_a,
            "correct_b": correct_b,
            "correctness_bucket": _correctness_bucket(correct_a, correct_b),
            "problem_text": problem_text,  # used by judge, dropped before save
        })
    return pairs


def print_quick_summary(df: pd.DataFrame):
    """Print a compact summary table after judging."""
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)

    total = len(df)
    a_wins = (df["effort_judgment"] == "A").sum()
    b_wins = (df["effort_judgment"] == "B").sum()
    ties = (df["effort_judgment"] == "Neither").sum()
    errors = (df["effort_judgment"] == "error").sum()

    name_a = df["model_a"].iloc[0]
    name_b = df["model_b"].iloc[0]

    print(f"  {name_a} (A) tries harder: {a_wins}/{total} ({a_wins/total*100:.1f}%)")
    print(f"  {name_b} (B) tries harder: {b_wins}/{total} ({b_wins/total*100:.1f}%)")
    print(f"  Similar effort:             {ties}/{total} ({ties/total*100:.1f}%)")
    if errors > 0:
        print(f"  Errors:                     {errors}/{total}")
    print(f"  Mean judge confidence:      {df['judge_confidence'].mean():.3f}")

    print(f"\n  {'Benchmark':<15} {'A':>6} {'B':>6} {'Tie':>6} {'N':>6}")
    print(f"  {'-'*39}")
    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        n = len(sub)
        a = (sub["effort_judgment"] == "A").sum()
        b = (sub["effort_judgment"] == "B").sum()
        t = (sub["effort_judgment"] == "Neither").sum()
        print(f"  {bench:<15} {a:>5} {b:>6} {t:>6} {n:>6}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pairwise effort comparison")
    parser.add_argument("--model-a", required=True, help="Model A results dir name (under results/)")
    parser.add_argument("--model-b", required=True, help="Model B results dir name (under results/)")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark keys")
    parser.add_argument("--name-a", default=None, help="Display name for model A")
    parser.add_argument("--name-b", default=None, help="Display name for model B")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit per benchmark (for testing)")
    parser.add_argument("--concurrency", type=int, default=10, help="Per-model concurrency")
    return parser.parse_args()


async def async_main():
    args = parse_args()

    model_a_dir = RESULTS_DIR / args.model_a
    model_b_dir = RESULTS_DIR / args.model_b
    name_a = args.name_a or args.model_a
    name_b = args.name_b or args.model_b
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    all_pairs = []
    for benchmark in benchmarks:
        if benchmark not in BENCHMARK_CONFIG:
            print(f"WARNING: Unknown benchmark '{benchmark}', skipping")
            continue
        print(f"\nLoading {benchmark}...")
        problem_lookup = load_problem_text(benchmark)
        pairs = load_and_join(
            model_a_dir, model_b_dir, benchmark,
            name_a, name_b, problem_lookup, args.max_problems,
        )
        print(f"  {len(pairs)} paired rows")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("No pairs to judge. Exiting.")
        return

    print(f"\nTotal: {len(all_pairs)} pairs across {len(benchmarks)} benchmarks")
    print("Running ensemble judging...")

    judged = await run_pairwise_judging(all_pairs, concurrency=args.concurrency)

    # Build output DataFrame
    df = pd.DataFrame(judged)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_a = args.model_a.replace("/", "_")
        safe_b = args.model_b.replace("/", "_")
        output_path = OUTPUT_DIR / f"{safe_a}_vs_{safe_b}.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    print_quick_summary(df)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
