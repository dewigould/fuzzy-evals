"""Analyze eval results from a parquet folder.

Reports completion rate and accuracy for math and code benchmarks.

Usage:
  # Single folder (defaults to trained mode)
  python analyze_results.py results/math_4k_lr5_step100

  # Base model (must specify --base before the folder)
  python analyze_results.py --base results/qwen3_base

  # Multiple folders (comparison table) — base folders marked with --base prefix
  python analyze_results.py --base results/qwen3_base results/math_4k_lr5_step100 results/code_4k_lr5_step100
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ── Detection helpers ────────────────────────────────────────────────────────

def has_boxed(text: str) -> bool:
    """Check if text contains a complete \\boxed{...} expression."""
    # Match \boxed{ followed by content and a closing }
    # Handle nested braces by counting
    idx = text.find("\\boxed{")
    if idx == -1:
        return False
    depth = 0
    for i in range(idx + len("\\boxed{") - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


def has_complete_code_block(text: str) -> bool:
    """Check if text contains a complete ```python ... ``` block."""
    return bool(re.search(r"```python\s*\n.*?```", text, re.DOTALL))


def has_complete_answer(text: str) -> bool:
    """Check if text contains a complete \\boxed{} or ```python``` block."""
    return has_boxed(text) or has_complete_code_block(text)


def is_thinking_closed(raw_output: str) -> bool:
    """Check if </think> tag is present (thinking section was closed)."""
    return "</think>" in raw_output


def get_after_think(raw_output: str) -> str:
    """Get content after the last </think> tag."""
    idx = raw_output.rfind("</think>")
    if idx == -1:
        return ""
    return raw_output[idx + len("</think>"):]


# ── Per-dataset analysis ─────────────────────────────────────────────────────

def analyze_parquet(path: Path, is_trained: bool) -> dict:
    """Analyze a single parquet file. Returns metrics dict."""
    df = pd.read_parquet(path)
    n = len(df)

    # Determine accuracy column
    if "correct" in df.columns:
        acc_col = "correct"
        task_type = "math"
    elif "passed" in df.columns:
        acc_col = "passed"
        task_type = "code"
    else:
        return {"n": n, "error": "no accuracy column found"}

    total_correct = df[acc_col].sum()
    accuracy_all = total_correct / n if n > 0 else 0

    if is_trained:
        # Trained model: completion = </think> closed AND complete answer after thinking
        completed = []
        for _, row in df.iterrows():
            raw = row["raw_output"] or ""
            if is_thinking_closed(raw):
                after = get_after_think(raw)
                if task_type == "math":
                    completed.append(has_boxed(after))
                else:
                    completed.append(has_complete_code_block(after))
            else:
                completed.append(False)
        df["completed"] = completed
    else:
        # Base model: completion = complete answer anywhere in output
        if task_type == "math":
            df["completed"] = df["raw_output"].fillna("").apply(has_boxed)
        else:
            df["completed"] = df["raw_output"].fillna("").apply(has_complete_code_block)

    n_completed = df["completed"].sum()
    completion_rate = n_completed / n if n > 0 else 0

    # Accuracy on completed only
    completed_mask = df["completed"]
    n_completed_correct = df.loc[completed_mask, acc_col].sum() if n_completed > 0 else 0
    accuracy_completed = n_completed_correct / n_completed if n_completed > 0 else 0

    return {
        "n": n,
        "task_type": task_type,
        "completion_rate": completion_rate,
        "n_completed": int(n_completed),
        "accuracy_all": accuracy_all,
        "n_correct": int(total_correct),
        "accuracy_completed": accuracy_completed,
        "n_completed_correct": int(n_completed_correct),
    }


def analyze_folder(folder: Path, is_trained: bool = True) -> dict:
    """Analyze all parquet files in a results folder."""
    folder = Path(folder)
    if not folder.exists():
        return {"error": f"folder not found: {folder}"}

    results = {"name": folder.name, "is_trained": is_trained}
    for pq in sorted(folder.glob("results_*.parquet")):
        ds_name = pq.stem.replace("results_", "")
        results[ds_name] = analyze_parquet(pq, is_trained)

    return results


# ── Display ──────────────────────────────────────────────────────────────────

def print_single(results: dict):
    """Print results for a single folder."""
    name = results["name"]
    is_trained = results["is_trained"]
    mode = "trained" if is_trained else "base"

    print(f"\n{'='*70}")
    print(f"  {name}  ({mode})")
    print(f"{'='*70}")
    print(f"  {'Dataset':<18} {'N':>5}  {'Completed':>12}  {'Acc (all)':>10}  {'Acc (completed)':>16}")
    print(f"  {'-'*18} {'-'*5}  {'-'*12}  {'-'*10}  {'-'*16}")

    for key, val in results.items():
        if key in ("name", "is_trained", "error"):
            continue
        if "error" in val:
            print(f"  {key:<18} {val['error']}")
            continue
        n = val["n"]
        cr = val["completion_rate"]
        aa = val["accuracy_all"]
        ac = val["accuracy_completed"]
        nc = val["n_completed"]
        print(
            f"  {key:<18} {n:>5}  "
            f"{nc:>4}/{n:<4} {cr*100:5.1f}%  "
            f"{aa*100:9.1f}%  "
            f"{ac*100:15.1f}%"
        )
    print()


def print_comparison(all_results: list[dict]):
    """Print a comparison table across multiple folders."""
    # Collect all dataset names
    all_ds = []
    for r in all_results:
        for key in r:
            if key not in ("name", "is_trained", "error") and key not in all_ds:
                all_ds.append(key)

    # Header
    name_width = max(len(r["name"]) for r in all_results) + 2
    print(f"\n{'='*80}")
    print(f"  Comparison Table")
    print(f"{'='*80}")

    for ds in all_ds:
        print(f"\n  --- {ds} ---")
        print(f"  {'Model':<{name_width}} {'Mode':>7}  {'N':>5}  {'Completed':>12}  {'Acc(all)':>9}  {'Acc(cmplt)':>11}")
        print(f"  {'-'*name_width} {'-'*7}  {'-'*5}  {'-'*12}  {'-'*9}  {'-'*11}")
        for r in all_results:
            name = r["name"]
            mode = "train" if r["is_trained"] else "base"
            if ds not in r:
                print(f"  {name:<{name_width}} {mode:>7}  {'—':>5}")
                continue
            val = r[ds]
            if "error" in val:
                print(f"  {name:<{name_width}} {mode:>7}  {val['error']}")
                continue
            n = val["n"]
            cr = val["completion_rate"]
            aa = val["accuracy_all"]
            ac = val["accuracy_completed"]
            nc = val["n_completed"]
            print(
                f"  {name:<{name_width}} {mode:>7}  {n:>5}  "
                f"{nc:>4}/{n:<4} {cr*100:4.0f}%  "
                f"{aa*100:8.1f}%  "
                f"{ac*100:10.1f}%"
            )
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze eval results from parquet folders")
    parser.add_argument("folders", nargs="+", type=str,
                        help="Result folder(s) containing results_*.parquet files (treated as trained)")
    parser.add_argument("--base", nargs="+", type=str, default=[],
                        help="Base model folder(s) (no </think> check)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save output to file (in addition to printing)")
    parsed = parser.parse_args()

    base_set = set()
    all_folders = []

    for f in parsed.base:
        base_set.add(f)
        all_folders.append(f)
    for f in parsed.folders:
        all_folders.append(f)

    all_results = []
    for folder_path in all_folders:
        folder = Path(folder_path)
        if not folder.is_absolute():
            exp_dir = Path(__file__).parent
            if (exp_dir / folder).exists():
                folder = exp_dir / folder
        is_trained = folder_path not in base_set
        results = analyze_folder(folder, is_trained=is_trained)
        all_results.append(results)

    # Optionally tee output to file
    if parsed.output:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = type("Tee", (), {
            "write": lambda self, s: (buf.write(s), old_stdout.write(s)),
            "flush": lambda self: (buf.flush(), old_stdout.flush()),
        })()

    if len(all_results) == 1:
        print_single(all_results[0])
    else:
        # Print individual + comparison
        for r in all_results:
            print_single(r)
        print_comparison(all_results)

    if parsed.output:
        sys.stdout = old_stdout
        out_path = Path(parsed.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(buf.getvalue())
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
