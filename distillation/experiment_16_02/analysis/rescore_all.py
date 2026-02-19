"""Re-score all MBPP+, HumanEval+, and MATH-500 parquets for Sonnet Qwen models.

MBPP+ / HumanEval+: Re-extract code using fixed extract_code (filters empty blocks,
prefers blocks with 'def'), re-run tests for rows where extraction changed.

MATH-500: Run math_verify + LLM judge on stage1 failures.
"""

import asyncio
import concurrent.futures
import sys
from pathlib import Path

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, EVAL_CONCURRENCY
from utils import (
    extract_code_from_response, parse_think_tags,
    call_openrouter, load_or_build_cache,
)

# Import test runners
from eval_tools.mbppplus import run_mbppplus_test_sync, _build_mbppplus_cache
from eval_tools.humanevalplus import run_humanevalplus_test_sync, _build_humanevalplus_cache

sys.path.insert(0, '/workspace/tinker-cookbook')
from tinker_cookbook.recipes.math_rl.math_grading import grade_answer_math_verify

RESULTS_DIR = Path(__file__).parent.parent / "results"

# All Sonnet Qwen model families (all checkpoints)
MODEL_PREFIXES = [
    "sonnet_math_qwen_4k",
    "sonnet_code_qwen_4k",
    "sonnet_math_qwen_instruct_4k",
    "sonnet_code_qwen_instruct_4k",
]
# Also include baselines
EXTRA_MODELS = ["qwen3_base", "qwen3_instruct_base"]

_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

JUDGE_SYSTEM_MSG = (
    "You are a math answer verification system. You will be given a math competition "
    "question, the expected correct answer, and a student's answer. Determine whether "
    "the student's answer is mathematically equivalent to the expected answer.\n\n"
    "Account for:\n"
    "- Different variable names (e.g., n vs k)\n"
    "- Different but equivalent formalisms (e.g., \\lceil x \\rceil vs ceil(x))\n"
    "- Different formatting (e.g., 1/2 vs \\frac{1}{2} vs 0.5)\n"
    "- Simplified vs unsimplified forms (e.g., 2+3 vs 5)\n"
    "- Equivalent expressions (e.g., n(n+1)/2 vs \\binom{n+1}{2})\n\n"
    "Respond with ONLY a JSON object: {\"equivalent\": true} or {\"equivalent\": false}\n"
    "Do NOT include any explanation outside the JSON."
)


def get_all_model_dirs():
    """Get all model directories matching our prefixes + extras."""
    dirs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name in EXTRA_MODELS:
            dirs.append(name)
        elif any(name.startswith(p) for p in MODEL_PREFIXES):
            dirs.append(name)
    return dirs


# ── MBPP+ re-scoring ─────────────────────────────────────────────────────────

def rescore_mbppplus(model_name, problems_by_task):
    """Re-score MBPP+ parquet using fixed extract_code."""
    pq_path = RESULTS_DIR / model_name / "results_mbppplus.parquet"
    if not pq_path.exists():
        return None

    df = pd.read_parquet(pq_path)
    original_passed = df["passed"].sum()
    total = len(df)
    changed = 0
    upgraded = 0

    for idx, row in df.iterrows():
        if row["passed"]:
            continue  # already passing, skip

        raw = row["raw_output"]
        if not isinstance(raw, str) or raw.startswith("ERROR:"):
            continue

        new_code = extract_code_from_response(raw)
        if new_code is None:
            continue

        # Look up test code from problems cache
        task_id = row.get("task_id")
        if task_id not in problems_by_task:
            continue

        problem = problems_by_task[task_id]
        passed, detail = run_mbppplus_test_sync(new_code, problem["test"])
        if passed:
            df.at[idx, "passed"] = True
            df.at[idx, "detail"] = "passed (re-scored)"
            upgraded += 1
        changed += 1

    new_passed = df["passed"].sum()
    if upgraded > 0:
        df.to_parquet(pq_path, index=False)

    return {
        "model": model_name, "benchmark": "mbppplus",
        "total": total, "original": int(original_passed),
        "retested": changed, "upgraded": upgraded,
        "final": int(new_passed),
    }


# ── HumanEval+ re-scoring ────────────────────────────────────────────────────

def rescore_humanevalplus(model_name, problems_by_task):
    """Re-score HumanEval+ parquet using fixed extract_code."""
    pq_path = RESULTS_DIR / model_name / "results_humanevalplus.parquet"
    if not pq_path.exists():
        return None

    df = pd.read_parquet(pq_path)
    original_passed = df["passed"].sum()
    total = len(df)
    changed = 0
    upgraded = 0

    for idx, row in df.iterrows():
        if row["passed"]:
            continue

        raw = row["raw_output"]
        if not isinstance(raw, str) or raw.startswith("ERROR:"):
            continue

        new_code = extract_code_from_response(raw)
        if new_code is None:
            continue

        task_id = row.get("task_id")
        if task_id not in problems_by_task:
            continue

        problem = problems_by_task[task_id]
        passed, detail = run_humanevalplus_test_sync(
            new_code, problem["test"], problem["entry_point"]
        )
        if passed:
            df.at[idx, "passed"] = True
            df.at[idx, "detail"] = "passed (re-scored)"
            upgraded += 1
        changed += 1

    new_passed = df["passed"].sum()
    if upgraded > 0:
        df.to_parquet(pq_path, index=False)

    return {
        "model": model_name, "benchmark": "humanevalplus",
        "total": total, "original": int(original_passed),
        "retested": changed, "upgraded": upgraded,
        "final": int(new_passed),
    }


# ── MATH-500 re-scoring ──────────────────────────────────────────────────────

async def judge_answer_llm(session, semaphore, question, expected, predicted):
    """Use GPT-5.2 to judge mathematical equivalence."""
    prompt = (
        f"## Question\n\n{question}\n\n"
        f"## Expected Correct Answer\n\n{expected}\n\n"
        f"## Student's Answer\n\n{predicted}\n\n"
        f"Are these answers mathematically equivalent?"
    )
    resp = await call_openrouter(
        session, JUDGE_MODEL,
        [
            {"role": "system", "content": JUDGE_SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0, max_tokens=128,
        semaphore=semaphore,
        api_key=OPENROUTER_API_KEY, api_url=OPENROUTER_URL,
    )
    if str(resp).startswith("ERROR:"):
        return False
    resp_lower = str(resp).lower()
    if '"equivalent": true' in resp_lower or '"equivalent":true' in resp_lower:
        return True
    return False


async def rescore_math500(model_name):
    """Re-score MATH-500 with math_verify + LLM judge."""
    pq_path = RESULTS_DIR / model_name / "results_math_500.parquet"
    if not pq_path.exists():
        return None

    df = pd.read_parquet(pq_path)
    total = len(df)
    original_correct = int(df["correct"].sum())

    # Add grading_method column if missing
    if "grading_method" not in df.columns:
        df["grading_method"] = df["correct"].apply(
            lambda x: "stage1" if x else "stage1_fail"
        )

    # Find candidates
    mask = (~df["correct"]) & (~df["predicted_answer"].isin(["no_boxed", "error"]))
    candidates = df[mask].index.tolist()

    if not candidates:
        return {
            "model": model_name, "benchmark": "math_500",
            "total": total, "original": original_correct,
            "math_verify": 0, "llm_judge": 0,
            "final": original_correct,
        }

    # Stage 1b: math_verify
    mv_upgraded = 0
    still_need_judge = []
    for idx in candidates:
        predicted = df.at[idx, "predicted_answer"]
        ground_truth = df.at[idx, "ground_truth"]
        try:
            if grade_answer_math_verify(predicted, ground_truth):
                df.at[idx, "correct"] = True
                df.at[idx, "grading_method"] = "math_verify"
                mv_upgraded += 1
                continue
        except Exception:
            pass
        still_need_judge.append(idx)

    # Stage 2: LLM judge
    llm_upgraded = 0
    if still_need_judge:
        semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
        async with aiohttp.ClientSession() as session:
            async def judge_one(idx):
                result = await judge_answer_llm(
                    session, semaphore,
                    question=df.at[idx, "question"],
                    expected=df.at[idx, "ground_truth"],
                    predicted=df.at[idx, "predicted_answer"],
                )
                return idx, result

            tasks = [judge_one(idx) for idx in still_need_judge]
            results = await asyncio.gather(*tasks)

            for idx, is_correct in results:
                if is_correct:
                    df.at[idx, "correct"] = True
                    df.at[idx, "grading_method"] = "llm_judge"
                    llm_upgraded += 1

    final_correct = int(df["correct"].sum())
    if mv_upgraded > 0 or llm_upgraded > 0:
        df.to_parquet(pq_path, index=False)

    return {
        "model": model_name, "benchmark": "math_500",
        "total": total, "original": original_correct,
        "math_verify": mv_upgraded, "llm_judge": llm_upgraded,
        "final": final_correct,
    }


async def main():
    models = get_all_model_dirs()
    print(f"Found {len(models)} model directories to process\n")

    # Load problem caches for code benchmarks
    print("Loading MBPP+ problem cache...")
    mbpp_problems = load_or_build_cache("mbppplus", _build_mbppplus_cache, 42)
    mbpp_by_task = {p["task_id"]: p for p in mbpp_problems}

    print("Loading HumanEval+ problem cache...")
    he_problems = load_or_build_cache("humanevalplus", _build_humanevalplus_cache, 42)
    he_by_task = {p["task_id"]: p for p in he_problems}

    # ── MBPP+ ──
    print("\n" + "=" * 70)
    print("MBPP+ RE-SCORING")
    print("=" * 70)
    mbpp_results = []
    for model in models:
        r = rescore_mbppplus(model, mbpp_by_task)
        if r:
            delta = r["final"] - r["original"]
            tag = f" (+{delta})" if delta > 0 else ""
            print(f"  {model:<50} {r['original']:>3}/{r['total']} → {r['final']:>3}/{r['total']}{tag}")
            mbpp_results.append(r)

    # ── HumanEval+ ──
    print("\n" + "=" * 70)
    print("HumanEval+ RE-SCORING")
    print("=" * 70)
    he_results = []
    for model in models:
        r = rescore_humanevalplus(model, he_by_task)
        if r:
            delta = r["final"] - r["original"]
            tag = f" (+{delta})" if delta > 0 else ""
            print(f"  {model:<50} {r['original']:>3}/{r['total']} → {r['final']:>3}/{r['total']}{tag}")
            he_results.append(r)

    # ── MATH-500 ──
    print("\n" + "=" * 70)
    print("MATH-500 RE-SCORING (math_verify + LLM judge)")
    print("=" * 70)
    math_results = []
    for model in models:
        r = await rescore_math500(model)
        if r:
            delta = r["final"] - r["original"]
            tag = f" (+{delta})" if delta > 0 else ""
            mv = f"mv:{r['math_verify']}" if r["math_verify"] else ""
            llm = f"llm:{r['llm_judge']}" if r["llm_judge"] else ""
            upgrades = " ".join(filter(None, [mv, llm]))
            print(f"  {model:<50} {r['original']:>3}/{r['total']} → {r['final']:>3}/{r['total']}{tag}  {upgrades}")
            math_results.append(r)

    # ── Summary of changes ──
    print("\n" + "=" * 70)
    print("MODELS WITH CHANGES")
    print("=" * 70)
    for results, bench in [(mbpp_results, "MBPP+"), (he_results, "HE+"), (math_results, "MATH")]:
        changed = [r for r in results if r["final"] != r["original"]]
        if changed:
            print(f"\n{bench}:")
            for r in changed:
                delta = r["final"] - r["original"]
                total = r["total"]
                print(f"  {r['model']:<50} {r['original']/total*100:5.1f}% → {r['final']/total*100:5.1f}% (+{delta})")


if __name__ == "__main__":
    asyncio.run(main())
