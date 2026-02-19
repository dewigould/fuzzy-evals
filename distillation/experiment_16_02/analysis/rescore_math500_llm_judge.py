"""Re-score MATH-500 parquets with two-stage grading (math_verify + LLM judge).

For each parquet:
1. Keep all correct=True rows unchanged
2. For correct=False rows with a predicted answer (not no_boxed/error):
   a. Try grade_answer_math_verify as fallback
   b. If still wrong, send to GPT-5.2 LLM judge for equivalence check
3. Update parquet with new correct values and grading_method column
"""

import asyncio
import sys
from pathlib import Path

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, EVAL_CONCURRENCY
from utils import call_openrouter

sys.path.insert(0, '/workspace/tinker-cookbook')
from tinker_cookbook.recipes.math_rl.math_grading import grade_answer_math_verify

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

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Models in the base_vs_instruct plot
PLOT_MODELS = [
    "qwen3_base",
    "sonnet_math_qwen_4k_step300",
    "sonnet_code_qwen_4k_step200",
    "qwen3_instruct_base",
    "sonnet_math_qwen_instruct_4k_step200",
    "sonnet_code_qwen_instruct_4k_final",
]


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


async def rescore_model(model_name):
    """Re-score a single model's MATH-500 parquet."""
    pq_path = RESULTS_DIR / model_name / "results_math_500.parquet"
    if not pq_path.exists():
        print(f"  {model_name}: MISSING")
        return None

    df = pd.read_parquet(pq_path)
    total = len(df)
    original_correct = df["correct"].sum()

    # Add grading_method column if missing
    if "grading_method" not in df.columns:
        df["grading_method"] = df["correct"].apply(lambda x: "stage1" if x else "stage1_fail")

    # Find candidates for re-grading
    mask = (~df["correct"]) & (~df["predicted_answer"].isin(["no_boxed", "error"]))
    candidates = df[mask].index.tolist()
    print(f"  {model_name}: {original_correct}/{total} correct, {len(candidates)} to re-check")

    if not candidates:
        return {"model": model_name, "original": original_correct, "total": total,
                "math_verify_upgrades": 0, "llm_judge_upgrades": 0,
                "final": original_correct}

    # Stage 1b: Try math_verify
    math_verify_upgraded = 0
    still_need_judge = []
    for idx in candidates:
        predicted = df.at[idx, "predicted_answer"]
        ground_truth = df.at[idx, "ground_truth"]
        try:
            if grade_answer_math_verify(predicted, ground_truth):
                df.at[idx, "correct"] = True
                df.at[idx, "grading_method"] = "math_verify"
                math_verify_upgraded += 1
                continue
        except Exception:
            pass
        still_need_judge.append(idx)

    print(f"    math_verify upgraded: {math_verify_upgraded}, still need judge: {len(still_need_judge)}")

    # Stage 2: LLM judge
    llm_judge_upgraded = 0
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
                    llm_judge_upgraded += 1

    final_correct = df["correct"].sum()
    print(f"    LLM judge upgraded: {llm_judge_upgraded}")
    print(f"    Final: {final_correct}/{total} ({final_correct/total*100:.1f}%) [was {original_correct} ({original_correct/total*100:.1f}%)]")

    # Save updated parquet
    df.to_parquet(pq_path, index=False)

    return {
        "model": model_name, "original": int(original_correct), "total": total,
        "math_verify_upgrades": math_verify_upgraded,
        "llm_judge_upgrades": llm_judge_upgraded,
        "final": int(final_correct),
    }


async def main():
    print("Re-scoring MATH-500 with math_verify + LLM judge...\n")
    summaries = []
    for model in PLOT_MODELS:
        summary = await rescore_model(model)
        if summary:
            summaries.append(summary)
        print()

    print("=" * 70)
    print(f"{'Model':<45} {'Was':>5} {'Now':>5} {'MV':>4} {'LLM':>4} {'Total':>5}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['model']:<45} {s['original']:>5} {s['final']:>5} "
              f"{s['math_verify_upgrades']:>4} {s['llm_judge_upgrades']:>4} /{s['total']:>4}")


if __name__ == "__main__":
    asyncio.run(main())
