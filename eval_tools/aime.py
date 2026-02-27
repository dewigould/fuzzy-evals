"""AIME evaluation: 90 problems from AI-MO/aimo-validation-aime.

Two-stage verification (same as math_500 / omni_math):
  1. Extract \\boxed{} answer, run through grade_answer (normalize + sympy)
     and grade_answer_math_verify (math_verify package). If either passes, correct.
  2. If stage 1 fails but we extracted an answer, call GPT-5.2 judge to check
     equivalence accounting for different variables/formalism/formatting.
"""

import asyncio
import random
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset

EVAL_SEED = 42

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAX_TOKENS, EVAL_CONCURRENCY, MATH_SYSTEM_PROMPT,
    JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL,
)
from infer import generate
from utils import parse_think_tags, save_results_parquet, load_or_build_cache, call_openrouter

sys.path.insert(0, '/workspace/tinker-cookbook')
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

# Import shared two-stage grading from math_500
from eval_tools.math_500 import grade_math_answer_stage1, judge_answer_llm


async def run(sampling_client, renderer, tokenizer, results_dir: Path, model_name: str,
              think_prefix: bool = True, max_tokens: int = MAX_TOKENS,
              max_problems: int | None = None, **kwargs) -> dict:
    """Evaluate on AIME (90 problems)."""
    def _build():
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        indices = list(range(len(ds)))
        random.Random(EVAL_SEED).shuffle(indices)
        return [{"problem": ds[i]["problem"], "answer": str(ds[i]["answer"])} for i in indices]

    problems = load_or_build_cache("aime", _build, EVAL_SEED)
    if max_problems is not None:
        problems = problems[:min(max_problems, len(problems))]
    total = len(problems)
    print(f"[aime] Evaluating {model_name} on {total} problems")

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    counter = {"done": 0, "correct": 0, "need_judge": 0}
    t0 = time.time()

    # Phase 1: Generate + stage 1 grading
    async def eval_one(i):
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": problems[i]["problem"]},
        ]
        async with semaphore:
            try:
                completions = await generate(
                    sampling_client, renderer, tokenizer,
                    messages=messages, max_tokens=max_tokens,
                    temperature=0.0, think_prefix=think_prefix,
                )
                completion = completions[0]
            except Exception as e:
                completion = f"ERROR: {e}"

        cot, user_output = parse_think_tags(completion)

        if completion.startswith("ERROR:"):
            correct, predicted, method = False, "error", "error"
        else:
            correct, predicted = grade_math_answer_stage1(completion, problems[i]["answer"])
            method = "stage1" if correct else "stage1_fail"

        counter["done"] += 1
        if correct:
            counter["correct"] += 1
        if not correct and predicted not in ("no_boxed", "error"):
            counter["need_judge"] += 1
        if counter["done"] % 10 == 0 or counter["done"] == total:
            elapsed = time.time() - t0
            pct = counter["correct"] / counter["done"] * 100
            print(f"  [aime] {counter['done']}/{total} done, "
                  f"{counter['correct']} correct ({pct:.1f}%), "
                  f"{counter['need_judge']} pending judge [{elapsed:.0f}s]")

        return {
            "model": model_name,
            "dataset": "aime",
            "question": problems[i]["problem"],
            "raw_output": completion,
            "cot": cot,
            "user_output": user_output,
            "correct": correct,
            "predicted_answer": predicted,
            "ground_truth": problems[i]["answer"],
            "grading_method": method,
        }

    tasks = [eval_one(i) for i in range(total)]
    results = list(await asyncio.gather(*tasks))

    stage1_correct = sum(1 for r in results if r["correct"])
    need_judge = [i for i, r in enumerate(results)
                  if not r["correct"] and r["predicted_answer"] not in ("no_boxed", "error")]
    print(f"  [aime] Stage 1: {stage1_correct}/{total} correct, "
          f"{len(need_judge)} need LLM judge")

    # Phase 2: LLM judge
    judge_upgraded = 0
    if need_judge:
        judge_sem = asyncio.Semaphore(EVAL_CONCURRENCY)
        async with aiohttp.ClientSession() as session:
            async def judge_one(i):
                return i, await judge_answer_llm(
                    session, judge_sem,
                    question=results[i]["question"],
                    expected=results[i]["ground_truth"],
                    predicted=results[i]["predicted_answer"],
                )

            judge_tasks = [judge_one(i) for i in need_judge]
            judge_results = await asyncio.gather(*judge_tasks)

            for i, is_correct in judge_results:
                if is_correct:
                    results[i]["correct"] = True
                    results[i]["grading_method"] = "llm_judge"
                    judge_upgraded += 1

        elapsed_judge = time.time() - t0
        print(f"  [aime] LLM judge upgraded {judge_upgraded} answers [{elapsed_judge:.0f}s]")

    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / total
    print(f"[aime] FINAL: {n_correct}/{total} ({accuracy*100:.1f}%)")

    save_results_parquet(results, results_dir / "results_aime.parquet")
    return {
        "dataset": "aime", "accuracy": accuracy,
        "n_correct": n_correct, "total": total,
        "judge_upgraded": judge_upgraded,
    }
