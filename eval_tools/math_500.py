"""MATH-500 evaluation: 500 problems from HuggingFaceH4/MATH-500.

Two-stage verification (same as omni_math):
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


def grade_math_answer_stage1(model_output: str, ground_truth: str) -> tuple[bool, str]:
    """Stage 1: extract boxed answer and grade with math grading + math_verify.

    Returns (correct, predicted_answer).
    """
    try:
        predicted = extract_boxed(model_output)
    except (ValueError, Exception):
        return False, "no_boxed"

    # Try grade_answer (normalize + sympy)
    try:
        if grade_answer(predicted, ground_truth):
            return True, predicted
    except Exception:
        pass

    # Try math_verify as fallback
    try:
        from tinker_cookbook.recipes.math_rl.math_grading import grade_answer_math_verify
        if grade_answer_math_verify(predicted, ground_truth):
            return True, predicted
    except Exception:
        pass

    return False, predicted


async def judge_answer_llm(session, semaphore, question: str, expected: str,
                           predicted: str) -> bool:
    """Stage 2: Use GPT-5.2 to judge mathematical equivalence."""
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


async def run(sampling_client, renderer, tokenizer, results_dir: Path, model_name: str,
              think_prefix: bool = True, max_tokens: int = MAX_TOKENS,
              max_problems: int | None = None, **kwargs) -> dict:
    """Evaluate on MATH-500."""
    def _build():
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        indices = list(range(len(ds)))
        random.Random(EVAL_SEED).shuffle(indices)
        return [{"problem": ds[i]["problem"], "answer": ds[i]["answer"]} for i in indices]

    problems = load_or_build_cache("math_500", _build, EVAL_SEED)
    if max_problems is not None:
        problems = problems[:min(max_problems, len(problems))]
    total = len(problems)
    print(f"[math_500] Evaluating {model_name} on {total} problems")

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
                gen_kwargs = dict(
                    messages=messages, max_tokens=max_tokens,
                    temperature=kwargs.get("temperature", 0.0),
                    think_prefix=think_prefix,
                )
                if kwargs.get("top_p") is not None:
                    gen_kwargs["top_p"] = kwargs["top_p"]
                completions = await generate(
                    sampling_client, renderer, tokenizer,
                    **gen_kwargs,
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
        if counter["done"] % 50 == 0 or counter["done"] == total:
            elapsed = time.time() - t0
            pct = counter["correct"] / counter["done"] * 100
            print(f"  [math_500] {counter['done']}/{total} done, "
                  f"{counter['correct']} correct ({pct:.1f}%), "
                  f"{counter['need_judge']} pending judge [{elapsed:.0f}s]")

        return {
            "model": model_name,
            "dataset": "math_500",
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
    print(f"  [math_500] Stage 1: {stage1_correct}/{total} correct, "
          f"{len(need_judge)} need LLM judge")

    # Phase 2: LLM judge for stage1 failures with an extracted answer
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
        print(f"  [math_500] LLM judge upgraded {judge_upgraded} answers [{elapsed_judge:.0f}s]")

    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / total
    print(f"[math_500] FINAL: {n_correct}/{total} ({accuracy*100:.1f}%)")

    save_results_parquet(results, results_dir / "results_math_500.parquet")
    return {
        "dataset": "math_500", "accuracy": accuracy,
        "n_correct": n_correct, "total": total,
        "judge_upgraded": judge_upgraded,
    }
