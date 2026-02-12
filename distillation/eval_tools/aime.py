"""AIME evaluation: 90 problems from AI-MO/aimo-validation-aime."""

import asyncio
import time
from pathlib import Path

from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MAX_TOKENS, EVAL_CONCURRENCY, MATH_SYSTEM_PROMPT
from infer import generate
from utils import parse_think_tags, save_results_parquet

sys.path.insert(0, '/workspace/tinker-cookbook')
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer


def grade_math_answer(model_output: str, ground_truth: str) -> tuple[bool, str]:
    """Grade a math answer by extracting boxed content and comparing."""
    try:
        predicted = extract_boxed(model_output)
    except (ValueError, Exception):
        return False, "no_boxed"
    try:
        correct = grade_answer(predicted, ground_truth)
        return correct, predicted
    except Exception:
        return False, predicted


async def run(sampling_client, renderer, tokenizer, results_dir: Path, model_name: str,
              think_prefix: bool = True, max_tokens: int = MAX_TOKENS,
              max_problems: int | None = None, **kwargs) -> dict:
    """Evaluate on AIME (90 problems)."""
    aime_ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    if max_problems is not None:
        aime_ds = aime_ds.select(range(min(max_problems, len(aime_ds))))
    total = len(aime_ds)
    print(f"[aime] Evaluating {model_name} on {total} problems")

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    counter = {"done": 0, "correct": 0}
    t0 = time.time()

    async def eval_one(i):
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": aime_ds[i]["problem"]},
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
        correct, predicted = (
            (False, "error") if completion.startswith("ERROR:")
            else grade_math_answer(completion, str(aime_ds[i]["answer"]))
        )

        counter["done"] += 1
        if correct:
            counter["correct"] += 1
        if counter["done"] % 10 == 0 or counter["done"] == total:
            elapsed = time.time() - t0
            pct = counter["correct"] / counter["done"] * 100
            print(f"  [aime] {counter['done']}/{total} done, "
                  f"{counter['correct']} correct ({pct:.1f}%) [{elapsed:.0f}s]")

        return {
            "model": model_name,
            "dataset": "aime",
            "question": aime_ds[i]["problem"],
            "raw_output": completion,
            "cot": cot,
            "user_output": user_output,
            "correct": correct,
            "predicted_answer": predicted,
            "ground_truth": str(aime_ds[i]["answer"]),
        }

    tasks = [eval_one(i) for i in range(total)]
    results = list(await asyncio.gather(*tasks))

    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / total
    print(f"[aime] FINAL: {n_correct}/{total} ({accuracy*100:.1f}%)")

    save_results_parquet(results, results_dir / "results_aime.parquet")
    return {"dataset": "aime", "accuracy": accuracy, "n_correct": n_correct, "total": total}
