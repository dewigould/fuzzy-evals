"""Philosophy fuzzy evaluation: 10 questions, judged via GPT-5.2 rubric."""

import asyncio
import json
import time
from pathlib import Path

import aiohttp

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAX_TOKENS, EVAL_CONCURRENCY, FUZZY_SYSTEM_PROMPT,
    FUZZY_N_SAMPLES, JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL,
    FUZZY_DATA_DIR, RUBRICS_DIR,
)
from infer import generate
from utils import parse_think_tags, save_results_parquet, call_openrouter, parse_judge_score

DATASET_NAME = "fuzzy_philosophy"
QUESTION_KEY = "question"
RUBRIC_KEYS = [
    "thesis_clarity", "argumentative_soundness", "dialectical_engagement",
    "precision_distinctions", "substantive_contribution", "example_quality",
]
JUDGE_SYS_MSG = "You are grading a philosophical answer for quality of argumentation and exposition."


async def run(sampling_client, renderer, tokenizer, results_dir: Path, model_name: str,
              think_prefix: bool = True, max_tokens: int = MAX_TOKENS,
              max_problems: int | None = None, fuzzy_samples: int | None = None,
              **kwargs) -> dict:
    """Evaluate on philosophy questions."""
    with open(FUZZY_DATA_DIR / "philosophy_questions.json") as f:
        questions = json.load(f)
    if max_problems is not None:
        questions = questions[:max_problems]
    rubric = (RUBRICS_DIR / "philosophy_rubric.md").read_text()

    n_samples = fuzzy_samples if fuzzy_samples is not None else FUZZY_N_SAMPLES
    total_samples = len(questions) * n_samples
    print(f"[{DATASET_NAME}] Evaluating {model_name}: {len(questions)}Q x {n_samples}S = {total_samples}")

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    results = []
    t0 = time.time()

    # Generate all responses
    async def gen_question(q_idx, q):
        messages = [
            {"role": "system", "content": FUZZY_SYSTEM_PROMPT},
            {"role": "user", "content": q[QUESTION_KEY]},
        ]
        async with semaphore:
            try:
                samples = await generate(
                    sampling_client, renderer, tokenizer,
                    messages=messages, max_tokens=max_tokens,
                    temperature=0.7, num_samples=n_samples,
                    think_prefix=think_prefix,
                )
                return list(samples)
            except Exception as e:
                return [f"ERROR: {e}"] * n_samples

    gen_tasks = [gen_question(i, q) for i, q in enumerate(questions)]
    all_completions = list(await asyncio.gather(*gen_tasks))

    elapsed_gen = time.time() - t0
    print(f"  [{DATASET_NAME}] Generation done in {elapsed_gen:.0f}s, judging...")

    # Judge all responses
    async with aiohttp.ClientSession() as session:
        for q_idx, q in enumerate(questions):
            for s_idx, completion in enumerate(all_completions[q_idx]):
                cot, user_output = parse_think_tags(completion)

                if completion.startswith("ERROR:"):
                    row = {
                        "model": model_name,
                        "dataset": DATASET_NAME,
                        "question": q[QUESTION_KEY],
                        "raw_output": completion,
                        "cot": cot,
                        "user_output": user_output,
                        "total_score": None,
                        "judge_raw": "",
                    }
                    for key in RUBRIC_KEYS:
                        row[key] = None
                    results.append(row)
                    continue

                grading_prompt = f"""{JUDGE_SYS_MSG}

## Question

{q[QUESTION_KEY]}

## Answer to Grade

{completion}

## Grading Rubric

{rubric}"""
                judge_resp = await call_openrouter(
                    session, JUDGE_MODEL,
                    [{"role": "user", "content": grading_prompt}],
                    temperature=0.0, max_tokens=1024,
                    semaphore=semaphore,
                    api_key=OPENROUTER_API_KEY, api_url=OPENROUTER_URL,
                )

                total, scores = parse_judge_score(judge_resp) if not str(judge_resp).startswith("ERROR:") else (None, {})

                row = {
                    "model": model_name,
                    "dataset": DATASET_NAME,
                    "question": q[QUESTION_KEY],
                    "raw_output": completion,
                    "cot": cot,
                    "user_output": user_output,
                    "total_score": total,
                    "judge_raw": str(judge_resp),
                }
                for key in RUBRIC_KEYS:
                    row[key] = scores.get(key, None)
                results.append(row)

            done_q = q_idx + 1
            elapsed = time.time() - t0
            print(f"  [{DATASET_NAME}] {done_q}/{len(questions)} questions judged [{elapsed:.0f}s]")

    # Compute summary
    valid_scores = [r["total_score"] for r in results if r["total_score"] is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"[{DATASET_NAME}] FINAL: mean={mean_score:.2f} (n={len(valid_scores)})")

    save_results_parquet(results, results_dir / f"results_{DATASET_NAME}.parquet")
    return {"dataset": DATASET_NAME, "mean_score": mean_score, "n_scored": len(valid_scores)}
