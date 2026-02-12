"""
Generate SFT formatting examples using Qwen3-30B-A3B via OpenRouter.

Takes the first 10 questions from each dataset, generates responses using the
same system prompts as distillation training, and stores reasoning traces + user outputs.

The OpenRouter API returns reasoning in a separate `message.reasoning` field
(not <think> tags), so we extract both fields from the response.

Run:
  conda activate fuzzy-evals && python distillation/sft_formatting_examples/generate_examples.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
from dotenv import load_dotenv

# ── Paths & env ───────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-30b-a3b-thinking-2507"

DATASET_JSONS_DIR = PROJECT_ROOT / "dataset_jsons"
OUTPUT_DIR = SCRIPT_DIR / "results"

N_QUESTIONS = 10
MAX_TOKENS = 16384
TEMPERATURE = 0.7
CONCURRENCY = 5

# ── System prompts (same as distillation training) ────────────────────

MATH_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question, "
    "putting your final answer inside \\boxed{}"
)
CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question. "
    "Enclose your code within delimiters as follows: "
    "```python \n #YOUR CODE HERE \n ``` \n\n"
)
FUZZY_SYSTEM_PROMPT = "Answer the question"


# ── OpenRouter async call with retry ──────────────────────────────────

async def call_openrouter(session: aiohttp.ClientSession, messages: list[dict],
                          semaphore: asyncio.Semaphore) -> dict:
    """Call OpenRouter API with retry. Returns dict with 'reasoning' and 'content' fields."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    retry_delays = [2, 5, 15, 30, 60, 120]
    for delay in retry_delays + [120]:
        try:
            async with semaphore:
                async with session.post(
                    OPENROUTER_URL, headers=headers, json=body,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    data = await resp.json()

            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0]["message"]
                return {
                    "reasoning": msg.get("reasoning", None),
                    "content": msg.get("content", ""),
                }
            elif "error" in data:
                err = data["error"]
                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                if "rate" in err_msg.lower() or "429" in err_msg:
                    print(f"  Rate limited, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                return {"reasoning": None, "content": f"ERROR: {err_msg}"}
            return {"reasoning": None, "content": "ERROR: unexpected response"}
        except asyncio.TimeoutError:
            print(f"  Timeout, retrying in {delay}s...")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"  Exception: {e}, retrying in {delay}s...")
            await asyncio.sleep(delay)
    return {"reasoning": None, "content": "ERROR: max retries exceeded"}


# ── Dataset loaders ───────────────────────────────────────────────────

def load_math_500(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        questions.append({
            "question": row["problem"],
            "ground_truth": row["answer"],
        })
    return questions


def load_aime(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    questions = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        questions.append({
            "question": row["problem"],
            "ground_truth": str(row["answer"]),
        })
    return questions


def load_codeforces(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        name="codeforces", split="test", streaming=True,
    )
    questions = []
    for row in ds:
        question = row.get("question") or row.get("prompt") or row.get("problem")
        if not question:
            continue
        questions.append({"question": question})
        if len(questions) >= n:
            break
    return questions


def load_kodcode(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    tasks = []
    for row in ds:
        if row["gpt_difficulty"] != "easy":
            continue
        test = row["test"]
        if "from solution import" not in test:
            continue
        test_lower = test.lower()
        if any(lib in test_lower for lib in ["import numpy", "import pandas", "import torch"]):
            continue
        tasks.append({"question": row["question"]})
        if len(tasks) >= n:
            break
    return tasks


def load_fuzzy_weird_qs(n: int) -> list[dict]:
    with open(DATASET_JSONS_DIR / "weird_questions.json") as f:
        data = json.load(f)
    return [{"question": q["prompt"]} for q in data[:n]]


# ── Dataset registry ──────────────────────────────────────────────────

DATASETS = {
    "math_500":       {"loader": load_math_500,       "system_prompt": MATH_SYSTEM_PROMPT},
    "aime":           {"loader": load_aime,           "system_prompt": MATH_SYSTEM_PROMPT},
    "codeforces_500": {"loader": load_codeforces,     "system_prompt": CODE_SYSTEM_PROMPT},
    "kodcode_500":    {"loader": load_kodcode,         "system_prompt": CODE_SYSTEM_PROMPT},
    "fuzzy_weird_qs": {"loader": load_fuzzy_weird_qs, "system_prompt": FUZZY_SYSTEM_PROMPT},
}


# ── Main generation loop ─────────────────────────────────────────────

async def generate_for_dataset(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    dataset_name: str,
    questions: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Generate responses for all questions in one dataset."""

    async def gen_one(idx: int, q: dict) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["question"]},
        ]
        result = await call_openrouter(session, messages, semaphore)
        print(f"  [{dataset_name}] {idx+1}/{len(questions)} done")
        return {
            "dataset": dataset_name,
            "question": q["question"],
            "reasoning": result["reasoning"],
            "user_output": result["content"],
            "ground_truth": q.get("ground_truth", None),
        }

    tasks = [gen_one(i, q) for i, q in enumerate(questions)]
    return list(await asyncio.gather(*tasks))


async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set. Add it to .env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_results = {}

    async with aiohttp.ClientSession() as session:
        for ds_name, ds_cfg in DATASETS.items():
            print(f"\n{'='*60}")
            print(f"Loading {ds_name} (first {N_QUESTIONS} questions)...")
            t0 = time.time()

            questions = ds_cfg["loader"](N_QUESTIONS)
            print(f"  Loaded {len(questions)} questions, generating with {MODEL}...")

            results = await generate_for_dataset(
                session, semaphore, ds_name, questions, ds_cfg["system_prompt"],
            )

            # Save per-dataset parquet
            df = pd.DataFrame(results)
            out_path = OUTPUT_DIR / f"{ds_name}.parquet"
            df.to_parquet(out_path, index=False)

            elapsed = time.time() - t0
            print(f"  Saved {len(df)} rows to {out_path} [{elapsed:.1f}s]")
            all_results[ds_name] = results

    # Save combined parquet
    all_rows = [r for rows in all_results.values() for r in rows]
    combined_df = pd.DataFrame(all_rows)
    combined_path = OUTPUT_DIR / "all_datasets.parquet"
    combined_df.to_parquet(combined_path, index=False)
    print(f"\nSaved combined {len(combined_df)} rows to {combined_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    for ds_name, results in all_results.items():
        n_with_reasoning = sum(1 for r in results if r["reasoning"])
        n_errors = sum(1 for r in results if r["user_output"].startswith("ERROR:"))
        print(f"  {ds_name}: {len(results)} responses, {n_with_reasoning} with reasoning, {n_errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
