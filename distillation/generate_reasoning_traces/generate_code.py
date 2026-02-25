"""Generate coding reasoning traces via OpenRouter.

Calls Sonnet 4.5 (or thinking model) with coding problems and saves the full responses.
Resumable: skips already-processed question_ids.

Usage:
    python generate_code.py [--concurrency N] [--max-tokens N] [--model MODEL] [--effortful] [--thinking]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

# Import local config BEFORE adding parent to sys.path (parent also has config.py)
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    GENERATION_MODEL, THINKING_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL,
    CONCURRENCY, CODE_MAX_TOKENS, CODE_SYSTEM_PROMPT, EFFORTFUL_CODE_PROMPT,
    THINKING_MODEL_CODE_PROMPT, THINKING_REASONING_TOKENS, THINKING_CONTENT_TOKENS,
    CODE_DATA_DIR, CODE_QUESTIONS_PATH, CODE_GENERATIONS_PATH,
    EFFORTFUL_CODE_DATA_DIR, EFFORTFUL_CODE_QUESTIONS_PATH, EFFORTFUL_CODE_GENERATIONS_PATH,
    THINKING_CODE_DATA_DIR, THINKING_CODE_QUESTIONS_PATH, THINKING_CODE_GENERATIONS_PATH,
)

# Now add parent for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import call_openrouter, call_openrouter_reasoning


def load_done_ids(generations_path: Path) -> set[str]:
    """Load question_ids already processed (for resume support)."""
    done = set()
    if generations_path.exists():
        with open(generations_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add(row["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


async def generate_one(session, semaphore, question: dict, model: str,
                       max_tokens: int, system_prompt: str,
                       outfile, lock, counters: dict, thinking_mode: bool = False):
    """Generate a trace for one question and append to output file."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question["question"]},
    ]

    if thinking_mode:
        reasoning, content = await call_openrouter_reasoning(
            session, model, messages,
            temperature=0.0, max_tokens=THINKING_CONTENT_TOKENS,
            reasoning_max_tokens=THINKING_REASONING_TOKENS,
            semaphore=semaphore,
            api_key=OPENROUTER_API_KEY, api_url=OPENROUTER_URL,
        )
        if str(reasoning).startswith("ERROR:"):
            response = reasoning
        else:
            response = f"<think>\n{reasoning}\n</think>\n\n{content}"
    else:
        response = await call_openrouter(
            session, model, messages,
            temperature=0.0, max_tokens=max_tokens,
            semaphore=semaphore,
            api_key=OPENROUTER_API_KEY, api_url=OPENROUTER_URL,
        )

    result = {
        "question_id": question["question_id"],
        "question": question["question"],
        "test": question["test"],
        "solution": question.get("solution", ""),
        "generation": response,
        "model": model,
    }

    async with lock:
        outfile.write(json.dumps(result) + "\n")
        outfile.flush()
        counters["done"] += 1
        if counters["done"] % 100 == 0 or counters["done"] == counters["total"]:
            elapsed = time.time() - counters["start"]
            err_str = f" (errors so far: {counters['errors']})" if counters["errors"] else ""
            print(f"  [{counters['done']}/{counters['total']}] {elapsed:.0f}s{err_str}")

    if str(response).startswith("ERROR:"):
        counters["errors"] += 1

    return result


async def main_async(args):
    # Select paths and prompt based on flags
    if args.output_dir:
        data_dir = Path(args.output_dir)
        questions_path = data_dir / "questions.jsonl"
        generations_path = data_dir / "generations.jsonl"
        system_prompt = THINKING_MODEL_CODE_PROMPT
        model = args.model if args.model != GENERATION_MODEL else THINKING_MODEL
        print(f"Using custom output dir: {data_dir}")
        print(f"Using model: {model}")
    elif args.thinking:
        data_dir = THINKING_CODE_DATA_DIR
        questions_path = THINKING_CODE_QUESTIONS_PATH
        generations_path = THINKING_CODE_GENERATIONS_PATH
        system_prompt = THINKING_MODEL_CODE_PROMPT
        model = args.model if args.model != GENERATION_MODEL else THINKING_MODEL
        print(f"Using THINKING model: {model}")
    elif args.effortful:
        data_dir = EFFORTFUL_CODE_DATA_DIR
        questions_path = EFFORTFUL_CODE_QUESTIONS_PATH
        generations_path = EFFORTFUL_CODE_GENERATIONS_PATH
        system_prompt = EFFORTFUL_CODE_PROMPT
        model = args.model
        print("Using EFFORTFUL code prompt")
    else:
        data_dir = CODE_DATA_DIR
        questions_path = CODE_QUESTIONS_PATH
        generations_path = CODE_GENERATIONS_PATH
        system_prompt = CODE_SYSTEM_PROMPT
        model = args.model

    data_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    if not questions_path.exists():
        print(f"ERROR: {questions_path} not found. Run extract_questions_code.py first.")
        sys.exit(1)

    questions = []
    with open(questions_path) as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Resume support
    done_ids = load_done_ids(generations_path)
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} already done, {len(remaining)} remaining")

    if not remaining:
        print("All questions already processed!")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counters = {
        "done": 0, "total": len(remaining), "errors": 0,
        "start": time.time(),
    }

    print(f"Generating with {model}, concurrency={args.concurrency}, "
          f"max_tokens={args.max_tokens}")

    outfile = open(generations_path, "a")
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                generate_one(session, semaphore, q, model,
                             args.max_tokens, system_prompt,
                             outfile, lock, counters,
                             thinking_mode=args.thinking)
                for q in remaining
            ]
            await asyncio.gather(*tasks)
    finally:
        outfile.close()

    elapsed = time.time() - counters["start"]
    print(f"\nDone: {counters['done']} generations in {elapsed:.0f}s "
          f"({counters['errors']} errors)")


def main():
    parser = argparse.ArgumentParser(description="Generate coding traces via OpenRouter")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max-tokens", type=int, default=CODE_MAX_TOKENS)
    parser.add_argument("--model", type=str, default=GENERATION_MODEL)
    parser.add_argument("--effortful", action="store_true",
                        help="Use effortful code prompt and write to data_code_effortful/")
    parser.add_argument("--thinking", action="store_true",
                        help="Use thinking model with reasoning API and write to data_code_thinking/")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (must contain questions.jsonl)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
