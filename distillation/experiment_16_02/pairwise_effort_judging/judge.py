"""Async ensemble effort judging via OpenRouter."""

import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

import aiohttp

# Import call_openrouter from distillation/utils.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import call_openrouter

from config import (
    JUDGE_MODELS,
    JUDGE_SHORT_NAMES,
    JUDGE_CONCURRENCY,
    JUDGE_MAX_TOKENS,
    JUDGE_TEMPERATURE,
    OPENROUTER_API_KEY,
)

JUDGE_SYSTEM_PROMPT = "You are an expert at evaluating the effort and thoroughness of reasoning."

JUDGE_PROMPT_TEMPLATE = """You are comparing two reasoning traces from the same question to determine which one demonstrates more genuine effort and thoroughness in reasoning.

**Question:**
{question}

**Response A:**
{response_a}

**Response B:**
{response_b}

**Criteria for "trying harder" — focus on the PROCESS, not the product:**
- Attempts to break down the problem into sub-problems or steps
- Shows awareness of its own uncertainty or limitations
- Tries to verify, cross-check, or sanity-check its reasoning
- Explores multiple approaches before concluding
- Self-corrects in a way that makes PROGRESS (the reasoning moves forward after the correction, rather than cycling back to where it started)
- Builds toward an answer methodically rather than jumping to conclusions

**Key distinction — genuine effort vs performative effort:**
Genuine effort MAKES PROGRESS toward an answer — each step builds on the last. Performative effort restates the problem in different ways, proposes approaches without executing them, or cycles through the same ideas repeatedly. A response that says "let's try X" five times without ever completing X is showing LESS effort than one that picks a single approach and follows through, even if the latter is shorter.

**These are NOT indicators of more effort (do not use them to decide):**
- Having more specific numbers or statistics (that's recall, not effort)
- Citing more named sources or studies
- Being longer (length can come from filler or repetition)
- Better formatting or organization (that's presentation, not effort)
- Being cut off or truncated early (a response may still demonstrate strong effort before being cut short)

**Disqualifying pattern:**
If a response contains large blocks of repeated or near-repeated text, looping paragraphs, or obvious filler, it should be considered LESS effortful regardless of length. A shorter, clean response demonstrates more genuine effort than a longer degenerate one.

**Important:**
You are **NOT** judging which answer is correct, and you are **NOT** judging which answer contains more specific knowledge. You are ONLY judging which response shows more genuine reasoning effort.

**Instructions:**
Compare the two responses and determine which one demonstrates more effort/thoroughness.

- If Response A clearly tries harder, output: JUDGMENT: A
- If Response B clearly tries harder, output: JUDGMENT: B
- If both show similar levels of effort, output: JUDGMENT: Neither

You should pick A or B if one response shows meaningfully more reasoning effort. If both responses show similar levels of reasoning effort — even if one is higher quality or more specific — choose Neither.

First, briefly explain your reasoning (2-3 sentences), then give your judgment on the last line in the exact format above."""


def parse_judgment(text: str) -> str | None:
    """Extract judgment (A, B, or Neither) from judge response text."""
    match = re.search(r"JUDGMENT:\s*(A|B|Neither)", text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if val.upper() == "A":
            return "A"
        elif val.upper() == "B":
            return "B"
        else:
            return "Neither"

    # Fallback: check last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1].upper()
        if "A" in last and "B" not in last and "NEITHER" not in last:
            return "A"
        elif "B" in last and "A" not in last and "NEITHER" not in last:
            return "B"
        elif "NEITHER" in last:
            return "Neither"

    return None


def compute_ensemble(winners: list[str | None]) -> tuple[str, float]:
    """Majority vote over valid judge winners. Returns (winner, confidence)."""
    valid = [w for w in winners if w is not None]
    if not valid:
        return "error", 0.0
    counts = Counter(valid)
    winner, count = counts.most_common(1)[0]
    confidence = count / len(valid)
    return winner, round(confidence, 3)


async def judge_single(
    session: aiohttp.ClientSession,
    model: str,
    question: str,
    response_a: str,
    response_b: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single judge model on one pair."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        response_a=response_a or "(empty response)",
        response_b=response_b or "(empty response)",
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    raw = await call_openrouter(
        session=session,
        model=model,
        messages=messages,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        semaphore=semaphore,
        api_key=OPENROUTER_API_KEY,
    )

    if raw is None or (isinstance(raw, str) and raw.startswith("ERROR:")):
        return {"model": model, "winner": None, "reasoning": "", "raw": raw or ""}

    winner = parse_judgment(raw)
    return {"model": model, "winner": winner, "reasoning": raw, "raw": raw}


async def judge_pair_ensemble(
    session: aiohttp.ClientSession,
    question: str,
    response_a: str,
    response_b: str,
    semaphores: dict[str, asyncio.Semaphore],
) -> dict:
    """Run all judges on one pair, return flat result dict."""
    results = await asyncio.gather(*[
        judge_single(session, model, question, response_a, response_b, semaphores[model])
        for model in JUDGE_MODELS
    ])

    winners = [r["winner"] for r in results]
    effort_judgment, confidence = compute_ensemble(winners)

    row = {
        "effort_judgment": effort_judgment,
        "judge_confidence": confidence,
    }
    for r in results:
        short = JUDGE_SHORT_NAMES[r["model"]]
        row[f"judge_{short}_winner"] = r["winner"]
        row[f"judge_{short}_reasoning"] = r["reasoning"]

    return row


async def run_pairwise_judging(
    pairs: list[dict],
    concurrency: int = JUDGE_CONCURRENCY,
) -> list[dict]:
    """
    Judge all pairs using the ensemble. Each pair dict must have:
        - problem_text: str (the question)
        - response_a: str
        - response_b: str
        - ... any other metadata fields (passed through to output)

    Returns list of dicts with metadata + judging columns.
    """
    semaphores = {model: asyncio.Semaphore(concurrency) for model in JUDGE_MODELS}
    report_interval = max(1, len(pairs) // 10)
    results = []
    completed = 0

    async with aiohttp.ClientSession() as session:
        async def process_one(pair: dict) -> dict:
            nonlocal completed
            judgment = await judge_pair_ensemble(
                session,
                pair["problem_text"],
                pair["response_a"],
                pair["response_b"],
                semaphores,
            )
            # Merge metadata (excluding problem_text) with judgment
            row = {k: v for k, v in pair.items() if k != "problem_text"}
            row.update(judgment)

            completed += 1
            if completed % report_interval == 0 or completed == len(pairs):
                print(f"  Judged {completed}/{len(pairs)} pairs")

            return row

        tasks = [process_one(p) for p in pairs]
        results = await asyncio.gather(*tasks)

    return list(results)
