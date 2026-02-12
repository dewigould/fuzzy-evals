"""Shared utilities: think-tag parsing, parquet I/O, code extraction."""

import json
import re
from pathlib import Path

import pandas as pd


# ── Think tag parsing ────────────────────────────────────────────────────────

def parse_think_tags(raw_output: str) -> tuple[str | None, str]:
    """Parse <think>...</think> from model output.

    Returns (cot, user_output):
      - If </think> is present: cot = content inside tags, user_output = rest
      - Otherwise: cot = None, user_output = raw_output
    """
    if "</think>" not in raw_output:
        return None, raw_output

    # Find the last </think> to handle nested or malformed tags
    idx = raw_output.rfind("</think>")
    think_content = raw_output[:idx]
    user_output = raw_output[idx + len("</think>"):].strip()

    # Strip leading <think> tag if present
    if think_content.startswith("<think>"):
        think_content = think_content[len("<think>"):]
    think_content = think_content.strip()

    return think_content, user_output


# ── Code extraction ──────────────────────────────────────────────────────────

def extract_code(text: str) -> str | None:
    """Extract the last fenced code block from text."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return None


def extract_code_from_response(response: str) -> str | None:
    """Extract code from model response, handling <think>...</think> wrapper."""
    _, user_output = parse_think_tags(response)
    return extract_code(user_output)


# ── Parquet I/O ──────────────────────────────────────────────────────────────

def save_results_parquet(results: list[dict], path: str | Path):
    """Save results list as a parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return df


def load_results_parquet(path: str | Path) -> pd.DataFrame:
    """Load results from a parquet file."""
    return pd.read_parquet(path)


# ── OpenRouter API ───────────────────────────────────────────────────────────

async def call_openrouter(session, model, messages, temperature=0.0,
                          max_tokens=1024, semaphore=None, api_key=None,
                          api_url="https://openrouter.ai/api/v1/chat/completions"):
    """Call OpenRouter API with retry logic."""
    import asyncio
    import aiohttp

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    retry_delays = [2, 5, 15, 30, 60, 120]
    for delay in retry_delays + [120]:
        try:
            if semaphore:
                async with semaphore:
                    async with session.post(api_url, headers=headers, json=body,
                                           timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        data = await resp.json()
            else:
                async with session.post(api_url, headers=headers, json=body,
                                       timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    data = await resp.json()

            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            elif "error" in data:
                err = data["error"]
                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                if "rate" in err_msg.lower() or "429" in err_msg:
                    await asyncio.sleep(delay)
                    continue
                return f"ERROR: {err_msg}"
            return "ERROR: unexpected response"
        except asyncio.TimeoutError:
            await asyncio.sleep(delay)
        except Exception:
            await asyncio.sleep(delay)
    return "ERROR: max retries exceeded"


def parse_judge_score(response_text: str) -> tuple[int | None, dict]:
    """Parse JSON judge scores from response text."""
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        scores = json.loads(text)
        total = scores.get("total", None)
        if total is not None:
            return int(total), scores
        total = sum(v for v in scores.values() if isinstance(v, (int, float)))
        return total, scores
    except Exception:
        return None, {}
