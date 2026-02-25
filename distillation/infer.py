"""Central tinker inference module.

Provides a single generate() function used by all eval scripts.
Supports base model and checkpoint-based sampling clients.

Think prefix seeding:
  When think_prefix=True, <think>\\n is inserted as the first tokens of the
  assistant turn via the renderer's prefill parameter. This biases the model
  into the <think>...</think> format. The model's decoded output is returned
  exactly as generated — no post-hoc prepending.
"""

import asyncio
from typing import Optional

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import MODEL_NAME, MAX_TOKENS


def setup_renderer_and_tokenizer(model_name: str = MODEL_NAME):
    """Create renderer and tokenizer for the model."""
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return renderer, tokenizer


def create_base_client(model_name: str = MODEL_NAME):
    """Create a sampling client for the base model."""
    sc = tinker.ServiceClient()
    return sc.create_sampling_client(base_model=model_name)


def create_checkpoint_client(sampler_path: str):
    """Create a sampling client for a fine-tuned checkpoint."""
    sc = tinker.ServiceClient()
    return sc.create_sampling_client(model_path=sampler_path)


def _content_to_string(content) -> str:
    """Convert parse_response content to a flat string.

    The Qwen3 renderer's parse_response() returns content as either:
      - a plain str (when no <think> tags found in output)
      - a list of ContentPart TypedDicts (ThinkingPart / TextPart) when
        <think>...</think> blocks are present

    ContentPart items are TypedDicts (plain dicts) with a "type" discriminator:
      ThinkingPart: {"type": "thinking", "thinking": str}
      TextPart:     {"type": "text", "text": str}
    """
    if isinstance(content, str):
        return content

    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "thinking":
                parts.append(f"<think>{part['thinking']}</think>")
            elif part.get("type") == "text":
                parts.append(part["text"])
            else:
                parts.append(str(part))
        elif hasattr(part, "thinking"):
            parts.append(f"<think>{part.thinking}</think>")
        elif hasattr(part, "text"):
            parts.append(part.text)
        else:
            parts.append(str(part))
    return "".join(parts)


async def generate(
    sampling_client,
    renderer,
    tokenizer,
    prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    num_samples: int = 1,
    think_prefix: bool = True,
) -> list[str]:
    """Generate completions from a model, optionally seeding with <think>\\n.

    Args:
        sampling_client: Tinker sampling client (base or checkpoint).
        renderer: Model renderer for tokenization.
        tokenizer: HF tokenizer for encoding think prefix.
        prompt: Simple text prompt (used if messages is None).
        messages: Full message list [{role, content}, ...].
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        num_samples: Number of completions to generate.
        think_prefix: Whether to seed the assistant turn with <think>\\n.

    Returns:
        List of generated text strings (model output returned as-is).
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    # Build the generation prompt. When think_prefix is True, use the
    # renderer's prefill parameter to insert <think>\n as the first
    # assistant content — this is the correct structural position.
    prefill = "<think>\n" if think_prefix else None
    model_input = renderer.build_generation_prompt(messages, prefill=prefill)

    params = dict(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    if top_p is not None:
        params["top_p"] = top_p
    if top_k is not None:
        params["top_k"] = top_k

    response = await sampling_client.sample_async(
        model_input,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(**params),
    )

    # Decode responses. The model output is returned exactly as generated —
    # no post-hoc modification. If parse_response returns ContentPart dicts,
    # we faithfully reconstruct the text with <think>...</think> tags.
    results = []
    for seq in response.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = _content_to_string(parsed_msg["content"])
        results.append(content)

    return results
