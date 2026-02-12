"""Central tinker inference module.

Provides a single generate() function used by all eval scripts.
Supports base model and checkpoint-based sampling clients.
Always prepends <think>\\n tokens to seed reasoning.
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


async def generate(
    sampling_client,
    renderer,
    tokenizer,
    prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
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
        think_prefix: Whether to prepend <think>\\n tokens.

    Returns:
        List of generated text strings.
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    model_input = renderer.build_generation_prompt(messages)

    if think_prefix and tokenizer is not None:
        prefix_tokens = tokenizer.encode("<think>\n", add_special_tokens=False)
        for token_id in prefix_tokens:
            model_input.append_int(token_id)

    response = await sampling_client.sample_async(
        model_input,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
        ),
    )

    results = []
    for seq in response.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = parsed_msg["content"]
        # parse_response may return content as a list of ContentPart objects
        # (ThinkingPart, TextPart) when the output contains <think> tags.
        # Convert to a flat string in that case.
        if isinstance(content, list):
            parts = []
            for part in content:
                if hasattr(part, "thinking"):
                    parts.append(f"<think>{part.thinking}</think>")
                elif hasattr(part, "text"):
                    parts.append(part.text)
                else:
                    parts.append(str(part))
            content = "".join(parts)
        if think_prefix:
            content = "<think>\n" + content
        results.append(content)

    return results
