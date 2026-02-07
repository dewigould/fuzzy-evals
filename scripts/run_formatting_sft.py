"""
Formatting SFT: Apply 10 steps of SFT on formatting examples to all 5 models.
- base (fresh LoRA from base model)
- math_rlvr, code_rlvr, math_sft, code_sft (load from training checkpoints)
"""

import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from tinker_cookbook import model_info, checkpoint_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR = Path('/workspace/results_06_02_v2')
FORMATTING_DATA = RESULTS_DIR / 'data' / 'formatting_combined.jsonl'

# Models: key -> (checkpoint_log_path or None for base)
MODELS = {
    'base': None,
    'math_rlvr': RESULTS_DIR / 'training' / 'math_rlvr',
    'code_rlvr': RESULTS_DIR / 'training' / 'code_rlvr',
    'math_sft': RESULTS_DIR / 'training' / 'math_sft',
    'code_sft': RESULTS_DIR / 'training' / 'code_sft',
}


def get_checkpoint_path(model_key: str) -> str | None:
    """Get the state_path from a training log directory's checkpoints."""
    log_path = MODELS[model_key]
    if log_path is None:
        return None  # base model
    resume_info = checkpoint_utils.get_last_checkpoint(str(log_path))
    return resume_info['state_path']


async def run_formatting_sft_for_model(model_key: str):
    """Run 10 steps of formatting SFT for one model."""
    log.info(f"=== Formatting SFT for: {model_key} ===")

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    output_dir = RESULTS_DIR / 'training' / 'format_sft' / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=10,  # 100 examples / 10 = 10 batches = 10 steps
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(FORMATTING_DATA),
    )

    checkpoint_path = get_checkpoint_path(model_key) if model_key != 'base' else None

    config = train.Config(
        log_path=str(output_dir),
        model_name=MODEL_NAME,
        load_checkpoint_path=checkpoint_path,
        dataset_builder=dataset_builder,
        learning_rate=5e-5,
        lr_schedule="constant",
        num_epochs=1,
        lora_rank=32,
        save_every=0,  # Save handled at end
        eval_every=0,
    )

    await train.main(config)
    log.info(f"=== Done formatting SFT for: {model_key} ===")


async def main():
    # Run formatting SFT for each model sequentially
    for model_key in MODELS:
        log_path = RESULTS_DIR / 'training' / 'format_sft' / model_key
        # Skip if already done (checkpoint exists)
        ckpt_file = log_path / 'checkpoints.jsonl'
        if ckpt_file.exists():
            log.info(f"Skipping {model_key}: already done ({ckpt_file})")
            continue
        # Skip if training not yet finished (no source checkpoint)
        if model_key != 'base':
            src_ckpt = MODELS[model_key] / 'checkpoints.jsonl'
            if not src_ckpt.exists():
                log.info(f"Skipping {model_key}: training not yet complete")
                continue
        await run_formatting_sft_for_model(model_key)


if __name__ == '__main__':
    asyncio.run(main())
