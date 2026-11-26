# SPDX-License-Identifier: Apache-2.0
"""Regenerate the fixed prompt cache; inference itself does not load the text encoder."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

DEFAULT_PROMPT = "A sharp natural image with clear edges and fine details."


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", default="assets/prompt.safetensors")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    base = Path(args.base)
    encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            base / "text_encoder",
            dtype=torch.bfloat16,
        )
        .eval()
        .requires_grad_(False)
        .to(args.device)
    )
    processor = Qwen3VLProcessor.from_pretrained(base / "processor")
    pipeline = QwenImage21Pipeline(
        scheduler=None, vae=None, text_encoder=encoder, processor=processor, transformer=None
    )
    embeddings, _, _ = pipeline.encode_prompt(DEFAULT_PROMPT, device=torch.device(args.device))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"embeddings": embeddings.cpu().contiguous()},
        args.output,
        metadata={"prompt": DEFAULT_PROMPT, "base_model": "Qwen-Image-2.1"},
    )


if __name__ == "__main__":
    main()
