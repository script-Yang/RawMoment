# SPDX-License-Identifier: Apache-2.0
"""Build the RawMoment latent adaptation and restoration models."""

import torch
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Transformer2DModel,
)
from safetensors.torch import load_file

from ..checkpoints import load_weights
from .stage2 import ConditionalDenoiser, Stage2Model
from .vae import Stage1Model


def build_stage1(config, load_trained=True, perceptual=False):
    vae = AutoencoderKLQwenImage21.from_pretrained(
        config["model"]["base"],
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    model = Stage1Model(vae, loss_weights=config.get("loss"))
    if load_trained:
        load_weights(model, "stage1", config["model"]["stage1_weights"])
    if perceptual and config["loss"].get("lpips", 0) > 0:
        from ..perceptual import VGGPerceptualLoss

        model.perceptual_loss = VGGPerceptualLoss(config.get("perceptual", {}).get("vgg_weights"))
    return model.to(config["device"])


def build_stage2(config, load_trained=False):
    options = config["model"]
    stage1 = build_stage1(config)
    prompt = load_file(config["prompt"]["cache"])["embeddings"]
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        options["base"],
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    denoiser = ConditionalDenoiser(
        transformer,
        prompt,
        rank=options["lora_rank"],
        alpha=options["lora_alpha"],
        spatial_width=options["spatial_width"],
        spatial_patch_size=options["spatial_patch_size"],
        physics_tokens=options["physics_tokens"],
    )
    if options.get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        options["base"], subfolder="scheduler"
    )
    model = Stage2Model(stage1, denoiser, scheduler).to(config["device"])
    if load_trained:
        load_weights(model, "stage2", options["stage2_weights"])
    return model
