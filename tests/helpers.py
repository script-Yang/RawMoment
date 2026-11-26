# SPDX-License-Identifier: Apache-2.0
import torch
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Transformer2DModel,
)

from rawmoment.models.stage2 import ConditionalDenoiser, Stage2Model
from rawmoment.models.vae import Stage1Model


def tiny_model():
    vae = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=64,
        dim_mult=[1, 2, 2, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False] * 4,
        latents_mean=[0.1] * 64,
        latents_std=[2.0] * 64,
    )
    stage1 = Stage1Model(vae)
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        context_in_dim=16,
        axes_dims_rope=(2, 2, 4),
        mlp_ratio=2,
        causal_condition=True,
    )
    denoiser = ConditionalDenoiser(
        transformer, torch.randn(1, 3, 16), rank=2, alpha=2, spatial_width=16, physics_tokens=2
    )
    return Stage2Model(stage1, denoiser, FlowMatchEulerDiscreteScheduler())


def batch():
    values = torch.rand(1, 4, 64, 64)
    mask = torch.zeros_like(values)
    for c, (y, x) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        mask[:, c, y::2, x::2] = 1
    return {
        "raw_values": values * mask,
        "cfa_mask": mask,
        "physical_params": torch.randn(1, 8),
        "rgb_clean": torch.rand(1, 3, 64, 64),
    }
