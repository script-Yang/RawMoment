# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import PhysicalTokenEncoder, RawSpatialEncoder, patchify_spatial
from .dit import spatial_dit_forward


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        residual = F.linear(F.linear(inputs.float(), self.lora_a), self.lora_b)
        return self.base(inputs) + (self.scale * residual).to(inputs.dtype)


def pack_latents(latents):
    return latents[:, :, 0].flatten(2).transpose(1, 2)


def make_noisy_latents(clean_latents, noise, sigma):
    sigma = sigma.reshape(-1, 1, 1, 1, 1)
    return ((1 - sigma) * clean_latents + sigma * noise, noise - clean_latents)


def configure_schedule(scheduler, steps, image_tokens, device):
    config = scheduler.config
    slope = (config.max_shift - config.base_shift) / (
        config.max_image_seq_len - config.base_image_seq_len
    )
    mu = config.base_shift + slope * (image_tokens - config.base_image_seq_len)
    scheduler.set_timesteps(sigmas=np.linspace(1, 1 / steps, steps), device=device, mu=mu)
    scheduler.set_begin_index(0)
    return scheduler


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        transformer,
        prompt,
        rank=128,
        alpha=128,
        spatial_width=64,
        spatial_patch_size=2,
        physics_tokens=64,
    ):
        super().__init__()
        self.transformer = transformer.requires_grad_(False)
        for block in self.transformer.transformer_blocks:
            attention = block.attn
            for name in ("to_q", "to_k", "to_v"):
                setattr(attention, name, LoRALinear(getattr(attention, name), rank, alpha))
            attention.to_out[0] = LoRALinear(attention.to_out[0], rank, alpha)
        context_dim = transformer.config.context_in_dim
        self.spatial_encoder = RawSpatialEncoder(spatial_width)
        self.spatial_patch_size = spatial_patch_size
        patch_channels = spatial_width * spatial_patch_size**2
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(patch_channels), nn.Linear(patch_channels, transformer.inner_dim)
        )
        self.spatial_type_embedding = nn.Parameter(torch.randn(1, 1, transformer.inner_dim) * 0.02)
        self.physics_mlp = PhysicalTokenEncoder(context_dim, physics_tokens)
        self.register_buffer("prompt", prompt.detach().float(), persistent=False)

    def encode_conditions(self, raw_values, cfa_mask, physical_params):
        features = self.spatial_encoder(raw_values, cfa_mask)
        raw_spatial_tokens = self.spatial_projection(
            patchify_spatial(features, self.spatial_patch_size)
        )
        raw_spatial_tokens = raw_spatial_tokens + self.spatial_type_embedding
        text_embeddings = self.prompt.expand(raw_values.shape[0], -1, -1)
        physical_tokens = self.physics_mlp(physical_params)
        condition_tokens = torch.cat([text_embeddings, physical_tokens], dim=1)
        return (raw_spatial_tokens, condition_tokens)

    def forward(self, noisy_latents, raw_latents, raw_spatial_tokens, condition_tokens, sigma):
        height, width = raw_latents.shape[-2:]
        noisy_tokens = pack_latents(noisy_latents)
        raw_tokens = pack_latents(raw_latents)
        output = spatial_dit_forward(
            self.transformer,
            raw_spatial_tokens,
            raw_tokens,
            noisy_tokens,
            condition_tokens,
            sigma,
            height,
            width,
            spatial_shape=(height // self.spatial_patch_size, width // self.spatial_patch_size),
        )
        return output.float().transpose(1, 2).reshape_as(noisy_latents)


class Stage2Model(nn.Module):
    def __init__(self, stage1, denoiser, scheduler):
        super().__init__()
        self.stage1 = stage1.eval().requires_grad_(False)
        self.denoiser, self.scheduler = (denoiser, scheduler)

    def train(self, mode=True):
        super().train(mode)
        self.stage1.eval()
        return self

    def training_loss(self, batch, generator=None):
        from diffusers import FlowMatchEulerDiscreteScheduler

        with torch.no_grad():
            raw_latents = self.stage1.raw_posterior(batch["raw_values"], batch["cfa_mask"])[0]
            clean_latents = self.stage1.rgb_posterior(batch["rgb_clean"])[0]
        schedule = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        configure_schedule(
            schedule,
            schedule.config.num_train_timesteps,
            clean_latents.shape[-2] * clean_latents.shape[-1],
            clean_latents.device,
        )
        indices = torch.randint(
            len(schedule.timesteps),
            (clean_latents.shape[0],),
            device=clean_latents.device,
            generator=generator,
        )
        sigma = schedule.sigmas.to(clean_latents.device)[indices].float()
        noise = torch.randn(
            clean_latents.shape,
            device=clean_latents.device,
            dtype=clean_latents.dtype,
            generator=generator,
        )
        noisy_latents, target_velocity = make_noisy_latents(clean_latents, noise, sigma)
        raw_spatial_tokens, condition_tokens = self.denoiser.encode_conditions(
            batch["raw_values"], batch["cfa_mask"], batch["physical_params"]
        )
        predicted_velocity = self.denoiser(
            noisy_latents, raw_latents, raw_spatial_tokens, condition_tokens, sigma
        )
        return (
            F.mse_loss(predicted_velocity, target_velocity),
            {"sigma_mean": sigma.mean().item()},
        )

    @torch.no_grad()
    def restore(self, batch, steps=40, seed=42):
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.eval()
        raw_latents = self.stage1.raw_posterior(batch["raw_values"], batch["cfa_mask"])[0]
        generator = torch.Generator(device=raw_latents.device).manual_seed(seed)
        latents = torch.randn(
            raw_latents.shape,
            device=raw_latents.device,
            dtype=raw_latents.dtype,
            generator=generator,
        )
        raw_spatial_tokens, condition_tokens = self.denoiser.encode_conditions(
            batch["raw_values"], batch["cfa_mask"], batch["physical_params"]
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        configure_schedule(
            scheduler, steps, raw_latents.shape[-2] * raw_latents.shape[-1], raw_latents.device
        )
        for timestep in scheduler.timesteps:
            sigma = (timestep / scheduler.config.num_train_timesteps).expand(raw_latents.shape[0])
            velocity = self.denoiser(
                latents, raw_latents, raw_spatial_tokens, condition_tokens, sigma
            )
            latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
        return self.stage1.decode(latents).clamp(0, 1)
