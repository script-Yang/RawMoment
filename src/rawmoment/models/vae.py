# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from torch import nn
from torch.nn import functional as F

from ..raw import prepare_raw_input


def gaussian_kl(mean_p, logvar_p, mean_q, logvar_q):
    """Elementwise KL(p || q), evaluated in FP32."""
    mean_p, logvar_p, mean_q, logvar_q = (x.float() for x in (mean_p, logvar_p, mean_q, logvar_q))
    return 0.5 * (
        logvar_q
        - logvar_p
        + torch.exp(logvar_p - logvar_q)
        + (mean_p - mean_q).square() * torch.exp(-logvar_q)
        - 1
    )


class RawEncoder(nn.Module):
    """Copy the Qwen RGB encoder, remapping its RGBA input to sparse R/G1/G2/B."""

    def __init__(self, vae):
        super().__init__()
        self.encoder = copy.deepcopy(vae.encoder).requires_grad_(True)
        self.quant_conv = copy.deepcopy(vae.quant_conv).requires_grad_(True)
        with torch.no_grad():
            conv = self.encoder.conv_in
            weight = conv.weight.clone()
            conv.weight.copy_(
                torch.stack(
                    [4 * weight[:, 0], 2 * weight[:, 1], 2 * weight[:, 1], 4 * weight[:, 2]], dim=1
                )
            )
            if conv.bias is not None:
                conv.bias.add_(weight[:, 3].flatten(1).sum(1))
            channels = len(vae.config.latents_std)
            self.quant_conv.weight[channels:].zero_()
            std = self.quant_conv.bias.new_tensor(vae.config.latents_std)
            self.quant_conv.bias[channels:].copy_(2 * (0.5 * std).log())

    def forward(self, raw_values, cfa_mask):
        encoded = self.encoder(prepare_raw_input(raw_values, cfa_mask).float().unsqueeze(2))
        return self.quant_conv(encoded)


class Stage1Model(nn.Module):
    def __init__(self, vae, raw_encoder=None, loss_weights=None, perceptual_loss=None):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.raw_encoder = raw_encoder if raw_encoder is not None else RawEncoder(vae)
        self.register_buffer(
            "latent_mean", torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        )
        self.register_buffer(
            "latent_std", torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        )
        self.loss_weights = dict(loss_weights or {"image_l1": 1.0, "latent_l1": 1.0, "kl": 1.0})
        self.perceptual_loss = perceptual_loss
        if self.perceptual_loss is not None:
            self.perceptual_loss.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        if self.perceptual_loss is not None:
            self.perceptual_loss.eval()
        return self

    def normalize(self, mean, logvar):
        return (
            (mean.float() - self.latent_mean) / self.latent_std,
            logvar.float().clamp(-30, 20) - 2 * self.latent_std.log(),
        )

    @torch.no_grad()
    def rgb_posterior(self, rgb):
        rgba = torch.cat([2 * rgb.float() - 1, torch.ones_like(rgb[:, :1])], dim=1)
        posterior = self.vae.encode(rgba.unsqueeze(2)).latent_dist
        return self.normalize(posterior.mean, posterior.logvar)

    def raw_posterior(self, raw_values, cfa_mask):
        mean, logvar = self.raw_encoder(raw_values, cfa_mask).chunk(2, dim=1)
        return self.normalize(mean, logvar)

    def decode(self, latent):
        native = latent * self.latent_std + self.latent_mean
        return self.vae.decode(native.float()).sample[:, :3, 0].add(1).div(2)

    def training_loss(self, batch, generator=None):
        terms = {}
        for domain in ("clean", "blur"):
            target = batch[f"rgb_{domain}"]
            mean_rgb, logvar_rgb = self.rgb_posterior(target)
            mean_raw, logvar_raw = self.raw_posterior(batch[f"raw_{domain}"], batch["cfa_mask"])
            noise = torch.randn(
                mean_raw.shape, device=mean_raw.device, dtype=mean_raw.dtype, generator=generator
            )
            reconstructed = self.decode(mean_raw + (0.5 * logvar_raw).exp() * noise)
            terms[f"{domain}/image_l1"] = F.l1_loss(reconstructed, target)
            terms[f"{domain}/latent_l1"] = F.l1_loss(mean_raw, mean_rgb)
            terms[f"{domain}/kl"] = gaussian_kl(mean_rgb, logvar_rgb, mean_raw, logvar_raw).mean()
            if "lpips" in self.loss_weights:
                terms[f"{domain}/lpips"] = (
                    self.perceptual_loss(reconstructed, target)
                    if self.loss_weights["lpips"] > 0
                    else reconstructed.new_zeros(())
                )
        loss = (
            sum(
                (
                    self.loss_weights[name] * terms[f"{domain}/{name}"]
                    for domain in ("clean", "blur")
                    for name in self.loss_weights
                )
            )
            / 2
        )
        return (loss, {name: value.detach().item() for name, value in terms.items()})
