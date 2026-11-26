# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch
from torch import nn


class VGGPerceptualLoss(nn.Module):
    """LPIPS v0.1 VGG, with an optional local ImageNet VGG16 checkpoint."""

    def __init__(self, weights_path=None):
        super().__init__()
        import lpips

        if weights_path:
            self.metric = lpips.LPIPS(net="vgg", version="0.1", pnet_rand=True, verbose=False)
            weights = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
            backbone = {
                name: weights["features." + name.split(".", 1)[1]]
                for name in self.metric.net.state_dict()
            }
            self.metric.net.load_state_dict(backbone)
        else:
            self.metric = lpips.LPIPS(net="vgg", version="0.1", verbose=False)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, reconstructed, target):
        with torch.autocast(device_type=reconstructed.device.type, enabled=False):
            return self.metric(
                reconstructed.float().clamp(0, 1) * 2 - 1, target.detach().float() * 2 - 1
            ).mean()
