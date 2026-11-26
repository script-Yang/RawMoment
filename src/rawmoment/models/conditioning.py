# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
from torch.nn import functional as F

from ..raw import prepare_raw_input


def pack_bayer(raw_values, cfa_mask):
    """Keep one sample per RGGB phase; preserve the original CFA origin."""
    raw = prepare_raw_input(raw_values, cfa_mask).float()
    return torch.stack(
        [
            raw[:, channel, y::2, x::2]
            for channel, (y, x) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
        ],
        dim=1,
    )


def patchify_spatial(features, patch_size=2):
    """Pack adjacent features into channels without averaging spatial samples."""
    return F.pixel_unshuffle(features, patch_size).flatten(2).transpose(1, 2)


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, features):
        return features + self.net(features)


class RawSpatialEncoder(nn.Module):
    """Encode RAW into an ordered [B,C,H/16,W/16] feature grid."""

    def __init__(self, width=64):
        super().__init__()
        self.width = width
        stem, middle = (max(8, width // 4), max(8, width // 2))
        layers = [nn.Conv2d(4, stem, 3, padding=1), nn.SiLU(), SpatialResidualBlock(stem)]
        previous = stem
        for channels in (middle, width, width):
            layers.extend(
                [
                    nn.Conv2d(previous, channels, 3, stride=2, padding=1),
                    nn.SiLU(),
                    SpatialResidualBlock(channels),
                ]
            )
            previous = channels
        self.features = nn.Sequential(*layers)

    def forward(self, raw_values, cfa_mask):
        return self.features(pack_bayer(raw_values, cfa_mask))


class PhysicalTokenEncoder(nn.Module):
    def __init__(self, context_dim, tokens=64, input_dim=8):
        super().__init__()
        self.tokens, self.context_dim = (tokens, context_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, tokens * context_dim)
        )

    def forward(self, physical_params):
        return self.net(physical_params.float()).reshape(-1, self.tokens, self.context_dim)
