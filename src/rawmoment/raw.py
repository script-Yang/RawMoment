# SPDX-License-Identifier: Apache-2.0
"""RGGB normalization and the physical-condition vector used in training."""

import math

import numpy as np
import torch

CFA = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2))


def normalize_raw(raw):
    """Sparse uint16 RGB RGGB -> four normalized sparse channels and CFA mask."""
    height, width = raw.shape[:2]
    values = np.zeros((4, height, width), np.float32)
    mask = np.zeros_like(values)
    for channel, (y, x, color) in enumerate(CFA):
        values[channel, y::2, x::2] = (
            raw[y::2, x::2, color].astype(np.float32) / 65535 - 1 / 16
        ) / (15 / 16)
        mask[channel, y::2, x::2] = 1
    return torch.from_numpy(values), torch.from_numpy(mask)


def prepare_raw_input(raw_values, cfa_mask):
    return cfa_mask * (2 * raw_values - 1)


def physical_vector(iso, shot, read, white_balance):
    return torch.tensor(
        [
            math.log2(iso / 100) / 8,
            math.log10(shot) / 12,
            math.log10(read + 1e-12) / 12,
            *[math.log2(gain) / 4 for gain in white_balance],
            1 / 16,
            1.0,
        ],
        dtype=torch.float32,
    )


def realblur_input(dn, row):
    """Prepared Sony bin4 DN mosaic; WB once, no clipping or extra exposure gain."""
    black, white = row["black_level"], row["white_level"]
    wb = [row["wb_gain"][0], 1.0, row["wb_gain"][1]]
    mosaic = (dn.astype(np.float32) - black) / (white - black)
    for y, x, color in CFA:
        mosaic[y::2, x::2] *= wb[color]
    values = np.zeros((4, *dn.shape), np.float32)
    mask = np.zeros_like(values)
    for channel, (y, x, _) in enumerate(CFA):
        values[channel, y::2, x::2] = mosaic[y::2, x::2]
        mask[channel, y::2, x::2] = 1
    # Saved noise coefficients already include bin4's /16 factor. Convert to
    # the training intensity domain exactly once; WB is a separate condition.
    shot, read = row["blur_noise_ab"]
    k = 15 / 16
    physical = physical_vector(row["blur_iso"], shot * k, read * k**2, [1 / gain for gain in wb])
    return torch.from_numpy(values), torch.from_numpy(mask), physical
