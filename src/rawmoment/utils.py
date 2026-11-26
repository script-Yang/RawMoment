# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def read_config(path):
    """Paths in YAML are relative to the current working directory."""
    return yaml.safe_load(Path(path).read_text())


def seed_for(sample_id, seed=42):
    return (seed + int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:4], "big")) % (
        2**63 - 1
    )


def read_rows(path, split=None, split_key="split"):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows if split is None else [r for r in rows if r[split_key] == split]


def image_path(root, row, key):
    entry = row["files"][key]
    relative = entry["path"] if isinstance(entry, dict) else entry
    return Path(root) / relative


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16:
        raise ValueError(f"Expected a uint16 PNG: {path}")
    return image[..., ::-1].copy()


def rgb_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 65535


def save_rgb(path, rgb):
    """Save the clipped native model output as RGB PNG16."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rgb.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    pixels = np.rint(pixels * 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), pixels[..., ::-1]):
        raise OSError(f"Could not write {path}")


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
