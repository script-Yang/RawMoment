# SPDX-License-Identifier: Apache-2.0
"""Prepared PNG datasets. No demosaicing, resizing or GT-dependent input fitting."""

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from .raw import normalize_raw, physical_vector, realblur_input
from .utils import image_path, read_rgb, read_rows, rgb_tensor, seed_for


class RestorationDataset(Dataset):
    """Stage2 training or testing on prepared RawMoment, GoPro and RealBlur data."""

    def __init__(
        self,
        manifest,
        dataset,
        size=512,
        split="test",
        training=False,
        seed=42,
        include_target=False,
    ):
        if dataset not in ("rawmoment_bench", "gopro", "realblur"):
            raise ValueError("Dataset must be rawmoment_bench, gopro or realblur")
        if size not in (512, 1024) or (dataset in ("gopro", "realblur") and size != 512):
            raise ValueError("Use 512/1024 for rawmoment_bench, or 512 for GoPro/RealBlur")
        self.root = Path(manifest).parent
        self.rows = read_rows(manifest, split)
        if not self.rows:
            raise ValueError(f"No {split} samples in {manifest}")
        self.dataset, self.size, self.training = dataset, size, training
        self.seed, self.epoch, self.include_target = seed, 0, include_target

    def __len__(self):
        return len(self.rows)

    def crop(self, height, width, index, row):
        if min(height, width) < self.size:
            raise ValueError(f"{row['sample_id']} is smaller than {self.size}")
        if self.training:
            rng = torch.Generator().manual_seed(seed_for(f"{self.epoch}:{index}", self.seed))
            origins = row.get("valid_crop_origins_512") if self.dataset == "realblur" else None
            if origins:
                return origins[torch.randint(len(origins), (), generator=rng).item()]
            return [
                16 * torch.randint((d - self.size) // 16 + 1, (), generator=rng).item()
                for d in (height, width)
            ]
        if self.dataset == "realblur":
            return row["eval_crop_yxhw"][:2]
        return [16 * ((d - self.size) // 32) for d in (height, width)]

    def __getitem__(self, index):
        row = self.rows[index]
        key = "BLUR_RAW" if self.dataset == "realblur" else "RAW"
        raw = (
            read_rgb(image_path(self.root, row, key))
            if key == "RAW"
            else cv2.imread(str(image_path(self.root, row, key)), cv2.IMREAD_UNCHANGED)
        )
        if raw is None:
            raise FileNotFoundError(image_path(self.root, row, key))
        y, x = self.crop(*raw.shape[:2], index, row)
        raw = raw[y : y + self.size, x : x + self.size]
        if self.dataset != "realblur":
            values, mask = normalize_raw(raw)
            physical = physical_vector(
                row["noise_iso"], row["noise_a"], row["noise_b"], row["wb_gain"]
            )
            seed_key = row["sample_id"]
        else:
            values, mask, physical = realblur_input(raw, row)
            seed_key = f"{row['sample_id']}:{y}:{x}"
        sample = {
            "id": row["sample_id"],
            "raw_values": values,
            "cfa_mask": mask,
            "physical_params": physical,
            "seed": seed_for(seed_key, self.seed),
        }
        if self.include_target:
            sample["rgb_clean"] = self.target(index)
        return sample

    def target(self, index):
        """Opened separately after inference, or as supervision during training."""
        row = self.rows[index]
        key = "CLEAN_RGB" if self.dataset == "realblur" else "RGB"
        image = read_rgb(image_path(self.root, row, key))
        y, x = self.crop(*image.shape[:2], index, row)
        return rgb_tensor(image[y : y + self.size, x : x + self.size])


class Stage1Dataset(Dataset):
    """Paired clean/blur RAW and RGB for VAE latent alignment."""

    def __init__(self, manifest, size=512, split="train", seed=42, split_key="split"):
        self.root = Path(manifest).parent
        self.rows = read_rows(manifest, split, split_key)
        if not self.rows:
            raise ValueError(f"No {split} samples in {manifest}")
        self.size, self.seed, self.epoch = size, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = {
            k: read_rgb(image_path(self.root, row, k))
            for k in ("CLEAN_RAW", "BLUR_RAW", "CLEAN_RGB", "BLUR_RGB")
        }
        h, w = images["BLUR_RAW"].shape[:2]
        if min(h, w) < self.size:
            raise ValueError(f"{row['sample_id']} is smaller than {self.size}")
        rng = torch.Generator().manual_seed(seed_for(f"{self.epoch}:{index}", self.seed))
        y, x = [
            16 * torch.randint((d - self.size) // 16 + 1, (), generator=rng).item() for d in (h, w)
        ]
        images = {k: v[y : y + self.size, x : x + self.size] for k, v in images.items()}
        clean, mask = normalize_raw(images["CLEAN_RAW"])
        blur, _ = normalize_raw(images["BLUR_RAW"])
        return {
            "id": row["sample_id"],
            "raw_clean": clean,
            "raw_blur": blur,
            "cfa_mask": mask,
            "rgb_clean": rgb_tensor(images["CLEAN_RGB"]),
            "rgb_blur": rgb_tensor(images["BLUR_RGB"]),
        }
