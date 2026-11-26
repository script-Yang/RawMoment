# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from rawmoment.data import RestorationDataset
from rawmoment.raw import CFA, normalize_raw, realblur_input
from rawmoment.utils import seed_for


class DataTest(unittest.TestCase):
    def test_bayer_black_level(self):
        raw = np.zeros((64, 64, 3), np.uint16)
        for y, x, c in CFA:
            raw[y::2, x::2, c] = 4096
        values, mask = normalize_raw(raw)
        expected = (4096 / 65535 - 1 / 16) / (15 / 16)
        torch.testing.assert_close(values[mask.bool()], torch.full((4096,), expected))
        self.assertEqual(values[~mask.bool()].count_nonzero(), 0)

    def test_native_and_center_crops_keep_cfa_and_gt_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.zeros((1024, 1024, 3), np.uint16)
            for y, x, c in CFA:
                raw[y::2, x::2, c] = np.arange(512, dtype=np.uint16)[None] + 5000
            cv2.imwrite(str(root / "raw.png"), raw[..., ::-1])
            row = {
                "sample_id": "example",
                "split": "test",
                "noise_iso": 400,
                "noise_a": 0.001,
                "noise_b": 0.0001,
                "wb_gain": [0.5, 1.0, 0.6],
                "files": {"RAW": "raw.png", "RGB": "missing_gt.png"},
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            small = RestorationDataset(manifest, "rawmoment_bench", 512)[0]
            full = RestorationDataset(manifest, "rawmoment_bench", 1024)[0]
            torch.testing.assert_close(
                small["raw_values"], full["raw_values"][:, 256:768, 256:768], rtol=0, atol=0
            )
            torch.testing.assert_close(small["cfa_mask"], full["cfa_mask"][:, 256:768, 256:768])
            self.assertEqual(small["seed"], full["seed"])
            self.assertEqual(small["seed"], seed_for("example"))

    def test_realblur_wb_and_noise_scaling_once(self):
        row = {
            "black_level": 512,
            "white_level": 16383,
            "wb_gain": [2.0, 1.5],
            "blur_iso": 400,
            "blur_noise_ab": [0.0001, 0.0000001],
        }
        raw = np.full((512, 512), 2048, np.uint16)
        values, mask, physical = realblur_input(raw, row)
        sensor = (2048 - 512) / (16383 - 512)
        self.assertAlmostEqual(values[0, 0, 0].item(), sensor * 2, places=7)
        self.assertAlmostEqual(values[1, 0, 1].item(), sensor, places=7)
        self.assertAlmostEqual(values[3, 1, 1].item(), sensor * 1.5, places=7)
        self.assertAlmostEqual(physical[1].item(), np.log10(0.0001 * 15 / 16) / 12, places=7)
        self.assertEqual(physical[3].item(), -0.25)
        self.assertEqual(mask.sum(), 512 * 512)

    def test_realblur_1024_is_not_upsampled(self):
        for dataset in ("realblur", "gopro"):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                RestorationDataset("unused.jsonl", dataset, 1024)

    def test_gopro_native_inputs_training_split_and_optional_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.zeros((512, 512, 3), np.uint16)
            for y, x, c in CFA:
                raw[y::2, x::2, c] = 5000 + c * 1000
            target = np.full((512, 512, 3), [10000, 20000, 30000], np.uint16)
            cv2.imwrite(str(root / "raw.png"), raw[..., ::-1])
            cv2.imwrite(str(root / "target.png"), target[..., ::-1])
            common = {
                "sample_id": "GoPro/train/sequence/frame",
                "noise_iso": 400,
                "noise_a": 0.001,
                "noise_b": 0.0001,
                "wb_gain": [0.5, 1.0, 0.6],
                "files": {"RAW": {"path": "raw.png"}, "RGB": {"path": "target.png"}},
            }
            rows = [{**common, "split": "train", "record_id": f"frame_v{i}"} for i in range(3)]
            test = {**common, "sample_id": "GoPro/test/sequence/frame", "split": "test"}
            test["files"] = {"RAW": {"path": "raw.png"}, "RGB": {"path": "missing_gt.png"}}
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in [*rows, test]))
            inference = RestorationDataset(manifest, "gopro")[0]
            values, mask = normalize_raw(raw)
            torch.testing.assert_close(inference["raw_values"], values, rtol=0, atol=0)
            torch.testing.assert_close(inference["cfa_mask"], mask)
            self.assertEqual(inference["seed"], seed_for(test["sample_id"]))
            training = RestorationDataset(
                manifest, "gopro", split="train", training=True, include_target=True
            )
            self.assertEqual(len(training), 3)
            sample = training[2]
            torch.testing.assert_close(sample["raw_values"], values, rtol=0, atol=0)
            expected = torch.tensor([10000, 20000, 30000]).float() / 65535
            torch.testing.assert_close(sample["rgb_clean"], expected[:, None, None].expand(3, 512, 512))


if __name__ == "__main__":
    unittest.main()
