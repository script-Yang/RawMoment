# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rawmoment.align_realblur import PROTOCOL, align_brightness, align_run, read_png16


class RealBlurAlignmentTest(unittest.TestCase):
    def test_single_rgb_scalar_instead_of_luminance_or_per_channel_gains(self):
        prediction = np.broadcast_to([0.1, 0.3, 0.2], (8, 8, 3)).copy()
        target = np.broadcast_to([0.8, 0.2, 0.6], prediction.shape)
        original = prediction.copy()
        aligned, gain = align_brightness(prediction, target)
        self.assertAlmostEqual(gain, 13 / 7, places=12)
        np.testing.assert_allclose(aligned, prediction * (13 / 7))
        np.testing.assert_array_equal(prediction, original)

    def test_known_exposure_and_highlight_clipping(self):
        target = np.random.default_rng(42).uniform(0.01, 0.8, (8, 8, 3))
        aligned, gain = align_brightness(target / 3, target)
        self.assertAlmostEqual(gain, 3, places=12)
        np.testing.assert_allclose(aligned, target, atol=1e-14)
        prediction = np.full((8, 8, 3), 0.1)
        prediction[0, 0] = 0.9
        aligned, gain = align_brightness(prediction, np.full_like(prediction, 0.7))
        self.assertGreater(gain, 1)
        np.testing.assert_array_equal(aligned[0, 0], [1, 1, 1])

    def test_degenerate_and_invalid_images_are_rejected(self):
        image = np.full((4, 4, 3), 0.2)
        for prediction in (np.zeros_like(image), image * np.nan, image - 1, image + 1):
            with self.subTest(prediction=prediction[0, 0, 0]), self.assertRaises(ValueError):
                align_brightness(prediction, image)
        with self.assertRaises(ValueError):
            align_brightness(image, image[:2])

    def fixture(self, root):
        run = root / "run"
        (run / "predictions").mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"dataset": "realblur", "size": 512, "count": 1}))
        (run / "predictions.jsonl").write_text(
            json.dumps({"sample_id": "scene001_01", "seed": 42, "prediction": "predictions/a.png"})
            + "\n"
        )
        prediction = np.broadcast_to([5000, 7000, 11000], (512, 512, 3)).copy()
        prediction[0, 0] = 60000
        self.assertTrue(
            cv2.imwrite(str(run / "predictions/a.png"), prediction[..., ::-1].astype(np.uint16))
        )
        target = np.full((544, 560, 3), 65535, np.uint16)
        target[16:528, 32:544] = [30000, 20000, 45000]
        self.assertTrue(cv2.imwrite(str(root / "gt.png"), target[..., ::-1]))
        row = {
            "sample_id": "scene001_01",
            "split": "test",
            "eval_crop_yxhw": [16, 32, 512, 512],
            "files": {"CLEAN_RGB": {"path": "gt.png"}},
        }
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        return run, manifest

    def test_cli_crop_png16_roundtrip_and_native_outputs_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            run, manifest = self.fixture(Path(directory))
            originals = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
            # Also verifies postprocessing needs neither model weights nor RAW files.
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rawmoment.align_realblur",
                    "--run",
                    str(run),
                    "--manifest",
                    str(manifest),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = run / "brightness_rgb_gt"
            record = json.loads((output / "predictions.jsonl").read_text())
            prediction = read_png16(run / "predictions/a.png")
            target = read_png16(manifest.parent / "gt.png")[16:528, 32:544]
            expected_gain = np.linalg.lstsq(prediction.reshape(-1, 1), target.ravel(), rcond=None)[
                0
            ][0]
            self.assertAlmostEqual(record["brightness_gain"], expected_gain, delta=1e-10)
            expected = np.rint(np.clip(prediction * expected_gain, 0, 1) * 65535).astype(np.uint16)
            saved = cv2.imread(str(output / record["prediction"]), cv2.IMREAD_UNCHANGED)[..., ::-1]
            np.testing.assert_array_equal(saved, expected)
            self.assertEqual(record["seed"], 42)
            settings = json.loads((output / "run.json").read_text())
            self.assertEqual(settings["brightness_alignment"], PROTOCOL)
            self.assertTrue(settings["uses_clean_gt_for_gain"])
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)
            with self.assertRaises(FileExistsError):
                align_run(run, manifest)
            with self.assertRaisesRegex(ValueError, "already"):
                align_run(output, manifest, run / "twice")

    def test_rawmoment_bench_is_rejected_before_reading_gt_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for size in (512, 1024):
                (root / "run.json").write_text(
                    json.dumps({"dataset": "rawmoment_bench", "size": size})
                )
                before = {p: p.read_bytes() for p in root.iterdir()}
                with self.assertRaisesRegex(ValueError, "only for RealBlur"):
                    align_run(root, root / "missing_manifest.jsonl")
                self.assertEqual({p: p.read_bytes() for p in root.iterdir()}, before)

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, manifest = self.fixture(Path(directory))
            (run / "run.json").write_text(
                json.dumps({"dataset": "realblur", "size": 512, "count": 2})
            )
            with self.assertRaisesRegex(ValueError, "complete"):
                align_run(run, manifest)
            self.assertFalse((run / "brightness_rgb_gt").exists())


if __name__ == "__main__":
    unittest.main()
