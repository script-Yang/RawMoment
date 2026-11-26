# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path

import torch
from helpers import batch, tiny_model

from rawmoment.checkpoints import load_weights, save_weights, weight_state
from rawmoment.models.conditioning import patchify_spatial
from rawmoment.models.stage2 import make_noisy_latents


class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_training_gradients_and_frozen_stage1(self):
        torch.manual_seed(7)
        model = tiny_model().train()
        loss, _ = model.training_loss(batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in model.stage1.parameters()))
        for module in [
            model.denoiser.spatial_encoder,
            model.denoiser.spatial_projection,
            model.denoiser.physics_mlp,
        ]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_released_weights_roundtrip_preserves_prediction(self):
        torch.manual_seed(3)
        model, inputs = tiny_model().eval(), batch()
        with torch.no_grad():
            for stage in ["stage1", "stage2"]:
                for tensor in weight_state(
                    model.stage1 if stage == "stage1" else model, stage
                ).values():
                    tensor.add_(0.01)
        with torch.no_grad():
            expected = model.restore(inputs, steps=2, seed=42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ["stage1", "stage2"]:
                save_weights(
                    model.stage1 if stage == "stage1" else model,
                    stage,
                    root / f"{stage}.safetensors",
                    512,
                    10,
                )
            torch.manual_seed(3)
            other = tiny_model().eval()
            for stage in ["stage1", "stage2"]:
                load_weights(
                    other.stage1 if stage == "stage1" else other,
                    stage,
                    root / f"{stage}.safetensors",
                )
                for key, value in weight_state(
                    model.stage1 if stage == "stage1" else model, stage
                ).items():
                    torch.testing.assert_close(
                        value,
                        weight_state(other.stage1 if stage == "stage1" else other, stage)[key],
                        rtol=0,
                        atol=0,
                    )
            with torch.no_grad():
                actual = other.restore(inputs, steps=2, seed=42)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_flow_endpoints(self):
        clean, noise = torch.randn(2, 64, 1, 4, 4), torch.randn(2, 64, 1, 4, 4)
        noisy, target = make_noisy_latents(clean, noise, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(noisy[0], clean[0])
        torch.testing.assert_close(noisy[1], noise[1])
        torch.testing.assert_close(target, noise - clean)

    def test_patch_tokens_preserve_pixels(self):
        features = torch.arange(72).float().reshape(1, 3, 4, 6)
        tokens = patchify_spatial(features)
        restored = torch.nn.functional.pixel_shuffle(tokens.transpose(1, 2).reshape(1, 12, 2, 3), 2)
        torch.testing.assert_close(restored, features, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
