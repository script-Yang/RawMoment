# SPDX-License-Identifier: Apache-2.0
"""Deployment weights are safetensors; optimizer state stays in training checkpoints."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parameters_for_stage(model, stage):
    if stage == "stage1":
        return dict(model.raw_encoder.named_parameters())
    return {k: p for k, p in model.denoiser.named_parameters() if p.requires_grad}


def weight_state(model, stage):
    if stage == "stage1":
        return model.raw_encoder.state_dict()
    return parameters_for_stage(model, stage)


def load_weights(model, stage, path):
    weights = load_file(str(path), device="cpu")
    if stage == "stage1":
        model.raw_encoder.load_state_dict(weights)
    else:
        parameters = parameters_for_stage(model, stage)
        if parameters.keys() != weights.keys():
            raise ValueError(
                "Stage2 weights do not match the configured LoRA/conditioning architecture"
            )
        with torch.no_grad():
            for key, parameter in parameters.items():
                parameter.copy_(weights[key])


def save_weights(model, stage, path, resolution, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        k: v.detach().cpu().contiguous().clone() for k, v in weight_state(model, stage).items()
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "format": "pt",
            "stage": stage,
            "training_resolution": str(resolution),
            "training_step": str(step),
        },
    )
