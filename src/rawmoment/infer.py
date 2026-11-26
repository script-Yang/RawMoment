# SPDX-License-Identifier: Apache-2.0
"""Restore all test images using native 40-step Euler flow matching."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate
from tqdm import tqdm

from .data import RestorationDataset
from .models import build_stage2
from .utils import read_config, save_rgb, to_device


@torch.inference_mode()
def run(config, manifest, dataset, size, output, steps=40, seed=42, limit=None, save_gt=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    model = build_stage2(config, load_trained=True).eval().requires_grad_(False)
    images = RestorationDataset(manifest, dataset, size, seed=seed)
    count = len(images) if limit is None else min(limit, len(images))
    # Match the evaluation precision: BF16 DiT, FP32 VAE/adapters, TF32 off.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (output / "predictions.jsonl").open("w") as records:
        for index in tqdm(range(count), desc=f"{dataset} {size}"):
            sample = images[index]
            batch = to_device(default_collate([sample]), config["device"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.restore(batch, steps=steps, seed=sample["seed"])
            path = output / "predictions" / (sample["id"] + ".png")
            save_rgb(path, prediction[0])
            # GT is decoded only after the RAW-only forward pass has finished.
            if save_gt:
                save_rgb(output / "gt" / (sample["id"] + ".png"), images.target(index))
            records.write(
                json.dumps(
                    {
                        "sample_id": sample["id"],
                        "seed": sample["seed"],
                        "prediction": str(path.relative_to(output)),
                    }
                )
                + "\n"
            )
            records.flush()
    (output / "run.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "size": size,
                "count": count,
                "steps": steps,
                "base_seed": seed,
                "stage1_weights": config["model"]["stage1_weights"],
                "stage2_weights": config["model"]["stage2_weights"],
                "output": "native RGB PNG16; no GT brightness/color/geometry fitting",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["rawmoment_bench", "gopro", "realblur"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--size", type=int, choices=[512, 1024], default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-gt", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.dataset in ("gopro", "realblur") and args.size != 512:
        parser.error("The prepared GoPro and RealBlur releases support 512 crops only")
    config = read_config(args.config)
    config["device"] = args.device
    run(
        config,
        args.manifest,
        args.dataset,
        args.size,
        args.output,
        args.steps,
        args.seed,
        args.limit,
        args.save_gt,
    )


if __name__ == "__main__":
    main()
