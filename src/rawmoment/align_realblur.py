# SPDX-License-Identifier: Apache-2.0
"""Apply the paper's GT-assisted brightness alignment to saved RealBlur predictions."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROTOCOL = "rgb_scalar_clip_png16_v1"


def align_brightness(prediction, target):
    """Fit one scalar over all RGB values, then clip; inputs are display RGB [0,1]."""
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[2] != 3
        or prediction.size == 0
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
        or min(prediction.min(), target.min()) < 0
        or max(prediction.max(), target.max()) > 1
    ):
        raise ValueError("Expected matching, finite RGB images in [0,1]")
    denominator = float(np.square(prediction).sum())
    if denominator <= 1e-16:
        raise ValueError("Cannot fit brightness to an all-black prediction")
    gain = float(np.sum(prediction * target) / denominator)
    return np.clip(gain * prediction, 0, 1), gain


def read_png16(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB PNG16 image: {path}")
    return image[..., ::-1].astype(np.float64) / 65535


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def align_run(run, manifest, output=None):
    """Postprocess native RealBlur runs only; no model or RAW input is loaded."""
    run, manifest = Path(run).resolve(), Path(manifest).resolve()
    settings = json.loads((run / "run.json").read_text())
    if settings.get("dataset") != "realblur" or settings.get("size") != 512:
        raise ValueError("Brightness alignment is supported only for RealBlur 512 runs")
    if settings.get("brightness_alignment"):
        raise ValueError("This run has already been brightness aligned")
    records = read_rows(run / "predictions.jsonl")
    ids = [record["sample_id"] for record in records]
    if not ids or len(ids) != len(set(ids)) or len(ids) != settings["count"]:
        raise ValueError("Prediction records must be complete and have unique sample IDs")
    rows = [row for row in read_rows(manifest) if row["split"] == "test"]
    targets = {row["sample_id"]: row for row in rows}
    if len(targets) != len(rows) or not set(ids).issubset(targets):
        raise ValueError("Predictions must match unique test samples in the RealBlur manifest")
    output = Path(output).resolve() if output else run / "brightness_rgb_gt"
    # A fresh directory preserves native outputs and prevents applying the gain twice.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w") as stream:
        for record in records:
            sid = record["sample_id"]
            row = targets[sid]
            relative = Path(record["prediction"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Prediction paths must be relative to the source run")
            y, x, height, width = row["eval_crop_yxhw"]
            if min(y, x) < 0 or (height, width) != (512, 512):
                raise ValueError(f"Expected a fixed fully valid 512 crop: {sid}")
            entry = row["files"]["CLEAN_RGB"]
            target_path = manifest.parent / (entry["path"] if isinstance(entry, dict) else entry)
            target = read_png16(target_path)[y : y + height, x : x + width]
            prediction = read_png16(run / relative)
            if prediction.shape != (height, width, 3) or target.shape != prediction.shape:
                raise ValueError(f"Prediction/GT crop shape mismatch: {sid}")
            # Released evaluation crops are fully valid: fit all pixels and RGB jointly.
            aligned, gain = align_brightness(prediction, target)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.rint(aligned * 65535).astype(np.uint16)
            if not cv2.imwrite(str(destination), pixels[..., ::-1]):
                raise OSError(destination)
            stream.write(json.dumps({**record, "brightness_gain": gain}) + "\n")
    (output / "run.json").write_text(
        json.dumps(
            {
                **settings,
                "source_run": str(run),
                "manifest": str(manifest),
                "brightness_alignment": PROTOCOL,
                "uses_clean_gt_for_gain": True,
                "output": "GT brightness-aligned RGB PNG16; no color or geometry fitting",
                "formula": "gain=sum(pred*GT)/sum(pred^2); aligned=clip(gain*pred,0,1)",
                "fit_domain": "saved RGB PNG16 read as float64/65535; all RGB; full valid crop",
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Native RealBlur inference output")
    parser.add_argument("--manifest", required=True, type=Path, help="RealBlur test manifest")
    parser.add_argument(
        "--output", type=Path, help="Fresh directory; default RUN/brightness_rgb_gt"
    )
    args = parser.parse_args()
    print(align_run(args.run, args.manifest, args.output))


if __name__ == "__main__":
    main()
