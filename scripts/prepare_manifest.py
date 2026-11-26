# SPDX-License-Identifier: Apache-2.0
"""Keep portable fields from the prepared dataset manifest; images stay in place."""

import argparse
import json
from pathlib import Path

FIELDS = {
    "rawmoment_bench": ["noise_iso", "noise_a", "noise_b", "wb_gain"],
    "gopro": ["noise_iso", "noise_a", "noise_b", "wb_gain"],
    "stage1": [],
    "realblur": [
        "black_level",
        "white_level",
        "blur_iso",
        "blur_noise_ab",
        "wb_gain",
        "eval_crop_yxhw",
        "valid_crop_origins_512",
    ],
}
KEYS = {
    "rawmoment_bench": ["RAW", "RGB", "BLUR_RGB"],
    "gopro": ["RAW", "RGB", "BLUR_RGB"],
    "stage1": ["CLEAN_RAW", "BLUR_RAW", "CLEAN_RGB", "BLUR_RGB"],
    "realblur": ["BLUR_RAW", "CLEAN_RGB", "BLUR_RGB", "VALID_MASK"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Save beside the images/old manifest")
    parser.add_argument("--dataset", choices=list(FIELDS), required=True)
    parser.add_argument("--split-key", default="split")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    result = []
    for row in rows:
        files = {}
        for key in KEYS[args.dataset]:
            if key in row["files"]:
                entry = row["files"][key]
                relative = entry["path"] if isinstance(entry, dict) else entry
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("Prepared image paths must be relative to the manifest")
                files[key] = relative
        public = {"sample_id": row["sample_id"], "split": row[args.split_key], "files": files}
        if "group" in row:
            public["group"] = row["group"]
        if "record_id" in row:
            public["record_id"] = row["record_id"]
        public.update({key: row[key] for key in FIELDS[args.dataset]})
        result.append(public)
    Path(args.output).write_text("".join(json.dumps(row) + "\n" for row in result))
    print(f"Wrote {len(result)} records")


if __name__ == "__main__":
    main()
