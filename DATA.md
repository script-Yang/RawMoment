# Data formats

The [RawMoment dataset repository](https://huggingface.co/datasets/yscript/RawMoment) contains `rawmoment_data/`, `realblur_processed_data/` and `gopro_processed_data/`. Images are PNG; RAW and RGB images use uint16 storage, and RealBlur validity masks use uint8. Manifests are JSON Lines, with one record per sample. Image paths are relative to the manifest directory.

RawMoment provides 29,634 training samples, 366 validation samples and 200 RawMoment-Bench test samples. Train and val each provide `stage1.jsonl` and `stage2.jsonl`; test uses `manifest.jsonl`. Source videos are disjoint across splits. Training IDs distinguish sample variants; benchmark IDs determine inference seeds. Processed RealBlur provides 3,758 training pairs and 980 test pairs, each split with its own `manifest.jsonl`.

## RawMoment-Bench

The paper benchmark uses 200 held-out scenes, one fixed sample per scene. Inputs are native 1024×1024 sparse RGB RGGB images; targets are clean RGB.

RawMoment training, validation and RawMoment-Bench test data follow the same
data-construction protocol described in the paper. Use the released prepared images
directly; the dataset loader handles the required input conversion.

Minimal record (the image paths are illustrative):

```json
{"sample_id":"scene001","split":"test","group":"video001","files":{"RAW":"blur_raw/scene001.png","RGB":"clean_rgb/scene001.png","BLUR_RGB":"blur_rgb/scene001.png"},"noise_iso":400,"noise_a":0.001,"noise_b":0.00001,"wb_gain":[0.5,1.0,0.6]}
```

The four sparse channels are R/G1/G2/B at RGGB phases (0,0), (0,1), (1,0), (1,1). Input normalization follows the paper and is implemented by the dataset loader.

`--size 512` reads the center `[256:768,256:768]`. `--size 1024` reads the whole native image. There is no resize or padding. During training, paired crops have origins aligned to 16 pixels. Physical conditions are loaded from the manifest following the paper.

`BLUR_RGB` is optional for inference and unused by the model. `RGB` is required for training or `--save-gt` only; RAW-only inference can run without it.

## Stage 1 training

Stage 1 uses separate clean and blurred RAW/RGB pairs:

```json
{"sample_id":"scene001_v00","split":"train","group":"video001","files":{"CLEAN_RAW":"clean_raw/scene001.png","BLUR_RAW":"blur_raw/scene001.png","CLEAN_RGB":"clean_rgb/scene001.png","BLUR_RGB":"blur_rgb/scene001.png"}}
```

All four images share the same native geometry and sparse RAW representation. The crop is shared across all four images. Data construction follows the paper. Both clean and blur domains contribute equally to image L1, latent L1, KL and VGG LPIPS losses. Stage1 trains only the copied RAW encoder and quantization convolution; the RGB VAE remains frozen.

## Prepared RealBlur bin4

Use the existing registered RAW/RGB release. The Sony RGGB mosaic is uint16 storage of 14-bit DN, black 512, white 16383. The prepared full images are typically height 744 × width 648; the evaluation manifest specifies the fixed, fully valid 512 crop. This entry does not support a native 1024 crop or reconstruct the registration from original ARW files.

```json
{"sample_id":"scene001_01","split":"test","files":{"BLUR_RAW":"blur_raw/scene001_01.png","CLEAN_RGB":"clean_rgb/scene001_01.png"},"black_level":512,"white_level":16383,"blur_iso":400,"blur_noise_ab":[0.000005175,0.000000003675],"wb_gain":[2.0,1.5],"eval_crop_yxhw":[112,64,512,512],"valid_crop_origins_512":[[112,64]]}
```

`wb_gain` is [R,B], with green fixed to 1. The dataset loader uses the supplied camera metadata for input conversion and physical conditioning, following the paper.

Ground truth is the prepared RGB target, including its existing radiometric and geometric processing. Do not repeat that processing. `eval_crop_yxhw` is used for inference, and `valid_crop_origins_512` for training crops. The inference entry saves native model output, without using GT to fit its brightness or geometry.

## Prepared GoPro

GoPro data preparation follows the same protocol as the RawMoment training and test data described in the paper. The official sequence split is retained: 2,103 source training pairs provide 6,309 samples (three variants per pair), and 1,111 test pairs provide 1,111 fixed samples. All images are 512×512; there is no separate validation split.

Each split contains `manifest.jsonl`, `blur_raw/`, `blur_rgb/` and `clean_rgb/`, mapped by manifest keys `RAW`, `BLUR_RGB` and `RGB`. Images are PNG16; source RGB images are 8-bit. Paths are relative to the manifest directory. Original `sample_id` values are preserved, and `record_id` distinguishes training variants. The released loader reads the physical conditions from the manifest.

Use `--dataset gopro --size 512` for inference and `configs/stage2_gopro_512.yaml` for Stage 2 training. The loader accepts the published manifests directly, keeps the original sample IDs for inference seeds, and reads targets only for training or `--save-gt`. GoPro uses native predictions without RealBlur's brightness alignment; native 1024 inference is not supported by this release.

There is no clean RAW modality; these data support Stage 2 training. See the [GoPro data notes](https://huggingface.co/datasets/yscript/RawMoment/blob/main/gopro_processed_data/README.md). Source attribution and CC BY 4.0 terms follow GOPRO_Large by Nah, Kim and Lee (CVPR 2017).

## Evaluation outputs

Inference saves native RGB predictions. **Only RealBlur** uses the paper's GT-fitted
scalar brightness alignment. Run `rawmoment-align-realblur --run outputs/realblur_512
--manifest data/realblur_processed_data/test/manifest.jsonl` after inference.

Data preparation and evaluation follow the protocols described in the paper.

Protocol `rgb_scalar_clip_png16_v1`: read native PNG16 predictions and the matching
`CLEAN_RGB` crop as float64 RGB in `[0,1]`; fit `g = sum(pred * GT) / sum(pred²)` over
all pixels and RGB channels jointly; clip `g * pred` to `[0,1]`; round to PNG16.
The released fixed 512 evaluation crops are fully valid. There is no per-channel gain,
additive offset or extra geometry alignment. All-black predictions fail explicitly.
Compute RealBlur metrics from the saved images in `RUN/brightness_rgb_gt/predictions/`.
The new `predictions.jsonl` records each gain, and `run.json` identifies the GT-assisted
protocol. Native predictions are preserved. A fresh output directory is required.

**RawMoment-Bench predictions receive no brightness alignment.** The postprocessing
command checks the source `run.json` and rejects non-RealBlur runs and already aligned runs.

## Acknowledgments

We thank the teams behind [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M), [RealBlur](https://github.com/rimchang/RealBlur), and [GoPro](https://github.com/SeungjunNah/DeepDeblur_release) for making their datasets available.

For the RawMoment paper citation, see [Citation](README.md#citation).
