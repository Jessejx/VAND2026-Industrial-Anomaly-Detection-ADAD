# VAND 2026 Industrial Anomaly Detection

## Overview

This repository contains a one-command training and inference package for the VAND 2026 Industrial Anomaly Detection submission. The method uses an INP-Former style DINOv2 feature reconstruction branch with a residual anomaly head. Training synthesizes pseudo anomalies from random noise and Perlin masks, so no external image dataset is required.

## Project Structure

```text
VAND_2026/
  configs/
    mvtecad2_inp_residual.yaml
  datasets/
    anomaly_synthesis.py
    mvtecad2.py
  losses/
    loss_cdo.py
  models/
    dino_backbones/
    inp_former.py
    residual_head.py
    vand4_model.py
  utils/
    analysis.py
    io_utils.py
    metrics.py
    seed.py
    submission.py
    visualization.py
  checkpoints/
    vand2026_inp_residual_noise.pth
  train.py
  test.py
  evaluate.py
  requirements.txt
```

## Environment

```bash
conda create -n vand2026 python=3.10 -y
conda activate vand2026
pip install -r requirements.txt
```

Install the correct PyTorch build for your CUDA version if the default `pip install` build is not suitable for your machine.

## Dataset Preparation

Set `data.root` in `configs/mvtecad2_inp_residual.yaml`, or pass `--data_root` on the command line.

Expected layout:

```text
data/mvtec_ad_2/
  can/
    train/good/
    validation/good/
    test_public/good/
    test_public/bad/
    test_public/ground_truth/
    test_private/
    test_private_mixed/
  fabric/
  fruit_jelly/
  rice/
  sheet_metal/
  vial/
  wallplugs/
  walnuts/
```

## Training

```bash
python train.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --data_root /path/to/mvtec_ad_2 \
  --device cuda
```

Training saves periodic checkpoints under `checkpoints/`. The final checkpoint is also saved as:

```text
checkpoints/vand2026_inp_residual_noise.pth
```

## Inference

Generate anomaly maps for the private splits with the default checkpoint:

```bash
python test.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --data_root /path/to/mvtec_ad_2 \
  --checkpoint checkpoints/vand2026_inp_residual_noise.pth \
  --output_dir outputs/vand2026_submit \
  --device cuda
```

If `--checkpoint` is omitted, `test.py` uses `checkpoints/vand2026_inp_residual_noise.pth`.

## Submission Generation

`test.py` writes competition-style anomaly maps to:

```text
outputs/vand2026_submit/anomaly_images/<category>/<split>/*.tiff
```

When enabled in the config or with `--save_thresholded`, binary segmentation masks are written to:

```text
outputs/vand2026_submit/anomaly_images_thresholded/<category>/<split>/*.png
```

Configured per-category thresholds are used first for binary masks. If the threshold table is removed, `test.py` estimates a threshold from `validation/good`.

## Local Evaluation

Evaluate on the public split:

```bash
python evaluate.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --data_root /path/to/mvtec_ad_2 \
  --checkpoint checkpoints/vand2026_inp_residual_noise.pth \
  --splits test_public \
  --device cuda
```

## Checkpoint

Default submission checkpoint:

```text
checkpoints/vand2026_inp_residual_noise.pth
```

Backbone weights are cached under `backbones/weights`. If the DINOv2 weight file is not already present, the model loader downloads it automatically.

## Notes

This submission version only supports `synthetic.anomaly_mode: random_noise`. Pseudo anomaly masks are generated from random Perlin masks, and pseudo anomaly patterns are generated from random noise.
