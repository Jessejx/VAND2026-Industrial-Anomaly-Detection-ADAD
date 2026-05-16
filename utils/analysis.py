from pathlib import Path

import cv2
import numpy as np
import torch

from .visualization import denormalize_chw, make_anomaly_heatmap


ANALYSIS_PANEL_SIZE = (392, 392)


def _resize_panel_image(image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(image, ANALYSIS_PANEL_SIZE, interpolation=interpolation)


def _image_to_bgr(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read analysis source image: {image}")
        return bgr

    if isinstance(image, torch.Tensor):
        rgb = denormalize_chw(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = arr.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _mask_to_bgr(mask) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        arr = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Could not read analysis mask: {mask}")
    elif isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3:
            arr = arr[..., 0]

    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    mask_u8 = (arr > 0).astype(np.uint8) * 255
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def save_analysis_panel(anomaly_map, image, threshold, item, img_type, name, root_path, alpha=0.45):
    image_bgr = _resize_panel_image(_image_to_bgr(image))

    binary_map = (np.asarray(anomaly_map) > threshold).astype(np.uint8) * 255
    binary_bgr = cv2.cvtColor(_resize_panel_image(binary_map, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    heatmap_bgr = _resize_panel_image(make_anomaly_heatmap(anomaly_map, image=image, alpha=alpha))
    panel = np.concatenate([image_bgr, binary_bgr, heatmap_bgr], axis=1)

    out_name = Path(name).with_suffix(".png")
    out_path = Path(root_path) / "analysis_panels" / item / img_type / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    return out_path


def save_evaluate_analysis_panel(anomaly_map, image, gt_mask, threshold, item, img_type, name, root_path, alpha=0.45):
    image_bgr = _resize_panel_image(_image_to_bgr(image))
    gt_bgr = _resize_panel_image(_mask_to_bgr(gt_mask), interpolation=cv2.INTER_NEAREST)

    binary_map = (np.asarray(anomaly_map) > threshold).astype(np.uint8) * 255
    binary_bgr = cv2.cvtColor(_resize_panel_image(binary_map, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    heatmap_bgr = _resize_panel_image(make_anomaly_heatmap(anomaly_map, image=image, alpha=alpha))
    panel = np.concatenate([image_bgr, gt_bgr, binary_bgr, heatmap_bgr], axis=1)

    out_name = Path(name).with_suffix(".png")
    out_path = Path(root_path) / "eval_analysis_panels" / item / img_type / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    return out_path
