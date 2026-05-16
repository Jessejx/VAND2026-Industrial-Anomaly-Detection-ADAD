from pathlib import Path

import cv2
import numpy as np
import torch


def denormalize_chw(tensor):
    arr = tensor.detach().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr * std + mean).clip(0, 1)
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def smooth_anomaly_map(anomaly_map: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(anomaly_map, sigma=sigma)
    except Exception:
        k = max(3, int(round(sigma * 4)) | 1)
        return cv2.GaussianBlur(anomaly_map.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)


def normalize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    """Normalize one anomaly map to uint8 for visualization only."""
    arr = np.asarray(anomaly_map, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    valid = arr[finite]
    min_val = float(valid.min())
    max_val = float(valid.max())
    if max_val - min_val < 1.0e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - min_val) / (max_val - min_val)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def _image_to_rgb(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image for heatmap: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image, torch.Tensor):
        return denormalize_chw(image)

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def make_anomaly_heatmap(anomaly_map: np.ndarray, image=None, alpha: float = 0.45) -> np.ndarray:
    """Build a JET heatmap overlay in BGR format for cv2.imwrite."""
    heatmap_u8 = normalize_anomaly_map(anomaly_map)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    if image is None:
        return heatmap_bgr

    base_rgb = _image_to_rgb(image)
    height, width = heatmap_bgr.shape[:2]
    if base_rgb.shape[:2] != (height, width):
        base_rgb = cv2.resize(base_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def save_anomaly_heatmap(
    anomaly_map: np.ndarray,
    image,
    item: str,
    img_type: str,
    name: str,
    root_path: str,
    alpha: float = 0.45,
):
    out_name = Path(name).with_suffix(".png")
    out_path = Path(root_path) / "anomaly_heatmaps" / item / img_type / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = make_anomaly_heatmap(anomaly_map, image=image, alpha=alpha)
    cv2.imwrite(str(out_path), heatmap)
    return out_path
