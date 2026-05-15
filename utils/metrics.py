from typing import Optional

import numpy as np
import torch
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def min_max_norm_np(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + eps)


def image_score_from_map(anomaly_map: torch.Tensor, max_ratio: float = 0.01) -> torch.Tensor:
    flat = anomaly_map.flatten(1)
    if max_ratio <= 0:
        return flat.max(dim=1)[0]
    k = max(1, int(flat.shape[1] * max_ratio))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def f1_score_max(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    if thresholds.size == 0 or f1.size == 0:
        return 0.0, 0.5
    idx = int(np.nanargmax(f1))
    return float(f1[idx]), float(thresholds[idx])


def binary_metrics(gt_px: np.ndarray, pred_px: np.ndarray, gt_sp: Optional[np.ndarray] = None, pred_sp=None):
    gt_flat = gt_px.reshape(-1).astype(np.uint8)
    pred_flat = pred_px.reshape(-1)
    out = {}
    if len(np.unique(gt_flat)) > 1:
        out["pixel_auroc"] = float(roc_auc_score(gt_flat, pred_flat))
        out["pixel_ap"] = float(average_precision_score(gt_flat, pred_flat))
        out["pixel_f1"], out["pixel_threshold"] = f1_score_max(gt_flat, pred_flat)
    if gt_sp is not None and pred_sp is not None and len(np.unique(gt_sp)) > 1:
        out["image_auroc"] = float(roc_auc_score(gt_sp, pred_sp))
        out["image_ap"] = float(average_precision_score(gt_sp, pred_sp))
        out["image_f1"], out["image_threshold"] = f1_score_max(gt_sp, pred_sp)
    return out


def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> float:
    """Compute pixel AUPRO up to 0.3 FPR, following the Vand3 evaluation style."""
    masks = masks.astype(np.uint8)
    amaps = amaps.astype(np.float32)
    if masks.ndim != 3 or amaps.ndim != 3:
        raise ValueError("masks and amaps must have shape [N,H,W]")
    if masks.shape != amaps.shape:
        raise ValueError(f"masks/amaps shape mismatch: {masks.shape} vs {amaps.shape}")
    if masks.max() == 0:
        return float("nan")

    min_th, max_th = float(amaps.min()), float(amaps.max())
    if max_th <= min_th:
        return 0.0

    pros = []
    fprs = []
    inverse_masks = 1 - masks
    inverse_area = inverse_masks.sum()
    if inverse_area == 0:
        return float("nan")

    for threshold in np.linspace(min_th, max_th, num_th, endpoint=False):
        binary_amaps = (amaps > threshold).astype(bool)
        pro_values = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pro_values.append(tp_pixels / region.area)
        if not pro_values:
            continue
        fp_pixels = np.logical_and(inverse_masks.astype(bool), binary_amaps).sum()
        fpr = fp_pixels / inverse_area
        if fpr < 0.3:
            pros.append(float(np.mean(pro_values)))
            fprs.append(float(fpr))

    if len(fprs) < 2:
        return 0.0
    fprs = np.asarray(fprs, dtype=np.float32)
    pros = np.asarray(pros, dtype=np.float32)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    max_fpr = fprs.max()
    if max_fpr <= 0:
        return 0.0
    return float(auc(fprs / max_fpr, pros))
