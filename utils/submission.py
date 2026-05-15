import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import tifffile as tiff


DEFAULT_THRESHOLDS_PRIVATE = {
    "can": 0.203,
    "fabric": 0.122,
    "fruit_jelly": 0.177,
    "rice": 0.153,
    "sheet_metal": 0.216,
    "vial": 0.219,
    "wallplugs": 0.195,
    "walnuts": 0.134,
}

DEFAULT_THRESHOLDS_MIXED = {
    "can": 0.203,
    "fabric": 0.122,
    "fruit_jelly": 0.177,
    "rice": 0.153,
    "sheet_metal": 0.216,
    "vial": 0.219,
    "wallplugs": 0.155,
    "walnuts": 0.134,
}


def _output_name(name: str, suffix: str) -> Path:
    return Path(name).with_suffix(suffix)


def save_anomaly_tiff(anomaly_map, item: str, img_type: str, name: str, ext: str, root_path: str):
    out_name = _output_name(name, ".tiff")
    out_path = Path(root_path) / "anomaly_images" / item / img_type / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), np.asarray(anomaly_map).astype(np.float16))
    return out_path


def threshold_for(item: str, img_type: str, thresholds: Optional[Dict] = None, default: float = 0.5) -> float:
    thresholds = thresholds or {}
    if item in thresholds:
        value = thresholds[item]
        if isinstance(value, dict):
            if img_type in value:
                return float(value[img_type])
            if "default" in value:
                return float(value["default"])
        return float(value)
    if img_type == "test_private":
        return float(DEFAULT_THRESHOLDS_PRIVATE.get(item, default))
    if img_type == "test_private_mixed":
        return float(DEFAULT_THRESHOLDS_MIXED.get(item, default))
    return float(default)


def save_thresholded_mask(anomaly_map, threshold, item, img_type, name, ext, root_path, verbose=False):
    """Save Vand3.0-style thresholded binary mask with category-specific morphology."""
    anomaly_map = np.asarray(anomaly_map)
    thred = np.where(anomaly_map > threshold, 255, 0).astype(np.uint8)
    if item == "fabric":
        if img_type == "test_private":
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
        else:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (111, 111))
    elif item == "sheet_metal":
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    else:
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    dilated = cv2.dilate(thred, kernel1, iterations=1)
    fill_holes = dilated.copy()
    contours, _ = cv2.findContours(fill_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(fill_holes, [cnt], 0, 255, thickness=cv2.FILLED)
    thred = cv2.erode(fill_holes, kernel2, iterations=1)

    if verbose:
        print(threshold, thred.shape, anomaly_map.shape)
    out_name = _output_name(name, ".png")
    out_path = Path(root_path) / "anomaly_images_thresholded" / item / img_type / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), thred.astype(np.uint8))
    return out_path


def collect_submission_files(root_path: str):
    root = Path(root_path) / "anomaly_images"
    return sorted(str(p) for p in root.rglob("*.tiff")) if root.exists() else []
