import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.mvtecad2 import MVTecAD2Dataset, MVTecAD2Transform, build_dataloader, discover_categories
from models.vand4_model import VAND4Model
from utils.analysis import save_evaluate_analysis_panel
from utils.io_utils import load_checkpoint, load_config
from utils.metrics import f1_score_max
from utils.seed import seed_everything
from utils.visualization import smooth_anomaly_map


def _resolve_path(value, base: Path):
    if value in (None, ""):
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def _config_path(path: str) -> str:
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())


def normalize_config_paths(cfg, project_root: Path):
    backbone = cfg.setdefault("model", {}).setdefault("backbone", {})
    backbone["source_root"] = _resolve_path(backbone.get("source_root"), project_root)
    backbone["weights_dir"] = _resolve_path(backbone.get("weights_dir"), project_root)
    synthetic = cfg.setdefault("synthetic", {})
    synthetic["dtd_root"] = _resolve_path(synthetic.get("dtd_root"), project_root)
    return cfg


def parse_categories(value, data_root):
    if value:
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return list(value)
    return discover_categories(data_root)


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_id was set, but CUDA is not available")
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(args, cfg, device):
    model = VAND4Model.from_config(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=args.strict)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def compute_pixel_auroc(gt_masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    gt_flat = (gt_masks > 0).astype(np.uint8).reshape(-1)
    pred_flat = anomaly_maps.astype(np.float32).reshape(-1)
    if len(np.unique(gt_flat)) <= 1:
        return float("nan")
    return float(roc_auc_score(gt_flat, pred_flat))


def _as_binary_uint8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _find_external_contours(mask: np.ndarray):
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def get_adaptive_kernel(mask, kernel_ratio=0.01):
    shape = np.asarray(mask).shape
    if len(shape) < 2:
        raise ValueError(f"mask must have at least 2 dimensions, got shape {shape}")
    height, width = shape[-2], shape[-1]
    kernel_size = max(3, int(round(min(height, width) * float(kernel_ratio))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def compute_boundary_coverage(original_mask, filled_mask, tolerance=3) -> float:
    original = _as_binary_uint8(original_mask)
    filled = _as_binary_uint8(filled_mask)
    if original.size == 0 or filled.size == 0:
        return 0.0

    contours = _find_external_contours(filled)
    if not contours:
        return 0.0

    boundary = np.zeros_like(filled, dtype=np.uint8)
    cv2.drawContours(boundary, contours, -1, 255, thickness=1)
    boundary_pixels = boundary > 0
    boundary_count = boundary_pixels.sum(dtype=np.float64)
    if boundary_count <= 0:
        return 0.0

    tolerance = max(0, int(round(float(tolerance))))
    if tolerance > 0:
        kernel_size = 2 * tolerance + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        original_cover = cv2.dilate(original, kernel, iterations=1) > 0
    else:
        original_cover = original > 0

    covered = np.logical_and(boundary_pixels, original_cover).sum(dtype=np.float64)
    return float(covered / boundary_count)


def semi_closed_fill_componentwise(binary_mask, kernel_ratio=0.01, coverage_thresh=0.8, tolerance=3):
    binary = _as_binary_uint8(binary_mask)
    if binary.ndim != 2:
        raise ValueError(f"binary_mask must have shape (H, W), got {binary.shape}")
    if binary.size == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    component_seed = (binary > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(component_seed, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    output = np.zeros_like(binary, dtype=np.uint8)
    kernel = get_adaptive_kernel(binary, kernel_ratio=kernel_ratio)

    for label_idx in range(1, num_labels):
        component = np.zeros_like(binary, dtype=np.uint8)
        component[labels == label_idx] = 255

        closed = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)
        contours = _find_external_contours(closed)
        if not contours:
            output = cv2.bitwise_or(output, component)
            continue

        filled_component = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(filled_component, contours, -1, 255, thickness=cv2.FILLED)

        coverage = compute_boundary_coverage(component, filled_component, tolerance=tolerance)
        if coverage >= float(coverage_thresh):
            output = cv2.bitwise_or(output, filled_component)
        else:
            output = cv2.bitwise_or(output, component)

    return output


def postprocess_binary_maps(binary_maps, kernel_ratio=0.01, coverage_thresh=0.8, tolerance=3):
    binary = _as_binary_uint8(binary_maps)
    if binary.ndim == 2:
        return semi_closed_fill_componentwise(
            binary,
            kernel_ratio=kernel_ratio,
            coverage_thresh=coverage_thresh,
            tolerance=tolerance,
        )
    if binary.ndim != 3:
        raise ValueError(f"binary_maps must have shape (H, W) or (N, H, W), got {binary.shape}")
    if binary.shape[0] == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    processed = [
        semi_closed_fill_componentwise(
            binary[i],
            kernel_ratio=kernel_ratio,
            coverage_thresh=coverage_thresh,
            tolerance=tolerance,
        )
        for i in range(binary.shape[0])
    ]
    return np.stack(processed, axis=0).astype(np.uint8, copy=False)


def compute_seg_f1_at_threshold_2(anomaly_maps: np.ndarray, gt_masks: np.ndarray, threshold: float) -> float:
    binary_maps = (anomaly_maps > threshold).astype(np.uint8) * 255
    binary_maps = postprocess_binary_maps(binary_maps, kernel_ratio=0.01, coverage_thresh=0.7, tolerance=3)
    pred = binary_maps > 0
    gt = gt_masks > 0
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.float64)
    denom = 2.0 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return float((2.0 * tp) / denom)


def compute_seg_f1_at_threshold(anomaly_maps: np.ndarray, gt_masks: np.ndarray, threshold: float) -> float:
    return compute_seg_f1_at_threshold_2(anomaly_maps, gt_masks, threshold)


def compute_seg_f1_max(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> Tuple[float, float]:
    gt_flat = (gt_masks > 0).astype(np.uint8).reshape(-1)
    pred_flat = anomaly_maps.astype(np.float32).reshape(-1)
    if pred_flat.size == 0:
        return 0.0, float("nan")
    return f1_score_max(gt_flat, pred_flat)


def quantile_from_score_arrays(score_arrays, quantile: float) -> float:
    quantile = float(quantile)
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    values = np.concatenate(score_arrays).astype(np.float32, copy=False)
    if values.size == 0:
        raise ValueError("Cannot compute quantile from empty score array")

    position = quantile * float(values.size - 1)
    lower_idx = int(np.floor(position))
    upper_idx = int(np.ceil(position))
    if lower_idx == upper_idx:
        values.partition(lower_idx)
        return float(values[lower_idx])

    values.partition((lower_idx, upper_idx))
    lower = float(values[lower_idx])
    upper = float(values[upper_idx])
    return lower + (upper - lower) * (position - lower_idx)


def compute_normal_val_threshold(
    model,
    data_root,
    categories,
    transform,
    batch_size,
    num_workers,
    device,
    gaussian_sigma,
    quantile=0.995,
):
    normal_val_data = MVTecAD2Dataset(
        data_root=data_root,
        categories=categories,
        mode="test",
        test_splits=["validation"],
        transform=transform,
        enable_synthetic=False,
    )
    normal_val_loader = build_dataloader(
        normal_val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    all_scores = []
    category_label = ",".join(categories)
    with torch.no_grad():
        for batch in tqdm(normal_val_loader, desc=f"threshold validation {category_label}", ncols=100, leave=False):
            image = batch["image"].to(device, non_blocking=True)
            output = model(image, mode="test")
            anomaly_np = output["anomaly_map"][:, 0].detach().cpu().numpy()
            for i in range(anomaly_np.shape[0]):
                amap = smooth_anomaly_map(anomaly_np[i].astype(np.float32), sigma=gaussian_sigma)
                all_scores.append(amap.reshape(-1).astype(np.float32, copy=False))

    if not all_scores:
        raise ValueError(f"No normal validation scores found for categories: {categories}")
    return quantile_from_score_arrays(all_scores, quantile)


def evaluate_public(
    model,
    data_root,
    categories,
    cfg,
    device,
    splits=None,
    metric_mode="segf1",
    save_analysis=False,
    output_dir=None,
    heatmap_alpha=0.45,
):
    transform = MVTecAD2Transform.from_config(cfg.get("transform", {}))
    train_cfg = cfg.get("train", {})
    test_cfg = cfg.get("test", {})
    batch_size = int(train_cfg.get("eval_batch_size", test_cfg.get("batch_size", 1)))
    num_workers = int(train_cfg.get("eval_num_workers", test_cfg.get("num_workers", 0)))
    gaussian_sigma = float(train_cfg.get("eval_gaussian_sigma", test_cfg.get("gaussian_sigma", 4.0)))
    threshold_quantile = float(train_cfg.get("eval_threshold_quantile", 0.995))
    splits = ["test_public"] if splits is None else list(splits)
    split_label = ",".join(splits)
    metric_prefix = "Public " if splits == ["test_public"] else f"{split_label} "
    metric_mode = str(metric_mode).lower()
    if metric_mode not in ("segf1", "segf1max"):
        raise ValueError(f"metric_mode must be 'segf1' or 'segf1max', got {metric_mode}")
    save_analysis = bool(save_analysis)
    if save_analysis:
        output_dir = str(output_dir or test_cfg.get("output_dir", "outputs/vand4_eval"))
        heatmap_alpha = float(heatmap_alpha)

    model.eval()
    category_metrics = []
    written_analysis = 0
    threshold = None
    if metric_mode == "segf1":
        try:
            threshold = compute_normal_val_threshold(
                model=model,
                data_root=data_root,
                categories=categories,
                transform=transform,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                gaussian_sigma=gaussian_sigma,
                quantile=threshold_quantile,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[eval] cannot compute normal validation threshold: {exc}")
            return {}

        print(f"{metric_prefix}SegF1 threshold@{threshold_quantile:.3f}: {threshold:.6f}")
    else:
        print(f"{metric_prefix}SegF1Max: selecting the best threshold from labeled eval masks")

    with torch.no_grad():
        for item in categories:
            try:
                public_data = MVTecAD2Dataset(
                    data_root=data_root,
                    categories=[item],
                    mode="test",
                    test_splits=splits,
                    transform=transform,
                    enable_synthetic=False,
                )
            except FileNotFoundError as exc:
                print(f"[eval] skip {item}: {exc}")
                continue

            if len(public_data) == 0:
                print(f"[eval] skip {item}: no samples in splits {split_label}")
                continue

            public_loader = build_dataloader(
                public_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )
            gt_px_list, pr_px_list = [], []
            analysis_samples = []
            for batch in tqdm(public_loader, desc=f"eval {split_label} {item}", ncols=100, leave=False):
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"]
                output = model(image, mode="test")
                anomaly = output["anomaly_map"]
                if anomaly.shape[-2:] != mask.shape[-2:]:
                    anomaly = F.interpolate(anomaly, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                anomaly_np = anomaly[:, 0].detach().cpu().numpy()
                mask_np = mask[:, 0].numpy().astype(np.uint8)

                for i in range(anomaly_np.shape[0]):
                    amap = smooth_anomaly_map(anomaly_np[i].astype(np.float32), sigma=gaussian_sigma)
                    pr_px_list.append(amap)
                    gt_px_list.append(mask_np[i])
                    if save_analysis:
                        analysis_samples.append(
                            (
                                batch["image"][i].detach().cpu(),
                                batch["item"][i],
                                batch["img_type"][i],
                                batch["filename"][i],
                            )
                        )

            gt_px = np.stack(gt_px_list, axis=0)
            pr_px = np.stack(pr_px_list, axis=0)
            auroc_px = compute_pixel_auroc(gt_px, pr_px)
            if metric_mode == "segf1max":
                f1_px, f1_threshold = compute_seg_f1_max(pr_px, gt_px)
            else:
                f1_px = compute_seg_f1_at_threshold(pr_px, gt_px, threshold)
                f1_threshold = threshold
            category_metrics.append((auroc_px, f1_px))
            print(
                "{} {}Pixel Auroc:{:.3f},  Pixel {}:{:.3f},  threshold:{:.6f}".format(
                    item,
                    metric_prefix,
                    auroc_px,
                    "SegF1Max" if metric_mode == "segf1max" else "SegF1",
                    f1_px,
                    f1_threshold,
                )
            )
            if save_analysis:
                for (analysis_image, sample_item, img_type, name), amap, gt_mask in zip(
                    analysis_samples, pr_px, gt_px
                ):
                    save_evaluate_analysis_panel(
                        amap,
                        analysis_image,
                        gt_mask,
                        f1_threshold,
                        sample_item,
                        img_type,
                        name,
                        output_dir,
                        alpha=heatmap_alpha,
                    )
                    written_analysis += 1

    if save_analysis:
        print(f"{metric_prefix}analysis panels: {written_analysis}, output_dir={output_dir}")
    if category_metrics:
        arr = np.asarray(category_metrics, dtype=np.float32)
        mean_metrics = np.nanmean(arr, axis=0)
        print(
            "{}Mean Pixel Auroc:{:.3f},  Pixel {}:{:.3f}".format(
                metric_prefix,
                mean_metrics[0],
                "SegF1Max" if metric_mode == "segf1max" else "SegF1",
                mean_metrics[1],
            )
        )
        result = {
            "pixel_auroc": float(mean_metrics[0]),
            "pixel_f1": float(mean_metrics[1]),
        }
        if metric_mode == "segf1max":
            result["pixel_f1max"] = float(mean_metrics[1])
        return result
    return {}


def evaluate(args):
    cfg = normalize_config_paths(load_config(_config_path(args.config)), PROJECT_ROOT)
    if args.data_root:
        cfg.setdefault("data", {})["root"] = args.data_root

    train_cfg = cfg.setdefault("train", {})
    test_cfg = cfg.get("test", {})
    if args.batch_size is not None:
        train_cfg["eval_batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["eval_num_workers"] = args.num_workers
    if args.gaussian_sigma is not None:
        train_cfg["eval_gaussian_sigma"] = args.gaussian_sigma
    if args.threshold_quantile is not None:
        train_cfg["eval_threshold_quantile"] = args.threshold_quantile

    seed_everything(int(cfg.get("seed", 1)))
    data_root = cfg.get("data", {}).get("root")
    if not data_root:
        raise ValueError("data_root is required via --data_root or config data.root")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")

    device = resolve_device(args)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    categories = parse_categories(args.categories or cfg.get("data", {}).get("categories"), data_root)
    if not categories:
        raise ValueError(f"No categories found under {data_root}")
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split name")

    print(f"using device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"categories: {categories}")
    print(f"splits: {splits}")
    model = _load_model(args, cfg, device)
    output_dir_arg = getattr(args, "output_dir", None)
    heatmap_alpha_arg = getattr(args, "heatmap_alpha", None)
    output_dir = output_dir_arg or test_cfg.get("output_dir", "outputs/vand4_eval")
    heatmap_alpha = (
        float(test_cfg.get("heatmap_alpha", 0.45)) if heatmap_alpha_arg is None else float(heatmap_alpha_arg)
    )
    return evaluate_public(
        model,
        data_root,
        categories,
        cfg,
        device,
        splits=splits,
        save_analysis=True,
        output_dir=output_dir,
        heatmap_alpha=heatmap_alpha,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mvtecad2_inp_residual.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--splits", type=str, default="test_public")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gaussian_sigma", type=float, default=None)
    parser.add_argument("--threshold_quantile", type=float, default=None)

    parser.add_argument("--heatmap_alpha", type=float, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=2, help="CUDA device index, for example --gpu_id 0")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
