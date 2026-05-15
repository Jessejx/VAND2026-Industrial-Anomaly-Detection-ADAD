import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIFF_OUTPUT_SIZE = (392, 392)

from datasets.mvtecad2 import MVTecAD2Dataset, MVTecAD2Transform, build_dataloader, discover_categories
from evaluate import compute_normal_val_threshold, normalize_config_paths, parse_categories
from evaluate import postprocess_binary_maps
from models.vand4_model import VAND4Model
from utils.analysis import save_analysis_panel
from utils.io_utils import load_checkpoint, load_config
from utils.submission import save_anomaly_tiff
from utils.visualization import save_anomaly_heatmap, smooth_anomaly_map


def _config_path(path: str) -> str:
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())


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


def _resize_if_needed(anomaly_map: np.ndarray, original_size, enabled: bool) -> np.ndarray:
    if not enabled:
        return anomaly_map
    height, width = int(original_size[0]), int(original_size[1])
    if anomaly_map.shape[:2] == (height, width):
        return anomaly_map
    return cv2.resize(anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_tiff_map(anomaly_map: np.ndarray) -> np.ndarray:
    height, width = TIFF_OUTPUT_SIZE
    if anomaly_map.shape[:2] == (height, width):
        return anomaly_map
    return cv2.resize(anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_id was set, but CUDA is not available")
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_binary_thresholded_mask(anomaly_map, threshold, item, img_type, name, root_path):
    binary_map = (np.asarray(anomaly_map) > threshold).astype(np.uint8) * 255
    binary_map = postprocess_binary_maps(binary_map, kernel_ratio=0.01, coverage_thresh=0.7, tolerance=3)
    out_name = Path(name).with_suffix(".png")
    out_path = Path(root_path) / "anomaly_images_thresholded" / item / img_type / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), binary_map)
    return out_path


def test(args):
    config_path = _config_path(args.config)
    cfg = normalize_config_paths(load_config(config_path), PROJECT_ROOT)
    if args.data_root:
        cfg.setdefault("data", {})["root"] = args.data_root
    test_cfg = cfg.get("test", {})
    data_root = cfg.get("data", {}).get("root")
    if not data_root:
        raise ValueError("data_root is required via --data_root or config data.root")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")

    device = resolve_device(args)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"using device: {device}")
    categories = parse_categories(args.categories or cfg.get("data", {}).get("categories"), data_root)
    if not categories:
        categories = discover_categories(data_root)
    splits = [x.strip() for x in (args.splits or ",".join(test_cfg.get("splits", ["test_private", "test_private_mixed"]))).split(",") if x.strip()]

    transform = MVTecAD2Transform.from_config(cfg.get("transform", {}))
    test_batch_size = int(test_cfg.get("batch_size", 1))
    test_num_workers = int(test_cfg.get("num_workers", 0))
    dataset = MVTecAD2Dataset(
        data_root=data_root,
        categories=categories,
        mode="test",
        test_splits=splits,
        transform=transform,
        enable_synthetic=False,
    )
    loader = build_dataloader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=test_num_workers)
    model = _load_model(args, cfg, device)

    output_dir = args.output_dir or test_cfg.get("output_dir", "outputs/vand4_submit")
    save_tiff = bool(test_cfg.get("save_tiff", True)) if args.save_tiff is None else bool(args.save_tiff)
    save_thresholded = (
        bool(test_cfg.get("save_thresholded", False)) if args.save_thresholded is None else bool(args.save_thresholded)
    )
    resize_to_original = (
        bool(test_cfg.get("resize_to_original", False)) if args.resize_to_original is None else bool(args.resize_to_original)
    )
    save_heatmap = bool(test_cfg.get("save_heatmap", False)) if args.save_heatmap is None else bool(args.save_heatmap)
    save_analysis = bool(test_cfg.get("save_analysis", True)) if args.save_analysis is None else bool(args.save_analysis)
    heatmap_alpha = float(test_cfg.get("heatmap_alpha", 0.45))
    gaussian_sigma = float(test_cfg.get("gaussian_sigma", 4.0))
    threshold_quantile = (
        float(test_cfg.get("threshold_quantile", cfg.get("train", {}).get("eval_threshold_quantile", 0.995)))
        if args.threshold_quantile is None
        else float(args.threshold_quantile)
    )
    binary_threshold = args.threshold
    needs_binary_threshold = save_thresholded or save_analysis
    if needs_binary_threshold and binary_threshold is None:
        binary_threshold = compute_normal_val_threshold(
            model=model,
            data_root=data_root,
            categories=categories,
            transform=transform,
            batch_size=test_batch_size,
            num_workers=test_num_workers,
            device=device,
            gaussian_sigma=gaussian_sigma,
            quantile=threshold_quantile,
        )
        print(f"binary threshold validation threshold@{threshold_quantile:.3f}: {binary_threshold:.6f}")
    elif needs_binary_threshold:
        binary_threshold = float(binary_threshold)
        print(f"binary threshold: {binary_threshold:.6f}")

    written_tiff = 0
    written_masks = 0
    written_heatmaps = 0
    written_analysis = 0
    processed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="test", ncols=100):
            image = batch["image"].to(device, non_blocking=True)
            output = model(image, mode="test")
            maps = output["anomaly_map"].detach().cpu().numpy()
            for i in range(maps.shape[0]):
                if args.max_samples is not None and processed >= args.max_samples:
                    break
                anomaly_map = maps[i, 0].astype(np.float32)
                anomaly_map = smooth_anomaly_map(anomaly_map, sigma=gaussian_sigma)
                anomaly_map = _resize_if_needed(anomaly_map, batch["original_size"][i].tolist(), resize_to_original)

                item = batch["item"][i]
                img_type = batch["img_type"][i]
                name = batch["filename"][i]
                ext = batch["ext"][i]

                ######################################### Saving #########################################
                # if save_heatmap:
                #     heatmap_image = batch["path"][i] if resize_to_original else batch["image"][i]
                #     save_anomaly_heatmap(anomaly_map, heatmap_image, item, img_type, name, output_dir, alpha=heatmap_alpha)
                #     written_heatmaps += 1
                # if save_analysis:
                #     analysis_image = batch["path"][i] if resize_to_original else batch["image"][i]
                #     save_analysis_panel(
                #         anomaly_map,
                #         analysis_image,
                #         binary_threshold,
                #         item,
                #         img_type,
                #         name,
                #         output_dir,
                #         alpha=heatmap_alpha,
                #     )
                #     written_analysis += 1
                if save_tiff:
                    save_anomaly_tiff(_resize_tiff_map(anomaly_map), item, img_type, name, ext, output_dir)
                    written_tiff += 1
                if save_thresholded:
                    save_binary_thresholded_mask(anomaly_map, binary_threshold, item, img_type, name, output_dir)
                    written_masks += 1
                
    

                processed += 1
            if args.max_samples is not None and processed >= args.max_samples:
                break

    print(
        f"completed: {processed}/{len(dataset)} samples, "
        f"tiff={written_tiff}, thresholded={written_masks}, heatmap={written_heatmaps}, "
        f"analysis={written_analysis}, output_dir={output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mvtecad2_inp_residual.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Manual binary threshold; default uses normal validation quantile")
    parser.add_argument("--threshold_quantile", type=float, default=None, help="Normal validation pixel quantile for binary masks")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_tiff", dest="save_tiff", action="store_true", default=None)
    parser.add_argument("--no_save_tiff", dest="save_tiff", action="store_false")
    parser.add_argument("--save_thresholded", dest="save_thresholded", action="store_true", default=None)
    parser.add_argument("--no_save_thresholded", dest="save_thresholded", action="store_false")
    parser.add_argument("--save_heatmap", dest="save_heatmap", action="store_true", default=None)
    parser.add_argument("--no_save_heatmap", dest="save_heatmap", action="store_false")
    parser.add_argument("--save_analysis", dest="save_analysis", action="store_true", default=None)
    parser.add_argument("--no_save_analysis", dest="save_analysis", action="store_false")
    parser.add_argument("--thresholded_from_heatmap", dest="thresholded_from_heatmap", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no_thresholded_from_heatmap", dest="thresholded_from_heatmap", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--heatmap_mask_level", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resize_to_original", dest="resize_to_original", action="store_true", default=None)
    parser.add_argument("--no_resize_to_original", dest="resize_to_original", action="store_false")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=2, help="CUDA device index, for example --gpu_id 0")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
