from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .anomaly_synthesis import SyntheticAnomalyGenerator


MVTECAD2_CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resample_bilinear():
    return getattr(Image, "Resampling", Image).BILINEAR


def _resample_nearest():
    return getattr(Image, "Resampling", Image).NEAREST


def _center_crop_box(width: int, height: int, crop_w: int, crop_h: int) -> Tuple[int, int, int, int]:
    left = max(0, int(round((width - crop_w) / 2.0)))
    top = max(0, int(round((height - crop_h) / 2.0)))
    return left, top, left + min(crop_w, width), top + min(crop_h, height)


def _to_chw_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask.convert("L"), dtype=np.float32)
    arr = (arr > 127.5).astype(np.float32)
    return torch.from_numpy(arr[None]).contiguous()


class MVTecAD2Transform:
    """PIL transform matching this project without depending on torchvision."""

    def __init__(
        self,
        resize: Optional[int] = 448,
        crop: Optional[int] = 392,
        resize_ratio: Optional[float] = None,
        multiple: int = 14,
    ) -> None:
        self.resize = resize
        self.crop = crop
        self.resize_ratio = resize_ratio
        self.multiple = int(multiple)

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "MVTecAD2Transform":
        cfg = cfg or {}
        strategy = cfg.get("strategy", "resize_crop")
        if strategy == "resize_only":
            return cls(
                resize=cfg.get("resize", 392),
                crop=None,
                resize_ratio=None,
                multiple=int(cfg.get("multiple", 14)),
            )
        if strategy == "ratio_multiple":
            return cls(
                resize=None,
                crop=None,
                resize_ratio=float(cfg.get("resize_ratio", 0.5)),
                multiple=int(cfg.get("multiple", 14)),
            )
        return cls(
            resize=cfg.get("resize", 448),
            crop=cfg.get("crop", 392),
            resize_ratio=None,
            multiple=int(cfg.get("multiple", 14)),
        )

    def _resize_size(self, image: Image.Image) -> Optional[Tuple[int, int]]:
        if self.resize_ratio is not None:
            width = int(round(image.width * self.resize_ratio))
            height = int(round(image.height * self.resize_ratio))
            width = max(self.multiple, int(np.ceil(width / self.multiple) * self.multiple))
            height = max(self.multiple, int(np.ceil(height / self.multiple) * self.multiple))
            return width, height
        if self.resize is None:
            return None
        if isinstance(self.resize, (list, tuple)):
            return int(self.resize[0]), int(self.resize[1])
        return int(self.resize), int(self.resize)

    def apply_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        size = self._resize_size(image)
        if size is not None:
            image = image.resize(size, _resample_bilinear())
        if self.crop is not None:
            crop_w = crop_h = int(self.crop)
            image = image.crop(_center_crop_box(image.width, image.height, crop_w, crop_h))
        return _to_chw_tensor(image)

    def apply_mask(self, mask: Image.Image, reference: Image.Image) -> torch.Tensor:
        mask = mask.convert("L")
        size = self._resize_size(reference)
        if size is not None:
            mask = mask.resize(size, _resample_nearest())
        if self.crop is not None:
            crop_w = crop_h = int(self.crop)
            mask = mask.crop(_center_crop_box(mask.width, mask.height, crop_w, crop_h))
        return _mask_to_tensor(mask)


def _list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


class MVTecAD2Dataset(Dataset):
    """MVTec AD 2 / VAND 4.0 dataset reader based on the Vand3.0 layout.

    Train reads `{root}/{item}/train/good`.
    Public eval reads `{root}/{item}/test_public/{good,bad}` and optional `ground_truth` or `gt` masks.
    Submission splits read flat image files from `test_private` and `test_private_mixed`.
    """

    def __init__(
        self,
        data_root: str,
        categories: Optional[Sequence[str]] = None,
        mode: str = "train",
        split: Optional[str] = None,
        test_splits: Optional[Sequence[str]] = None,
        transform: Optional[MVTecAD2Transform] = None,
        synthetic_generator: Optional[SyntheticAnomalyGenerator] = None,
        enable_synthetic: bool = True,
        include_validation_in_train: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")
        self.categories = list(categories or MVTECAD2_CATEGORIES)
        self.mode = mode
        self.split = split
        self.test_splits = list(test_splits or ([split] if split else ["test_private", "test_private_mixed"]))
        self.transform = transform or MVTecAD2Transform()
        self.synthetic_generator = synthetic_generator
        self.enable_synthetic = enable_synthetic
        self.include_validation_in_train = include_validation_in_train
        self.samples = self._scan()
        if not self.samples:
            raise FileNotFoundError(f"No samples found for mode={mode}, split={split}, root={self.data_root}")

    def _scan(self) -> List[Dict]:
        samples: List[Dict] = []
        for item in self.categories:
            item_root = self.data_root / item
            if not item_root.exists():
                continue
            if self.mode == "train":
                samples.extend(self._scan_train_item(item, item_root))
                continue

            for split in self.test_splits:
                if split in ("test_public", "public"):
                    samples.extend(self._scan_public_item(item, item_root / "test_public", "test_public"))
                elif split == "validation":
                    samples.extend(self._scan_good_folder(item, item_root / "validation" / "good", "validation"))
                elif split in ("test_private", "private", "test_private_mixed"):
                    canonical = "test_private" if split in ("private", "test_private") else "test_private_mixed"
                    samples.extend(self._scan_flat_split(item, item_root / canonical, canonical))
                elif split in ("test", "submission"):
                    samples.extend(self._scan_flat_split(item, item_root / "test_private", "test_private"))
                    samples.extend(self._scan_flat_split(item, item_root / "test_private_mixed", "test_private_mixed"))
                else:
                    samples.extend(self._scan_flat_split(item, item_root / split, split))
        return samples

    def _scan_train_item(self, item: str, item_root: Path) -> List[Dict]:
        paths = _list_images(item_root / "train" / "good")
        if self.include_validation_in_train:
            paths.extend(_list_images(item_root / "validation" / "good"))
        return [self._sample_dict(path, item, "train", "good", None, 0) for path in sorted(paths)]

    def _scan_good_folder(self, item: str, folder: Path, img_type: str) -> List[Dict]:
        return [self._sample_dict(path, item, img_type, "good", None, 0) for path in _list_images(folder)]

    def _scan_flat_split(self, item: str, split_root: Path, img_type: str) -> List[Dict]:
        return [self._sample_dict(path, item, img_type, img_type, None, -1, split_root) for path in _list_images(split_root)]

    def _scan_public_item(self, item: str, split_root: Path, img_type: str) -> List[Dict]:
        samples: List[Dict] = []
        for defect_type in ("bad", "good"):
            img_dir = split_root / defect_type
            for path in _list_images(img_dir):
                label = 0 if defect_type == "good" else 1
                gt_path = self._find_public_gt(split_root, path) if label == 1 else None
                samples.append(self._sample_dict(path, item, img_type, defect_type, gt_path, label, split_root))
        return samples

    @staticmethod
    def _find_public_gt(split_root: Path, image_path: Path) -> Optional[Path]:
        stems = [f"{image_path.stem}_mask", image_path.stem]
        gt_roots = [
            split_root / "ground_truth" / "bad",
            split_root / "ground_truth",
            split_root / "gt" / "bad",
            split_root / "gt",
        ]
        for root in gt_roots:
            for stem in stems:
                for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                    candidate = root / f"{stem}{ext}"
                    if candidate.exists():
                        return candidate
        return None

    def _sample_dict(
        self,
        path: Path,
        item: str,
        img_type: str,
        defect_type: str,
        gt_path: Optional[Path],
        label: int,
        relative_root: Optional[Path] = None,
    ) -> Dict:
        relative_root = relative_root or path.parent
        try:
            name = str(path.relative_to(relative_root))
        except ValueError:
            name = path.name
        return {
            "path": path,
            "item": item,
            "category": item,
            "filename": name,
            "img_type": img_type,
            "defect_type": defect_type,
            "gt_path": gt_path,
            "label": label,
            "ext": path.suffix.lstrip("."),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        image_pil = Image.open(sample["path"]).convert("RGB")
        original_size = torch.tensor([image_pil.height, image_pil.width], dtype=torch.long)
        image = self.transform.apply_image(image_pil)

        out = {
            "image": image,
            "item": sample["item"],
            "category": sample["category"],
            "filename": sample["filename"],
            "img_type": sample["img_type"],
            "defect_type": sample["defect_type"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "original_size": original_size,
            "ext": sample["ext"],
            "path": str(sample["path"]),
        }

        if self.mode == "train":
            if self.enable_synthetic and self.synthetic_generator is not None:
                anomaly_np, mask_np = self.synthetic_generator(image_pil)
                anomaly_pil = Image.fromarray(anomaly_np)
                mask_pil = Image.fromarray(mask_np, mode="L")
            else:
                anomaly_pil = image_pil
                mask_pil = Image.fromarray(np.zeros((image_pil.height, image_pil.width), dtype=np.uint8), mode="L")
            out["anomaly_image"] = self.transform.apply_image(anomaly_pil)
            out["synthetic_mask"] = self.transform.apply_mask(mask_pil, image_pil)
        else:
            if sample["gt_path"] is not None:
                mask_pil = Image.open(sample["gt_path"]).convert("L")
            else:
                mask_pil = Image.fromarray(np.zeros((image_pil.height, image_pil.width), dtype=np.uint8), mode="L")
            out["mask"] = self.transform.apply_mask(mask_pil, image_pil)

        return out

    def set_worker_seed(self, worker_id: int) -> None:
        if self.synthetic_generator is not None:
            self.synthetic_generator.reseed(seed=(torch.initial_seed() + worker_id) % (2**32))


def _pad_tensor(tensor: torch.Tensor, target_hw: Tuple[int, int], value: float = 0.0) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    pad_h = max(0, target_hw[0] - h)
    pad_w = max(0, target_hw[1] - w)
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=value)


def mvtecad2_collate(batch: Sequence[Dict]) -> Dict:
    result: Dict = {}
    tensor_keys = [key for key, value in batch[0].items() if isinstance(value, torch.Tensor)]
    image_like = {"image", "anomaly_image", "synthetic_mask", "mask"}
    max_h = max(int(item["image"].shape[-2]) for item in batch)
    max_w = max(int(item["image"].shape[-1]) for item in batch)

    for key in tensor_keys:
        values = [item[key] for item in batch]
        if key in image_like:
            values = [_pad_tensor(v, (max_h, max_w), 0.0) for v in values]
        result[key] = torch.stack(values, dim=0)

    for key in batch[0].keys():
        if key not in result:
            result[key] = [item[key] for item in batch]
    return result


def _worker_init(worker_id: int) -> None:
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed(worker_id)


def build_dataloader(
    dataset: MVTecAD2Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        collate_fn=mvtecad2_collate,
        worker_init_fn=_worker_init,
    )


def discover_categories(data_root: str) -> List[str]:
    root = Path(data_root)
    categories = [p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []
    known = [item for item in MVTECAD2_CATEGORIES if item in categories]
    return known or sorted(categories)
