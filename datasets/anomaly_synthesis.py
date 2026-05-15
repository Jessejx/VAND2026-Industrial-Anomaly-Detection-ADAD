import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _fade(x: np.ndarray) -> np.ndarray:
    return ((6.0 * x - 15.0) * x + 10.0) * x**3


def rand_perlin_2d_np(shape: Tuple[int, int], res: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Fast numpy Perlin noise, adapted from the DRAEM-style code used by the source framework."""
    height, width = int(shape[0]), int(shape[1])
    rx, ry = max(1, int(res[0])), max(1, int(res[1]))

    angles = (2.0 * np.pi * rng.random((rx + 1, ry + 1))).astype(np.float32)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    uu = np.linspace(0, rx, height, endpoint=False, dtype=np.float32)
    vv = np.linspace(0, ry, width, endpoint=False, dtype=np.float32)
    u, v = np.meshgrid(uu, vv, indexing="ij")
    ix = np.floor(u).astype(np.int32)
    iy = np.floor(v).astype(np.int32)
    fu = u - ix
    fv = v - iy

    ix0 = ix % rx
    iy0 = iy % ry
    ix1 = (ix + 1) % rx
    iy1 = (iy + 1) % ry

    g00 = gradients[ix0, iy0]
    g10 = gradients[ix1, iy0]
    g01 = gradients[ix0, iy1]
    g11 = gradients[ix1, iy1]

    n00 = g00[..., 0] * fu + g00[..., 1] * fv
    n10 = g10[..., 0] * (fu - 1.0) + g10[..., 1] * fv
    n01 = g01[..., 0] * fu + g01[..., 1] * (fv - 1.0)
    n11 = g11[..., 0] * (fu - 1.0) + g11[..., 1] * (fv - 1.0)

    su = _fade(fu)
    sv = _fade(fv)
    nx0 = n00 * (1.0 - su) + n10 * su
    nx1 = n01 * (1.0 - su) + n11 * su
    return (np.sqrt(2.0) * (nx0 * (1.0 - sv) + nx1 * sv)).astype(np.float32)


def _shape_from_value(value) -> Optional[Tuple[int, ...]]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [int(x) for x in re.split(r"[xX,\s]+", value.strip()) if x]
    else:
        parts = [int(x) for x in value]
    return tuple(parts)


def _target_size_from_value(value) -> Optional[Tuple[int, int]]:
    shape = _shape_from_value(value)
    if shape is None:
        return None
    if len(shape) != 2:
        raise ValueError(f"target_size must be (W, H), got {value}")
    return int(shape[0]), int(shape[1])


def _load_mmap_shape_sidecar(path: Path) -> Optional[Tuple[int, int, int, int]]:
    candidates = [
        Path(str(path) + ".json"),
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        shape = data.get("shape", data) if isinstance(data, dict) else data
        parts = _shape_from_value(shape)
        if parts is not None and len(parts) == 4:
            return tuple(int(x) for x in parts)
    return None


def _infer_hwc_from_mmap_name(path: Path) -> Optional[Tuple[int, int, int]]:
    match = re.search(r"(\d+)x(\d+)x(\d+)", path.name)
    if match is None:
        return None
    height, width, channels = (int(match.group(i)) for i in range(1, 4))
    return height, width, channels


def _infer_mmap_shape(path: Path, mmap_shape, target_size: Optional[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    shape = _shape_from_value(mmap_shape)
    if shape is None:
        shape = _load_mmap_shape_sidecar(path)
    if shape is None:
        hwc = _infer_hwc_from_mmap_name(path)
        if hwc is None and target_size is not None:
            width, height = target_size
            hwc = (height, width, 3)
        if hwc is None:
            raise ValueError(
                "backend=memmap requires mmap_shape=(N,H,W,C), or a mmap filename like dtd_518x518x3.mmap"
            )
        height, width, channels = hwc
        bytes_per_image = height * width * channels
        file_size = os.path.getsize(path)
        if file_size % bytes_per_image != 0:
            raise ValueError(
                f"Cannot infer N for {path}: file size {file_size} is not divisible by H*W*C={bytes_per_image}"
            )
        shape = (file_size // bytes_per_image, height, width, channels)
    elif len(shape) == 3:
        height, width, channels = shape
        bytes_per_image = height * width * channels
        file_size = os.path.getsize(path)
        if file_size % bytes_per_image != 0:
            raise ValueError(
                f"Cannot infer N for {path}: file size {file_size} is not divisible by H*W*C={bytes_per_image}"
            )
        shape = (file_size // bytes_per_image, height, width, channels)
    elif len(shape) != 4:
        raise ValueError(f"mmap_shape must be (N,H,W,C) or (H,W,C), got {mmap_shape}")

    n_images, height, width, channels = (int(x) for x in shape)
    if n_images <= 0 or height <= 0 or width <= 0 or channels <= 0:
        raise ValueError(f"mmap_shape values must be positive, got {shape}")
    return n_images, height, width, channels


class SyntheticAnomalyGenerator:
    """Perlin-mask texture paste generator extracted and simplified from Residual_Prototype.

    `anomaly_ratio` is the probability of creating a synthetic anomaly for one normal image.
    Use `backend="memmap"` with `mmap_path`/`mmap_shape` to sample prebuilt DTD textures
    directly from uint8 mmap instead of silently falling back to random noise.
    """

    def __init__(
        self,
        dtd_root: Optional[str] = None,
        dtd_dir: Optional[str] = None,
        anomaly_ratio: float = 0.1,
        perlin_percentile: float = 99.0,
        perlin_scale_min_pow: int = 1,
        perlin_scale_max_pow: int = 4,
        seed: Optional[int] = None,
        rng_seed: Optional[int] = None,
        backend: str = "disk",
        mmap_path: Optional[str] = None,
        mmap_shape=None,
        mmap_color_order: str = "bgr",
        target_size=None,
        jpeg_quality: int = 90,
        max_side: int = 512,
        allow_noise_fallback: bool = True,
    ) -> None:
        if dtd_root is None:
            dtd_root = dtd_dir
        if seed is None:
            seed = rng_seed

        self.dtd_root = Path(dtd_root).expanduser() if dtd_root else None
        self.anomaly_ratio = float(anomaly_ratio)
        self.perlin_percentile = float(perlin_percentile)
        self.perlin_scale_min_pow = int(perlin_scale_min_pow)
        self.perlin_scale_max_pow = int(perlin_scale_max_pow)
        self.backend = str(backend).lower()
        self.target_size = _target_size_from_value(target_size)
        self.jpeg_quality = int(jpeg_quality)
        self.max_side = int(max_side)
        self.allow_noise_fallback = bool(allow_noise_fallback)
        self.mmap_color_order = str(mmap_color_order).lower()
        self.rng = np.random.default_rng(seed)
        self.texture_paths: Sequence[Path] = []
        self._bank = None
        self._mmap = None
        self._mmap_shape = None
        self.N = 0

        if not 0.0 <= self.anomaly_ratio <= 1.0:
            raise ValueError(f"anomaly_ratio must be in [0, 1], got {self.anomaly_ratio}")
        if not 0.0 < self.perlin_percentile < 100.0:
            raise ValueError(f"perlin_percentile must be in (0, 100), got {self.perlin_percentile}")
        if self.backend not in {"memmap", "disk", "bytes", "uint8", "noise"}:
            raise ValueError("backend must be one of {'memmap','bytes','uint8','disk','noise'}")
        if self.mmap_color_order not in {"bgr", "rgb"}:
            raise ValueError(f"mmap_color_order must be 'bgr' or 'rgb', got {mmap_color_order}")

        if self.backend == "memmap":
            self._init_memmap(mmap_path, mmap_shape)
        elif self.backend in {"disk", "bytes", "uint8"}:
            self.texture_paths = self._scan_textures(self.dtd_root)
            if not self.texture_paths and not self.allow_noise_fallback:
                raise FileNotFoundError(
                    f"No DTD images found under {self.dtd_root}; set backend='memmap' or provide a valid dtd_root"
                )
            if self.backend in {"bytes", "uint8"}:
                self._bank = self._build_texture_bank()

    @staticmethod
    def _scan_textures(root: Optional[Path]) -> Sequence[Path]:
        if root is None or not root.exists():
            return []
        image_root = root / "images"
        search_root = image_root if image_root.exists() else root
        return sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    def _init_memmap(self, mmap_path: Optional[str], mmap_shape) -> None:
        if mmap_path in (None, ""):
            raise ValueError("backend=memmap requires mmap_path")
        path = Path(mmap_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"DTD memmap not found: {path}")
        n_images, height, width, channels = _infer_mmap_shape(path, mmap_shape, self.target_size)
        if channels != 3:
            raise ValueError(f"DTD memmap must have 3 channels, got {channels}")
        self._mmap = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_images, height, width, channels))
        self._mmap_shape = (n_images, height, width, channels)
        self.N = n_images
        self.mmap_path = path

    def _build_texture_bank(self) -> Sequence[np.ndarray]:
        bank = []
        for path in self.texture_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if self.max_side and self.max_side > 0:
                height, width = image.shape[:2]
                scale = self.max_side / max(height, width)
                if scale < 1.0:
                    image = cv2.resize(
                        image,
                        (int(round(width * scale)), int(round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
            if self.backend == "bytes":
                ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    bank.append(buffer)
            else:
                bank.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not bank and not self.allow_noise_fallback:
            raise RuntimeError("Failed to build DTD anomaly texture bank")
        return bank

    def reseed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _texture(self, size: Tuple[int, int]) -> np.ndarray:
        width, height = int(size[0]), int(size[1])
        texture_array = self._sample_anomaly_uint8((width, height))
        return self._augment_texture(Image.fromarray(texture_array, mode="RGB"))

    def _sample_anomaly_uint8(self, target_wh: Tuple[int, int]) -> np.ndarray:
        width, height = int(target_wh[0]), int(target_wh[1])
        if self.backend == "memmap":
            idx = int(self.rng.integers(0, self.N))
            image = np.asarray(self._mmap[idx])
            if (image.shape[1], image.shape[0]) != (width, height):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if self.mmap_color_order == "bgr":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(image)

        if self.backend == "bytes" and self._bank:
            idx = int(self.rng.integers(0, len(self._bank)))
            image = cv2.imdecode(self._bank[idx], cv2.IMREAD_COLOR)
            if (image.shape[1], image.shape[0]) != (width, height):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.backend == "uint8" and self._bank:
            idx = int(self.rng.integers(0, len(self._bank)))
            image = self._bank[idx]
            if (image.shape[1], image.shape[0]) != (width, height):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            else:
                image = image.copy()
            return image

        if self.texture_paths:
            path = self.texture_paths[int(self.rng.integers(0, len(self.texture_paths)))]
            with Image.open(path) as im:
                texture = im.convert("RGB")
            texture = self._random_crop_resize(texture, width, height)
            return np.asarray(texture, dtype=np.uint8)

        if self.allow_noise_fallback or self.backend == "noise":
            return np.asarray(self._noise_texture(width, height), dtype=np.uint8)

        raise RuntimeError("No DTD anomaly textures are available")

    def _texture_from_disk_or_noise(self, width: int, height: int) -> Image.Image:
        if self.texture_paths:
            path = self.texture_paths[int(self.rng.integers(0, len(self.texture_paths)))]
            with Image.open(path) as im:
                texture = im.convert("RGB")
            return self._random_crop_resize(texture, width, height)
        else:
            return self._noise_texture(width, height)

    def _random_crop_resize(self, image: Image.Image, width: int, height: int) -> Image.Image:
        if image.width < 2 or image.height < 2:
            return image.resize((width, height), Image.BILINEAR)
        scale = float(self.rng.uniform(0.4, 1.0))
        crop_w = max(1, min(image.width, int(round(image.width * scale))))
        crop_h = max(1, min(image.height, int(round(image.height * scale))))
        left = int(self.rng.integers(0, max(1, image.width - crop_w + 1)))
        top = int(self.rng.integers(0, max(1, image.height - crop_h + 1)))
        image = image.crop((left, top, left + crop_w, top + crop_h))
        return image.resize((width, height), Image.BILINEAR)

    def _noise_texture(self, width: int, height: int) -> Image.Image:
        base = self.rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=float(self.rng.uniform(1.0, 4.0)))
        return Image.fromarray(base, mode="RGB")

    def _augment_texture(self, image: Image.Image) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        if self.rng.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.rng.random() < 0.5:
            image = ImageOps.flip(image)
        image = ImageEnhance.Color(image).enhance(float(self.rng.uniform(0.5, 1.8)))
        image = ImageEnhance.Contrast(image).enhance(float(self.rng.uniform(0.6, 1.8)))
        image = ImageEnhance.Brightness(image).enhance(float(self.rng.uniform(0.7, 1.3)))
        arr = np.asarray(image, dtype=np.uint8)
        if self.rng.random() < 0.25:
            arr = 255 - arr
        return arr

    def _perlin_mask(self, height: int, width: int) -> np.ndarray:
        low = min(self.perlin_scale_min_pow, self.perlin_scale_max_pow)
        high = max(self.perlin_scale_min_pow, self.perlin_scale_max_pow)
        kx = int(self.rng.integers(low, high + 1))
        ky = int(self.rng.integers(low, high + 1))
        perlin = rand_perlin_2d_np((height, width), (2**kx, 2**ky), self.rng)

        angle = float(self.rng.uniform(-5.0, 5.0))
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        perlin = cv2.warpAffine(perlin, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        threshold = np.percentile(perlin, self.perlin_percentile)
        mask = (perlin > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def __call__(self, normal_image, normal_mask=None):
        if isinstance(normal_image, Image.Image):
            image = normal_image.convert("RGB")
        else:
            normal_arr = np.asarray(normal_image)
            if normal_arr.dtype != np.uint8:
                normal_float = normal_arr.astype(np.float32)
                if normal_float.max(initial=0.0) <= 1.5:
                    normal_float *= 255.0
                normal_arr = np.clip(normal_float, 0, 255).astype(np.uint8)
            image = Image.fromarray(normal_arr).convert("RGB")

        if self.target_size is not None and image.size != self.target_size:
            image = image.resize(self.target_size, Image.BILINEAR)

        normal = np.asarray(image, dtype=np.uint8)
        height, width = normal.shape[:2]
        if self.rng.random() >= self.anomaly_ratio:
            return normal.copy(), np.zeros((height, width), dtype=np.uint8)

        texture = self._texture((width, height)).astype(np.float32)
        mask = self._perlin_mask(height, width)
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        beta = float(self.rng.uniform(0.0, 0.8))
        normal_f = normal.astype(np.float32)
        anomaly = normal_f * (1.0 - mask_f) + ((1.0 - beta) * texture + beta * normal_f) * mask_f
        return anomaly.clip(0, 255).astype(np.uint8), mask

    def perlin_and_paste(self, normal_image, normal_mask=None):
        return self(normal_image, normal_mask)


class PerlinPaste(SyntheticAnomalyGenerator):
    """Compatibility wrapper matching the Residual_Prototype PerlinPaste constructor."""

    def __init__(
        self,
        dtd_dir=None,
        backend: str = "memmap",
        mmap_path=None,
        mmap_shape=None,
        target_size=(518, 518),
        perlin_scale_min_pow: int = 1,
        perlin_scale_max_pow: int = 3,
        rng_seed=None,
        jpeg_quality: int = 90,
        max_side: int = 512,
        perlin_percentile: float = 99.0,
        mmap_color_order: str = "bgr",
    ) -> None:
        super().__init__(
            dtd_root=dtd_dir,
            anomaly_ratio=1.0,
            perlin_percentile=perlin_percentile,
            perlin_scale_min_pow=perlin_scale_min_pow,
            perlin_scale_max_pow=perlin_scale_max_pow,
            seed=rng_seed,
            backend=backend,
            mmap_path=mmap_path,
            mmap_shape=mmap_shape,
            mmap_color_order=mmap_color_order,
            target_size=target_size,
            jpeg_quality=jpeg_quality,
            max_side=max_side,
            allow_noise_fallback=False,
        )

    def reseed(self, worker_id: int = 0, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
