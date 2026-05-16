from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _fade(x: np.ndarray) -> np.ndarray:
    return ((6.0 * x - 15.0) * x + 10.0) * x**3


def rand_perlin_2d_np(shape: Tuple[int, int], res: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Fast numpy Perlin noise used to build irregular synthetic anomaly masks."""
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


def _target_size_from_value(value) -> Optional[Tuple[int, int]]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [int(x) for x in value.replace("x", ",").replace("X", ",").split(",") if x.strip()]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 2:
        raise ValueError(f"target_size must be (W, H), got {value}")
    return int(parts[0]), int(parts[1])


class SyntheticAnomalyGenerator:
    """Random-noise synthetic anomaly generator for VAND 2026 training.

    The generator samples a Perlin-style binary mask with probability `anomaly_ratio`,
    fills the masked region with augmented random noise, and returns the synthetic
    image plus the mask used as supervision. No external anomaly source is required.
    """

    def __init__(
        self,
        anomaly_mode: str = "random_noise",
        anomaly_ratio: float = 0.1,
        perlin_percentile: float = 99.0,
        perlin_scale_min_pow: int = 1,
        perlin_scale_max_pow: int = 4,
        seed: Optional[int] = None,
        rng_seed: Optional[int] = None,
        target_size=None,
        noise_blur_min: float = 1.0,
        noise_blur_max: float = 4.0,
    ) -> None:
        if seed is None:
            seed = rng_seed

        self.anomaly_mode = str(anomaly_mode).lower()
        if self.anomaly_mode != "random_noise":
            raise ValueError("Only synthetic.anomaly_mode='random_noise' is supported")

        self.anomaly_ratio = float(anomaly_ratio)
        self.perlin_percentile = float(perlin_percentile)
        self.perlin_scale_min_pow = int(perlin_scale_min_pow)
        self.perlin_scale_max_pow = int(perlin_scale_max_pow)
        self.target_size = _target_size_from_value(target_size)
        self.noise_blur_min = float(noise_blur_min)
        self.noise_blur_max = float(noise_blur_max)
        self.rng = np.random.default_rng(seed)

        if not 0.0 <= self.anomaly_ratio <= 1.0:
            raise ValueError(f"anomaly_ratio must be in [0, 1], got {self.anomaly_ratio}")
        if not 0.0 < self.perlin_percentile < 100.0:
            raise ValueError(f"perlin_percentile must be in (0, 100), got {self.perlin_percentile}")
        if self.noise_blur_min < 0.0 or self.noise_blur_max < self.noise_blur_min:
            raise ValueError("noise_blur_min/noise_blur_max must define a valid non-negative range")

    def reseed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _noise_texture(self, width: int, height: int) -> Image.Image:
        base = self.rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        sigma = float(self.rng.uniform(self.noise_blur_min, self.noise_blur_max))
        if sigma > 0.0:
            base = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return Image.fromarray(base, mode="RGB")

    def _augment_texture(self, image: Image.Image) -> np.ndarray:
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

        texture = self._augment_texture(self._noise_texture(width, height)).astype(np.float32)
        mask = self._perlin_mask(height, width)
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        beta = float(self.rng.uniform(0.0, 0.8))
        normal_f = normal.astype(np.float32)
        anomaly = normal_f * (1.0 - mask_f) + ((1.0 - beta) * texture + beta * normal_f) * mask_f
        return anomaly.clip(0, 255).astype(np.uint8), mask

    def perlin_and_paste(self, normal_image, normal_mask=None):
        return self(normal_image, normal_mask)


class PerlinPaste(SyntheticAnomalyGenerator):
    """Compatibility wrapper retained for older local imports."""

    def __init__(
        self,
        anomaly_mode: str = "random_noise",
        target_size=(518, 518),
        perlin_scale_min_pow: int = 1,
        perlin_scale_max_pow: int = 3,
        rng_seed=None,
        perlin_percentile: float = 99.0,
    ) -> None:
        super().__init__(
            anomaly_mode=anomaly_mode,
            anomaly_ratio=1.0,
            perlin_percentile=perlin_percentile,
            perlin_scale_min_pow=perlin_scale_min_pow,
            perlin_scale_max_pow=perlin_scale_max_pow,
            seed=rng_seed,
            target_size=target_size,
        )

    def reseed(self, worker_id: int = 0, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
