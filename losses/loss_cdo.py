from functools import partial
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_mask(mask: Optional[torch.Tensor], batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    if mask is None:
        return torch.zeros((batch, 1, height, width), device=device, dtype=dtype)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    mask = mask.to(device=device, dtype=dtype)
    return F.interpolate(mask, size=(height, width), mode="nearest")


class CDOLoss(nn.Module):
    """Contrastive discrepancy optimization loss from the residual prototype framework.

    Normal pixels minimize encoder-decoder discrepancy with adaptive hard weighting.
    Synthetic abnormal pixels use a hinge term so their discrepancy remains above the
    normal distribution boundary.
    """

    def __init__(self, gamma: float = 2.0, use_adaptive_weight: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.use_adaptive_weight = bool(use_adaptive_weight)
        self.eps = float(eps)

    def _weighted_discrepancy(self, fe: torch.Tensor, fd: torch.Tensor, normal: bool) -> torch.Tensor:
        if fe.numel() == 0:
            return fd.new_tensor(0.0)
        fe = F.normalize(fe, p=2, dim=1)
        fd = F.normalize(fd, p=2, dim=1)
        d = torch.sum((fe - fd) ** 2, dim=1)
        if not self.use_adaptive_weight:
            return d.mean()
        mean_d = d.mean().clamp_min(self.eps)
        if normal:
            weight = (d / mean_d).clamp_min(self.eps).pow(self.gamma)
        else:
            weight = (mean_d / d.clamp_min(self.eps)).pow(self.gamma)
        weight = weight.detach()
        return torch.sum(d * weight) / (torch.sum(weight) + self.eps)

    def forward(self, encoder_features: Iterable[torch.Tensor], decoder_features: Iterable[torch.Tensor], mask=None):
        total = None
        for fe, fd in zip(encoder_features, decoder_features):
            batch, channels, height, width = fe.shape
            mask_map = _ensure_mask(mask, batch, height, width, fe.device, fe.dtype)
            mask_vec = mask_map.permute(0, 2, 3, 1).reshape(-1) > 0.5

            fe_flat = fe.detach().permute(0, 2, 3, 1).reshape(-1, channels)
            fd_flat = fd.permute(0, 2, 3, 1).reshape(-1, channels)

            normal_sel = ~mask_vec
            abnormal_sel = mask_vec
            loss_normal = self._weighted_discrepancy(fe_flat[normal_sel], fd_flat[normal_sel], normal=True)

            if abnormal_sel.any() and normal_sel.any():
                with torch.no_grad():
                    d_normal = torch.sum(
                        (
                            F.normalize(fe_flat[normal_sel], p=2, dim=1)
                            - F.normalize(fd_flat[normal_sel], p=2, dim=1)
                        )
                        ** 2,
                        dim=1,
                    )
                    margin = d_normal.mean() + d_normal.std(unbiased=False) + self.eps
                d_abnormal = torch.sum(
                    (
                        F.normalize(fe_flat[abnormal_sel], p=2, dim=1)
                        - F.normalize(fd_flat[abnormal_sel], p=2, dim=1)
                    )
                    ** 2,
                    dim=1,
                )
                loss_abnormal = F.relu(margin - d_abnormal).mean()
            else:
                loss_abnormal = fd.new_tensor(0.0)

            item_loss = loss_normal + loss_abnormal
            total = item_loss if total is None else total + item_loss

        if total is None:
            raise ValueError("CDOLoss received empty feature lists")
        return total


def loss_cdo(encoder_features: List[torch.Tensor], decoder_features: List[torch.Tensor], mask=None, **kwargs):
    return CDOLoss(**kwargs)(encoder_features, decoder_features, mask)


def normal_reconstruction_loss(encoder_features: Iterable[torch.Tensor], decoder_features: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for fe, fd in zip(encoder_features, decoder_features):
        cos = F.cosine_similarity(fe.detach().flatten(1), fd.flatten(1), dim=1)
        item_loss = torch.mean(1.0 - cos)
        total = item_loss if total is None else total + item_loss
    if total is None:
        raise ValueError("normal_reconstruction_loss received empty feature lists")
    return total


def _modify_grad(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return x * factor.expand_as(x)


def global_cosine_hm_adaptive(encoder_features: Iterable[torch.Tensor], decoder_features: Iterable[torch.Tensor], y: float = 3.0):
    """INP-Former adaptive cosine reconstruction loss."""
    total = None
    for fe, fd in zip(encoder_features, decoder_features):
        fe_detached = fe.detach()
        point_dist = (1.0 - F.cosine_similarity(fe_detached, fd, dim=1, eps=1e-8)).unsqueeze(1).detach()
        factor = (point_dist / (point_dist.mean() + 1e-6)).pow(y)
        if fd.requires_grad:
            fd.register_hook(partial(_modify_grad, factor=factor))
        item_loss = torch.mean(1.0 - F.cosine_similarity(fe_detached.flatten(1), fd.flatten(1), dim=1, eps=1e-8))
        total = item_loss if total is None else total + item_loss
    if total is None:
        raise ValueError("global_cosine_hm_adaptive received empty feature lists")
    return total


def residual_margin_loss(residual_map: torch.Tensor, mask: torch.Tensor, margin: float = 1.0, lambda_abn: float = 1.0):
    """Residual supervision migrated from `loss/L2.py::margin_loss`."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=residual_map.device, dtype=residual_map.dtype)
    if mask.shape[-2:] != residual_map.shape[-2:]:
        mask = F.interpolate(mask, size=residual_map.shape[-2:], mode="nearest")
    mask = (mask > 0.5).to(dtype=residual_map.dtype)
    normal_loss = ((1.0 - mask) * residual_map.pow(2)).mean()
    abnormal_loss = (mask * F.relu(margin - residual_map)).mean()
    return normal_loss + lambda_abn * abnormal_loss
