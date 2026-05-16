from typing import Dict, Optional

import torch
import torch.nn as nn

from .inp_former import build_inp_former, cal_anomaly_maps, residual_feature
from .residual_head import ResidualHead


def _minmax_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(1)
    mn = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def _kurtosis_per_image(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = x.flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    z = (flat - mean) / std
    return z.pow(4).mean(dim=1)


class KurtosisPriorResidualFusion(nn.Module):
    def __init__(self, base_lambda: float = 1.0, alpha: float = 2.0, k0: float = 50.0) -> None:
        super().__init__()
        self.base_lambda = float(base_lambda)
        self.alpha = float(alpha)
        self.k0 = float(k0)

    def forward(self, recon: torch.Tensor, residual: torch.Tensor):
        residual_kurtosis = _kurtosis_per_image(residual)
        fusion_lambda = self.base_lambda * torch.sigmoid(self.alpha * (residual_kurtosis - self.k0))
        anomaly_map = recon + fusion_lambda.view(-1, 1, 1, 1) * residual
        return anomaly_map, fusion_lambda, residual_kurtosis


class VAND4Model(nn.Module):
    """Unified VAND 4.0 model.

    The INP-Former branch produces DINO feature reconstruction maps. The residual branch
    predicts a synthetic anomaly residual map from encoder-decoder feature differences.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        weights_dir: Optional[str] = None,
        inp_num: int = 6,
        residual_hidden: int = 128,
        reconstruction_weight: float = 1.0,
        residual_weight: float = 0.2,
        normalize_fusion: bool = False,
        fusion_mode: str = "linear",
        prior_base_lambda: float = 1.0,
        prior_alpha: float = 2.0,
        prior_k0: float = 50.0,
    ) -> None:
        super().__init__()
        self.inp_branch = build_inp_former(
            encoder_name=encoder_name,
            weights_dir=weights_dir,
            inp_num=inp_num,
        )
        self.residual_layer_weights = nn.Parameter(torch.ones(2, dtype=torch.float32) / 2.0)
        self.residual_head = ResidualHead(in_channels=self.inp_branch.embed_dim, hidden_channels=residual_hidden)
        self.reconstruction_weight = float(reconstruction_weight)
        self.residual_weight = float(residual_weight)
        self.normalize_fusion = bool(normalize_fusion)
        self.fusion_mode = str(fusion_mode)
        self.prior_residual_fusion = KurtosisPriorResidualFusion(
            base_lambda=prior_base_lambda,
            alpha=prior_alpha,
            k0=prior_k0,
        )

    @classmethod
    def from_config(cls, cfg: Dict) -> "VAND4Model":
        model_cfg = (cfg or {}).get("model", cfg or {})
        backbone_cfg = model_cfg.get("backbone", {})
        fusion_cfg = model_cfg.get("fusion", {})
        return cls(
            encoder_name=backbone_cfg.get("encoder", model_cfg.get("encoder", "dinov2reg_vit_base_14")),
            weights_dir=backbone_cfg.get("weights_dir"),
            inp_num=int(model_cfg.get("inp_num", 6)),
            residual_hidden=int(model_cfg.get("residual_hidden", 128)),
            reconstruction_weight=float(fusion_cfg.get("reconstruction_weight", 1.0)),
            residual_weight=float(fusion_cfg.get("residual_weight", 0.2)),
            normalize_fusion=bool(fusion_cfg.get("normalize", False)),
            fusion_mode=fusion_cfg.get("mode", "linear"),
            prior_base_lambda=float(fusion_cfg.get("base_lambda", fusion_cfg.get("prior_base_lambda", 1.0))),
            prior_alpha=float(fusion_cfg.get("alpha", fusion_cfg.get("prior_alpha", 2.0))),
            prior_k0=float(fusion_cfg.get("k0", fusion_cfg.get("prior_k0", 50.0))),
        )

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def forward(self, image, anomaly_image=None, mask=None, mode: str = "train"):
        x = anomaly_image if mode == "train" and anomaly_image is not None else image
        encoder_features, decoder_features, gather_loss = self.inp_branch(x)

        recon_map, recon_parts = cal_anomaly_maps(encoder_features, decoder_features, out_size=x.shape[-2:])
        residual_fea = residual_feature(encoder_features, decoder_features, self.residual_layer_weights)
        residual_logits = self.residual_head(residual_fea, out_size=x.shape[-2:])
        residual_map = residual_logits.sigmoid()

        recon_for_fusion = _minmax_per_image(recon_map) if self.normalize_fusion else recon_map
        residual_for_fusion = _minmax_per_image(residual_map) if self.normalize_fusion else residual_map

        if self.fusion_mode in ("kurtosis_prior", "prior_residual"):
            anomaly_map, fusion_lambda, residual_kurtosis = self.prior_residual_fusion(
                self.reconstruction_weight * recon_for_fusion,
                residual_for_fusion,
            )
        elif self.fusion_mode == "linear":
            anomaly_map = self.reconstruction_weight * recon_for_fusion + self.residual_weight * residual_for_fusion
            batch_size = anomaly_map.size(0)
            fusion_lambda = torch.full(
                (batch_size,),
                self.residual_weight,
                dtype=anomaly_map.dtype,
                device=anomaly_map.device,
            )
            residual_kurtosis = None
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        return {
            "encoder_features": encoder_features,
            "decoder_features": decoder_features,
            "gather_loss": gather_loss,
            "reconstruction_map": recon_map,
            "reconstruction_parts": recon_parts,
            "residual_feature": residual_fea,
            "residual_logits": residual_logits,
            "residual_map": residual_map,
            "fusion_lambda": fusion_lambda,
            "residual_kurtosis": residual_kurtosis,
            "anomaly_map": anomaly_map,
        }
