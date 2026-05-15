import importlib.util
import math
import os
import sys
import types
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file
from torch.nn.init import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PrototypeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prototype_token):
        batch, tokens, channels = x.shape
        prototype_num = prototype_token.shape[1]
        q = self.q(x).reshape(batch, tokens, 1, self.num_heads, channels // self.num_heads)
        q = q.permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(prototype_token).reshape(batch, prototype_num, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x), attn


class AggregationAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        batch, proto_tokens, channels = x.shape
        feature_tokens = y.shape[1]
        q = self.q(x).reshape(batch, proto_tokens, 1, self.num_heads, channels // self.num_heads)
        q = q.permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y).reshape(batch, feature_tokens, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch, proto_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class AggregationBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AggregationAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PrototypeBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrototypeAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, prototype, return_attention=False):
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, attn) if return_attention else x


class DummyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return x + 0.1 * self.proj(self.norm(x))


class DummyViTEncoder(nn.Module):
    """Small smoke-test encoder. Real training should use a DINO encoder."""

    def __init__(self, embed_dim=64, depth=10, patch_size=14, num_register_tokens=0):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([DummyBlock(embed_dim) for _ in range(depth)])

    def prepare_tokens(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


def encoder_dimensions(encoder_name: str) -> Tuple[int, int, List[int]]:
    if "dummy" in encoder_name:
        return 64, 4, [2, 3, 4, 5, 6, 7, 8, 9]
    if "small" in encoder_name:
        return 384, 6, [2, 3, 4, 5, 6, 7, 8, 9]
    if "base" in encoder_name:
        return 768, 12, [2, 3, 4, 5, 6, 7, 8, 9]
    if "large" in encoder_name:
        return 1024, 16, [4, 6, 8, 10, 12, 14, 16, 18]
    raise ValueError(f"Unsupported encoder architecture: {encoder_name}")


def load_dino_encoder(encoder_name: str, source_root: Optional[str] = None, weights_dir: Optional[str] = None):
    if "dummy" in encoder_name:
        embed_dim, _, target_layers = encoder_dimensions(encoder_name)
        return DummyViTEncoder(embed_dim=embed_dim, depth=max(target_layers) + 1)

    if source_root is None:
        source_root = str(Path(__file__).resolve().parents[2] / "INP-Former-main")
    source_root_path = Path(source_root).expanduser().resolve()
    if weights_dir is None:
        weights_dir = source_root_path / "backbones" / "weights"
    weights_dir_path = Path(weights_dir).expanduser().resolve()
    weights_dir_path.mkdir(parents=True, exist_ok=True)

    if "dinov2" in encoder_name:
        return _load_dinov2_direct(encoder_name, source_root_path, weights_dir_path)
    if "dino" in encoder_name:
        return _load_dinov1_direct(encoder_name, source_root_path, weights_dir_path)

    vit_encoder_path = source_root_path / "models" / "vit_encoder.py"
    if not vit_encoder_path.exists():
        raise FileNotFoundError(f"Cannot find source vit_encoder.py: {vit_encoder_path}")

    if "timm" not in sys.modules:
        try:
            __import__("timm")
        except Exception:
            sys.modules["timm"] = types.ModuleType("timm")

    sys.path.insert(0, str(source_root_path))
    try:
        spec = importlib.util.spec_from_file_location("_vand4_source_vit_encoder", vit_encoder_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {vit_encoder_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if weights_dir is not None:
            module._WEIGHTS_DIR = str(Path(weights_dir).expanduser().resolve())
            Path(module._WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
        encoder = module.load(encoder_name)
    finally:
        try:
            sys.path.remove(str(source_root_path))
        except ValueError:
            pass
    return encoder


def _download_weight(url: str, weights_dir: Path) -> Path:
    filename = url.rstrip("/").split("/")[-1]
    target = weights_dir / filename
    if not target.exists():
        download_url_to_file(url, str(target), progress=True)
    return target


def _load_dinov2_direct(encoder_name: str, source_root: Path, weights_dir: Path):
    arch, patchsize = encoder_name.split("_")[-2], encoder_name.split("_")[-1]
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    sys.path.insert(0, str(source_root))
    try:
        from dinov2.models import vision_transformer as vision_transformer_dinov2

        model = vision_transformer_dinov2.__dict__[f"vit_{arch}"](
            patch_size=int(patchsize),
            img_size=518,
            block_chunks=0,
            init_values=1e-8,
            num_register_tokens=4 if "reg" in encoder_name else 0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass

    arch_short = {"small": "s", "base": "b", "large": "l"}.get(arch)
    if arch_short is None:
        raise ValueError(f"Unsupported DINOv2 architecture: {arch}")
    reg = "_reg4" if "reg" in encoder_name else ""
    filename = f"dinov2_vit{arch_short}{patchsize}{reg}_pretrain.pth"
    path = weights_dir / filename
    if not path.exists():
        url = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vit{arch_short}{patchsize}/{filename}"
        path = _download_weight(url, weights_dir)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def _load_dinov1_direct(encoder_name: str, source_root: Path, weights_dir: Path):
    arch, patchsize = encoder_name.split("_")[-2], encoder_name.split("_")[-1]
    sys.path.insert(0, str(source_root))
    try:
        from dinov1 import vision_transformer

        model = vision_transformer.__dict__[f"vit_{arch}"](patch_size=int(patchsize))
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass
    if arch == "base":
        url = f"https://dl.fbaipublicfiles.com/dino/dino_vit{arch}{patchsize}_pretrain/dino_vit{arch}{patchsize}_pretrain.pth"
    elif arch == "small":
        url = f"https://dl.fbaipublicfiles.com/dino/dino_deit{arch}{patchsize}_pretrain/dino_deit{arch}{patchsize}_pretrain.pth"
    else:
        raise ValueError(f"Unsupported DINO architecture: {arch}")
    path = weights_dir / url.rstrip("/").split("/")[-1]
    if not path.exists():
        path = _download_weight(url, weights_dir)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


class INPFormerCore(nn.Module):
    """INP-Former feature reconstruction branch from INP_Former_Multi_Class.py."""

    def __init__(
        self,
        encoder,
        embed_dim: int,
        num_heads: int,
        inp_num: int = 6,
        target_layers: Optional[Sequence[int]] = None,
        fuse_layer_encoder: Optional[Sequence[Sequence[int]]] = None,
        fuse_layer_decoder: Optional[Sequence[Sequence[int]]] = None,
        remove_class_token: bool = True,
        encoder_require_grad_layers: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.target_layers = list(target_layers or [2, 3, 4, 5, 6, 7, 8, 9])
        self.fuse_layer_encoder = [list(x) for x in (fuse_layer_encoder or [[0, 1, 2, 3], [4, 5, 6, 7]])]
        self.fuse_layer_decoder = [list(x) for x in (fuse_layer_decoder or [[0, 1, 2, 3], [4, 5, 6, 7]])]
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layers = set(encoder_require_grad_layers or [])
        self.prototype_token = nn.Parameter(torch.randn(inp_num, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-8)
        self.bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
        self.aggregation = nn.ModuleList(
            [AggregationBlock(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm_layer)]
        )
        self.decoder = nn.ModuleList(
            [PrototypeBlock(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm_layer) for _ in range(8)]
        )

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self._init_trainable()
        self._freeze_encoder()

    def _init_trainable(self) -> None:
        for module in [self.bottleneck, self.aggregation, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        trunc_normal_(self.prototype_token, std=0.01, a=-0.03, b=0.03)

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _patch_grid(self, image: torch.Tensor, token_count: int) -> Tuple[int, int]:
        patch_size = getattr(self.encoder, "patch_size", None)
        if patch_size is None and hasattr(self.encoder, "patch_embed"):
            patch_size = getattr(self.encoder.patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            patch_h = patch_w = int(patch_size or 14)
        grid_h = max(1, image.shape[-2] // patch_h)
        grid_w = max(1, image.shape[-1] // patch_w)
        if grid_h * grid_w != token_count:
            side = int(math.sqrt(token_count))
            if side * side == token_count:
                return side, side
            raise RuntimeError(f"Cannot reshape {token_count} patch tokens into grid from image {tuple(image.shape)}")
        return grid_h, grid_w

    def gather_loss(self, query, keys):
        distribution = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance = torch.min(distribution, dim=2)[0]
        return distance.mean()

    def fuse_feature(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(features), dim=1).mean(dim=1)

    def forward(self, image: torch.Tensor):
        tokens = self.encoder.prepare_tokens(image)
        encoded_layers = []
        for index, block in enumerate(self.encoder.blocks):
            if index > self.target_layers[-1]:
                break
            if index in self.encoder_require_grad_layers:
                tokens = block(tokens)
            else:
                with torch.no_grad():
                    tokens = block(tokens)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            if index in self.target_layers:
                encoded_layers.append(tokens)

        if not encoded_layers:
            raise RuntimeError("No encoder layers were collected. Check target_layers.")

        token_start = 1 + int(getattr(self.encoder, "num_register_tokens", 0))
        patch_token_count = encoded_layers[0].shape[1] - token_start
        grid_h, grid_w = self._patch_grid(image, patch_token_count)

        if self.remove_class_token:
            encoded_layers = [layer[:, token_start:, :] for layer in encoded_layers]

        x = self.fuse_feature(encoded_layers)
        prototype = self.prototype_token.unsqueeze(0).repeat(image.shape[0], 1, 1)
        for block in self.aggregation:
            prototype = block(prototype, x)
        gather_loss = 0
        for block in self.bottleneck:
            x = block(x)

        decoded_layers = []
        for block in self.decoder:
            x = block(x, prototype)
            decoded_layers.append(x)
        decoded_layers = decoded_layers[::-1]

        enc = [self.fuse_feature([encoded_layers[idx] for idx in group]) for group in self.fuse_layer_encoder]
        dec = [self.fuse_feature([decoded_layers[idx] for idx in group]) for group in self.fuse_layer_decoder]

        if not self.remove_class_token:
            enc = [item[:, token_start:, :] for item in enc]
            dec = [item[:, token_start:, :] for item in dec]

        enc_maps = [item.permute(0, 2, 1).reshape(image.shape[0], -1, grid_h, grid_w).contiguous() for item in enc]
        dec_maps = [item.permute(0, 2, 1).reshape(image.shape[0], -1, grid_h, grid_w).contiguous() for item in dec]
        return enc_maps, dec_maps, gather_loss


def cal_anomaly_maps(encoder_features: List[torch.Tensor], decoder_features: List[torch.Tensor], out_size):
    if isinstance(out_size, int):
        out_size = (out_size, out_size)
    maps = []
    for fe, fd in zip(encoder_features, decoder_features):
        anomaly = 1.0 - F.cosine_similarity(fe, fd, dim=1, eps=1e-8)
        anomaly = anomaly.unsqueeze(1)
        anomaly = F.interpolate(anomaly, size=out_size, mode="bilinear", align_corners=True)
        maps.append(anomaly)
    return torch.cat(maps, dim=1).mean(dim=1, keepdim=True), maps


def residual_feature(encoder_features: List[torch.Tensor], decoder_features: List[torch.Tensor], weights=None):
    maps = []
    for fe, fd in zip(encoder_features, decoder_features):
        cosine = (1.0 - F.cosine_similarity(fe, fd, dim=1, eps=1e-8)).unsqueeze(1)
        maps.append(torch.abs(fe - fd) * cosine)
    if weights is None:
        return torch.stack(maps, dim=0).mean(dim=0)
    weights = torch.softmax(weights[: len(maps)], dim=0)
    total = None
    for weight, item in zip(weights, maps):
        total = weight * item if total is None else total + weight * item
    return total


def build_inp_former(
    encoder_name: str,
    source_root: Optional[str] = None,
    weights_dir: Optional[str] = None,
    inp_num: int = 6,
    target_layers: Optional[Sequence[int]] = None,
    fuse_layer_encoder: Optional[Sequence[Sequence[int]]] = None,
    fuse_layer_decoder: Optional[Sequence[Sequence[int]]] = None,
):
    embed_dim, num_heads, default_layers = encoder_dimensions(encoder_name)
    encoder = load_dino_encoder(encoder_name, source_root=source_root, weights_dir=weights_dir)
    return INPFormerCore(
        encoder=encoder,
        embed_dim=embed_dim,
        num_heads=num_heads,
        inp_num=inp_num,
        target_layers=target_layers or default_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
