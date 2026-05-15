import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.anomaly_synthesis import SyntheticAnomalyGenerator
from datasets.mvtecad2 import MVTecAD2Dataset, MVTecAD2Transform, build_dataloader, discover_categories
from evaluate import evaluate_public
from losses.loss_cdo import CDOLoss, residual_margin_loss
from models.vand4_model import VAND4Model
from utils.io_utils import ensure_dir, load_checkpoint, load_config, save_checkpoint
from utils.seed import seed_everything


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
    synthetic["mmap_path"] = _resolve_path(synthetic.get("mmap_path"), project_root)
    return cfg


def parse_categories(value, data_root):
    if value:
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return list(value)
    return discover_categories(data_root)


def _synthetic_target_size(syn_cfg, transform_cfg):
    target_size = syn_cfg.get("target_size")
    if target_size not in (None, ""):
        return target_size

    resize = transform_cfg.get("resize")
    if resize in (None, ""):
        return None
    if isinstance(resize, (list, tuple)):
        if len(resize) != 2:
            raise ValueError(f"transform.resize must be int or (W, H), got {resize}")
        return [int(resize[0]), int(resize[1])]
    return [int(resize), int(resize)]


def _log_synthetic_source(generator, enabled: bool) -> None:
    if not enabled:
        print("[synthetic] disabled")
        return

    if generator.backend == "memmap":
        print(
            "[synthetic] enabled; using DTD memmap: yes "
            f"path={generator.mmap_path} shape={generator._mmap_shape} target_size={generator.target_size}"
        )
        return

    texture_count = len(generator.texture_paths)
    if texture_count:
        print(
            "[synthetic] enabled; using DTD images: yes "
            f"backend={generator.backend} count={texture_count} target_size={generator.target_size}"
        )
    else:
        print(
            "[synthetic] enabled; using DTD: no "
            f"backend={generator.backend} fallback=random_noise target_size={generator.target_size}"
        )


def build_dataset(cfg, data_root, categories):
    transform_cfg = cfg.get("transform", {})
    transform = MVTecAD2Transform.from_config(transform_cfg)
    syn_cfg = cfg.get("synthetic", {})
    dtd_root = syn_cfg.get("dtd_root")
    backend = str(syn_cfg.get("backend", "memmap" if syn_cfg.get("mmap_path") else "disk")).lower()
    mmap_path = syn_cfg.get("mmap_path")
    target_size = _synthetic_target_size(syn_cfg, transform_cfg)
    if backend == "memmap":
        if not mmap_path:
            raise ValueError("synthetic.backend=memmap requires synthetic.mmap_path")
        if not Path(mmap_path).exists():
            raise FileNotFoundError(f"DTD memmap not found: {mmap_path}")
    elif dtd_root and not Path(dtd_root).exists():
        print(f"[warn] DTD root not found, synthetic generator will use image-noise fallback: {dtd_root}")
    generator = SyntheticAnomalyGenerator(
        dtd_root=dtd_root if dtd_root and Path(dtd_root).exists() else None,
        anomaly_ratio=float(syn_cfg.get("anomaly_ratio", 0.1)),
        perlin_percentile=float(syn_cfg.get("perlin_percentile", 99.0)),
        perlin_scale_min_pow=int(syn_cfg.get("perlin_scale_min_pow", 1)),
        perlin_scale_max_pow=int(syn_cfg.get("perlin_scale_max_pow", 4)),
        seed=syn_cfg.get("seed"),
        backend=backend,
        mmap_path=mmap_path,
        mmap_shape=syn_cfg.get("mmap_shape"),
        mmap_color_order=syn_cfg.get("mmap_color_order", "bgr"),
        target_size=target_size,
        jpeg_quality=int(syn_cfg.get("jpeg_quality", 90)),
        max_side=int(syn_cfg.get("max_side", 512)),
        allow_noise_fallback=bool(syn_cfg.get("allow_noise_fallback", backend != "memmap")),
    )
    synthetic_enabled = bool(syn_cfg.get("enabled", True))
    _log_synthetic_source(generator, synthetic_enabled)
    return MVTecAD2Dataset(
        data_root=data_root,
        categories=categories,
        mode="train",
        transform=transform,
        synthetic_generator=generator,
        enable_synthetic=synthetic_enabled,
        include_validation_in_train=bool(cfg.get("data", {}).get("include_validation_in_train", False)),
    )


def load_resume(path, model, optimizer, device):
    if not path:
        return 0, math.inf
    checkpoint = load_checkpoint(path, device=device)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[resume] missing keys: {len(missing)}")
    if unexpected:
        print(f"[resume] unexpected keys: {len(unexpected)}")
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_loss", math.inf))


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_id was set, but CUDA is not available")
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    cfg = normalize_config_paths(load_config(_config_path(args.config)), PROJECT_ROOT)
    if args.data_root:
        cfg.setdefault("data", {})["root"] = args.data_root
    if args.save_dir:
        cfg.setdefault("train", {})["save_dir"] = args.save_dir

    seed_everything(int(cfg.get("seed", 1)))
    device = resolve_device(args)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_root = cfg.get("data", {}).get("root")
    if not data_root:
        raise ValueError("data_root is required via --data_root or config data.root")
    categories = parse_categories(args.categories or cfg.get("data", {}).get("categories"), data_root)
    if not categories:
        raise ValueError(f"No categories found under {data_root}")

    train_cfg = cfg.get("train", {})
    dataset = build_dataset(cfg, data_root, categories)
    loader = build_dataloader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 16)),
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    print(f"train samples: {len(dataset)} categories: {categories}")

    model = VAND4Model.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        amsgrad=True,
        eps=float(train_cfg.get("eps", 1e-10)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg.get("epochs", 200)) * max(1, len(loader))),
        eta_min=float(train_cfg.get("min_lr", 1e-5)),
    )
    cdo_loss = CDOLoss(**cfg.get("loss", {}).get("cdo", {})).to(device)

    start_epoch, _ = load_resume(args.resume or train_cfg.get("resume"), model, optimizer, device)
    save_dir = ensure_dir(train_cfg.get("save_dir", "checkpoints/vand4"))
    epochs = int(train_cfg.get("epochs", 200))
    log_interval = int(train_cfg.get("log_interval", 20))
    max_steps_per_epoch = train_cfg.get("max_steps_per_epoch")
    max_steps_per_epoch = int(max_steps_per_epoch) if max_steps_per_epoch else None
    clip_grad = float(train_cfg.get("clip_grad", 0.1))
    save_interval = int(train_cfg.get("save_interval", 10))
    eval_interval = int(train_cfg.get("eval_interval", 10))
    eval_public = bool(train_cfg.get("eval_public", True))
    eval_metric = str(train_cfg.get("eval_metric", "segf1max")).lower()
    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"total": 0.0, "cdo": 0.0, "residual": 0.0}
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", ncols=100)
        for step, batch in enumerate(progress, start=1):
            if max_steps_per_epoch is not None and step > max_steps_per_epoch:
                break
            image = batch["image"].to(device, non_blocking=True)
            anomaly_image = batch["anomaly_image"].to(device, non_blocking=True)
            mask = batch["synthetic_mask"].to(device, non_blocking=True)

            output = model(image, anomaly_image=anomaly_image, mask=mask, mode="train")
            loss_cdo_value = cdo_loss(output["encoder_features"], output["decoder_features"], mask)
            loss_residual = residual_margin_loss(output["residual_map"], mask, margin=float(cfg.get("loss", {}).get("residual_margin", 1.0)))
            loss_total = loss_cdo_value + loss_residual

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_norm=clip_grad)
            optimizer.step()
            scheduler.step()

            running["total"] += float(loss_total.detach().cpu())
            running["cdo"] += float(loss_cdo_value.detach().cpu())
            running["residual"] += float(loss_residual.detach().cpu())
            if step % log_interval == 0 or step == 1:
                denom = float(step)
                progress.set_postfix(
                    total=f"{running['total'] / denom:.4f}",
                    cdo=f"{running['cdo'] / denom:.4f}",
                    residual=f"{running['residual'] / denom:.4f}",
                )

        steps_done = min(len(loader), max_steps_per_epoch or len(loader))
        epoch_loss = running["total"] / max(1, steps_done)
        print(
            "epoch [{}/{}] loss_total:{:.4f} loss_cdo:{:.4f} loss_residual:{:.4f}".format(
                epoch + 1,
                epochs,
                epoch_loss,
                running["cdo"] / max(1, steps_done),
                running["residual"] / max(1, steps_done),
            )
        )

        if eval_public and ((epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs):
            evaluate_public(model, data_root, categories, cfg, device, metric_mode=eval_metric)

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            checkpoint_extra = {"epoch_loss": epoch_loss, "config": cfg}
            save_checkpoint(str(Path(save_dir) / "latest.pth"), model, optimizer, epoch, checkpoint_extra)
            epoch_path = Path(save_dir) / f"epoch_{epoch + 1:03d}.pth"
            save_checkpoint(str(epoch_path), model, optimizer, epoch, checkpoint_extra)
            print(f"saved checkpoint: {epoch_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mvtecad2_inp_residual.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=5, help="CUDA device index, for example --gpu_id 0")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
