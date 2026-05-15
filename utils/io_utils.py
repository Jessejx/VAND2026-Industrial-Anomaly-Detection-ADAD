import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: Dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_checkpoint(path: str, device="cpu") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {p}")
    checkpoint = torch.load(p, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model": checkpoint}


def save_checkpoint(path: str, model, optimizer=None, epoch: int = 0, extra=None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": int(epoch)}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, p)
