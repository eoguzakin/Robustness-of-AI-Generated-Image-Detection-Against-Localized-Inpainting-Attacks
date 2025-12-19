from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


IMG_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(d: str | Path, exts: Iterable[str] = IMG_EXTS_DEFAULT) -> list[Path]:
    d = Path(d)
    exts = {e.lower() for e in exts}
    paths: list[Path] = []
    for dp, _, files in os.walk(d):
        for f in files:
            p = Path(dp) / f
            if p.suffix.lower() in exts:
                paths.append(p)
    paths.sort()
    return paths


def get_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dimd_available():
    try:
        from dimd.models import DIMDBackbone  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "DIMD is not importable.\n"
            "Expected: `from dimd.models import DIMDBackbone` to work.\n"
            "Fix: install/clone DIMD and add it to PYTHONPATH (or pip install it), then retry.\n"
            f"Original import error: {type(e).__name__}: {e}"
        )


def real_id_from_file(fn: str) -> str:
    return os.path.splitext(fn)[0].split("_")[0]


def fake_id_from_file(fn: str) -> str:
    return os.path.splitext(fn)[0]


class DIMDWrapper(nn.Module):
    """
    Wrapper matching your stub contract: higher logit => more fake. [file:1]
    """
    def __init__(self, ckpt_path: str | Path, device: torch.device):
        super().__init__()
        ensure_dimd_available()
        from dimd.models import DIMDBackbone  # type: ignore

        self.backbone = DIMDBackbone()
        state = torch.load(str(ckpt_path), map_location="cpu")
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.to(device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.device = device

    @torch.no_grad()
    def score_paths(self, paths: list[Path], batch_size: int = 32) -> list[float]:
        out: list[float] = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            imgs = [self.preprocess(Image.open(p).convert("RGB")) for p in batch]
            x = torch.stack(imgs, 0).to(self.device)
            logits = self.backbone(x)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            out.extend(logits.detach().cpu().tolist())
        return out
