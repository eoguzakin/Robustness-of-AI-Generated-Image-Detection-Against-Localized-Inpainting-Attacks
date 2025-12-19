from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image


IMG_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def add_ufd_to_pythonpath(third_party_dir: str | Path) -> None:
    third_party_dir = Path(third_party_dir)
    ufd_root = third_party_dir / "UniversalFakeDetect"
    if not ufd_root.exists():
        raise FileNotFoundError(
            f"Missing UFD repo at: {ufd_root}\n"
            f"Expected third_party/UniversalFakeDetect with models/clip/clip.py inside."
        )
    sys.path.insert(0, str(ufd_root))


def list_images(d: str | Path, exts: Iterable[str] = IMG_EXTS_DEFAULT) -> list[Path]:
    d = Path(d)
    exts = {e.lower() for e in exts}
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def get_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_ufd(ckpt_path: str | Path, third_party_dir: str | Path, device: str = "cuda"):
    """
    Loads upstream UFD CLIP ViT-L/14 backbone + upstream linear head (fc_weights.pth).
    Returns (clip_model, preprocess, fc_head, device_str).
    """
    device = get_device(device)
    add_ufd_to_pythonpath(third_party_dir)

    from models.clip import clip as ufd_clip  # upstream import

    clip_model, preprocess = ufd_clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()

    state = torch.load(Path(ckpt_path), map_location=device)
    if "weight" not in state or "bias" not in state:
        raise KeyError(f"Checkpoint missing keys. Found: {list(state.keys())}")

    fc = nn.Linear(768, 1)
    fc.weight.data.copy_(state["weight"])
    fc.bias.data.copy_(state["bias"])
    fc.to(device).eval()

    return clip_model, preprocess, fc, device


@torch.no_grad()
def ufd_score_image(img_path: str | Path, clip_model, preprocess, fc, device: str) -> float:
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    feats = clip_model.encode_image(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats = feats.to(torch.float32)

    logit = fc(feats)
    return torch.sigmoid(logit).item()


def real_id_from_file(fn: str) -> str:
    # Matches your thesis convention: real filenames can include suffixes; keep stable prefix.
    return os.path.splitext(fn)[0].split("_")[0]


def fake_id_from_file(fn: str) -> str:
    return os.path.splitext(fn)[0]
