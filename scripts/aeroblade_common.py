from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
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


def load_vae(vae_id_or_path: str, device: torch.device, dtype: str = "fp16"):
    # Uses diffusers AutoencoderKL (users install dependencies themselves, like other detectors).
    from diffusers import AutoencoderKL

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    vae = AutoencoderKL.from_pretrained(vae_id_or_path, torch_dtype=torch_dtype)
    vae.to(device).eval()
    return vae


def make_preprocess(image_size: int = 512):
    # SD-style: resize+center crop, map to [-1, 1]
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                  # [0,1]
        transforms.Lambda(lambda x: x * 2 - 1),  # [-1,1]
    ])


@torch.no_grad()
def recon_error_batch(vae, x: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """
    x: (B,3,H,W) in [-1,1]
    returns: (B,) reconstruction error
    """
    dist = vae.encode(x).latent_dist
    latents = dist.sample() * getattr(vae.config, "scaling_factor", 0.18215)
    recon = vae.decode(latents / getattr(vae.config, "scaling_factor", 0.18215)).sample

    if metric == "l1":
        err = (x - recon).abs().mean(dim=(1, 2, 3))
    else:
        err = ((x - recon) ** 2).mean(dim=(1, 2, 3))
    return err
