from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from scripts.config_utils import load_yaml
from scripts.aeroblade_common import get_device, list_images, load_vae, make_preprocess, recon_error_batch


def score_dir(vae, files, preprocess, device, batch_size: int, metric: str):
    scores = []
    for i in tqdm(range(0, len(files), batch_size), desc="AEROBLADE scoring"):
        batch = files[i:i + batch_size]
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(imgs, 0).to(device)
        err = recon_error_batch(vae, x, metric=metric)
        scores.extend(err.detach().cpu().tolist())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_aeroblade.yaml")
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse", help="HF model id or local path to VAE.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--metric", choices=["l1", "l2"], default="l2")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="outputs/aeroblade/baseline")
    args = ap.parse_args()

    P = load_yaml(args.paths)["aeroblade"]
    real_dir = P["baseline"]["reals"]
    fake_dir = P["baseline"]["fakes"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    vae = load_vae(args.vae, device=device, dtype=args.dtype)
    preprocess = make_preprocess(args.image_size)

    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)

    real_scores = score_dir(vae, real_files, preprocess, device, args.batch_size, args.metric)
    fake_scores = score_dir(vae, fake_files, preprocess, device, args.batch_size, args.metric)

    rows = []
    for p, s in zip(real_files, real_scores):
        rows.append({"file": p.name, "score_aeroblade": float(s), "label": 0})
    for p, s in zip(fake_files, fake_scores):
        rows.append({"file": p.name, "score_aeroblade": float(s), "label": 1})

    pd.DataFrame(rows).to_csv(out_dir / "aeroblade_baseline.csv", index=False)
    print("Wrote:", out_dir / "aeroblade_baseline.csv")


if __name__ == "__main__":
    main()
