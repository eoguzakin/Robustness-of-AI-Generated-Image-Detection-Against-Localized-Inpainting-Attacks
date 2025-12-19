from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from scripts.config_utils import load_yaml


class DIMDModel(nn.Module):
    """
    Adapter: replace the import in __init__ to match the upstream DIMD repo you use.
    This mirrors your existing DIMD stub structure. [file:1]
    """
    def __init__(self, ckpt_path: str, device: torch.device):
        super().__init__()

        # TODO: CHANGE THIS IMPORT TO MATCH YOUR DIMD REPO
        # Example (as in your stub): from dimd.models import DIMDBackbone [file:1]
        from dimd.models import DIMDBackbone  # noqa: F401

        self.backbone = DIMDBackbone()
        state = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.to(device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def forward_scores(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return logits


def list_images(root: str | Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    root = Path(root)
    paths = []
    for dp, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(Path(dp) / f)
    paths.sort()
    return paths


def score_pair(model: DIMDModel, real_dir: str, fake_dir: str, batch_size: int, device: torch.device) -> pd.DataFrame:
    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)
    data = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]

    scores, labels, fnames = [], [], []
    for i in tqdm(range(0, len(data), batch_size), desc="Scoring DIMD"):
        batch = data[i:i + batch_size]
        imgs, lbs, names = [], [], []
        for p, lb in batch:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            imgs.append(model.preprocess(img))
            lbs.append(lb)
            names.append(p.name)
        if not imgs:
            continue
        x = torch.stack(imgs, 0).to(device)
        logits = model.forward_scores(x)
        scores.extend(logits.detach().cpu().numpy().tolist())
        labels.extend(lbs)
        fnames.extend(names)

    return pd.DataFrame({"file": fnames, "score_dimd": scores, "label": labels})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_dimd.yaml")
    ap.add_argument("--ckpt", required=True, help="DIMD checkpoint path (from upstream repo).")
    ap.add_argument("--third_party_dimd", default="third_party/DIMD", help="Added to PYTHONPATH.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mode", choices=["baseline", "robust"], required=True)
    ap.add_argument("--out_dir", default="outputs/dimd")
    args = ap.parse_args()

    # make upstream DIMD importable
    os.environ["PYTHONPATH"] = str(Path(args.third_party_dimd).resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")

    P = load_yaml(args.paths)["dimd"]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = DIMDModel(args.ckpt, device=device)

    if args.mode == "baseline":
        df = score_pair(model, P["baseline"]["reals"], P["baseline"]["fakes"], args.batch_size, device)
        out = out_root / "baseline" / "dimd_baseline.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print("Wrote:", out)
        return

    # robust: score each attacked dir against the matching clean counterpart (real_dir fixed, fake_dir changes)
    # Save one CSV per condition (same naming scheme as UFD).
    rob_dir = out_root / "robust"
    rob_dir.mkdir(parents=True, exist_ok=True)

    # attacked fakes: keep real_dir = baseline reals, vary fake_dir
    real_dir = P["baseline"]["reals"]
    AF = P["attacked_fakes"]
    fake_jobs = [
        ("lama_bin1_fake", AF["lama"]["bin1"]),
        ("lama_bin2_fake", AF["lama"]["bin2"]),
        ("lama_bin3_fake", AF["lama"]["bin3"]),
        ("lama_bin4_fake", AF["lama"]["bin4"]),
        ("lama_randrect_fake", AF["lama"]["randrect"]),
        ("zits_bin1_fake", AF["zits"]["bin1"]),
        ("zits_bin2_fake", AF["zits"]["bin2"]),
        ("zits_bin3_fake", AF["zits"]["bin3"]),
        ("zits_bin4_fake", AF["zits"]["bin4"]),
        ("zits_randrect_fake", AF["zits"]["randrect"]),
    ]
    for name, fake_dir in fake_jobs:
        df = score_pair(model, real_dir, fake_dir, args.batch_size, device)
        df[df["label"] == 1][["file", "score_dimd"]].to_csv(rob_dir / f"{name}.csv", index=False)
        print("Wrote:", rob_dir / f"{name}.csv")

    # attacked reals: keep fake_dir = baseline fakes, vary real_dir
    fake_dir = P["baseline"]["fakes"]
    AR = P["attacked_reals"]
    real_jobs = [
        ("lama_bin1_real", AR["lama"]["bin1_0_3"]),
        ("lama_bin2_real", AR["lama"]["bin2_3_10"]),
        ("lama_bin3_real", AR["lama"]["bin3_10_25"]),
        ("lama_bin4_real", AR["lama"]["bin4_25_40"]),
        ("lama_randrect_real", AR["lama"]["randrect"]),
        ("zits_bin1_real", AR["zits"]["bin1_0_3"]),
        ("zits_bin2_real", AR["zits"]["bin2_3_10"]),
        ("zits_bin3_real", AR["zits"]["bin3_10_25"]),
        ("zits_bin4_real", AR["zits"]["bin4_25_40"]),
        ("zits_randrect_real", AR["zits"]["randrect"]),
        ("semitruths_reals_semantic", AR["semantic"]["reals_inpainted"]),
    ]
    for name, real_dir2 in real_jobs:
        df = score_pair(model, real_dir2, fake_dir, args.batch_size, device)
        df[df["label"] == 0][["file", "score_dimd"]].to_csv(rob_dir / f"{name}.csv", index=False)
        print("Wrote:", rob_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
