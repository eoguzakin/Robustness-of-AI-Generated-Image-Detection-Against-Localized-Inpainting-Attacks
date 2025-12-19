from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scripts.config_utils import load_yaml
from scripts.ufd_common import list_images, load_ufd, ufd_score_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_ufd.yaml")
    ap.add_argument("--ckpt", default="pretrained_weights/fc_weights.pth")
    ap.add_argument("--third_party", default="third_party")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs/ufd/baseline")
    args = ap.parse_args()

    P = load_yaml(args.paths)["ufd"]
    real_dir = P["baseline"]["reals"]
    fake_dir = P["baseline"]["fakes"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess, fc, device = load_ufd(
        ckpt_path=args.ckpt, third_party_dir=args.third_party, device=args.device
    )

    rows = []
    for p in list_images(real_dir):
        rows.append({"file": p.name, "score_ufd": ufd_score_image(p, clip_model, preprocess, fc, device), "label": 0})
    for p in list_images(fake_dir):
        rows.append({"file": p.name, "score_ufd": ufd_score_image(p, clip_model, preprocess, fc, device), "label": 1})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ufd_baseline.csv", index=False)
    print("Wrote:", out_dir / "ufd_baseline.csv")


if __name__ == "__main__":
    main()
