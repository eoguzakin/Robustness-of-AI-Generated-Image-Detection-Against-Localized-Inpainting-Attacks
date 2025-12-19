from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ufd_common import list_images, load_ufd, ufd_score_image


def score_dir(in_dir: str, out_csv: Path, clip_model, preprocess, fc, device: str):
    files = list_images(in_dir)
    rows = [{"file": p.name, "score_ufd": ufd_score_image(p, clip_model, preprocess, fc, device)} for p in files]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Wrote:", out_csv, "N=", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="pretrained_weights/fc_weights.pth")
    ap.add_argument("--third_party", default="third_party")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs/ufd/robust")
    ap.add_argument("--jobs", required=True, help="Text file: each line is 'name,path'")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess, fc, device = load_ufd(
        ckpt_path=args.ckpt, third_party_dir=args.third_party, device=args.device
    )

    jobs = []
    for line in Path(args.jobs).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, path = [x.strip() for x in line.split(",", 1)]
        jobs.append((name, path))

    for name, path in jobs:
        score_dir(path, out_dir / f"{name}.csv", clip_model, preprocess, fc, device)


if __name__ == "__main__":
    main()
