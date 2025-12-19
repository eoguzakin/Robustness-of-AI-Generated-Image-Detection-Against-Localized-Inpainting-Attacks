from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scripts.config_utils import load_yaml
from scripts.dimd_common import DIMDWrapper, get_device, list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_dimd.yaml")
    ap.add_argument("--ckpt", required=True, help="Path to DIMD checkpoint (.pth).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs/dimd/baseline")
    args = ap.parse_args()

    P = load_yaml(args.paths)["dimd"]
    real_dir = P["baseline"]["reals"]
    fake_dir = P["baseline"]["fakes"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    model = DIMDWrapper(args.ckpt, device=device)

    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)

    real_scores = model.score_paths(real_files, batch_size=args.batch_size)
    fake_scores = model.score_paths(fake_files, batch_size=args.batch_size)

    rows = []
    for p, s in zip(real_files, real_scores):
        rows.append({"file": p.name, "score_dimd": float(s), "label": 0})
    for p, s in zip(fake_files, fake_scores):
        rows.append({"file": p.name, "score_dimd": float(s), "label": 1})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dimd_baseline.csv", index=False)
    print("Wrote:", out_dir / "dimd_baseline.csv")


if __name__ == "__main__":
    main()
