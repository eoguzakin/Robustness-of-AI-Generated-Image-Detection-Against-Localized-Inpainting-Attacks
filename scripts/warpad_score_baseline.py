from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import pandas as pd

from scripts.config_utils import load_yaml


def run_warpad(script: str, reals: str, fakes: str, out_csv: Path,
              prepsize: int, patchsize: int, noiselevel: float, batchsize: int):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", script,
        "--reals", reals,
        "--fakes", fakes,
        "--out", str(out_csv),
        "--prepsize", str(prepsize),
        "--patchsize", str(patchsize),
        "--noiselevel", str(noiselevel),
        "--batchsize", str(batchsize),
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"WaRPAD scoring failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_warpad.yaml")
    ap.add_argument("--warpad_script", default="third_party/WaRPAD/scoredirwarpad.py")
    ap.add_argument("--prepsize", type=int, default=896)
    ap.add_argument("--patchsize", type=int, default=224)
    ap.add_argument("--noiselevel", type=float, default=0.1)
    ap.add_argument("--batchsize", type=int, default=4)
    ap.add_argument("--out_dir", default="outputs/warpad/baseline")
    args = ap.parse_args()

    P = load_yaml(args.paths)["warpad"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = out_dir / "_tmp_warpad_baseline_raw.csv"
    run_warpad(
        args.warpad_script,
        P["baseline"]["reals"],
        P["baseline"]["fakes"],
        tmp,
        args.prepsize, args.patchsize, args.noiselevel, args.batchsize,
    )

    df = pd.read_csv(tmp)
    df = df.rename(columns={"filename": "file", "score": "score_warpad"})
    df = df[["file", "score_warpad", "label"]]
    df.to_csv(out_dir / "warpad_baseline.csv", index=False)
    print("Wrote:", out_dir / "warpad_baseline.csv")


if __name__ == "__main__":
    main()
