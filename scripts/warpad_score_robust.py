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


def postprocess(tmp_csv: Path, out_csv: Path):
    df = pd.read_csv(tmp_csv)
    df = df.rename(columns={"filename": "file", "score": "score_warpad"})
    # keep label too (useful if you want to sanity-check), but analyzer only needs file+score
    df = df[["file", "score_warpad", "label"]]
    df.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_warpad.yaml")
    ap.add_argument("--warpad_script", default="third_party/WaRPAD/scoredirwarpad.py")
    ap.add_argument("--prepsize", type=int, default=896)
    ap.add_argument("--patchsize", type=int, default=224)
    ap.add_argument("--noiselevel", type=float, default=0.1)
    ap.add_argument("--batchsize", type=int, default=4)
    ap.add_argument("--out_dir", default="outputs/warpad/robust")
    args = ap.parse_args()

    D = load_yaml(args.paths)["warpad"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_base = D["baseline"]["reals"]
    fake_base = D["baseline"]["fakes"]

    AF = D["attacked_fakes"]
    AR = D["attacked_reals"]

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

    # Score attacked fakes (reals fixed, fakes vary)
    for name, fake_dir in fake_jobs:
        tmp = out_dir / f"_tmp_{name}.csv"
        out = out_dir / f"{name}.csv"
        run_warpad(args.warpad_script, real_base, fake_dir, tmp,
                  args.prepsize, args.patchsize, args.noiselevel, args.batchsize)
        postprocess(tmp, out)
        print("Wrote:", out)

    # Score attacked reals (reals vary, fakes fixed)
    for name, real_dir in real_jobs:
        tmp = out_dir / f"_tmp_{name}.csv"
        out = out_dir / f"{name}.csv"
        run_warpad(args.warpad_script, real_dir, fake_base, tmp,
                  args.prepsize, args.patchsize, args.noiselevel, args.batchsize)
        postprocess(tmp, out)
        print("Wrote:", out)


if __name__ == "__main__":
    main()
