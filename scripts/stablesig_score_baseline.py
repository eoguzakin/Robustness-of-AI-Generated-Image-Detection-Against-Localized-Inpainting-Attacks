from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scripts.config_utils import load_yaml
from scripts.external_scorer import ensure_parent, run_cmd_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_stablesig.yaml")
    ap.add_argument(
        "--score_cmd",
        required=True,
        help="Command template that writes CSV with columns: file,score_stablesig. Use {in_dir} {out_csv}.",
    )
    ap.add_argument("--out_dir", default="outputs/stablesig/baseline")
    args = ap.parse_args()

    P = load_yaml(args.paths)["stablesig"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_tmp = out_dir / "_tmp_reals.csv"
    fake_tmp = out_dir / "_tmp_fakes.csv"
    ensure_parent(real_tmp)
    ensure_parent(fake_tmp)

    run_cmd_template(args.score_cmd, in_dir=P["baseline"]["reals"], out_csv=str(real_tmp))
    run_cmd_template(args.score_cmd, in_dir=P["baseline"]["fakes"], out_csv=str(fake_tmp))

    dr = pd.read_csv(real_tmp)[["file", "score_stablesig"]].copy()
    df = pd.read_csv(fake_tmp)[["file", "score_stablesig"]].copy()
    dr["label"] = 0
    df["label"] = 1

    out = pd.concat([dr, df], ignore_index=True)
    out.to_csv(out_dir / "stablesig_baseline.csv", index=False)
    print("Wrote:", out_dir / "stablesig_baseline.csv")


if __name__ == "__main__":
    main()
