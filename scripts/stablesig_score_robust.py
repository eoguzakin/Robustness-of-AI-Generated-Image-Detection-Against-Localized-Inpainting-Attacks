from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scripts.config_utils import load_yaml
from scripts.external_scorer import ensure_parent, run_cmd_template


def score_dir(score_cmd: str, in_dir: str, out_csv: Path):
    ensure_parent(out_csv)
    run_cmd_template(score_cmd, in_dir=in_dir, out_csv=str(out_csv))
    df = pd.read_csv(out_csv)[["file", "score_stablesig"]]
    df.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_stablesig.yaml")
    ap.add_argument(
        "--score_cmd",
        required=True,
        help="Command template that writes CSV with columns: file,score_stablesig. Use {in_dir} {out_csv}.",
    )
    ap.add_argument("--out_dir", default="outputs/stablesig/robust")
    args = ap.parse_args()

    D = load_yaml(args.paths)["stablesig"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    AF = D["attacked_fakes"]

    jobs = [
        ("lama_bin1_fake", AF["lama"]["bin1_0_3"]),
        ("lama_bin2_fake", AF["lama"]["bin2_3_10"]),
        ("lama_bin3_fake", AF["lama"]["bin3_10_25"]),
        ("lama_bin4_fake", AF["lama"]["bin4_25_40"]),
        ("lama_randrect_fake", AF["lama"]["randrect"]),
        ("zits_bin1_fake", AF["zits"]["bin1_0_3"]),
        ("zits_bin2_fake", AF["zits"]["bin2_3_10"]),
        ("zits_bin3_fake", AF["zits"]["bin3_10_25"]),
        ("zits_bin4_fake", AF["zits"]["bin4_25_40"]),
        ("zits_randrect_fake", AF["zits"]["randrect"]),
    ]

    for name, path in jobs:
        score_dir(args.score_cmd, path, out_dir / f"{name}.csv")
        print("Wrote:", out_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
