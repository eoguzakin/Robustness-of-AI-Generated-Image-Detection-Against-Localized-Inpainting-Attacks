from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scripts.config_utils import load_yaml
from scripts.dimd_common import DIMDWrapper, get_device, list_images


def score_dir(model: DIMDWrapper, in_dir: str, out_csv: Path, batch_size: int):
    files = list_images(in_dir)
    scores = model.score_paths(files, batch_size=batch_size)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": [p.name for p in files], "score_dimd": [float(s) for s in scores]}).to_csv(out_csv, index=False)
    print("Wrote:", out_csv, "N=", len(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_dimd.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs/dimd/robust")
    args = ap.parse_args()

    D = load_yaml(args.paths)["dimd"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    model = DIMDWrapper(args.ckpt, device=device)

    AF = D["attacked_fakes"]
    AR = D["attacked_reals"]

    jobs = [
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

    for name, path in jobs:
        score_dir(model, path, out_dir / f"{name}.csv", batch_size=args.batch_size)


if __name__ == "__main__":
    main()
