from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths_ufd.yaml")
    ap.add_argument("--out", default="configs/ufd_jobs_ufd.txt")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
    U = cfg["ufd"]

    lines = []

    # attacked fakes
    lines += [
        f"lama_bin1_fake,{U['attacked_fakes']['lama']['bin1']}",
        f"lama_bin2_fake,{U['attacked_fakes']['lama']['bin2']}",
        f"lama_bin3_fake,{U['attacked_fakes']['lama']['bin3']}",
        f"lama_bin4_fake,{U['attacked_fakes']['lama']['bin4']}",
        f"lama_randrect_fake,{U['attacked_fakes']['lama']['randrect']}",
        f"zits_bin1_fake,{U['attacked_fakes']['zits']['bin1']}",
        f"zits_bin2_fake,{U['attacked_fakes']['zits']['bin2']}",
        f"zits_bin3_fake,{U['attacked_fakes']['zits']['bin3']}",
        f"zits_bin4_fake,{U['attacked_fakes']['zits']['bin4']}",
        f"zits_randrect_fake,{U['attacked_fakes']['zits']['randrect']}",
    ]

    # attacked reals
    lines += [
        f"lama_bin1_real,{U['attacked_reals']['lama']['bin1_0_3']}",
        f"lama_bin2_real,{U['attacked_reals']['lama']['bin2_3_10']}",
        f"lama_bin3_real,{U['attacked_reals']['lama']['bin3_10_25']}",
        f"lama_bin4_real,{U['attacked_reals']['lama']['bin4_25_40']}",
        f"lama_randrect_real,{U['attacked_reals']['lama']['randrect']}",
        f"zits_bin1_real,{U['attacked_reals']['zits']['bin1_0_3']}",
        f"zits_bin2_real,{U['attacked_reals']['zits']['bin2_3_10']}",
        f"zits_bin3_real,{U['attacked_reals']['zits']['bin3_10_25']}",
        f"zits_bin4_real,{U['attacked_reals']['zits']['bin4_25_40']}",
        f"zits_randrect_real,{U['attacked_reals']['zits']['randrect']}",
        f"semitruths_reals_semantic,{U['attacked_reals']['semantic']['reals_inpainted']}",
    ]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", outp)


if __name__ == "__main__":
    main()
