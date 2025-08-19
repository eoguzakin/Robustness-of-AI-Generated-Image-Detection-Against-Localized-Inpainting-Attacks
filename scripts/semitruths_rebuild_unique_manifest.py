#!/usr/bin/env python3
# scripts/semitruths_rebuild_unique_manifest.py
'''
This script can be run with the following command:
    python scripts/semitruths_rebuild_unique_manifest.py \
  --base ~/data/semitruths/openimages_sd4_pairs_200
'''
import argparse, os, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="folder containing manifest_pairs.csv")
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

base = Path(os.path.expanduser(args.base))
src = base/"manifest_pairs.csv"
dst = base/f"manifest_pairs_unique{args.n}.csv"

df = pd.read_csv(src)
# original id = basename of original_local without extension
df["orig_id"] = df["original_local"].apply(lambda p: Path(p).stem)
df = df.drop_duplicates("orig_id")
df = df.head(args.n)
df[["edited_local","original_local","dataset","diffusion_model"]].to_csv(dst, index=False)
print(f"[rebuilt] wrote {len(df)} pairs → {dst}")
