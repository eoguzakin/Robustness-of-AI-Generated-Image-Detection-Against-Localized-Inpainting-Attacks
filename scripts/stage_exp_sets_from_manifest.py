#!/usr/bin/env python3
# scripts/stage_exp_sets_from_manifest.py
'''
This script can be run with the following command:
python scripts/stage_exp_sets_from_manifest.py \
  --manifest ~/data/semitruths/openimages_sd4_pairs_200/manifest_pairs_unique200.csv \
  --out ~/data/exp_sets \
  --fakes_dir ~/data/genimage/ai_200 \
  --copy
'''
import os, shutil, argparse, csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, help="manifest_pairs_uniqueN.csv")
ap.add_argument("--out", required=True, help="root out dir, e.g., ~/data/exp_sets")
ap.add_argument("--fakes_dir", required=True, help="dir with 200 GenImage fakes (jpg/png)")
ap.add_argument("--copy", action="store_true", help="copy files (default). If omitted, will hardlink when possible.")
args = ap.parse_args()

manifest = Path(os.path.expanduser(args.manifest))
out = Path(os.path.expanduser(args.out))
fakes = Path(os.path.expanduser(args.fakes_dir))

(basereals := out/"baseline/reals").mkdir(parents=True, exist_ok=True)
(basefakes := out/"baseline/fakes").mkdir(parents=True, exist_ok=True)
(rob_reals := out/"robustness/reals").mkdir(parents=True, exist_ok=True)
(rob_edits := out/"robustness/edited").mkdir(parents=True, exist_ok=True)

def place(src, dst):
    src = Path(src); dst = Path(dst)
    if args.copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

rows=[]
with open(manifest, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# Reals & Edited
for row in rows:
    o = row["original_local"]; e = row["edited_local"]
    place(o, basereals/Path(o).name)
    place(o, rob_reals/Path(o).name)
    place(e, rob_edits/Path(e).name)

# Fakes (assume already 200 in fakes folder)
fake_paths = sorted([p for p in fakes.iterdir() if p.is_file()])[:len(rows)]
for p in fake_paths:
    place(p, basefakes/p.name)

print(f"[stage] baseline reals={len(rows)} fakes={len(fake_paths)} | robustness reals={len(rows)} edited={len(rows)}")
print(f"[out] {out}")
