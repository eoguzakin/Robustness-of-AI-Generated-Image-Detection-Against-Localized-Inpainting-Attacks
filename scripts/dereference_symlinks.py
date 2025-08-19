#!/usr/bin/env python3
import os, shutil
from pathlib import Path
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
args = ap.parse_args()

src = Path(os.path.expanduser(args.src))
dst = Path(os.path.expanduser(args.dst)); dst.mkdir(parents=True, exist_ok=True)
for p in src.iterdir():
    if p.is_file():
        shutil.copy2(p.resolve(), dst/p.name)
print(f"[done] copied → {dst}")
