#!/usr/bin/env python3
# scripts/semitruths_fetch_pairs.py
'''
This script can be run with this command:
# Example (OpenImages + SDv4), verbose and purging tars after use
python scripts/semitruths_fetch_pairs.py \
  --source OpenImages \
  --model StableDiffusion_v4 \
  --n_pairs 200 \
  --oversample 2 \
  --max_edited_shards 3 \
  --max_original_shards 10 \
  --out ~/data/semitruths/openimages_sd4_pairs_200 \
  --purge_tars \
  --verbose \
  --log_every 10
'''
import os, tarfile, re, shutil, sys, argparse, time
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from huggingface_hub import hf_hub_download

REPO = "semi-truths/Semi-Truths"

def rm(path):
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception:
        pass

def purge_tar(local_tar):
    # Remove the symlink (snapshot) and the resolved blob, if possible
    try:
        real = os.path.realpath(local_tar)
        if os.path.exists(local_tar): os.unlink(local_tar)
        if real != local_tar and os.path.exists(real): os.unlink(real)
    except Exception:
        pass

def list_members(tar_path):
    with tarfile.open(tar_path, "r:*") as t:
        for m in t.getmembers():
            if m.isfile():
                yield m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["OpenImages","ADE20K","SUN_RGBD","CelebAHQ"])
    ap.add_argument("--model", required=True, help="e.g. StableDiffusion_v4 / SDv4 / SDv5 etc.")
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--oversample", type=int, default=2, help="multiplier for edited candidates")
    ap.add_argument("--max_edited_shards", type=int, default=3, help="upper bound to probe for edited shards (0..N-1)")
    ap.add_argument("--max_original_shards", type=int, default=10, help="upper bound to probe for original shards (use 0..N-1, will stop early)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--purge_tars", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_pairs.csv"

    # Expected shard names
    edited_shards = [f"inpainting/{args.source}/{args.model}/{args.source}_{args.model}_{i}.tar.bz2"
                     for i in range(args.max_edited_shards)]
    original_shards = [f"original/images/{args.source}_images_{i}.tar.bz2"
                       for i in range(args.max_original_shards)]

    # --- PLAN: collect edited candidates (unique originals)
    target_unique = args.n_pairs * args.oversample
    wanted_by_orig = defaultdict(list)   # orig_id -> [edited_suffixes]
    collected = 0
    used_edited_shards = []

    print(f"[plan] edited targets: {target_unique} across ≤{len(edited_shards)} shard(s)")
    for sh in edited_shards:
        try:
            local = hf_hub_download(REPO, sh)
        except Exception as e:
            if args.verbose: print(f"[skip] cannot fetch {sh}: {e}")
            continue

        cnt = 0
        with tarfile.open(local, "r:*") as t:
            members = [m for m in t.getmembers() if m.isfile() and m.name.endswith(".png")]
            # Filenames look like: {orig}_{class}_{editid}_{source}_{model}.png
            for m in members:
                fname = Path(m.name).name
                orig = fname.split("_", 1)[0]
                if re.fullmatch(r"[0-9a-fA-F]+", orig):
                    wanted_by_orig[orig].append(fname)
                    cnt += 1
                    if len(wanted_by_orig) >= target_unique:
                        used_edited_shards.append(local)
                        break
        used_edited_shards.append(local)
        if args.verbose:
            print(f"[scan] {Path(sh).name}: {cnt} edited files seen | unique-orig so far={len(wanted_by_orig)}")
        if len(wanted_by_orig) >= target_unique:
            break

    if len(wanted_by_orig) == 0:
        print("[error] no edited candidates found. Check --source/--model.")
        sys.exit(1)

    # pick up to n_pairs unique originals with at least 1 edited
    orig_ids = list(wanted_by_orig.keys())[:args.n_pairs]
    print(f"[state] selected unique originals: {len(orig_ids)}")

    # Build suffix map of what we want to extract from edited shards
    # We take the first edited per original to keep 1:1 pairing.
    want_edited_names = set()
    for oid in orig_ids:
        # choose a deterministic edited (first)
        want_edited_names.add(wanted_by_orig[oid][0])

    # --- EXTRACT EDITED
    saved_edited = set()
    for local in used_edited_shards:
        with tarfile.open(local, "r:*") as t:
            picked = 0
            for m in t.getmembers():
                if not m.isfile(): continue
                fname = Path(m.name).name
                if fname in want_edited_names and fname not in saved_edited:
                    t.extract(m, out_dir)
                    # move to flat dir
                    (out_dir / m.name).rename(out_dir / fname)
                    saved_edited.add(fname)
                    picked += 1
                    if args.verbose and (len(saved_edited) % args.log_every == 0):
                        print(f"[extract:edited] {len(saved_edited)}/{len(want_edited_names)} … last={fname}")
            if args.verbose:
                print(f"[extract:edited] exact got {picked} from {Path(local).name}")
        if args.purge_tars:
            purge_tar(local)
        if len(saved_edited) == len(want_edited_names):
            break

    if len(saved_edited) == 0:
        print("[error] failed to extract edited images.")
        sys.exit(2)

    # --- EXTRACT ORIGINALS
    needed_orig = {oid for oid in orig_ids}
    saved_orig = set()
    for sh in original_shards:
        try:
            local = hf_hub_download(REPO, sh)
        except Exception:
            continue
        with tarfile.open(local, "r:*") as t:
            picked = 0
            for m in t.getmembers():
                if not m.isfile(): continue
                fname = Path(m.name).name  # e.g., 0bc65d083d184788.jpg
                stem = fname.rsplit(".", 1)[0]
                if stem in needed_orig and stem not in saved_orig:
                    t.extract(m, out_dir)
                    (out_dir / m.name).rename(out_dir / fname)
                    saved_orig.add(stem)
                    picked += 1
                    if args.verbose and (len(saved_orig) % args.log_every == 0):
                        print(f"[extract:orig] {len(saved_orig)}/{len(needed_orig)} … last={fname}")
            if args.verbose:
                print(f"[extract:orig] exact got {picked} from {Path(local).name}")
        if args.purge_tars:
            purge_tar(local)
        if len(saved_orig) == len(needed_orig):
            break

    # --- BUILD MANIFEST (up to n_pairs with both sides present)
    rows = []
    have = 0
    for oid in orig_ids:
        # edited name starts with oid_
        edited = [n for n in saved_edited if n.startswith(f"{oid}_")]
        if edited and oid in saved_orig:
            rows.append((
                str(out_dir / edited[0]),
                str(out_dir / f"{oid}.jpg"),
                args.source,
                args.model
            ))
            have += 1
            if have == args.n_pairs:
                break

    if not rows:
        print("[done] wrote 0 pairs (no overlap).")
        return

    with open(manifest_path, "w") as f:
        f.write("edited_local,original_local,dataset,diffusion_model\n")
        for e,o,ds,dm in rows:
            f.write(f"{e},{o},{ds},{dm}\n")

    print(f"[done] wrote {len(rows)} pairs → {manifest_path}")

if __name__ == "__main__":
    main()
