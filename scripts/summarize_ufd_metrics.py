#!/usr/bin/env python3
# scripts/summarize_ufd_metrics.py
'''
This script can be run with the following command:
python scripts/summarize_ufd_metrics.py --base_auc 0.7780
'''
import os, json, time, numpy as np, pandas as pd
from pathlib import Path
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--base_auc", type=float, required=True, help="baseline AUC printed by ufd_score_and_metrics.py")
ap.add_argument("--runs_dir", default="/home/oakin/ufd_runs")
ap.add_argument("--out_json", default="/home/oakin/results/ufd_metrics.json")
args = ap.parse_args()

runs = Path(args.runs_dir)
bre = pd.read_csv(runs/"baseline_reals.csv")
bfk = pd.read_csv(runs/"baseline_fakes.csv") if (runs/"baseline_fakes.csv").exists() else None
rre = pd.read_csv(runs/"robustness_reals.csv")
rin = pd.read_csv(runs/"robustness_inpainted.csv")

def auc(scores, labels):
    s = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels); neg = len(labels)-pos
    rsum = 0.0; rank = 0
    for _,y in s:
        rank += 1
        if y==1: rsum += rank
    U = rsum - pos*(pos+1)/2.0
    return U / max(pos*neg, 1)

rob_auc = auc(np.r_[rre.score.values, rin.score.values],
              np.r_[np.zeros(len(rre)), np.ones(len(rin))])
delta_auc = args.base_auc - rob_auc
thr = float(np.median(rre.score.values))
asr = float(np.mean(rin.score.values < thr))

Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump({
        "detector": "UFD (CLIP ViT-L/14 + linear head)",
        "baseline_auc": round(args.base_auc,4),
        "robust_auc": round(float(rob_auc),4),
        "delta_auc": round(float(delta_auc),4),
        "asr": round(float(asr),4),
        "asr_threshold": round(float(thr),4),
        "n_baseline": int((len(bre) + (len(bfk) if bfk is not None else 0))),
        "n_robust": int(len(rre) + len(rin)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs_dir": str(runs)
    }, f, indent=2)

print(f"[UFD] Robust AUC = {rob_auc:.4f}")
print(f"[UFD] ΔAUC       = {delta_auc:.4f}")
print(f"[UFD] ASR        = {asr:.3f} (thr={thr:.4f})")
print(f"[UFD] → {args.out_json}")
