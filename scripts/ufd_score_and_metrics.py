#!/usr/bin/env python3
# scripts/ufd_score_and_metrics.py
import os, torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import clip
from tqdm import tqdm

# --------- folders (physical copies, not symlinks) ----------
BASE_REAL   = "/home/oakin/data/exp_sets_physical/baseline/reals"
BASE_FAKE   = "/home/oakin/data/exp_sets_physical/baseline/fakes"
ROB_REAL    = "/home/oakin/data/exp_sets_physical/robustness/reals"
ROB_EDITED  = "/home/oakin/data/exp_sets_physical/robustness/edited"
CKPT        = "/home/oakin/UniversalFakeDetect/pretrained_weights/fc_weights.pth"
OUTDIR      = "/home/oakin/ufd_runs"
BATCH       = 32

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

model, preprocess = clip.load("ViT-L/14", device=device)
model.eval(); model = model.float()   # ensure fp32

ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt.get("state_dict", ckpt)

w = b = None
for k,v in state.items():
    if k.endswith("weight") and v.ndim==2 and "fc." in k:
        w = v.clone().float(); b = state.get(k.replace("weight","bias")); break
if w is None:
    for k,v in state.items():
        if v.ndim==2:
            w = v.clone().float(); b = state.get(k.replace("weight","bias")); break
if w is None: raise RuntimeError("Couldn't locate linear head in checkpoint.")

w = w.to(device, dtype=torch.float32)
b = (b.clone().float().to(device) if b is not None else torch.zeros(w.shape[0], device=device))

if w.shape[0] == 2:
    w = w[1] - w[0]; b = b[1] - b[0]
elif w.shape[0] == 1:
    w = w[0]; b = b[0]
else:
    w = w[-1] - w[0]; b = b[-1] - b[0]

def list_imgs(folder):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    return sorted([str(p) for p in Path(folder).glob("*") if p.suffix.lower() in exts])

def score_paths(paths):
    scores, keep = [], []
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i+BATCH]
        ims, valid = [], []
        for p in chunk:
            try:
                ims.append(preprocess(Image.open(p).convert("RGB")))
                valid.append(p)
            except Exception:
                pass
        if not ims: continue
        ims = torch.stack(ims).to(device, dtype=torch.float32)
        feat = model.encode_image(ims).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        s = (feat @ w.unsqueeze(1)).squeeze(1) + b
        scores.extend(s.detach().cpu().tolist())
        keep.extend(valid)
    return pd.DataFrame({"path": keep, "score": scores})

def auc(scores, labels):
    s = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels); neg = len(labels)-pos
    rsum = 0.0; rank = 0
    for _,y in s:
        rank += 1
        if y==1: rsum += rank
    U = rsum - pos*(pos+1)/2.0
    return U / max(pos*neg, 1)

def run_split(real_dir, fake_dir, tag):
    reals = list_imgs(real_dir); fakes = list_imgs(fake_dir)
    print(f"[{tag}] found {len(reals)} reals, {len(fakes)} fakes")
    dr = score_paths(reals); df = score_paths(fakes)
    dr.to_csv(f"/home/oakin/ufd_runs/{tag}_reals.csv", index=False)
    df.to_csv(f"/home/oakin/ufd_runs/{tag}_fakes.csv", index=False)
    S = np.concatenate([dr["score"].values, df["score"].values])
    L = np.array([0]*len(dr) + [1]*len(df))
    A = auc(S, L)
    print(f"[{tag}] AUC={A:.4f}  N={len(S)}")
    return A

def run_robust(real_dir, edited_dir, tag):
    reals = list_imgs(real_dir); edits = list_imgs(edited_dir)
    print(f"[{tag}] found {len(reals)} reals, {len(edits)} inpainted")
    dr = score_paths(reals); de = score_paths(edits)
    dr.to_csv(f"/home/oakin/ufd_runs/{tag}_reals.csv", index=False)
    de.to_csv(f"/home/oakin/ufd_runs/{tag}_inpainted.csv", index=False)
    thr = float(np.median(dr["score"].values))
    asr = float(np.mean(de["score"].values < thr))
    print(f"[{tag}] ASR={asr:.3f}  (thr={thr:.4f})  N={len(de)}")
    return asr, thr

if __name__ == "__main__":
    run_split(BASE_REAL, BASE_FAKE, "baseline")
    run_robust(ROB_REAL, ROB_EDITED, "robustness")
