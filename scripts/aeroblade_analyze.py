# scripts/aeroblade_analyze.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def norm_real_name(name: str) -> str:
    # Real variants share a 16-char ID prefix. [file:3]
    base = os.path.splitext(os.path.basename(name))[0]
    prefix = base.split("_")[0]
    return prefix[:16]


def norm_fake_name(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def load_flat_csv(path: str | Path):
    df = pd.read_csv(path)
    scores = pd.to_numeric(df["score_aeroblade"], errors="coerce").to_numpy()
    labels = df["label"].astype(int).to_numpy()
    names = df["file"].astype(str).to_numpy()
    return scores, labels, names


def load_by_label(path: str | Path, sign: float):
    scores, labels, names = load_flat_csv(path)
    scores = scores * sign
    by = {0: {}, 1: {}}
    for s, y, n in zip(scores, labels, names):
        key = norm_real_name(n) if y == 0 else norm_fake_name(n)
        by[y][key] = float(s)
    return by


def best_threshold_by_accuracy(labels: np.ndarray, scores: np.ndarray):
    # Same threshold selection style as your WaRPAD script: maximize accuracy on baseline. [file:3]
    best_acc, best_thr = -1.0, float(np.min(scores))
    for t in np.unique(scores):
        preds = (scores >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(t)
    return best_thr, float(best_acc)


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, nboot: int, alpha: float, seed: int):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(labels)
    aucs = []
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    if not aucs:
        return np.nan, np.nan
    aucs = np.asarray(aucs)
    low = np.percentile(aucs, 100 * (alpha / 2))
    high = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(low), float(high)


def compute_asr_real(clean_real_scores: dict, cond_real_scores: dict, thr: float, nboot: int, alpha: float, seed: int):
    # ASRReal = pass rate on inpainted reals; denominator = baseline-correct reals. [file:3]
    keys = sorted(set(clean_real_scores) & set(cond_real_scores))
    if not keys:
        return np.nan, 0, np.nan, np.nan, np.nan, np.nan

    clean = np.array([clean_real_scores[k] for k in keys], dtype=float)
    cond = np.array([cond_real_scores[k] for k in keys], dtype=float)

    idx = np.where(clean < thr)[0]  # baseline predicted real
    denom = len(idx)
    if denom == 0:
        return np.nan, 0, np.nan, np.nan, np.nan, np.nan

    passed = (cond[idx] < thr).astype(int)
    point = passed.mean() * 100.0

    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(nboot):
        b = rng.randint(0, denom, denom)
        boots.append(passed[b].mean() * 100.0)
    boots = np.asarray(boots)

    return float(point), int(denom), float(boots.mean()), float(np.median(boots)), float(np.percentile(boots, 100 * (alpha / 2))), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def compute_asr_fake(clean_fake_scores: dict, cond_fake_scores: dict, thr: float, nboot: int, alpha: float, seed: int):
    # ASRFake = evasion rate on inpainted fakes; denominator = baseline-correct fakes. [file:3]
    keys = sorted(set(clean_fake_scores) & set(cond_fake_scores))
    if not keys:
        return np.nan, 0, np.nan, np.nan, np.nan, np.nan

    clean = np.array([clean_fake_scores[k] for k in keys], dtype=float)
    cond = np.array([cond_fake_scores[k] for k in keys], dtype=float)

    idx = np.where(clean >= thr)[0]  # baseline predicted fake
    denom = len(idx)
    if denom == 0:
        return np.nan, 0, np.nan, np.nan, np.nan, np.nan

    evaded = (cond[idx] < thr).astype(int)  # evasion => classified real
    point = evaded.mean() * 100.0

    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(nboot):
        b = rng.randint(0, denom, denom)
        boots.append(evaded[b].mean() * 100.0)
    boots = np.asarray(boots)

    return float(point), int(denom), float(boots.mean()), float(np.median(boots)), float(np.percentile(boots, 100 * (alpha / 2))), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", default="outputs/aeroblade/baseline/aeroblade_baseline.csv")
    ap.add_argument("--robust_dir", default="outputs/aeroblade/robust")
    ap.add_argument("--out_csv", default="outputs/aeroblade/aeroblade_summary.csv")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_scores_raw, base_labels, _ = load_flat_csv(args.baseline_csv)
    auc_raw = roc_auc_score(base_labels, base_scores_raw)

    # Orientation: flip if needed so higher = more fake. [file:3]
    sign = -1.0 if auc_raw < 0.5 else 1.0
    base_scores = base_scores_raw * sign

    auc_base = roc_auc_score(base_labels, base_scores)
    auc_lo, auc_hi = bootstrap_auc_ci(base_labels, base_scores, args.nboot, args.alpha, args.seed)
    thr, bestacc = best_threshold_by_accuracy(base_labels, base_scores)

    base_by = load_by_label(args.baseline_csv, sign=sign)
    clean_reals = base_by[0]
    clean_fakes = base_by[1]

    robust_dir = Path(args.robust_dir)
    cond_files = sorted([p for p in robust_dir.glob("*.csv") if not p.name.startswith("_tmp_")])

    rows = []
    for p in cond_files:
        cond = pd.read_csv(p)
        cond["score_aeroblade"] = pd.to_numeric(cond["score_aeroblade"], errors="coerce") * sign

        # These robust CSVs are file+score only, so label is inferred from filename suffix.
        # *_fake.csv => fake condition, *_real.csv or *semantic* => real condition.
        name = p.stem

        if name.endswith("_fake"):
            y = np.ones(len(cond), dtype=int)
            s = cond["score_aeroblade"].to_numpy()
            # AUC for fake splits is computed vs *baseline reals* in the original recipe,
            # but here we just report within-split AUC as undefined (single class).
            # To keep behavior consistent, set AUC NaN for single-class splits.
            auc_c = np.nan
            lo = hi = np.nan

            cond_fakes = {norm_fake_name(fn): float(sc) for fn, sc in zip(cond["file"].astype(str), s)}
            asr_point, denom, asr_mean, asr_med, asr_lo, asr_hi = compute_asr_fake(
                clean_fakes, cond_fakes, thr, args.nboot, args.alpha, args.seed
            )
            asr_kind = "ASRFake"

        else:
            # treat as real split
            y = np.zeros(len(cond), dtype=int)
            s = cond["score_aeroblade"].to_numpy()
            auc_c = np.nan
            lo = hi = np.nan

            cond_reals = {norm_real_name(fn): float(sc) for fn, sc in zip(cond["file"].astype(str), s)}
            asr_point, denom, asr_mean, asr_med, asr_lo, asr_hi = compute_asr_real(
                clean_reals, cond_reals, thr, args.nboot, args.alpha, args.seed
            )
            asr_kind = "ASRReal"

        rows.append({
            "Condition": name,
            "AUC": auc_c,
            "dAUC": np.nan,
            "AUCCI_low": lo,
            "AUCCI_high": hi,
            "baseline_AUC": round(auc_base * 100, 2),
            "baseline_AUCCI_low": round(auc_lo * 100, 2),
            "baseline_AUCCI_high": round(auc_hi * 100, 2),
            "sign": sign,
            "thr": thr,
            "baseline_bestacc": round(bestacc * 100, 2),
            "ASR_kind": asr_kind,
            "Nclean": int(denom),
            "ASR_point": round(float(asr_point), 2) if np.isfinite(asr_point) else np.nan,
            "ASR_mean": round(float(asr_mean), 2) if np.isfinite(asr_mean) else np.nan,
            "ASR_median": round(float(asr_med), 2) if np.isfinite(asr_med) else np.nan,
            "ASR_CI_low": round(float(asr_lo), 2) if np.isfinite(asr_lo) else np.nan,
            "ASR_CI_high": round(float(asr_hi), 2) if np.isfinite(asr_hi) else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("Condition")
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("=== AEROBLADE baseline ===")
    print(f"N={len(base_scores)} AUC={auc_base*100:.2f} 95%CI=[{auc_lo*100:.2f},{auc_hi*100:.2f}] bestAcc={bestacc*100:.2f} thr={thr:.6f} sign={sign:+.1f}")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
