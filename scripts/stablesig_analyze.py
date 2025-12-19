# scripts/stablesig_analyze.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def norm_real_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    prefix = base.split("_")[0]
    return prefix[:16]


def norm_fake_name(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def load_baseline(path: str | Path):
    df = pd.read_csv(path)
    s = pd.to_numeric(df["score_stablesig"], errors="coerce").to_numpy()
    y = df["label"].astype(int).to_numpy()
    f = df["file"].astype(str).to_numpy()
    return s, y, f


def best_threshold_by_accuracy(labels: np.ndarray, scores: np.ndarray):
    best_acc, best_thr = -1.0, float(np.min(scores))
    for t in np.unique(scores):
        preds = (scores >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(t)
    return best_thr, float(best_acc)


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, nboot: int, alpha: float, seed: int):
    rng = np.random.RandomState(seed)
    n = len(labels)
    vals = []
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(roc_auc_score(labels[idx], scores[idx]))
    if not vals:
        return np.nan, np.nan
    arr = np.asarray(vals)
    lo = np.percentile(arr, 100 * (alpha / 2))
    hi = np.percentile(arr, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def asr_fake(clean_fake_scores: dict, cond_fake_scores: dict, thr: float, nboot: int, alpha: float, seed: int):
    keys = sorted(set(clean_fake_scores) & set(cond_fake_scores))
    if not keys:
        return np.nan, 0, np.nan, np.nan, np.nan, np.nan

    clean = np.array([clean_fake_scores[k] for k in keys], dtype=float)
    cond = np.array([cond_fake_scores[k] for k in keys], dtype=float)

    idx = np.where(clean >= thr)[0]  # baseline-correct fakes (predicted fake on clean)
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
    ap.add_argument("--baseline_csv", default="outputs/stablesig/baseline/stablesig_baseline.csv")
    ap.add_argument("--robust_dir", default="outputs/stablesig/robust")
    ap.add_argument("--out_csv", default="outputs/stablesig/stablesig_summary.csv")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s_raw, y, _ = load_baseline(args.baseline_csv)
    auc_raw = roc_auc_score(y, s_raw)
    sign = -1.0 if auc_raw < 0.5 else 1.0  # enforce higher = more watermarked/fake
    s = s_raw * sign

    auc_base = roc_auc_score(y, s)
    auc_lo, auc_hi = bootstrap_auc_ci(y, s, args.nboot, args.alpha, args.seed)
    thr, bestacc = best_threshold_by_accuracy(y, s)

    dfb = pd.read_csv(args.baseline_csv)
    dfb["score_stablesig"] = pd.to_numeric(dfb["score_stablesig"], errors="coerce") * sign

    base_reals = {}
    base_fakes = {}
    for fn, lb, sc in zip(dfb["file"].astype(str), dfb["label"].astype(int), dfb["score_stablesig"].to_numpy()):
        if lb == 0:
            base_reals[norm_real_name(fn)] = float(sc)
        else:
            base_fakes[norm_fake_name(fn)] = float(sc)

    rows = []
    for p in sorted(Path(args.robust_dir).glob("*.csv")):
        name = p.stem
        df = pd.read_csv(p)
        df["score_stablesig"] = pd.to_numeric(df["score_stablesig"], errors="coerce") * sign

        cond_fakes = {norm_fake_name(fn): float(sc) for fn, sc in zip(df["file"].astype(str), df["score_stablesig"].to_numpy())}

        y_auc = np.concatenate([np.zeros(len(base_reals), dtype=int), np.ones(len(cond_fakes), dtype=int)])
        s_auc = np.concatenate([np.array(list(base_reals.values()), dtype=float), np.array(list(cond_fakes.values()), dtype=float)])
        auc_c = roc_auc_score(y_auc, s_auc)
        lo, hi = bootstrap_auc_ci(y_auc, s_auc, args.nboot, args.alpha, args.seed)
        dauc = (auc_c - auc_base) * 100.0

        asr_point, denom, asr_mean, asr_med, asr_lo, asr_hi = asr_fake(
            base_fakes, cond_fakes, thr, args.nboot, args.alpha, args.seed
        )

        rows.append({
            "Condition": name,
            "AUC": round(auc_c * 100, 2),
            "dAUC": round(dauc, 2),
            "AUCCI_low": round(lo * 100, 2),
            "AUCCI_high": round(hi * 100, 2),
            "baseline_AUC": round(auc_base * 100, 2),
            "baseline_AUCCI_low": round(auc_lo * 100, 2),
            "baseline_AUCCI_high": round(auc_hi * 100, 2),
            "sign": sign,
            "thr": thr,
            "baseline_bestacc": round(bestacc * 100, 2),
            "NcleanFake": int(denom),
            "ASRFake_point": round(asr_point, 2) if np.isfinite(asr_point) else np.nan,
            "ASRFake_mean": round(asr_mean, 2) if np.isfinite(asr_mean) else np.nan,
            "ASRFake_median": round(asr_med, 2) if np.isfinite(asr_med) else np.nan,
            "ASRFake_CI_low": round(asr_lo, 2) if np.isfinite(asr_lo) else np.nan,
            "ASRFake_CI_high": round(asr_hi, 2) if np.isfinite(asr_hi) else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("Condition")
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("=== Stable Signature baseline ===")
    print(f"N={len(s)} AUC={auc_base*100:.2f} 95%CI=[{auc_lo*100:.2f},{auc_hi*100:.2f}] bestAcc={bestacc*100:.2f} thr={thr:.6f} sign={sign:+.1f}")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
