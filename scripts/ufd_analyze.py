from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score

from ufd_common import real_id_from_file, fake_id_from_file

rng = np.random.default_rng(12345)


def auc_ci(reals, fakes, B=2000):
    reals = np.asarray(reals)
    fakes = np.asarray(fakes)
    nr, nf = len(reals), len(fakes)
    if nr == 0 or nf == 0:
        return np.nan, np.nan
    vals = []
    for _ in range(B):
        r = reals[rng.integers(0, nr, size=nr)]
        f = fakes[rng.integers(0, nf, size=nf)]
        y = np.concatenate([np.zeros(len(r), dtype=int), np.ones(len(f), dtype=int)])
        s = np.concatenate([r, f])
        fpr, tpr, _ = roc_curve(y, s)
        vals.append(auc(fpr, tpr))
    vals = np.sort(np.asarray(vals))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def asr_ci(successes, B=2000):
    successes = np.asarray(successes, dtype=int)
    n = len(successes)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    boots = []
    for _ in range(B):
        b = successes[rng.integers(0, n, size=n)]
        boots.append(b.mean() * 100.0)
    boots = np.sort(np.asarray(boots))
    point = successes.mean() * 100.0
    return float(point), float(boots.mean()), float(np.median(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", default="outputs/ufd/baseline/ufd_baseline.csv")
    ap.add_argument("--robust_dir", default="outputs/ufd/robust")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    B = args.B
    base = pd.read_csv(args.baseline_csv)
    base["rid"] = base["file"].apply(real_id_from_file)
    base["fid"] = base["file"].apply(fake_id_from_file)

    scores = base["score_ufd"].to_numpy()
    labels = base["label"].astype(int).to_numpy()

    ap100 = average_precision_score(labels, scores) * 100.0
    pred05 = (scores >= 0.5).astype(int)
    acc05 = accuracy_score(labels, pred05) * 100.0

    fpr, tpr, thr = roc_curve(labels, scores)
    youden = tpr - fpr
    t_opt = float(thr[int(np.argmax(youden))])
    auc_base = float(auc(fpr, tpr))

    pred_best = (scores >= t_opt).astype(int)
    acc_best = accuracy_score(labels, pred_best) * 100.0

    print("=== UFD baseline (fixed-threshold protocol) ===")
    print(f"AP={ap100:.2f}")
    print(f"acc@0.5={acc05:.2f}")
    print(f"AUC={auc_base*100:.2f}")
    print(f"thr_YoudenJ={t_opt:.6f}")
    print(f"acc@thr={acc_best:.2f}\n")

    reals = base[base["label"] == 0].copy()
    fakes = base[base["label"] == 1].copy()
    real_scores = reals["score_ufd"].to_numpy()
    fake_scores = fakes["score_ufd"].to_numpy()

    # baseline-correct sets at locked threshold
    ok_real_ids = set(reals.loc[real_scores < t_opt, "rid"])
    ok_fake_ids = set(fakes.loc[fake_scores >= t_opt, "fid"])

    print("Baseline-correct reals:", len(ok_real_ids))
    print("Baseline-correct fakes:", len(ok_fake_ids), "\n")

    robust_dir = Path(args.robust_dir)

    def load_attack_csv(name: str) -> pd.DataFrame:
        p = robust_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        return df

    # Convention: job names ending with "_fake" are attacked fakes; "_real" are attacked reals.
    fake_jobs = sorted([p.stem for p in robust_dir.glob("*_fake.csv")])
    real_jobs = sorted([p.stem for p in robust_dir.glob("*_real.csv")]) + \
                sorted([p.stem for p in robust_dir.glob("*semantic*.csv")])

    print("=== Fake-inpaint (ASRFake / evasion) ===")
    print("Condition,N_cleanFake,AUC,dAUC,AUCCI_low,AUCCI_high,ASR_point,ASR_mean,ASR_median,ASR_CI_low,ASR_CI_high")
    for name in fake_jobs:
        df = load_attack_csv(name)
        df["fid"] = df["file"].apply(fake_id_from_file)

        f_attack_all = df["score_ufd"].to_numpy()
        y = np.concatenate([np.zeros(len(real_scores), dtype=int), np.ones(len(f_attack_all), dtype=int)])
        s = np.concatenate([real_scores, f_attack_all])
        fpr_c, tpr_c, _ = roc_curve(y, s)
        auc_c = float(auc(fpr_c, tpr_c))
        lo, hi = auc_ci(real_scores, f_attack_all, B=B)
        dauc = (auc_c - auc_base) * 100.0

        df_ok = df[df["fid"].isin(ok_fake_ids)]
        # evasion success = predicted real under locked threshold
        success = (df_ok["score_ufd"].to_numpy() < t_opt).astype(int)
        asr = asr_ci(success, B=B)

        print(f"{name},{len(df_ok)},{auc_c*100:.2f},{dauc:.2f},{lo*100:.2f},{hi*100:.2f},{asr[0]:.2f},{asr[1]:.2f},{asr[2]:.2f},{asr[3]:.2f},{asr[4]:.2f}")

    print("\n=== Real-inpaint (ASRReal / pass rate) ===")
    print("Condition,N_cleanReal,AUC,dAUC,AUCCI_low,AUCCI_high,ASR_point,ASR_mean,ASR_median,ASR_CI_low,ASR_CI_high")
    for name in real_jobs:
        df = load_attack_csv(name)
        df["rid"] = df["file"].apply(real_id_from_file)

        r_attack_all = df["score_ufd"].to_numpy()
        y = np.concatenate([np.zeros(len(r_attack_all), dtype=int), np.ones(len(fake_scores), dtype=int)])
        s = np.concatenate([r_attack_all, fake_scores])
        fpr_c, tpr_c, _ = roc_curve(y, s)
        auc_c = float(auc(fpr_c, tpr_c))
        lo, hi = auc_ci(r_attack_all, fake_scores, B=B)
        dauc = (auc_c - auc_base) * 100.0

        df_ok = df[df["rid"].isin(ok_real_ids)]
        # pass success = predicted real under locked threshold
        success = (df_ok["score_ufd"].to_numpy() < t_opt).astype(int)
        asr = asr_ci(success, B=B)

        print(f"{name},{len(df_ok)},{auc_c*100:.2f},{dauc:.2f},{lo*100:.2f},{hi*100:.2f},{asr[0]:.2f},{asr[1]:.2f},{asr[2]:.2f},{asr[3]:.2f},{asr[4]:.2f}")


if __name__ == "__main__":
    main()
