"""
06_novel_contribution.py
AI4003 CEP - Part D: risk-controlled referral threshold.

Question: if we want referral sensitivity >= s* for the high-concern group, does a
threshold picked on a validation set actually deliver it on NEW lesions, and does a
statistically controlled threshold do better?

Threshold rules (refer a lesion if P(high-concern) >= t):
    naive       largest t whose sensitivity on the calibration set is >= s*
    img_lcb     largest t whose Clopper-Pearson lower bound (counting IMAGES) is >= s*
    lesion_lcb  same, but counting LESIONS (images of one lesion are not independent)

Each repetition re-splits the TRAIN+VALIDATION pool by lesion into
train / calibration (threshold-picking) / held-out. The locked TEST set is never used.

Also analyses RQ3: calibration of P(high-concern) for the cost-based rule t = 1/(1+r).

Run from the project root:
    python src/06_novel_contribution.py --reps 2     (quick check)
    python src/06_novel_contribution.py --reps 30    (real run, about 10 minutes)
Needs:  pip install matplotlib scipy   (scipy comes with scikit-learn)
"""
import argparse
import importlib.util
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (CACHE, CLASS_TO_IDX, HC_IDX, OUT, R_VALUES, SEED,
                    expected_cost, load_splits)

# reuse the metadata preprocessing from script 02 (file name starts with a digit)
_spec = importlib.util.spec_from_file_location("prep02", "src/02_preprocess_features.py")
prep02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep02)

TARGETS = (0.80, 0.85, 0.90)
DELTA = 0.10  # we want the target to hold with at least 90% confidence
METHODS = ("naive", "img_lcb", "lesion_lcb")
C_BEST = 0.1  # chosen on validation in script 04


# ---------------------------------------------------------------- helpers
def cp_lower(k, n, delta):
    """One-sided Clopper-Pearson lower confidence bound for a proportion k/n."""
    if n == 0 or k <= 0:
        return 0.0
    return float(beta.ppf(delta, k, n - k + 1))


def choose_threshold(p, hc, lesion, target, method):
    """Largest threshold whose criterion on the calibration set meets the target."""
    p_hc = p[hc]
    cands = np.unique(p_hc)[::-1]  # high to low
    _, uniq_idx = np.unique(lesion[hc], return_inverse=True)
    n_per_lesion = np.bincount(uniq_idx)
    for t in cands:
        caught = p_hc >= t
        if method == "naive":
            ok = caught.mean() >= target
        elif method == "img_lcb":
            ok = cp_lower(int(caught.sum()), len(caught), DELTA) >= target
        else:  # lesion_lcb: count lesions, not images
            frac = np.bincount(uniq_idx, weights=caught.astype(float)) / n_per_lesion
            k_eff = int(np.floor(frac.sum()))
            ok = cp_lower(k_eff, len(frac), DELTA) >= target
        if ok:
            return float(t)
    return 0.0  # cannot reach the target: refer everyone


def ece_binary(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    df = load_splits()
    pool = df[df["split"] != "test"].reset_index(drop=True)  # test set stays locked
    assert (pool["split"] != "test").all()

    z = np.load(CACHE / "emb_effb0.npz", allow_pickle=True)
    emb = dict(zip(z["ids"], z["emb"]))
    E = np.stack([emb[i] for i in pool["image_id"]])
    y = pool["dx"].map(CLASS_TO_IDX).to_numpy()
    hc_all = np.isin(y, HC_IDX)
    lesion = pool["lesion_id"].to_numpy()
    meta_raw = prep02.clean_meta(pool)
    print(f"Pool (train+val): {len(pool)} images, {len(np.unique(lesion))} lesions, "
          f"{hc_all.sum()} high-concern images. Test set not used.")

    rows, cost_rows = [], []
    t0 = time.time()
    for rep in range(args.reps):
        seed = SEED + rep
        outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        rest, held = next(outer.split(E, y, lesion))
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        tr_rel, ca_rel = next(inner.split(E[rest], y[rest], lesion[rest]))
        tr, ca = rest[tr_rel], rest[ca_rel]
        # proof: lesions are disjoint across the three parts
        assert not (set(lesion[tr]) & set(lesion[ca])) and not (set(lesion[tr]) & set(lesion[held])) \
            and not (set(lesion[ca]) & set(lesion[held])), "LEAK in resplit"

        # preprocessing and model fitted on this repetition's TRAIN part only
        pre = prep02.build_preprocessor()
        M_tr = pre.fit_transform(meta_raw.iloc[tr])
        X_tr = np.hstack([E[tr], M_tr])
        X_ca = np.hstack([E[ca], pre.transform(meta_raw.iloc[ca])])
        X_he = np.hstack([E[held], pre.transform(meta_raw.iloc[held])])
        model = Pipeline([("scale", StandardScaler()),
                          ("clf", LogisticRegression(C=C_BEST, class_weight="balanced",
                                                     max_iter=3000, random_state=seed))])
        model.fit(X_tr, y[tr])
        p_ca = model.predict_proba(X_ca)[:, HC_IDX].sum(axis=1)
        p_he = model.predict_proba(X_he)[:, HC_IDX].sum(axis=1)
        hc_ca, hc_he = hc_all[ca], hc_all[held]

        # ---- threshold rules, evaluated on the held-out lesions
        for target in TARGETS:
            for method in METHODS:
                t = choose_threshold(p_ca, hc_ca, lesion[ca], target, method)
                ref_ca = p_ca >= t
                ref_he = p_he >= t
                sens_he = float(ref_he[hc_he].mean())
                rows.append({
                    "rep": rep, "target": target, "method": method, "threshold": t,
                    "cal_sensitivity": float(ref_ca[hc_ca].mean()),
                    "heldout_sensitivity": sens_he,
                    "heldout_referral_rate": float(ref_he.mean()),
                    "missed_target": sens_he < target,
                    "shortfall": max(0.0, target - sens_he),
                })

        # ---- RQ3: calibration of P(high-concern) for the cost-based threshold 1/(1+r)
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(logit(p_ca).reshape(-1, 1), hc_ca.astype(int))
        pc_he = platt.predict_proba(logit(p_he).reshape(-1, 1))[:, 1]
        ece_raw = ece_binary(p_he, hc_he.astype(float))
        ece_cal = ece_binary(pc_he, hc_he.astype(float))
        for r in R_VALUES:
            thr = 1.0 / (1.0 + r)
            for name, scores, ece in (("raw", p_he, ece_raw), ("platt_calibrated", pc_he, ece_cal)):
                referred = scores >= thr
                cost_rows.append({
                    "rep": rep, "r": r, "version": name, "ece_hc": ece,
                    "referral_rate": float(referred.mean()),
                    "referral_sensitivity": float(referred[hc_he].mean()),
                    "cost_per_lesion": expected_cost(y[held], referred, r),
                })
        if (rep + 1) % 5 == 0 or rep == 0:
            print(f"  rep {rep + 1}/{args.reps} done ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    cost = pd.DataFrame(cost_rows)

    # ---------------------------------------------------------- summaries
    summ = (res.groupby(["target", "method"])
            .agg(miss_rate=("missed_target", "mean"),
                 mean_sens=("heldout_sensitivity", "mean"),
                 sd_sens=("heldout_sensitivity", "std"),
                 mean_referral_rate=("heldout_referral_rate", "mean"),
                 mean_shortfall=("shortfall", "mean"))
            .round(3))
    cost_summ = (cost.groupby(["r", "version"])
                 .agg(ece_hc=("ece_hc", "mean"), referral_rate=("referral_rate", "mean"),
                      referral_sens=("referral_sensitivity", "mean"),
                      cost_per_lesion=("cost_per_lesion", "mean"))
                 .round(3))

    print(f"\n=== Target miss rate on held-out lesions ({args.reps} lesion-grouped resplits) ===")
    print("(a valid 90%-confidence rule should miss about 10% of the time or less)")
    print(summ.to_string())
    print("\n=== RQ3: cost-based threshold 1/(1+r), raw vs calibrated P(high-concern) ===")
    print(cost_summ.to_string())

    # ---------------------------------------------------------- save + plot
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "metrics" / "novel_threshold_runs.csv", index=False)
    summ.to_csv(OUT / "metrics" / "novel_threshold_summary.csv")
    cost.to_csv(OUT / "metrics" / "novel_calibration_runs.csv", index=False)
    cost_summ.to_csv(OUT / "metrics" / "novel_calibration_summary.csv")
    (OUT / "metrics" / "novel_threshold_config.json").write_text(json.dumps(
        {"reps": args.reps, "targets": TARGETS, "delta": DELTA, "C": C_BEST,
         "pool": "train+val only; test locked"}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    tgt = 0.85
    sub = res[res["target"] == tgt]
    data = [sub[sub["method"] == m]["heldout_sensitivity"].to_numpy() for m in METHODS]
    axes[0].boxplot(data)
    axes[0].set_xticklabels(METHODS)
    axes[0].axhline(tgt, color="red", linestyle="--", label=f"target {tgt}")
    axes[0].set_ylabel("held-out referral sensitivity")
    axes[0].set_title("Does the threshold deliver the target?")
    axes[0].legend()
    data = [sub[sub["method"] == m]["heldout_referral_rate"].to_numpy() for m in METHODS]
    axes[1].boxplot(data)
    axes[1].set_xticklabels(METHODS)
    axes[1].set_ylabel("held-out referral rate")
    axes[1].set_title("What it costs in referrals")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "novel_threshold_control.png", dpi=150)
    print("\nSaved tables to outputs/metrics/ and the figure to outputs/plots/. Test set not used.")


if __name__ == "__main__":
    main()