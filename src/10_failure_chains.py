"""
10_failure_chains.py
AI4003 CEP - Part F.i: "one mistake spoils everything" experiments.

Each experiment runs the CORRECT pipeline and a version with ONE deliberate mistake, on the
same train+validation pool (the locked test set is never used), over several resplits.

  A. Split by image instead of by lesion      (images of one lesion leak across the split)
  B. Preprocessing fitted on ALL data          (imputer, encoder and scaler see the held-out rows)
  C. Wrong image-to-metadata match             (metadata rows no longer belong to their image)
  D. Cost-optimal threshold tuned on the evaluation set itself (instead of a separate set)
  Reference: image-only model, to show what the metadata is worth in C.

Run from the project root:
    python src/10_failure_chains.py --reps 5        (about 5 to 8 minutes)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (CACHE, CLASS_TO_IDX, HC_IDX, OUT, SEED, expected_cost,
                    load_splits, sens_at_rate)

_spec = importlib.util.spec_from_file_location("prep02", "src/02_preprocess_features.py")
prep02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep02)

C_BEST = 0.1  # chosen on validation in script 04
COST_R = 10


def new_model():
    return LogisticRegression(C=C_BEST, class_weight="balanced", max_iter=3000, random_state=SEED)


def score(y, proba):
    pred = proba.argmax(1)
    p_hc = proba[:, HC_IDX].sum(1)
    return {
        "macro_f1": f1_score(y, pred, labels=list(range(7)), average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(7))),
        "sens_at_30pct_referral": sens_at_rate(y, p_hc, 0.30),
    }


def fit_eval(E, meta_df, y, tr, ev, preproc_scope="train", use_meta=True):
    """Train on `tr`, return class probabilities for `ev`.
    preproc_scope='all' is the deliberate mistake: preprocessing also sees the `ev` rows."""
    if use_meta:
        pre = prep02.build_preprocessor()
        fit_rows = tr if preproc_scope == "train" else np.concatenate([tr, ev])
        pre.fit(meta_df.iloc[fit_rows])
        X_tr = np.hstack([E[tr], pre.transform(meta_df.iloc[tr])])
        X_ev = np.hstack([E[ev], pre.transform(meta_df.iloc[ev])])
    else:
        X_tr, X_ev = E[tr], E[ev]
    scaler_data = X_tr if preproc_scope == "train" else np.vstack([X_tr, X_ev])
    scaler = StandardScaler().fit(scaler_data)
    model = new_model().fit(scaler.transform(X_tr), y[tr])
    return model.predict_proba(scaler.transform(X_ev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    df = load_splits()
    pool = df[df["split"] != "test"].reset_index(drop=True)  # test set stays locked
    z = np.load(CACHE / "emb_effb0.npz", allow_pickle=True)
    emb = dict(zip(z["ids"], z["emb"]))
    E = np.stack([emb[i] for i in pool["image_id"]])
    y = pool["dx"].map(CLASS_TO_IDX).to_numpy()
    lesion = pool["lesion_id"].to_numpy()
    meta = prep02.clean_meta(pool)
    n = len(pool)
    print(f"Pool: {n} images, {len(np.unique(lesion))} lesions. Test set not used.")

    rows, d_rows, overlaps = [], [], []
    grid = np.linspace(0.02, 0.98, 97)
    t0 = time.time()
    for rep in range(args.reps):
        seed = SEED + rep
        rng = np.random.default_rng(seed)

        # correct split: by lesion, three parts (train / calibration / held-out)
        rest, he = next(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(E, y, lesion))
        tr_rel, ca_rel = next(StratifiedGroupKFold(6, shuffle=True, random_state=seed)
                              .split(E[rest], y[rest], lesion[rest]))
        tr, ca = rest[tr_rel], rest[ca_rel]
        assert not (set(lesion[tr]) & set(lesion[he])), "correct split must not leak"

        # mistake A: split by image
        rest_i, he_i = next(StratifiedKFold(7, shuffle=True, random_state=seed).split(E, y))
        tr_rel_i, _ = next(StratifiedKFold(6, shuffle=True, random_state=seed).split(E[rest_i], y[rest_i]))
        tr_i = rest_i[tr_rel_i]
        overlaps.append(float(np.isin(lesion[he_i], lesion[tr_i]).mean()))

        # correct pipeline (also gives calibration-set scores for experiment D)
        p_both = fit_eval(E, meta, y, tr, np.concatenate([ca, he]))
        p_ca, p_he = p_both[:len(ca)], p_both[len(ca):]

        variants = {
            "correct": (y[he], p_he),
            "A: split by image": (y[he_i], fit_eval(E, meta, y, tr_i, he_i)),
            "B: preprocessing on all data": (y[he], fit_eval(E, meta, y, tr, he, preproc_scope="all")),
            "C: wrong image-metadata match":
                (y[he], fit_eval(E, meta.iloc[rng.permutation(n)].reset_index(drop=True), y, tr, he)),
            "reference: image only": (y[he], fit_eval(E, meta, y, tr, he, use_meta=False)),
        }
        for name, (yy, pp) in variants.items():
            rows.append({"rep": rep, "variant": name, **score(yy, pp)})

        # D: pick the cost-optimal threshold (r = 10) on the evaluation set itself vs a separate set
        phc_ca, phc_he = p_ca[:, HC_IDX].sum(1), p_he[:, HC_IDX].sum(1)

        def best_t(p, yy):
            return grid[int(np.argmin([expected_cost(yy, p >= t, COST_R) for t in grid]))]
        t_honest, t_leak = best_t(phc_ca, y[ca]), best_t(phc_he, y[he])
        d_rows.append({
            "rep": rep,
            "cost_honest": expected_cost(y[he], phc_he >= t_honest, COST_R),
            "cost_reported_when_tuned_on_eval_set": expected_cost(y[he], phc_he >= t_leak, COST_R),
        })
        print(f"  rep {rep + 1}/{args.reps} done ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    order = ["correct", "A: split by image", "B: preprocessing on all data",
             "C: wrong image-metadata match", "reference: image only"]
    metrics = ["macro_f1", "balanced_acc", "macro_auc", "sens_at_30pct_referral"]
    mean = res.groupby("variant")[metrics].mean().loc[order]
    sd = res.groupby("variant")[metrics].std().loc[order]
    delta = mean - mean.loc["correct"]

    print(f"\n=== Mean over {args.reps} resplits (held-out lesions/images) ===")
    print(mean.round(3).to_string())
    print("\n=== Change versus the correct pipeline ===")
    print(delta.round(3).to_string())
    print("\n=== Spread (standard deviation across resplits) ===")
    print(sd.round(3).to_string())
    print(f"\nA: share of held-out images whose lesion also appears in training "
          f"(image-level split): {np.mean(overlaps) * 100:.1f}%  (correct split: 0.0%)")
    d = pd.DataFrame(d_rows)
    print(f"\nD: cost per lesion at r={COST_R}: honest {d['cost_honest'].mean():.3f}  vs  "
          f"reported when tuned on the evaluation set {d['cost_reported_when_tuned_on_eval_set'].mean():.3f}  "
          f"(optimism {(d['cost_honest'] - d['cost_reported_when_tuned_on_eval_set']).mean():.3f})")

    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "metrics" / "failure_chains_runs.csv", index=False)
    mean.round(4).to_csv(OUT / "metrics" / "failure_chains_mean.csv")
    d.to_csv(OUT / "metrics" / "failure_chain_D_runs.csv", index=False)
    (OUT / "metrics" / "failure_chains_summary.json").write_text(json.dumps({
        "reps": args.reps, "mean": mean.round(4).to_dict(), "delta_vs_correct": delta.round(4).to_dict(),
        "image_split_lesion_overlap_pct": round(float(np.mean(overlaps) * 100), 2),
        "D_cost_honest": float(d["cost_honest"].mean()),
        "D_cost_reported": float(d["cost_reported_when_tuned_on_eval_set"].mean()),
        "pool": "train+val only; test locked"}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m, title in zip(axes, ["macro_f1", "macro_auc", "sens_at_30pct_referral"],
                            ["macro-F1", "macro AUC", "referral sensitivity at 30% referral"]):
        ax.bar(range(len(order)), mean[m].to_numpy(), yerr=sd[m].to_numpy(), capsize=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["correct", "A split\\nby image", "B prep\\non all", "C wrong\\nmetadata", "image\\nonly"],
                           fontsize=8)
        ax.set_title(title)
        lo = float(mean[m].min() - 0.05)
        ax.set_ylim(max(0.0, lo), min(1.0, float(mean[m].max() + 0.05)))
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "failure_chains.png", dpi=150)
    print("\nSaved tables to outputs/metrics/ and the figure to outputs/plots/. Test set not used.")


if __name__ == "__main__":
    main()