"""
08_compare_designs.py
AI4003 CEP - Part C.ii / E: compare Design A and the Design B variants on VALIDATION.

Reads the saved validation logits of every Design B run (outputs/metrics/design_b_*_val_logits.npz)
and Design A's validation probabilities. Builds seed ensembles (average of probabilities),
reports seed-to-seed spread, and gives PAIRED lesion-level confidence intervals for:
    1. Design B (metadata on) vs Design A
    2. Design B metadata on vs image only (same seeds)
No models are trained. The TEST set is not used.

Run from the project root:  python src/08_compare_designs.py
"""
import json
import re

import numpy as np
import pandas as pd
from scipy.special import softmax

from common import (OUT, SEED, full_report, lesion_bootstrap, load_splits,
                    macro_f1_fn, p_high_concern, sens_at_rate)


def sens30_fn(y, p):
    return sens_at_rate(y, p_high_concern(p), 0.30)


def paired_ci(y, pa, pb, groups, fn, n_boot=1000, seed=SEED):
    """Difference fn(A) - fn(B), with a 95% interval from resampling whole lesions."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == g)[0] for g in range(len(uniq))]
    diffs = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[g] for g in chosen])
        diffs.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(fn(y, pa) - fn(y, pb)), float(lo), float(hi)


def row(name, y, p):
    r = full_report(y, p)
    return {"candidate": name, "macro_f1": r["macro_f1"], "balanced_acc": r["balanced_accuracy"],
            "macro_auc": r["macro_auc"], "sens@30%": r["sens_at_30pct_referral"],
            "cost_r5": r["cost_r5"], "cost_r10": r["cost_r10"]}


def main():
    df = load_splits()
    runs = {}
    for f in sorted((OUT / "metrics").glob("design_b_meta*_seed*_val_logits.npz")):
        m = re.match(r"design_b_meta(On|Off)_seed(\d+)_val_logits\.npz", f.name)
        z = np.load(f, allow_pickle=True)
        runs[(m.group(1), int(m.group(2)))] = (softmax(z["logits"], axis=1), z["y_val"], z["ids_val"])
    assert runs, "no Design B logits found in outputs/metrics"
    first = next(iter(runs.values()))
    y, ids = first[1], first[2]
    for k, (_, yy, ii) in runs.items():
        assert (yy == y).all() and (ii == ids).all(), f"validation rows differ for run {k}"
    groups = df.set_index("image_id")["lesion_id"].loc[ids].to_numpy()

    za = np.load(OUT / "metrics" / "design_a_val_probs.npz", allow_pickle=True)
    assert (za["ids_val"] == ids).all(), "Design A validation rows differ"
    pA = za["A_embeddings_plus_metadata"]

    on = {s: p for (t, s), (p, _, _) in runs.items() if t == "On"}
    off = {s: p for (t, s), (p, _, _) in runs.items() if t == "Off"}
    rows = [row("A: frozen embeddings + metadata (LR)", y, pA)]
    for s, p in sorted(on.items()):
        rows.append(row(f"B metadata-on, seed {s}", y, p))
    for s, p in sorted(off.items()):
        rows.append(row(f"B image-only, seed {s}", y, p))
    ens_on = np.mean([on[s] for s in sorted(on)], axis=0) if on else None
    ens_off = np.mean([off[s] for s in sorted(off)], axis=0) if off else None
    if len(on) > 1:
        rows.append(row(f"B metadata-on, ENSEMBLE of {len(on)} seeds", y, ens_on))
    if len(off) > 1:
        rows.append(row(f"B image-only, ENSEMBLE of {len(off)} seeds", y, ens_off))
    table = pd.DataFrame(rows).set_index("candidate")
    print("=== Validation results (no deferral; referral = predicted high-concern class) ===")
    print(table.round(3).to_string())

    # seed-to-seed spread
    print("\n=== Seed-to-seed spread (mean and standard deviation over seeds) ===")
    for label, group in (("metadata-on", on), ("image-only", off)):
        if len(group) > 1:
            reps = [full_report(y, p) for p in group.values()]
            for key in ("macro_f1", "sens_at_30pct_referral", "cost_r5"):
                v = np.array([r[key] for r in reps])
                print(f"  {label:<12} {key:<24} {v.mean():.3f} +/- {v.std(ddof=1):.3f}  (n={len(v)})")

    # lesion-level confidence intervals for the ensemble (or the single model)
    best_on = ens_on if len(on) > 1 else (next(iter(on.values())) if on else None)
    summary = {}
    if best_on is not None:
        lo, hi = lesion_bootstrap(y, best_on, groups, macro_f1_fn)
        slo, shi = lesion_bootstrap(y, best_on, groups, sens30_fn)
        print(f"\nB metadata-on (ensemble if available): macro-F1 95% CI [{lo:.3f}, {hi:.3f}]  "
              f"sens@30% 95% CI [{slo:.3f}, {shi:.3f}]")

    # paired comparisons
    print("\n=== Paired differences (lesion-level bootstrap, same validation lesions) ===")
    print("(a 95% interval that excludes 0 means the difference is not just sampling noise)")
    pairs = []
    if best_on is not None:
        pairs.append(("B metadata-on minus A", best_on, pA))
    common = sorted(set(on) & set(off))
    if common:
        e_on = np.mean([on[s] for s in common], axis=0)
        e_off = np.mean([off[s] for s in common], axis=0)
        pairs.append((f"B metadata-on minus B image-only (seeds {common})", e_on, e_off))
    for name, pa, pb in pairs:
        for mname, fn in (("macro-F1", macro_f1_fn), ("sens@30% referral", sens30_fn)):
            d, lo, hi = paired_ci(y, pa, pb, groups, fn)
            summary[f"{name} | {mname}"] = {"diff": d, "ci95": [lo, hi]}
            flag = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"  {name}\n      {mname:<20} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  ({flag})")

    table.round(4).to_csv(OUT / "metrics" / "compare_designs_val.csv")
    (OUT / "metrics" / "compare_designs_val.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved to outputs/metrics/. The test set was not used.")


if __name__ == "__main__":
    main()
