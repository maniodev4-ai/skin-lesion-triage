"""
06b_novel_stats.py
Part D.ii: confidence intervals and paired comparisons for the threshold experiment.
Reads outputs/metrics/novel_threshold_runs.csv (written by script 06). No models are trained.

Run from the project root:  python src/06b_novel_stats.py

Caveat to state in the report: the repetitions re-split the SAME pool of lesions, so
they are not independent. These intervals are therefore optimistic.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import OUT, SEED

runs = pd.read_csv(OUT / "metrics" / "novel_threshold_runs.csv")
rng = np.random.default_rng(SEED)


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


miss_rows, pair_rows = [], []
for target, g in runs.groupby("target"):
    miss = g.pivot(index="rep", columns="method", values="missed_target").astype(bool)
    ref = g.pivot(index="rep", columns="method", values="heldout_referral_rate")
    n = len(miss)

    for m in ["naive", "img_lcb", "lesion_lcb"]:
        k = int(miss[m].sum())
        lo, hi = wilson(k, n)
        miss_rows.append({"target": target, "method": m, "misses": k, "reps": n,
                          "miss_rate": round(k / n, 3),
                          "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)})

    for a, b in [("naive", "lesion_lcb"), ("naive", "img_lcb"), ("img_lcb", "lesion_lcb")]:
        a_only = int((miss[a] & ~miss[b]).sum())  # a missed, b did not
        b_only = int((~miss[a] & miss[b]).sum())  # b missed, a did not
        p = binomtest(a_only, a_only + b_only, 0.5).pvalue if (a_only + b_only) > 0 else 1.0
        extra = (ref[b] - ref[a]).to_numpy()  # extra referral rate paid by b relative to a
        boots = [rng.choice(extra, len(extra), replace=True).mean() for _ in range(5000)]
        pair_rows.append({
            "target": target, "comparison": f"{a} vs {b}",
            f"only_{a}_missed": a_only, f"only_{b}_missed": b_only,
            "sign_test_p": round(float(p), 4),
            "extra_referral_rate_pts": round(float(extra.mean() * 100), 2),
            "extra_referral_ci95": (round(float(np.percentile(boots, 2.5) * 100), 2),
                                    round(float(np.percentile(boots, 97.5) * 100), 2)),
        })

miss_df = pd.DataFrame(miss_rows)
pair_df = pd.DataFrame(pair_rows)
print("=== Miss rate with 95% Wilson intervals (30 repetitions, optimistic: shared pool) ===")
print(miss_df.to_string(index=False))
print("\n=== Paired comparisons on the same repetitions ===")
print("(sign test on discordant repetitions; extra referral = second method minus first)")
print(pair_df.to_string(index=False))

miss_df.to_csv(OUT / "metrics" / "novel_miss_rate_ci.csv", index=False)
pair_df.to_csv(OUT / "metrics" / "novel_paired_tests.csv", index=False)
print("\nSaved to outputs/metrics/.")
