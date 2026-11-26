"""
07_calibration_deferral.py
AI4003 CEP - Part E.i / E.ii (and the model files the app needs).

Takes the Design A candidate (frozen EfficientNet-B0 embeddings + metadata -> logistic
regression) and builds the complete decision system, using VALIDATION data only:

  1. Temperature scaling of the 7-class probabilities (cross-fitted within validation,
     so the reported ECE is not measured on the data the temperature was fitted on).
  2. Locked rule: referral CAP of 30% (R1: the cap is the hard limit). The cap is split
     between referral by P(high-concern) and deferral of the least-confident lesions.
     The risk-controlled threshold from script 06 is kept as a documented "safety mode".
  3. Deferral budget chosen on validation: the largest budget that costs at most 1 point
     of referral sensitivity.
  4. Plots: sensitivity vs referral rate, reliability diagram, risk-coverage curve,
     expected cost vs the unknown cost ratio r.
  5. Saves models/final_model.joblib, models/effb0_backbone.pt, models/final_config.json.

The TEST set is not used.

Run from the project root:  python src/07_calibration_deferral.py
"""
import importlib.util
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models

from common import (CACHE, CLASSES, HC_IDX, MODELS, OUT, SEED, expected_cost,
                    is_hc, load_splits, referral_stats)

# reuse the threshold rule and helpers from script 06
_spec = importlib.util.spec_from_file_location("nov06", "src/06_novel_contribution.py")
nov06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nov06)

C_BEST = 0.1            # chosen on validation in script 04
TARGET_SENS = 0.85      # R1 target stated in advance
DELTA = 0.10            # confidence level of the risk-controlled threshold
CAP = 0.30              # R1: hard limit on the share of lesions sent to doctors
DEFER_BUDGETS = [0.0, 0.025, 0.05, 0.075, 0.10]  # shares of ALL lesions that may be deferred
MAX_SENS_LOSS = 0.01    # deferral may cost at most 1 point of referral sensitivity
R_GRID = [1, 2, 5, 10, 20, 50]

(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- metrics
def ece_top(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def fit_temperature(logits, y):
    res = minimize_scalar(lambda T: nll(softmax(logits / T, axis=1), y),
                          bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


# ------------------------------------------------------------------- main
def main():
    df = load_splits()
    z = np.load(OUT / "meta_features.npz", allow_pickle=True)
    ids_tr, ids_va = z["ids_train"], z["ids_val"]
    y_tr, y_va = z["y_train"], z["y_val"]
    emb_z = np.load(CACHE / "emb_effb0.npz", allow_pickle=True)
    emb = dict(zip(emb_z["ids"], emb_z["emb"]))
    X_tr = np.hstack([np.stack([emb[i] for i in ids_tr]), z["F_train"]])
    X_va = np.hstack([np.stack([emb[i] for i in ids_va]), z["F_val"]])
    lesion_va = df.set_index("image_id")["lesion_id"].loc[ids_va].to_numpy()
    hc_va = is_hc(y_va)

    # ---- final model: fitted on TRAIN only
    t0 = time.time()
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(C=C_BEST, class_weight="balanced",
                                                 max_iter=3000, random_state=SEED))])
    model.fit(X_tr, y_tr)
    print(f"Final model fitted on train ({time.time() - t0:.0f}s).")
    logits = model.decision_function(X_va)
    probs_raw = softmax(logits, axis=1)
    assert np.allclose(probs_raw, model.predict_proba(X_va), atol=1e-6)

    # ---- 1. calibration, cross-fitted within validation (grouped by lesion)
    skf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
    probs_oof = np.zeros_like(probs_raw)
    for fit_idx, eval_idx in skf.split(logits, y_va, lesion_va):
        T_fold = fit_temperature(logits[fit_idx], y_va[fit_idx])
        probs_oof[eval_idx] = softmax(logits[eval_idx] / T_fold, axis=1)
    T = fit_temperature(logits, y_va)            # final temperature, all validation
    probs = softmax(logits / T, axis=1)
    calib = {
        "temperature": T,
        "ece_raw": ece_top(probs_raw, y_va), "ece_calibrated_cross_fitted": ece_top(probs_oof, y_va),
        "nll_raw": nll(probs_raw, y_va), "nll_calibrated_cross_fitted": nll(probs_oof, y_va),
        "brier_raw": brier(probs_raw, y_va), "brier_calibrated_cross_fitted": brier(probs_oof, y_va),
    }
    print("\n=== Calibration (7-class, top-label) ===")
    for k, v in calib.items():
        print(f"  {k:<32} {v:.4f}")

    # ---- 2. reference operating points (safety mode and simple rules)
    p_hc = probs[:, HC_IDX].sum(axis=1)
    pred = probs.argmax(1)
    conf = probs.max(1)
    n = len(y_va)
    t_naive = nov06.choose_threshold(p_hc, hc_va, lesion_va, TARGET_SENS, "naive")
    t_safety = nov06.choose_threshold(p_hc, hc_va, lesion_va, TARGET_SENS, "lesion_lcb")
    ops = {}
    for name, referred in {
        "argmax_rule": np.isin(pred, HC_IDX),
        "naive_threshold_target085": p_hc >= t_naive,
        "safety_mode_risk_controlled_target085": p_hc >= t_safety,
    }.items():
        rate, sens = referral_stats(y_va, referred)
        ops[name] = {"referral_rate": rate, "referral_sensitivity": sens,
                     **{f"cost_r{r}": expected_cost(y_va, referred, r) for r in (2, 5, 10, 20)}}
    print(f"\n=== Reference rules on validation (in-sample) ===")
    for k, v in ops.items():
        print(f"  {k:<40} rate={v['referral_rate']:.3f}  sens={v['referral_sensitivity']:.3f}")
    print(f"  thresholds: naive={t_naive:.4f}  safety_mode={t_safety:.4f}")

    # ---- 3. LOCKED RULE: 30% cap shared between referral and deferral
    order = np.argsort(-p_hc)

    def build_rule(d):
        k = int(round((CAP - d) * n))
        t = float(p_hc[order[k - 1]])
        refer = p_hc >= t
        pool = ~refer
        m = int(round(d * n))
        tau = float(np.sort(conf[pool])[m]) if m > 0 else 0.0
        defer = pool & (conf < tau)
        auto = pool & ~defer
        return t, tau, refer, defer, auto

    rows, sens_by_d = [], {}
    for d in DEFER_BUDGETS:
        t, tau, refer, defer, auto = build_rule(d)
        allref = refer | defer
        wrong_pool = int((pred[~refer] != y_va[~refer]).sum())
        wrong_def = int((pred[defer] != y_va[defer]).sum())
        sens = float(allref[hc_va].mean())
        sens_by_d[d] = sens
        rows.append({
            "defer_budget": d, "referral_threshold": round(t, 4), "tau": round(tau, 4),
            "referred": int(refer.sum()), "deferred": int(defer.sum()),
            "total_referral_rate": round(float(allref.mean()), 4),
            "referral_sensitivity": round(sens, 4),
            "hc_caught_by_deferral": int((defer & hc_va).sum()),
            "errors_in_pool": wrong_pool, "errors_avoided_by_deferral": wrong_def,
            "kept_error_rate": round(float((pred[auto] != y_va[auto]).mean()), 4),
        })
    print(f"\n=== Cap {CAP:.0%} shared between referral and deferral (validation) ===")
    for r_ in rows:
        print("  ", r_)
    base = sens_by_d[0.0]
    d_star = max(d for d in DEFER_BUDGETS if sens_by_d[d] >= base - MAX_SENS_LOSS)
    t_ref, best_tau, refer, defer, auto = build_rule(d_star)
    all_ref = refer | defer
    defer_stats = next(r_ for r_ in rows if r_["defer_budget"] == d_star)
    defer_stats = {**defer_stats, "chosen_rule":
                   f"largest deferral budget with sensitivity loss <= {MAX_SENS_LOSS} vs no deferral",
                   "sensitivity_without_deferral": round(base, 4)}
    ops["locked_rule_cap30_with_deferral"] = {
        "referral_rate": float(all_ref.mean()),
        "referral_sensitivity": float(all_ref[hc_va].mean()),
        **{f"cost_r{r}": expected_cost(y_va, all_ref, r) for r in (2, 5, 10, 20)}}
    print(f"\n  CHOSEN deferral budget: {d_star:.1%}  (referral threshold {t_ref:.4f}, tau {best_tau:.4f})")
    print("  Locked rule on validation (in-sample): "
          f"referral rate {all_ref.mean():.3f}, sensitivity {all_ref[hc_va].mean():.3f}")
    print("  Target 0.85 within the cap is not reached; report the best value and why.")

    # ---- 4. plots
    # (a) sensitivity vs referral rate
    srt = np.argsort(-p_hc)
    cum_hc = np.cumsum(hc_va[srt]) / hc_va.sum()
    rate_axis = np.arange(1, len(y_va) + 1) / len(y_va)
    plt.figure(figsize=(5.5, 4.2))
    plt.plot(rate_axis, cum_hc, label="rank by P(high-concern)")
    plt.plot([0, 1], [0, 1], "k:", label="random")
    for name, mk in (("argmax_rule", "s"), ("naive_threshold_target085", "o"),
                     ("safety_mode_risk_controlled_target085", "^"),
                     ("locked_rule_cap30_with_deferral", "*")):
        plt.scatter(ops[name]["referral_rate"], ops[name]["referral_sensitivity"],
                    marker=mk, s=70 if mk == "*" else 30, zorder=3, label=name)
    plt.axvline(0.30, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(TARGET_SENS, color="red", linestyle="--", linewidth=0.8)
    plt.xlabel("referral rate"); plt.ylabel("referral sensitivity (high-concern)")
    plt.title("Referral sensitivity vs referral rate (validation)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUT / "plots" / "referral_curve_val.png", dpi=150); plt.close()

    # (b) reliability diagram, raw vs cross-fitted calibrated
    plt.figure(figsize=(5.2, 4.4))
    edges = np.linspace(0, 1, 11)
    for name, pr in (("raw", probs_raw), ("temperature-scaled (cross-fitted)", probs_oof)):
        c, ok = pr.max(1), (pr.argmax(1) == y_va).astype(float)
        idx = np.clip(np.digitize(c, edges[1:-1]), 0, 9)
        xs = [c[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ys = [ok[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        plt.plot(xs, ys, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k:")
    plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("Reliability diagram (validation)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT / "plots" / "reliability_val.png", dpi=150); plt.close()

    # (c) risk-coverage curve
    def risk_cov(mask):
        c, wrong = conf[mask], (pred[mask] != y_va[mask]).astype(float)
        o = np.argsort(-c)
        cov = np.arange(1, len(c) + 1) / len(c)
        return cov, np.cumsum(wrong[o]) / np.arange(1, len(c) + 1)
    cov_all, risk_all = risk_cov(np.ones(len(y_va), bool))
    cov_pool, risk_pool = risk_cov(~refer)
    aurc = {"all_lesions": float(risk_all.mean()), "not_referred_pool": float(risk_pool.mean())}
    plt.figure(figsize=(5.5, 4.2))
    plt.plot(cov_all, risk_all, label=f"all lesions (AURC {aurc['all_lesions']:.3f})")
    plt.plot(cov_pool, risk_pool, label=f"not-referred pool (AURC {aurc['not_referred_pool']:.3f})")
    plt.xlabel("coverage (share kept)"); plt.ylabel("7-class error rate on kept")
    plt.title("Risk-coverage curve (validation)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT / "plots" / "risk_coverage_val.png", dpi=150); plt.close()

    # (d) expected cost vs the unknown r
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(nov06.logit(p_hc).reshape(-1, 1), hc_va.astype(int))
    p_hc_cal = platt.predict_proba(nov06.logit(p_hc).reshape(-1, 1))[:, 1]
    curves = {"argmax rule": [], "naive threshold (target 0.85)": [],
              "safety mode (risk-controlled 0.85)": [], "LOCKED rule (cap 30% + deferral)": [],
              "cost-optimal 1/(1+r), calibrated": []}
    for r in R_GRID:
        curves["argmax rule"].append(expected_cost(y_va, np.isin(pred, HC_IDX), r))
        curves["naive threshold (target 0.85)"].append(expected_cost(y_va, p_hc >= t_naive, r))
        curves["safety mode (risk-controlled 0.85)"].append(expected_cost(y_va, p_hc >= t_safety, r))
        curves["LOCKED rule (cap 30% + deferral)"].append(expected_cost(y_va, all_ref, r))
        curves["cost-optimal 1/(1+r), calibrated"].append(
            expected_cost(y_va, p_hc_cal >= 1.0 / (1.0 + r), r))
    plt.figure(figsize=(5.5, 4.2))
    for name, vals in curves.items():
        plt.plot(R_GRID, vals, marker="o", label=name)
    plt.xscale("log"); plt.xlabel("cost ratio r (missed high-concern : unnecessary referral)")
    plt.ylabel("expected cost per lesion"); plt.title("Cost vs unknown r (validation)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUT / "plots" / "cost_vs_r_val.png", dpi=150); plt.close()

    # ---- 5. save the model files and the locked config
    joblib.dump(model, MODELS / "final_model.joblib")
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    backbone.classifier = torch.nn.Identity()
    torch.save(backbone.state_dict(), MODELS / "effb0_backbone.pt")
    config = {
        "design": "A: frozen EfficientNet-B0 embeddings + metadata -> logistic regression",
        "class_order": CLASSES, "high_concern": [CLASSES[i] for i in HC_IDX],
        "input_size": 224, "imagenet_mean_std": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        "metadata_preprocessor": "models/meta_preprocessor.joblib",
        "logreg_C": C_BEST, "temperature": T,
        "operating_mode": "capacity-limited (R1: at most 30% of lesions sent to doctors)",
        "referral_cap": CAP, "defer_budget": d_star,
        "referral_threshold_p_hc": t_ref, "defer_confidence_tau": best_tau,
        "decision": "REFER if P(HC)>=referral_threshold; else DEFER if max prob < tau; else NOT REFERRED",
        "safety_mode": {"threshold_p_hc": t_safety, "target_sensitivity": TARGET_SENS, "delta": DELTA,
                        "rule": "lesion-level Clopper-Pearson lower bound (script 06)",
                        "note": "meets the target with confidence but needs about 44% referral"},
        "seed": SEED, "fitted_on": "train only; calibration and thresholds on validation only",
    }
    (MODELS / "final_config.json").write_text(json.dumps(config, indent=2))
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in MODELS.glob("*") if p.is_file()}

    summary = {"calibration": calib, "operating_points_val": ops, "deferral": defer_stats,
               "deferral_budget_table": rows, "aurc": aurc, "cost_vs_r_val": {k: dict(zip(map(str, R_GRID), v)) for k, v in curves.items()},
               "model_file_sizes_mb": sizes}
    (OUT / "metrics" / "final_val_decisions.json").write_text(json.dumps(summary, indent=2))
    print("\nModel files (MB):", sizes)
    print("Saved: models/final_model.joblib, effb0_backbone.pt, final_config.json;")
    print("       outputs/plots/*.png and outputs/metrics/final_val_decisions.json. Test set not used.")


if __name__ == "__main__":
    main()