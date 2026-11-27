"""
09_lock_final_model.py
AI4003 CEP - Part E.i / E.ii for the FINAL model: the Design B seed ensemble.

Uses VALIDATION data only (saved validation logits of the three seeds):
  1. Ensemble = mean of the seeds' logits. Temperature scaling, cross-fitted within validation
     (grouped by lesion) so the reported ECE is not measured on the data it was fitted on.
  2. Locked rule: referral CAP of 30% (R1), shared between referral by P(high-concern) and
     deferral of the least-confident lesions. Deferral budget = the largest one that costs at
     most 1 point of referral sensitivity.
  3. "Safety mode": risk-controlled threshold (script 06) for a 0.85 sensitivity target.
  4. Plots and the model files / config the app needs.
The TEST set is not used.

Run from the project root:
    python src/09_lock_final_model.py                 (image-only ensemble, the chosen model)
    python src/09_lock_final_model.py --variant On    (metadata-on ensemble, for comparison)
"""
import argparse
import importlib.util
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from common import (CLASSES, HC_IDX, MODELS, OUT, SEED, expected_cost, is_hc,
                    load_splits, referral_stats)

_spec = importlib.util.spec_from_file_location("nov06", "src/06_novel_contribution.py")
nov06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nov06)

CAP = 0.30
DEFER_BUDGETS = [0.0, 0.025, 0.05, 0.075, 0.10]
MAX_SENS_LOSS = 0.01
TARGET_SENS = 0.85
DELTA = 0.10
R_GRID = [1, 2, 5, 10, 20, 50]


# ---------------------------------------------------------------- metrics
def ece_top(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(m.mean() * abs(correct[m].mean() - conf[m].mean())
                     for m in (idx == b for b in range(bins)) if m.any()))


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(probs, y):
    return float(((probs - np.eye(probs.shape[1])[y]) ** 2).sum(1).mean())


def fit_temperature(logits, y):
    res = minimize_scalar(lambda T: nll(softmax(logits / T, axis=1), y),
                          bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["Off", "On"], default="Off")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    tag = f"meta{args.variant}"

    df = load_splits()
    logit_list, y_va, ids_va = [], None, None
    for s in args.seeds:
        z = np.load(OUT / "metrics" / f"design_b_{tag}_seed{s}_val_logits.npz", allow_pickle=True)
        if y_va is None:
            y_va, ids_va = z["y_val"], z["ids_val"]
        assert (z["y_val"] == y_va).all() and (z["ids_val"] == ids_va).all()
        logit_list.append(z["logits"])
    logits = np.mean(logit_list, axis=0)  # ensemble = mean of logits
    n = len(y_va)
    lesion_va = df.set_index("image_id")["lesion_id"].loc[ids_va].to_numpy()
    hc_va = is_hc(y_va)
    probs_raw = softmax(logits, axis=1)
    print(f"Ensemble of {len(args.seeds)} seeds ({tag}), {n} validation images.")

    # ---- 1. calibration (cross-fitted within validation)
    probs_oof = np.zeros_like(probs_raw)
    for fit_idx, eval_idx in StratifiedGroupKFold(2, shuffle=True, random_state=SEED).split(logits, y_va, lesion_va):
        T_fold = fit_temperature(logits[fit_idx], y_va[fit_idx])
        probs_oof[eval_idx] = softmax(logits[eval_idx] / T_fold, axis=1)
    T = fit_temperature(logits, y_va)
    probs = softmax(logits / T, axis=1)
    calib = {"temperature": T,
             "ece_raw": ece_top(probs_raw, y_va), "ece_calibrated_cross_fitted": ece_top(probs_oof, y_va),
             "nll_raw": nll(probs_raw, y_va), "nll_calibrated_cross_fitted": nll(probs_oof, y_va),
             "brier_raw": brier(probs_raw, y_va), "brier_calibrated_cross_fitted": brier(probs_oof, y_va)}
    print("\n=== Calibration (7-class, top-label) ===")
    for k, v in calib.items():
        print(f"  {k:<32} {v:.4f}")

    # ---- 2. reference rules
    p_hc = probs[:, HC_IDX].sum(axis=1)
    pred, conf = probs.argmax(1), probs.max(1)
    t_naive = nov06.choose_threshold(p_hc, hc_va, lesion_va, TARGET_SENS, "naive")
    t_safety = nov06.choose_threshold(p_hc, hc_va, lesion_va, TARGET_SENS, "lesion_lcb")
    ops = {}
    for name, referred in {"argmax_rule": np.isin(pred, HC_IDX),
                           "naive_threshold_target085": p_hc >= t_naive,
                           "safety_mode_risk_controlled_target085": p_hc >= t_safety}.items():
        rate, sens = referral_stats(y_va, referred)
        ops[name] = {"referral_rate": rate, "referral_sensitivity": sens,
                     **{f"cost_r{r}": expected_cost(y_va, referred, r) for r in (2, 5, 10, 20)}}
    print("\n=== Reference rules on validation (in-sample) ===")
    for k, v in ops.items():
        print(f"  {k:<40} rate={v['referral_rate']:.3f}  sens={v['referral_sensitivity']:.3f}")
    print(f"  thresholds: naive={t_naive:.4f}  safety_mode={t_safety:.4f}")

    # ---- 3. LOCKED RULE: cap shared between referral and deferral
    order = np.argsort(-p_hc)

    def build_rule(d):
        k = int(round((CAP - d) * n))
        t = float(p_hc[order[k - 1]])
        refer = p_hc >= t
        pool = ~refer
        m = int(round(d * n))
        tau = float(np.sort(conf[pool])[m]) if m > 0 else 0.0
        defer = pool & (conf < tau)
        return t, tau, refer, defer, pool & ~defer

    rows, sens_by_d = [], {}
    for d in DEFER_BUDGETS:
        t, tau, refer, defer, auto = build_rule(d)
        allref = refer | defer
        sens = float(allref[hc_va].mean())
        sens_by_d[d] = sens
        rows.append({
            "defer_budget": d, "referral_threshold": round(t, 4), "tau": round(tau, 4),
            "referred": int(refer.sum()), "deferred": int(defer.sum()),
            "total_referral_rate": round(float(allref.mean()), 4),
            "referral_sensitivity": round(sens, 4),
            "hc_caught_by_deferral": int((defer & hc_va).sum()),
            "errors_in_pool": int((pred[~refer] != y_va[~refer]).sum()),
            "errors_avoided_by_deferral": int((pred[defer] != y_va[defer]).sum()),
            "kept_error_rate": round(float((pred[auto] != y_va[auto]).mean()), 4)})
    print(f"\n=== Cap {CAP:.0%} shared between referral and deferral (validation) ===")
    for r_ in rows:
        print("  ", r_)
    base = sens_by_d[0.0]
    d_star = max(d for d in DEFER_BUDGETS if sens_by_d[d] >= base - MAX_SENS_LOSS)
    t_ref, best_tau, refer, defer, auto = build_rule(d_star)
    all_ref = refer | defer
    defer_stats = {**next(r_ for r_ in rows if r_["defer_budget"] == d_star),
                   "chosen_rule": f"largest deferral budget with sensitivity loss <= {MAX_SENS_LOSS}",
                   "sensitivity_without_deferral": round(base, 4)}
    ops["locked_rule_cap30_with_deferral"] = {
        "referral_rate": float(all_ref.mean()), "referral_sensitivity": float(all_ref[hc_va].mean()),
        **{f"cost_r{r}": expected_cost(y_va, all_ref, r) for r in (2, 5, 10, 20)}}
    print(f"\n  CHOSEN deferral budget: {d_star:.1%}  (referral threshold {t_ref:.4f}, tau {best_tau:.4f})")
    print(f"  Locked rule on validation (in-sample): referral rate {all_ref.mean():.3f}, "
          f"sensitivity {all_ref[hc_va].mean():.3f}  (target {TARGET_SENS})")

    # ---- 4. plots
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    srt = np.argsort(-p_hc)
    cum_hc = np.cumsum(hc_va[srt]) / hc_va.sum()
    plt.figure(figsize=(5.6, 4.3))
    plt.plot(np.arange(1, n + 1) / n, cum_hc, label="rank by P(high-concern)")
    plt.plot([0, 1], [0, 1], "k:", label="random")
    for name, mk in (("argmax_rule", "s"), ("naive_threshold_target085", "o"),
                     ("safety_mode_risk_controlled_target085", "^"), ("locked_rule_cap30_with_deferral", "*")):
        plt.scatter(ops[name]["referral_rate"], ops[name]["referral_sensitivity"], marker=mk,
                    s=70 if mk == "*" else 30, zorder=3, label=name)
    plt.axvline(CAP, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(TARGET_SENS, color="red", linestyle="--", linewidth=0.8)
    plt.xlabel("referral rate"); plt.ylabel("referral sensitivity (high-concern)")
    plt.title("Referral sensitivity vs referral rate (validation)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(plots / "final_referral_curve_val.png", dpi=150); plt.close()

    plt.figure(figsize=(5.2, 4.4))
    edges = np.linspace(0, 1, 11)
    for name, pr in (("raw (ensemble)", probs_raw), ("temperature-scaled (cross-fitted)", probs_oof)):
        c, ok = pr.max(1), (pr.argmax(1) == y_va).astype(float)
        idx = np.clip(np.digitize(c, edges[1:-1]), 0, 9)
        keep = [b for b in range(10) if (idx == b).sum() >= 10]
        plt.plot([c[idx == b].mean() for b in keep], [ok[idx == b].mean() for b in keep], marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k:")
    plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("Reliability diagram (validation)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(plots / "final_reliability_val.png", dpi=150); plt.close()

    def risk_cov(mask):
        c, wrong = conf[mask], (pred[mask] != y_va[mask]).astype(float)
        o = np.argsort(-c)
        return np.arange(1, len(c) + 1) / len(c), np.cumsum(wrong[o]) / np.arange(1, len(c) + 1)
    cov_all, risk_all = risk_cov(np.ones(n, bool))
    cov_pool, risk_pool = risk_cov(~refer)
    aurc = {"all_lesions": float(risk_all.mean()), "not_referred_pool": float(risk_pool.mean())}
    plt.figure(figsize=(5.6, 4.3))
    plt.plot(cov_all, risk_all, label=f"all lesions (AURC {aurc['all_lesions']:.3f})")
    plt.plot(cov_pool, risk_pool, label=f"not-referred pool (AURC {aurc['not_referred_pool']:.3f})")
    plt.xlabel("coverage (share kept)"); plt.ylabel("7-class error rate on kept")
    plt.title("Risk-coverage curve (validation)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(plots / "final_risk_coverage_val.png", dpi=150); plt.close()

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
        curves["cost-optimal 1/(1+r), calibrated"].append(expected_cost(y_va, p_hc_cal >= 1 / (1 + r), r))
    plt.figure(figsize=(5.6, 4.3))
    for name, vals in curves.items():
        plt.plot(R_GRID, vals, marker="o", label=name)
    plt.xscale("log"); plt.xlabel("cost ratio r (missed high-concern : unnecessary referral)")
    plt.ylabel("expected cost per lesion"); plt.title("Cost vs unknown r (validation)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(plots / "final_cost_vs_r_val.png", dpi=150); plt.close()

    # ---- 5. model files and locked config
    old_cfg = MODELS / "final_config.json"
    if old_cfg.exists() and json.loads(old_cfg.read_text()).get("design", "").startswith("A"):
        shutil.copy(old_cfg, MODELS / "design_a_candidate_config.json")
    old_sum = OUT / "metrics" / "final_val_decisions.json"
    if old_sum.exists():
        shutil.copy(old_sum, OUT / "metrics" / "design_a_candidate_decisions.json")
    members = []
    for s in args.seeds:
        dst = MODELS / f"final_seed{s}.pt"
        shutil.copy(MODELS / f"design_b_{tag}_seed{s}.pt", dst)
        members.append(f"models/{dst.name}")
    config = {
        "design": f"B: fine-tuned EfficientNet-B0 ensemble ({len(args.seeds)} seeds), "
                  f"{'with' if args.variant == 'On' else 'without'} metadata",
        "variant": tag, "members": members, "uses_metadata": args.variant == "On",
        "architecture": "FusionNet in src/05_design_b_finetune.py (image-only variant feeds a zero metadata vector)",
        "n_meta_features": 20, "metadata_preprocessor": "models/meta_preprocessor.joblib",
        "ensemble": "mean of member logits, divided by temperature, then softmax",
        "class_order": CLASSES, "high_concern": [CLASSES[i] for i in HC_IDX],
        "input_size": 224, "imagenet_mean_std": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        "temperature": T,
        "operating_mode": "capacity-limited (R1: at most 30% of lesions sent to doctors)",
        "referral_cap": CAP, "defer_budget": d_star,
        "referral_threshold_p_hc": t_ref, "defer_confidence_tau": best_tau,
        "decision": "REFER if P(HC)>=referral_threshold; else DEFER if max prob < tau; else NOT REFERRED",
        "safety_mode": {"threshold_p_hc": t_safety, "target_sensitivity": TARGET_SENS, "delta": DELTA,
                        "rule": "lesion-level Clopper-Pearson lower bound (script 06)"},
        "selection_note": "variant chosen after seeing validation results; metadata gave no detectable gain",
        "seed": SEED, "fitted_on": "train only; temperature, thresholds and deferral on validation only"}
    (MODELS / "final_config.json").write_text(json.dumps(config, indent=2))
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in MODELS.glob("final_*") if p.is_file()}
    summary = {"variant": tag, "seeds": args.seeds, "calibration": calib, "operating_points_val": ops,
               "deferral": defer_stats, "deferral_budget_table": rows, "aurc": aurc,
               "cost_vs_r_val": {k: dict(zip(map(str, R_GRID), v)) for k, v in curves.items()},
               "model_file_sizes_mb": sizes}
    (OUT / "metrics" / "final_val_decisions.json").write_text(json.dumps(summary, indent=2))
    print("\nFinal model files (MB):", sizes)
    print("Saved models/final_seed*.pt, models/final_config.json, outputs/plots/final_*.png, "
          "outputs/metrics/final_val_decisions.json. Test set not used.")


if __name__ == "__main__":
    main()
