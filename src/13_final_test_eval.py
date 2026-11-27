"""
13_final_test_eval.py
AI4003 CEP - Part E (final numbers), R5 and R6: the ONE evaluation on the locked TEST set.

Evaluates the locked final model (image-only Design B ensemble, temperature, referral
threshold and deferral rule from models/final_config.json) exactly as configured. Nothing is
tuned here. The script writes a lock file after the first complete run and refuses to run
again unless --force is given (and if you do, say so in the report).

Also measures R5: parameters, model file size, and CPU time per lesion (end to end), using
VALIDATION images for the timing so the test set is not touched more than needed.

Run from the project root:  python src/13_final_test_eval.py     (about 5 to 8 minutes on CPU)
"""
import argparse
import importlib.util
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

from common import (CLASS_TO_IDX, CLASSES, HC_IDX, MODELS, OUT, R_VALUES, SEED, expected_cost,
                    is_hc, lesion_bootstrap, load_splits, referral_stats)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b05 = load_module("b05", "src/05_design_b_finetune.py")
nov06 = load_module("nov06", "src/06_novel_contribution.py")
METRICS, PLOTS = OUT / "metrics", OUT / "plots"
LOCK = METRICS / "test_eval_LOCK.txt"
R_GRID = [1, 2, 5, 10, 20, 50]
torch.set_grad_enabled(False)


def ece_top(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(m.mean() * abs(correct[m].mean() - conf[m].mean())
                     for m in (idx == b for b in range(bins)) if m.any()))


def forward_logits(members, imgs, n_meta, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        x = b05.normalise(b05.to_input(torch.from_numpy(np.ascontiguousarray(imgs[i:i + bs]).copy())))
        zeros = torch.zeros(len(x), n_meta)
        out.append(np.mean([m.head(torch.cat([m.backbone(x), m.meta(zeros)], 1)).numpy() for m in members], axis=0))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if LOCK.exists() and not args.force:
        print("The test set has already been evaluated:\n" + LOCK.read_text() +
              "\nRefusing to run again. Using the test set twice invalidates R6. "
              "If you must, run with --force and disclose it in the report.")
        return

    cfg = json.loads((MODELS / "final_config.json").read_text())
    T, t_ref, tau, n_meta = cfg["temperature"], cfg["referral_threshold_p_hc"], cfg["defer_confidence_tau"], cfg["n_meta_features"]
    t_safety = cfg["safety_mode"]["threshold_p_hc"]
    df = load_splits()
    lookup = df.set_index("image_id")

    members = []
    for rel in cfg["members"]:
        m = b05.FusionNet(n_meta)
        m.load_state_dict(torch.load(rel, map_location="cpu"))
        members.append(m.eval())
    n_params = sum(p.numel() for m in members for p in m.parameters())

    # ---- the test set (read once, evaluated once)
    test = df[df["split"] == "test"].reset_index(drop=True)
    y = test["dx"].map(CLASS_TO_IDX).to_numpy()
    groups = test["lesion_id"].to_numpy()
    print(f"TEST SET: {len(test)} images, {len(np.unique(groups))} lesions. Reading images...")
    imgs = np.stack([b05.read_image(p) for p in test["path"]])
    t0 = time.time()
    logits = forward_logits(members, imgs, n_meta)
    print(f"Inference on the test set: {time.time() - t0:.0f}s")
    probs_raw, probs = softmax(logits, axis=1), softmax(logits / T, axis=1)
    pred, conf = probs.argmax(1), probs.max(1)
    p_hc = probs[:, HC_IDX].sum(1)
    hc = is_hc(y)
    res = {}

    # ---- locked decision rule, applied exactly as configured
    def locked_sent(p):
        ph, c = p[:, HC_IDX].sum(1), p.max(1)
        refer = ph >= t_ref
        return refer | (~refer & (c < tau))
    refer = p_hc >= t_ref
    defer = ~refer & (conf < tau)
    sent = refer | defer
    rate, sens = referral_stats(y, sent)

    # ---- headline metrics
    res["headline"] = {
        "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_auc": float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(7)))),
        "ece_raw": ece_top(probs_raw, y), "ece_calibrated": ece_top(probs, y)}
    res["locked_rule"] = {"referral_rate": rate, "referral_sensitivity": sens, "deferred_share": float(defer.mean()),
                          "referred_share": float(refer.mean()),
                          "sent_share_by_hc_class": {CLASSES[c]: float(sent[y == c].mean()) for c in HC_IDX},
                          "missed_high_concern": int((hc & ~sent).sum()), "high_concern_total": int(hc.sum())}
    res["locked_rule_ci95"] = {
        "referral_sensitivity": lesion_bootstrap(y, probs, groups, lambda yy, pp: locked_sent(pp)[is_hc(yy)].mean()),
        "referral_rate": lesion_bootstrap(y, probs, groups, lambda yy, pp: locked_sent(pp).mean()),
        "melanoma_sent_share": lesion_bootstrap(y, probs, groups, lambda yy, pp: locked_sent(pp)[yy == CLASS_TO_IDX["mel"]].mean()),
        "macro_f1": lesion_bootstrap(y, probs, groups, lambda yy, pp: f1_score(yy, pp.argmax(1), labels=list(range(7)), average="macro", zero_division=0))}
    rep = classification_report(y, pred, labels=list(range(7)), target_names=CLASSES, output_dict=True, zero_division=0)
    res["per_class"] = {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")} for c in CLASSES}

    # ---- rules compared on the test set (thresholds all chosen on validation)
    zv = [np.load(METRICS / f"design_b_metaOff_seed{s}_val_logits.npz", allow_pickle=True) for s in (42, 43, 44)]
    yv = zv[0]["y_val"]
    pv = softmax(np.mean([z["logits"] for z in zv], axis=0) / T, axis=1)
    lesion_v = lookup["lesion_id"].loc[zv[0]["ids_val"]].to_numpy()
    t_naive = nov06.choose_threshold(pv[:, HC_IDX].sum(1), is_hc(yv), lesion_v, 0.85, "naive")
    rules = {"argmax_rule": np.isin(pred, HC_IDX), "naive_threshold_target085": p_hc >= t_naive,
             "safety_mode_risk_controlled_target085": p_hc >= t_safety, "LOCKED_rule_cap30_with_deferral": sent}
    res["rules_on_test"] = {}
    for name, ref in rules.items():
        rr, ss = referral_stats(y, ref)
        res["rules_on_test"][name] = {"referral_rate": rr, "referral_sensitivity": ss,
                                      **{f"cost_r{r}": expected_cost(y, ref, r) for r in R_GRID}}
    mixups = sorted([(CLASSES[i], CLASSES[j], int(c)) for i, row in enumerate(confusion_matrix(y, pred, labels=list(range(7))))
                     for j, c in enumerate(row) if i != j and c > 0], key=lambda t: -t[2])[:6]
    res["top_mixups"] = mixups

    # ---- R5 resources (timing on validation images, batch of one, end to end)
    val_paths = lookup["path"].loc[zv[0]["ids_val"][:33]].tolist()
    times = []
    for p in val_paths:
        t0 = time.perf_counter()
        img = b05.read_image(p)
        lg = forward_logits(members, img[None], n_meta)
        softmax(lg / T, axis=1)
        times.append(time.perf_counter() - t0)
    times = np.array(times[3:])  # drop warm-up
    sizes_mb = sum((MODELS / rel.split("/")[-1]).stat().st_size for rel in cfg["members"]) / 1e6
    res["resources_R5"] = {
        "ensemble_members": len(members), "parameters_total": int(n_params),
        "model_files_total_mb": round(sizes_mb, 1), "limit_mb": 100,
        "cpu_seconds_per_lesion_mean": float(times.mean()), "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)), "max": float(times.max()), "limit_seconds": 1.0,
        "hardware": {"processor_string": platform.processor(), "platform": platform.platform(),
                     "torch_threads": torch.get_num_threads(), "note": "add CPU model and RAM manually"}}

    # ---- print
    print("\n=== TEST RESULTS (locked model, nothing tuned on the test set) ===")
    for k, v in res["headline"].items():
        print(f"  {k:<20} {v:.3f}")
    lr, ci = res["locked_rule"], res["locked_rule_ci95"]
    print(f"\nLocked rule: referral rate {lr['referral_rate']:.3f} [{ci['referral_rate'][0]:.3f}, {ci['referral_rate'][1]:.3f}]   "
          f"referral sensitivity {lr['referral_sensitivity']:.3f} [{ci['referral_sensitivity'][0]:.3f}, {ci['referral_sensitivity'][1]:.3f}]   "
          f"deferred {lr['deferred_share']:.3f}")
    print(f"  missed high-concern lesions: {lr['missed_high_concern']} of {lr['high_concern_total']}")
    print("  share sent by class:", {k: round(v, 3) for k, v in lr["sent_share_by_hc_class"].items()},
          f" melanoma 95% CI [{ci['melanoma_sent_share'][0]:.3f}, {ci['melanoma_sent_share'][1]:.3f}]")
    print(f"  macro-F1 95% CI [{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]")
    print("\nRules compared on the test set (thresholds chosen on validation):")
    print(pd.DataFrame(res["rules_on_test"]).T.round(3).to_string())
    print("\nPer-class (argmax):")
    print(pd.DataFrame(res["per_class"]).T.round(3).to_string())
    print("\nMost common mix-ups:", mixups)
    r5 = res["resources_R5"]
    print(f"\nR5: {r5['ensemble_members']} models, {r5['parameters_total']:,} parameters, {r5['model_files_total_mb']} MB "
          f"(limit 100); CPU time per lesion mean {r5['cpu_seconds_per_lesion_mean']:.3f}s, "
          f"p95 {r5['p95']:.3f}s, max {r5['max']:.3f}s (limit 1 s)")

    # ---- save and plot
    (METRICS / "test_final_results.json").write_text(json.dumps(res, indent=2, default=str))
    pd.DataFrame(res["rules_on_test"]).T.round(4).to_csv(METRICS / "test_rules_comparison.csv")
    pd.DataFrame(res["per_class"]).T.round(4).to_csv(METRICS / "test_per_class.csv")
    srt = np.argsort(-p_hc)
    plt.figure(figsize=(5.6, 4.3))
    plt.plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(hc[srt]) / hc.sum(), label="rank by P(high-concern)")
    plt.plot([0, 1], [0, 1], "k:", label="random")
    plt.scatter(rate, sens, marker="*", s=90, color="red", zorder=3, label="locked rule")
    plt.axvline(0.30, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("referral rate"); plt.ylabel("referral sensitivity"); plt.title("Test set: sensitivity vs referral rate")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(PLOTS / "test_referral_curve.png", dpi=150); plt.close()
    plt.figure(figsize=(5.2, 4.4))
    for name, pr in (("raw", probs_raw), ("temperature-scaled", probs)):
        c, ok = pr.max(1), (pr.argmax(1) == y).astype(float)
        idx = np.clip(np.digitize(c, np.linspace(0, 1, 11)[1:-1]), 0, 9)
        keep = [b for b in range(10) if (idx == b).sum() >= 10]
        plt.plot([c[idx == b].mean() for b in keep], [ok[idx == b].mean() for b in keep], marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("confidence"); plt.ylabel("accuracy")
    plt.title("Test set: reliability diagram"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(PLOTS / "test_reliability.png", dpi=150); plt.close()
    cm = confusion_matrix(y, pred, labels=list(range(7)))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(7), CLASSES, rotation=45); plt.yticks(range(7), CLASSES)
    for i in range(7):
        for j in range(7):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Test set: confusion matrix (row-normalised)")
    plt.tight_layout(); plt.savefig(PLOTS / "test_confusion_matrix.png", dpi=150); plt.close()

    LOCK.write_text(f"Test set evaluated once on {time.strftime('%Y-%m-%d %H:%M')} with the locked model.\n"
                    f"Config temperature {T:.4f}, referral threshold {t_ref:.4f}, tau {tau:.4f}.\n")
    print("\nSaved results to outputs/metrics/test_final_results.json and plots to outputs/plots/test_*.png.")
    print("The test set is now spent. Do not use these numbers to change the model, threshold or rule.")


if __name__ == "__main__":
    main()