"""
11_robustness.py
AI4003 CEP - Part E.iii: robustness on VALIDATION data (the test set is not used).

  Test 1  MISSING PATIENT DETAILS: metadata-on ensemble with 0%, 25%, 50%, 100% of the
          metadata fields hidden (three random masks per rate). The image-only final model
          does not use metadata, so it is the flat reference line.
  Test 2  IMAGE DEGRADATION on the FINAL image-only ensemble: blur, noise, darkening and
          heavy JPEG compression. The locked decision rule from models/final_config.json is
          applied, to see whether the system keeps its 30% referral cap and its sensitivity.

Runs on a laptop CPU (about 10 to 15 minutes). Backbone features are computed once per
model and image set; metadata only changes the small head, so Test 1 is cheap.

Run from the project root:  python src/11_robustness.py
"""
import argparse
import importlib.util
import io
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.special import softmax
from sklearn.metrics import f1_score, roc_auc_score

from common import HC_IDX, MODELS, OUT, SEED, is_hc, load_splits, p_high_concern, referral_stats, sens_at_rate


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b05 = load_module("b05", "src/05_design_b_finetune.py")   # FusionNet, read_image, normalise
prep02 = load_module("prep02", "src/02_preprocess_features.py")
torch.set_grad_enabled(False)


def load_members(tag, seeds, n_meta):
    members = []
    for s in seeds:
        m = b05.FusionNet(n_meta)
        m.load_state_dict(torch.load(MODELS / f"design_b_{tag}_seed{s}.pt", map_location="cpu"))
        members.append(m.eval())
    return members


def backbone_features(model, imgs, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        x = b05.normalise(b05.to_input(torch.from_numpy(imgs[i:i + bs])))
        out.append(model.backbone(x))
    return torch.cat(out)


def head_logits(model, feats, meta):
    return model.head(torch.cat([feats, model.meta(meta)], dim=1)).numpy()


def scores(y, probs):
    pred = probs.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, labels=list(range(7)), average="macro", zero_division=0)),
            "macro_auc": float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(7)))),
            "sens_at_30pct_referral": float(sens_at_rate(y, p_high_concern(probs), 0.30))}


def perturb(img, kind, rng):
    im = Image.fromarray(img)
    if kind == "blur":
        im = im.filter(ImageFilter.GaussianBlur(2))
    elif kind == "dark":
        im = ImageEnhance.Brightness(im).enhance(0.6)
    elif kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=15)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    elif kind == "noise":
        a = np.asarray(im, dtype=np.float32) + rng.normal(0, 13, img.shape)
        return np.clip(a, 0, 255).astype(np.uint8)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb", nargs="+", default=["blur", "noise", "dark", "jpeg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    df = load_splits()
    z = np.load(OUT / "meta_features.npz", allow_pickle=True)
    ids_va, y_va, F_va = z["ids_val"], z["y_val"], z["F_val"]
    lookup = df.set_index("image_id")
    val_rows = lookup.loc[ids_va].reset_index()
    print(f"Reading and resizing {len(ids_va)} validation images...")
    imgs = np.stack([b05.read_image(p) for p in lookup["path"].loc[ids_va]])
    hc = is_hc(y_va)
    n_meta = F_va.shape[1]
    results = {}

    # ------------------------------------------------ Test 1: missing metadata
    print("\n=== Test 1: missing patient details (metadata-on ensemble) ===")
    pre = joblib.load(MODELS / "meta_preprocessor.joblib")
    meta_raw = prep02.clean_meta(val_rows)
    on_members = load_members("metaOn", args.seeds, n_meta)
    t0 = time.time()
    on_feats = [backbone_features(m, imgs) for m in on_members]
    print(f"  backbone features for {len(on_members)} metadata-on models ({time.time() - t0:.0f}s)")
    rows1 = []
    for rate in (0.0, 0.25, 0.5, 1.0):
        per_mask = []
        for mask_seed in ([0] if rate in (0.0, 1.0) else [0, 1, 2]):
            Xm = meta_raw if rate == 0.0 else prep02.mask_metadata(meta_raw, rate, seed=mask_seed)
            M = torch.tensor(pre.transform(Xm), dtype=torch.float32)
            logits = np.mean([head_logits(m, f, M) for m, f in zip(on_members, on_feats)], axis=0)
            per_mask.append(scores(y_va, softmax(logits, axis=1)))
        row = {"missing_rate": rate, **{k: float(np.mean([p[k] for p in per_mask])) for k in per_mask[0]}}
        rows1.append(row)
        print(f"  missing {rate:>4.0%}: macro-F1 {row['macro_f1']:.3f}  AUC {row['macro_auc']:.3f}  "
              f"sens@30% {row['sens_at_30pct_referral']:.3f}")
    results["missing_metadata_metadata_on_ensemble"] = rows1

    # ------------------------------------------------ Test 2: image degradation
    print("\n=== Test 2: image degradation on the FINAL image-only ensemble ===")
    cfg = json.loads((MODELS / "final_config.json").read_text())
    T, t_ref, tau = cfg["temperature"], cfg["referral_threshold_p_hc"], cfg["defer_confidence_tau"]
    off_members = load_members("metaOff", args.seeds, n_meta)
    zeros = torch.zeros(len(imgs), n_meta)
    rng = np.random.default_rng(SEED)
    rows2 = []
    for cond in ["clean"] + args.perturb:
        t0 = time.time()
        x = imgs if cond == "clean" else np.stack([perturb(im, cond, rng) for im in imgs])
        logits = np.mean([head_logits(m, backbone_features(m, x), zeros) for m in off_members], axis=0)
        probs = softmax(logits / T, axis=1)
        p_hc, conf = p_high_concern(probs), probs.max(1)
        refer = p_hc >= t_ref
        defer = ~refer & (conf < tau)
        allref = refer | defer
        rate, sens = referral_stats(y_va, allref)
        row = {"condition": cond, **scores(y_va, probs),
               "locked_rule_referral_rate": rate, "locked_rule_referral_sensitivity": sens,
               "deferred_share": float(defer.mean()), "mean_confidence": float(conf.mean())}
        rows2.append(row)
        print(f"  {cond:<6} macro-F1 {row['macro_f1']:.3f}  AUC {row['macro_auc']:.3f}  | locked rule: "
              f"referral rate {rate:.3f}  sensitivity {sens:.3f}  deferred {row['deferred_share']:.3f}  "
              f"({time.time() - t0:.0f}s)")
    results["image_degradation_final_ensemble"] = rows2
    print("\n  Check: the 'clean' row should match script 09 (referral rate about 0.300, sensitivity about 0.882).")

    # ------------------------------------------------ save and plot
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows1).to_csv(OUT / "metrics" / "robustness_missing_metadata.csv", index=False)
    pd.DataFrame(rows2).to_csv(OUT / "metrics" / "robustness_image_degradation.csv", index=False)
    (OUT / "metrics" / "robustness_summary.json").write_text(json.dumps(results, indent=2))

    clean = rows2[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = [r["missing_rate"] * 100 for r in rows1]
    axes[0].plot(xs, [r["sens_at_30pct_referral"] for r in rows1], marker="o", label="metadata-on ensemble")
    axes[0].axhline(clean["sens_at_30pct_referral"], color="gray", linestyle="--", label="image-only (no metadata)")
    axes[0].set_xlabel("metadata fields hidden (%)"); axes[0].set_ylabel("sensitivity at 30% referral")
    axes[0].set_title("Missing patient details"); axes[0].legend(fontsize=8)
    names = [r["condition"] for r in rows2]
    xpos = np.arange(len(names))
    axes[1].bar(xpos - 0.2, [r["locked_rule_referral_sensitivity"] for r in rows2], 0.4, label="referral sensitivity")
    axes[1].bar(xpos + 0.2, [r["locked_rule_referral_rate"] for r in rows2], 0.4, label="referral rate")
    axes[1].axhline(0.30, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(names)
    axes[1].set_title("Locked rule under image degradation"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "robustness.png", dpi=150)
    print("\nSaved tables to outputs/metrics/ and the figure to outputs/plots/robustness.png. Test set not used.")


if __name__ == "__main__":
    main()