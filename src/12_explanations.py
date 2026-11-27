"""
12_explanations.py
AI4003 CEP - Part E.iv on VALIDATION data (the test set is not used).

  1. Grad-CAM heatmaps for 3 correct and 3 wrong predictions (final image-only ensemble's
     predicted class; the heatmap comes from ensemble member seed 42).
  2. Confusion matrix, the most common mix-ups, and which high-concern lesions the locked
     rule misses (and what they were mistaken for).
  3. Shortcut checks (member seed 42):
       - share of Grad-CAM attention that falls on the outer frame of the image
       - central-crop test   (only the middle 60% of the image is kept)
       - periphery-only test (the central lesion zone is hidden)
       - performance by amount of hair and by corner darkness (proxies for hair / vignette)
     We have NO ruler or ink annotations, so those artifacts are not tested directly.

Run from the project root:  python src/12_explanations.py   (about 10 minutes on CPU)
"""
import importlib.util
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import grey_closing
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score

from common import CLASS_TO_IDX, CLASSES, HC_IDX, MODELS, OUT, SEED, is_hc, load_splits


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b05 = load_module("b05", "src/05_design_b_finetune.py")
SIZE = b05.SIZE
PLOTS, METRICS = OUT / "plots", OUT / "metrics"
PLOTS.mkdir(parents=True, exist_ok=True)


def gradcam(model, img, cls, n_meta):
    x = b05.normalise(b05.to_input(torch.from_numpy(img[None])))
    with torch.enable_grad():
        A = model.backbone.features(x)
        A.retain_grad()
        f = torch.flatten(model.backbone.avgpool(A), 1)
        out = model.head(torch.cat([f, model.meta(torch.zeros(1, n_meta))], 1))
        out[0, cls].backward()
    w = A.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8)


@torch.inference_mode()
def member_logits(model, imgs, n_meta, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        x = b05.normalise(b05.to_input(torch.from_numpy(imgs[i:i + bs])))
        f = model.backbone(x)
        out.append(model.head(torch.cat([f, model.meta(torch.zeros(len(x), n_meta))], 1)).numpy())
    return np.concatenate(out)


def frame_mask(frac=0.15):
    m = np.ones((SIZE, SIZE), bool)
    k = int(SIZE * frac)
    m[k:-k, k:-k] = False
    return m


def main():
    cfg = json.loads((MODELS / "final_config.json").read_text())
    T, t_ref, tau, n_meta = cfg["temperature"], cfg["referral_threshold_p_hc"], cfg["defer_confidence_tau"], cfg["n_meta_features"]
    df = load_splits()
    lookup = df.set_index("image_id")

    # ensemble predictions from the saved validation logits (no inference needed)
    zs = [np.load(METRICS / f"design_b_metaOff_seed{s}_val_logits.npz", allow_pickle=True) for s in (42, 43, 44)]
    y, ids = zs[0]["y_val"], zs[0]["ids_val"]
    logits = np.mean([z["logits"] for z in zs], axis=0)
    probs = softmax(logits / T, axis=1)
    pred, conf = probs.argmax(1), probs.max(1)
    p_hc = probs[:, HC_IDX].sum(1)
    refer = p_hc >= t_ref
    defer = ~refer & (conf < tau)
    sent = refer | defer
    hc = is_hc(y)
    summary = {}

    # ---------------------------------------------------- 2. confusion analysis
    labels = list(range(7))
    cm = confusion_matrix(y, pred, labels=labels)
    off = [(CLASSES[i], CLASSES[j], int(cm[i, j]), float(cm[i, j] / cm[i].sum()))
           for i in labels for j in labels if i != j and cm[i, j] > 0]
    off.sort(key=lambda t: -t[2])
    print("=== Most common mix-ups (true -> predicted, count, share of the true class) ===")
    for t in off[:6]:
        print(f"  {t[0]:>5} -> {t[1]:<5} {t[2]:>3}  ({t[3]:.1%} of {t[0]})")
    missed = hc & ~sent
    missed_break = pd.Series([f"{CLASSES[a]} -> {CLASSES[b]}" for a, b in zip(y[missed], pred[missed])]).value_counts()
    print(f"\nHigh-concern lesions NOT sent to a doctor by the locked rule: {missed.sum()} of {hc.sum()}")
    print(missed_break.to_string())
    per_class_sent = {CLASSES[c]: float(sent[y == c].mean()) for c in HC_IDX}
    print("Share of each high-concern class that is sent:", {k: round(v, 3) for k, v in per_class_sent.items()})
    summary.update({"top_mixups": off[:8], "missed_high_concern": int(missed.sum()),
                    "missed_breakdown": missed_break.to_dict(), "sent_share_by_hc_class": per_class_sent})
    plt.figure(figsize=(6, 5))
    plt.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    plt.xticks(labels, CLASSES, rotation=45); plt.yticks(labels, CLASSES)
    for i in labels:
        for j in labels:
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Confusion matrix (validation, row-normalised)")
    plt.tight_layout(); plt.savefig(PLOTS / "confusion_matrix_val.png", dpi=150); plt.close()

    # ---------------------------------------------------- 1. Grad-CAM cases
    def pick(mask):
        idx = np.where(mask)[0]
        return int(idx[np.argmax(conf[idx])]) if len(idx) else None
    mel, bcc, nv = CLASS_TO_IDX["mel"], CLASS_TO_IDX["bcc"], CLASS_TO_IDX["nv"]
    rare = np.isin(y, [CLASS_TO_IDX[c] for c in ("akiec", "df", "vasc")])
    cases = [("correct", pick((y == mel) & (pred == mel))), ("correct", pick((y == bcc) & (pred == bcc))),
             ("correct", pick((y == nv) & (pred == nv))),
             ("wrong", pick((y == mel) & (pred == nv))), ("wrong", pick((y == nv) & (pred == mel))),
             ("wrong", pick(rare & (pred != y)))]
    cases = [(k, i) for k, i in cases if i is not None]
    member = b05.FusionNet(n_meta)
    member.load_state_dict(torch.load(MODELS / "design_b_metaOff_seed42.pt", map_location="cpu"))
    member.eval()
    fig, axes = plt.subplots(2, len(cases), figsize=(2.6 * len(cases), 5.6))
    case_info = []
    for c, (kind, i) in enumerate(cases):
        img = b05.read_image(lookup.loc[ids[i], "path"])
        cam = gradcam(member, img, int(pred[i]), n_meta)
        axes[0, c].imshow(img); axes[0, c].axis("off")
        axes[0, c].set_title(f"{kind}\ntrue {CLASSES[y[i]]} / pred {CLASSES[pred[i]]}\nconf {conf[i]:.2f}", fontsize=8)
        axes[1, c].imshow(img); axes[1, c].imshow(cam, cmap="jet", alpha=0.45); axes[1, c].axis("off")
        case_info.append({"kind": kind, "image_id": str(ids[i]), "true": CLASSES[y[i]], "pred": CLASSES[pred[i]],
                          "confidence": float(conf[i])})
    fig.suptitle("Grad-CAM for the predicted class (ensemble member seed 42)", fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS / "explanations_cases.png", dpi=150); plt.close()
    summary["cases"] = case_info
    print("\nGrad-CAM cases saved:", [(c["kind"], c["true"], c["pred"]) for c in case_info])

    # ---------------------------------------------------- 3. shortcut checks
    print("\nReading validation images for the shortcut checks...")
    imgs = np.stack([b05.read_image(lookup.loc[i, "path"]) for i in ids])
    frame = frame_mask(0.15)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(ids), 150, replace=False)
    shares = [float(c[frame].sum() / c.sum()) for c in
              (gradcam(member, imgs[i], int(pred[i]), n_meta) for i in sample)]
    frame_share = {"mean_attention_on_outer_frame": float(np.mean(shares)),
                   "area_share_of_frame_if_uniform": float(frame.mean()), "n_images": len(sample)}
    print(f"Attention on the outer 15% frame: {frame_share['mean_attention_on_outer_frame']:.3f} "
          f"(uniform attention would give {frame_share['area_share_of_frame_if_uniform']:.3f})")

    def evaluate(x, name):
        t0 = time.time()
        lg = member_logits(member, x, n_meta)
        p = lg.argmax(1)
        res = {"condition": name, "accuracy": float((p == y).mean()),
               "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0))}
        print(f"  {name:<18} accuracy {res['accuracy']:.3f}  macro-F1 {res['macro_f1']:.3f}  ({time.time() - t0:.0f}s)")
        return res
    centre = np.stack([np.asarray(Image.fromarray(im[45:179, 45:179]).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
                       for im in imgs])
    yy, xx = np.mgrid[:SIZE, :SIZE]
    hole = (yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2 < (0.6 * SIZE / 2) ** 2
    periph = imgs.copy()
    periph[:, hole] = np.array([124, 116, 104], dtype=np.uint8)
    print("\nShortcut tests (member seed 42, all validation images):")
    tests = [evaluate(imgs, "full image"), evaluate(centre, "central crop 60%"),
             evaluate(periph, "periphery only")]
    print("  Chance-level reference: always predicting nv gives accuracy "
          f"{(y == nv).mean():.3f} and macro-F1 0.115.")
    summary.update({"frame_attention": frame_share, "shortcut_tests": tests})

    gray = imgs.mean(axis=3)
    hair = np.array([(grey_closing(g, size=(9, 9)) - g > 12).mean() for g in gray])
    corners = np.array([np.mean([g[:24, :24].mean(), g[:24, -24:].mean(), g[-24:, :24].mean(), g[-24:, -24:].mean()])
                        for g in gray])
    print("\nPerformance by proxy bins (ensemble predictions, locked rule):")
    bins_out = {}
    for name, v in (("hair_amount", hair), ("corner_darkness", -corners)):
        q = np.quantile(v, [1 / 3, 2 / 3])
        b = np.digitize(v, q)
        rows = []
        for k, label in enumerate(["low", "mid", "high"]):
            m = b == k
            rows.append({"bin": label, "n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
                         "hc_sent_share": float(sent[m & hc].mean()) if (m & hc).any() else None})
            print(f"  {name:<16} {label:<5} n={m.sum():>4}  accuracy {rows[-1]['accuracy']:.3f}  "
                  f"high-concern sent {rows[-1]['hc_sent_share']:.3f}")
        bins_out[name] = rows
    summary["proxy_bins"] = bins_out
    (METRICS / "explanations_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(tests).to_csv(METRICS / "shortcut_tests_val.csv", index=False)
    print("\nWhat these do NOT prove: heatmaps show where the model looked, not why, and a heatmap on the "
          "lesion does not show the model reasons like a dermatologist. Rulers and ink were not tested.")
    print("Saved figures to outputs/plots/ and tables to outputs/metrics/. Test set not used.")


if __name__ == "__main__":
    main()