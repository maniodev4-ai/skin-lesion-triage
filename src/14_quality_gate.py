"""
14_quality_gate.py
Input-quality gate for the app. Script 11 showed the final model is brittle to blur, noise,
darkening and heavy JPEG compression, so the app checks every upload first.

Thresholds are set from CLEAN validation images (half of them; default limits are the 1st and
99th percentiles, see --pct), and detection rates are
measured on the OTHER half, both clean and degraded with the same degradations as script 11,
so the false-flag rate on clean images is an honest number. The test set is not used.

Checks (on the 224x224 image the model sees):
    sharpness   variance of the Laplacian           (flag if too LOW  -> blurry)
    noise       Immerkaer noise estimate            (flag if too HIGH -> noisy / compressed)
    brightness  mean grey level                     (flag if too low or too high)

Run from the project root:  python src/14_quality_gate.py     (about 2 minutes)
Writes models/quality_gate.json (used by the app) and outputs/metrics/quality_gate_eval.csv
"""
import argparse
import io
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import convolve, laplace

from common import MODELS, OUT, SEED, load_splits

SIZE = 224
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def read_image(path):
    im = Image.open(path)
    im.draft("RGB", (SIZE * 2, SIZE * 2))
    return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)


def perturb(img, kind, rng):
    im = Image.fromarray(img)
    if kind == "blur":
        im = im.filter(ImageFilter.GaussianBlur(2))
    elif kind == "dark":
        im = ImageEnhance.Brightness(im).enhance(0.6)
    elif kind == "bright":
        im = ImageEnhance.Brightness(im).enhance(1.6)
    elif kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=15)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    elif kind == "noise":
        a = np.asarray(im, dtype=np.float32) + rng.normal(0, 13, img.shape)
        return np.clip(a, 0, 255).astype(np.uint8)
    return np.asarray(im)


def quality_metrics(img):
    g = img.astype(np.float32).mean(axis=2)
    h, w = g.shape
    noise = np.sqrt(np.pi / 2) * np.abs(convolve(g, NOISE_KERNEL, mode="nearest"))[1:-1, 1:-1].sum() / (6 * (w - 2) * (h - 2))
    return {"sharpness": float(laplace(g).var()), "noise": float(noise), "brightness": float(g.mean())}


def flags(m, th):
    return {"blurry": m["sharpness"] < th["sharpness_min"], "noisy": m["noise"] > th["noise_max"],
            "too_dark": m["brightness"] < th["brightness_min"], "too_bright": m["brightness"] > th["brightness_max"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=1.0,
                    help="percentile limit taken from clean images: lower tail pct, upper tail 100-pct")
    args = ap.parse_args()
    lo_p, hi_p = args.pct, 100.0 - args.pct
    df = load_splits()
    val = df[df["split"] == "val"].reset_index(drop=True)
    print(f"Reading {len(val)} validation images...")
    imgs = np.stack([read_image(p) for p in val["path"]])
    fit_idx, eval_idx = np.arange(len(imgs))[0::2], np.arange(len(imgs))[1::2]

    m_fit = pd.DataFrame([quality_metrics(imgs[i]) for i in fit_idx])
    th = {"sharpness_min": float(np.percentile(m_fit["sharpness"], lo_p)),
          "noise_max": float(np.percentile(m_fit["noise"], hi_p)),
          "brightness_min": float(np.percentile(m_fit["brightness"], lo_p)),
          "brightness_max": float(np.percentile(m_fit["brightness"], hi_p))}
    print(f"Thresholds ({lo_p:g}th / {hi_p:g}th percentiles of clean validation images, even-indexed half):")
    for k, v in th.items():
        print(f"  {k:<16} {v:.3f}")

    rng = np.random.default_rng(SEED)
    rows = []
    for cond in ["clean", "blur", "noise", "dark", "bright", "jpeg"]:
        fl = []
        for i in eval_idx:
            img = imgs[i] if cond == "clean" else perturb(imgs[i], cond, rng)
            fl.append(flags(quality_metrics(img), th))
        f = pd.DataFrame(fl)
        rows.append({"condition": cond, "flagged_any": float(f.any(axis=1).mean()),
                     "blurry": float(f["blurry"].mean()), "noisy": float(f["noisy"].mean()),
                     "too_dark": float(f["too_dark"].mean()), "too_bright": float(f["too_bright"].mean())})
    table = pd.DataFrame(rows).set_index("condition")
    print(f"\n=== Share of images flagged, on the held-out half of validation ({len(eval_idx)} images) ===")
    print("(clean row = false-flag rate; degraded rows = detection rate)")
    print(table.round(3).to_string())

    out = {"thresholds": th,
           "description": "sharpness = variance of Laplacian of grey 224x224 image; noise = Immerkaer estimate; "
                          f"brightness = mean grey level. Thresholds = {lo_p:g}th/{hi_p:g}th percentiles of clean validation images.",
           "fit_on": "even-indexed half of the validation images",
           "evaluated_on": f"odd-indexed half ({len(eval_idx)} images)",
           "false_flag_rate_clean": float(table.loc["clean", "flagged_any"]),
           "detection_rates": {c: float(table.loc[c, "flagged_any"]) for c in table.index if c != "clean"}}
    MODELS.mkdir(exist_ok=True)
    (MODELS / "quality_gate.json").write_text(json.dumps(out, indent=2))
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT / "metrics" / "quality_gate_eval.csv")
    print("\nSaved models/quality_gate.json and outputs/metrics/quality_gate_eval.csv. Test set not used.")


if __name__ == "__main__":
    main()