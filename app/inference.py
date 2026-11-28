"""
app/inference.py
Self-contained, CPU-only inference for the skin-lesion triage prototype.
Loads ONLY the locked final model (models/final_config.json + models/final_seed*.pt)
and the quality gate (models/quality_gate.json). No internet access is needed.

Command-line test (no web app needed):
    python app/inference.py path/to/image.jpg

EDUCATIONAL RESEARCH PROTOTYPE. NOT A MEDICAL DEVICE.
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.ndimage import convolve, laplace
from scipy.special import softmax
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
MAX_BYTES = 25 * 1024 * 1024
FULL_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}


class BadImage(Exception):
    """Raised for files that cannot be used as an image."""


class FusionNet(nn.Module):
    """Same architecture as src/05_design_b_finetune.py (weights are loaded from disk)."""

    def __init__(self, n_meta):
        super().__init__()
        bb = models.efficientnet_b0(weights=None)
        bb.classifier = nn.Identity()
        self.backbone = bb
        self.meta = nn.Sequential(nn.Linear(n_meta, 64), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280 + 64, 7))


def load_system():
    cfg = json.loads((ROOT / "models" / "final_config.json").read_text())
    gate = json.loads((ROOT / "models" / "quality_gate.json").read_text())
    members = []
    for rel in cfg["members"]:
        m = FusionNet(cfg["n_meta_features"])
        m.load_state_dict(torch.load(ROOT / rel, map_location="cpu"))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        members.append(m)
    return cfg, gate, members


def prepare_image(file_bytes):
    """Decode an upload safely. Raises BadImage with a readable message on any problem."""
    if not file_bytes:
        raise BadImage("The file is empty.")
    if len(file_bytes) > MAX_BYTES:
        raise BadImage("The file is larger than 25 MB.")
    try:
        im = Image.open(io.BytesIO(file_bytes))
        if im.format == "JPEG":
            im.draft("RGB", (SIZE * 2, SIZE * 2))  # same fast decode as in training
        im.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise BadImage("This file is not a readable image (use a JPG or PNG photo).")
    im = ImageOps.exif_transpose(im).convert("RGB")
    if min(im.size) < 64:
        raise BadImage("The image is too small (the shorter side must be at least 64 pixels).")
    return im


def quality_metrics(img):
    g = img.astype(np.float32).mean(axis=2)
    h, w = g.shape
    noise = np.sqrt(np.pi / 2) * np.abs(convolve(g, NOISE_KERNEL, mode="nearest"))[1:-1, 1:-1].sum() / (6 * (w - 2) * (h - 2))
    return {"sharpness": float(laplace(g).var()), "noise": float(noise), "brightness": float(g.mean())}


def quality_check(img, gate):
    m, th, problems = quality_metrics(img), gate["thresholds"], []
    if m["sharpness"] < th["sharpness_min"]:
        problems.append("the image looks blurry or out of focus")
    if m["noise"] > th["noise_max"]:
        problems.append("the image looks noisy or heavily compressed")
    if m["brightness"] < th["brightness_min"]:
        problems.append("the image is too dark")
    if m["brightness"] > th["brightness_max"]:
        problems.append("the image is too bright")
    return m, problems


def make_overlay(img, cam, alpha=0.45):
    import matplotlib
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray((img * (1 - alpha) + heat * alpha).astype(np.uint8))


def analyse(pil_image, cfg, gate, members):
    """Run the locked model on one image. Returns a dict of results for display."""
    t0 = time.perf_counter()
    classes, T = cfg["class_order"], cfg["temperature"]
    arr = np.asarray(pil_image.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
    x = ((torch.from_numpy(arr[None].copy()).permute(0, 3, 1, 2).float() / 255.0) - MEAN) / STD
    x.requires_grad_(True)
    zeros = torch.zeros(1, cfg["n_meta_features"])
    acts, outs = [], []
    with torch.enable_grad():
        for m in members:
            A = m.backbone.features(x)
            A.retain_grad()
            out = m.head(torch.cat([torch.flatten(m.backbone.avgpool(A), 1), m.meta(zeros)], 1))
            acts.append(A)
            outs.append(out)
        logits = torch.stack([o.detach() for o in outs]).mean(0).numpy()[0]
        probs = softmax(logits / T)
        pred = int(probs.argmax())
        cams = []
        for A, out in zip(acts, outs):
            out[0, pred].backward()
            w = A.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * A).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
            cams.append(cam / (cam.max() + 1e-8))
    cam = np.mean(cams, axis=0)
    cam = cam / (cam.max() + 1e-8)

    p_hc = float(sum(probs[classes.index(c)] for c in cfg["high_concern"]))
    conf = float(probs[pred])
    q_metrics, q_problems = quality_check(arr, gate)
    if p_hc >= cfg["referral_threshold_p_hc"]:
        decision = "REFER"
        reason = (f"The estimated probability of a high-concern lesion ({p_hc:.1%}) is at or above "
                  f"the referral cut-off ({cfg['referral_threshold_p_hc']:.1%}).")
    elif q_problems:
        decision = "DEFER"
        reason = "The image quality check failed: " + "; ".join(q_problems) + ". The result is not reliable."
    elif conf < cfg["defer_confidence_tau"]:
        decision = "DEFER"
        reason = (f"The model is not confident ({conf:.1%} for the top class, deferral limit "
                  f"{cfg['defer_confidence_tau']:.1%}).")
    else:
        decision = "NOT REFERRED"
        reason = (f"The estimated probability of a high-concern lesion ({p_hc:.1%}) is below the cut-off and "
                  "the model is confident. This is not a clinical all-clear.")
    return {"probs": {c: float(p) for c, p in zip(classes, probs)}, "pred": classes[pred], "confidence": conf,
            "p_high_concern": p_hc, "decision": decision, "reason": reason, "quality": q_metrics,
            "quality_problems": q_problems, "overlay": make_overlay(arr, cam), "resized": Image.fromarray(arr),
            "seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python app/inference.py path/to/image.jpg")
        sys.exit(1)
    cfg_, gate_, members_ = load_system()
    try:
        img_ = prepare_image(Path(sys.argv[1]).read_bytes())
    except BadImage as e:
        print("Bad input:", e)
        sys.exit(2)
    r = analyse(img_, cfg_, gate_, members_)
    print(json.dumps({k: v for k, v in r.items() if k not in ("overlay", "resized")}, indent=2))