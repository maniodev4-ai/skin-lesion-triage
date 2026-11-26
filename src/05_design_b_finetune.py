"""
05_design_b_finetune.py
AI4003 CEP - Part C: MAIN DESIGN B
    ImageNet-pretrained EfficientNet-B0, fine-tuned end to end, with a small metadata
    network fused inside the model (image features + metadata features -> classifier).

Trained on TRAIN. The best epoch is chosen on VALIDATION (macro-F1). The TEST set is never
loaded. Images are pre-resized to 224x224 once and cached, and augmentation runs on the GPU,
so no data-loader workers are needed.

Run on a GPU (Colab or Kaggle), from the project root:
    python src/05_design_b_finetune.py --seed 42 --epochs 12
    python src/05_design_b_finetune.py --seed 42 --use_meta 0      (image-only ablation)
"""
import argparse
import copy
import json
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, f1_score
from torchvision import models

from common import (CACHE, CLASSES, MODELS, OUT, full_report, lesion_bootstrap,
                    load_splits, macro_f1_fn, ref_sens_fn)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------ data loading
def read_image(path):
    im = Image.open(path)
    im.draft("RGB", (SIZE * 2, SIZE * 2))  # fast JPEG decode at reduced size
    return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)


def load_image_array(ids, paths):
    """Pre-resized uint8 array (N, 224, 224, 3), cached on disk after the first run."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"imgs{SIZE}_trainval.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        if list(z["ids"]) == list(ids):
            print("Loaded cached resized images.")
            return z["imgs"]
    print(f"Reading and resizing {len(ids)} images (first run only)...")
    t0 = time.time()
    arr = np.empty((len(ids), SIZE, SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        arr[i] = read_image(p)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(ids)} ({time.time() - t0:.0f}s)")
    np.savez(cache_file, ids=np.array(ids), imgs=arr)
    return arr


# ------------------------------------------------------------------- model
class FusionNet(nn.Module):
    def __init__(self, n_meta):
        super().__init__()
        bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        bb.classifier = nn.Identity()
        self.backbone = bb
        self.meta = nn.Sequential(nn.Linear(n_meta, 64), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280 + 64, len(CLASSES)))

    def forward(self, img, meta):
        return self.head(torch.cat([self.backbone(img), self.meta(meta)], dim=1))


def to_input(x_uint8):
    return x_uint8.permute(0, 3, 1, 2).float() / 255.0


def augment(x):
    """Flips, 90-degree rotation, small brightness/contrast jitter. All on the GPU."""
    B = x.shape[0]
    x = torch.where((torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1), x.flip(3), x)
    x = torch.where((torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1), x.flip(2), x)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.2
    bright = 1 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.2
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (((x - mean) * contrast + mean) * bright).clamp(0, 1)


def normalise(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.inference_mode()
def predict(model, imgs, meta, bs=128):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = normalise(to_input(torch.from_numpy(imgs[i:i + bs]).to(DEVICE)))
        with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            out.append(model(x, meta[i:i + bs]).float().cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--use_meta", type=int, default=1, help="0 = image-only ablation")
    ap.add_argument("--meta_dropout", type=float, default=0.2,
                    help="chance a sample's metadata is replaced by the all-missing vector")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tag = f"meta{'On' if args.use_meta else 'Off'}_seed{args.seed}"
    print(f"Device: {DEVICE} | run: {tag}")

    df = load_splits()
    z = np.load(OUT / "meta_features.npz", allow_pickle=True)
    ids_tr, ids_va = z["ids_train"], z["ids_val"]
    y_tr, y_va = z["y_train"], z["y_val"]
    lookup = df.set_index("image_id")
    groups_va = lookup["lesion_id"].loc[ids_va].to_numpy()

    ids_all = np.concatenate([ids_tr, ids_va])
    imgs = load_image_array(ids_all, lookup["path"].loc[ids_all].tolist())
    imgs_tr, imgs_va = imgs[:len(ids_tr)], imgs[len(ids_tr):]

    # metadata: features from script 02 (fitted on train only) and the all-missing vector
    pre = joblib.load(MODELS / "meta_preprocessor.joblib")
    m0 = pre.transform(pd.DataFrame({"age": [np.nan], "sex": [np.nan], "localization": [np.nan]}))
    m0 = torch.tensor(m0, dtype=torch.float32, device=DEVICE)
    F_tr = torch.tensor(z["F_train"], dtype=torch.float32, device=DEVICE)
    F_va = torch.tensor(z["F_val"], dtype=torch.float32, device=DEVICE)
    if not args.use_meta:
        F_tr, F_va, m0 = F_tr * 0, F_va * 0, m0 * 0
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)

    # class weights: inverse square-root frequency (a gentler fix than full balancing)
    counts = np.bincount(y_tr, minlength=len(CLASSES))
    w = (len(y_tr) / (len(CLASSES) * counts)) ** 0.5
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)

    model = FusionNet(F_tr.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    steps = len(ids_tr) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    history, best = [], {"f1": -1}
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(ids_tr))
        run_loss, t_ep = 0.0, time.time()
        for b in range(steps):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            x = normalise(augment(to_input(torch.from_numpy(imgs_tr[idx]).to(DEVICE))))
            m = F_tr[idx].clone()
            if args.use_meta and args.meta_dropout > 0:
                drop = torch.rand(len(idx), device=DEVICE) < args.meta_dropout
                m[drop] = m0
            with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                loss = F.cross_entropy(model(x, m), y_tr_t[idx], weight=w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
        logits = predict(model, imgs_va, F_va)
        pred = logits.argmax(1)
        val_loss = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y_va)))
        f1 = float(f1_score(y_va, pred, labels=list(range(7)), average="macro", zero_division=0))
        rec = {"epoch": epoch + 1, "train_loss": run_loss / steps, "val_loss": val_loss,
               "val_macro_f1": f1, "val_balanced_acc": float(balanced_accuracy_score(y_va, pred)),
               "seconds": round(time.time() - t_ep, 1)}
        history.append(rec)
        print(f"  epoch {epoch + 1:>2}/{args.epochs}  train loss {rec['train_loss']:.3f}  "
              f"val loss {val_loss:.3f}  val macro-F1 {f1:.3f}  ({rec['seconds']}s)")
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch + 1, "logits": logits,
                    "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
    train_seconds = time.time() - t_start
    print(f"Training time: {train_seconds / 60:.1f} min. Best epoch {best['epoch']} (val macro-F1 {best['f1']:.3f}).")

    # ---- evaluate the best epoch on validation
    probs = softmax(best["logits"], axis=1)
    rep = full_report(y_va, probs)
    rep["macro_f1_ci95"] = lesion_bootstrap(y_va, probs, groups_va, macro_f1_fn)
    rep["referral_sensitivity_ci95"] = lesion_bootstrap(y_va, probs, groups_va, ref_sens_fn)
    rep.update({"best_epoch": best["epoch"], "parameters": int(n_params),
                "weights_size_mb_fp32": round(n_params * 4 / 1e6, 1),
                "training_seconds": round(train_seconds, 1), "device": DEVICE,
                "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None})

    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (OUT / "metrics" / f"design_b_{tag}.json").write_text(
        json.dumps({"args": vars(args), "report": rep, "history": history}, indent=2))
    np.savez(OUT / "metrics" / f"design_b_{tag}_val_logits.npz",
             logits=best["logits"], y_val=y_va, ids_val=ids_va)
    torch.save(best["state"], MODELS / f"design_b_{tag}.pt")

    # ---- learning curves (C.iv)
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train (weighted)")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].legend()
    axes[0].set_title("Loss curves")
    axes[1].plot(ep, [h["val_macro_f1"] for h in history], marker="o")
    axes[1].axvline(best["epoch"], color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro-F1")
    axes[1].set_title("Validation macro-F1 (best epoch dashed)")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / f"design_b_{tag}_curves.png", dpi=150)

    # ---- summary and comparison with Design A
    print("\n=== Design B on validation ===")
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "referral_rate",
              "referral_sensitivity", "sens_at_30pct_referral", "cost_r5", "cost_r10"):
        print(f"  {k:<26} {rep[k]:.3f}")
    lo, hi = rep["macro_f1_ci95"]
    slo, shi = rep["referral_sensitivity_ci95"]
    print(f"  macro-F1 95% CI [{lo:.3f}, {hi:.3f}]   referral sens 95% CI [{slo:.3f}, {shi:.3f}]")
    print("  per-class recall:", {c: round(v, 3) for c, v in rep["per_class_recall"].items()})
    a_file = OUT / "metrics" / "design_a_val.json"
    if a_file.exists():
        a = json.loads(a_file.read_text())["results"]["A_embeddings_plus_metadata"]
        print("\n  Design A (embeddings + metadata) for comparison: "
              f"macro-F1 {a['macro_f1']:.3f}, AUC {a['macro_auc']:.3f}, "
              f"sens@30% {a['sens_at_30pct_referral']:.3f}, cost r=5 {a['cost_r5']:.3f}")
    print("\nSaved metrics, logits, curves and model. The test set was never loaded.")


if __name__ == "__main__":
    main()
