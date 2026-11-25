"""
04_design_a_embeddings.py
AI4003 CEP - Part C: MAIN DESIGN A
    frozen ImageNet-pretrained EfficientNet-B0  ->  1280-number embedding per image
    ->  logistic regression (classical), with and without metadata.

Runs on a laptop CPU (about 10 to 20 minutes the first time, cached after).
Uses train + validation only. The test set is not used.

Run from the project root:  python src/04_design_a_embeddings.py
Needs:  pip install torch torchvision
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import (CACHE, OUT, SEED, full_report, lesion_bootstrap,
                    load_splits, macro_f1_fn, ref_sens_fn)

CACHE.mkdir(exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet constants from the pretrained model (fixed numbers, not learned from HAM10000)
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF(Image.open(self.paths[i]).convert("RGB"))


def build_backbone():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Identity()  # output = 1280-number embedding
    return m.eval().to(DEVICE)


def load_or_build_embeddings(df, model):
    cache_file = CACHE / "emb_effb0.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        if set(z["ids"]) == set(df["image_id"]):
            print("Loaded cached embeddings.")
            return dict(zip(z["ids"], z["emb"]))
    print(f"Extracting embeddings on {DEVICE} (first run only)...")
    loader = DataLoader(ImageDataset(df["path"]), batch_size=64,
                        num_workers=0 if os.name == "nt" else 2)
    chunks, t0 = [], time.time()
    with torch.inference_mode():
        for b, x in enumerate(loader):
            chunks.append(model(x.to(DEVICE)).cpu().numpy())
            if (b + 1) % 20 == 0:
                print(f"  {min((b + 1) * 64, len(df))}/{len(df)} images ({time.time() - t0:.0f}s)")
    emb = np.concatenate(chunks).astype(np.float32)
    np.savez(cache_file, ids=df["image_id"].to_numpy(), emb=emb)
    return dict(zip(df["image_id"], emb))


def time_cpu_inference(model, path, n=30):
    """Seconds per lesion for the CNN forward pass on CPU (R5 evidence)."""
    m = model.to("cpu")
    x = TF(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(3):
            m(x)  # warm-up
        t0 = time.perf_counter()
        for _ in range(n):
            m(x)
    return (time.perf_counter() - t0) / n


def fit_best_logreg(X_tr, y_tr, X_va, y_va, name, Cs=(0.001, 0.01, 0.1)):
    """Tune C on VALIDATION only; class_weight='balanced' for the rare classes."""
    best = None
    for C in Cs:
        model = Pipeline([
            ("scale", StandardScaler()),  # fitted on train only
            ("clf", LogisticRegression(C=C, class_weight="balanced",
                                       max_iter=3000, random_state=SEED)),
        ])
        t0 = time.time()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)
        f1 = f1_score(y_va, proba.argmax(1), labels=list(range(7)),
                      average="macro", zero_division=0)
        print(f"  [{name}] C={C:<6} val macro-F1={f1:.3f}  ({time.time() - t0:.1f}s)")
        if best is None or f1 > best[0]:
            best = (f1, C, proba)
    print(f"  [{name}] chosen C={best[1]}")
    return best[1], best[2]


def headline_table(results):
    return pd.DataFrame({
        n: {
            "accuracy": r["accuracy"], "balanced_acc": r["balanced_accuracy"],
            "macro_F1": r["macro_f1"], "macro_AUC": r["macro_auc"],
            "referral_rate": r["referral_rate"], "referral_sens": r["referral_sensitivity"],
            "sens@30%_referral": r["sens_at_30pct_referral"],
            "cost_r=5": r["cost_r5"], "cost_r=10": r["cost_r10"],
        } for n, r in results.items()
    }).T.round(3)


def main():
    df = load_splits()
    z = np.load(OUT / "meta_features.npz", allow_pickle=True)
    ids_tr, ids_va = z["ids_train"], z["ids_val"]
    y_tr, y_va = z["y_train"], z["y_val"]
    M_tr, M_va = z["F_train"], z["F_val"]
    groups_va = df.set_index("image_id")["lesion_id"].loc[ids_va].to_numpy()

    model = build_backbone()
    emb = load_or_build_embeddings(df, model)
    E_tr = np.stack([emb[i] for i in ids_tr])
    E_va = np.stack([emb[i] for i in ids_va])

    print("\nTuning on validation:")
    sets = {
        "A_embeddings_only": (E_tr, E_va),
        "A_embeddings_plus_metadata": (np.hstack([E_tr, M_tr]), np.hstack([E_va, M_va])),
    }
    results, probas, chosen_C = {}, {}, {}
    for name, (Xtr, Xva) in sets.items():
        chosen_C[name], probas[name] = fit_best_logreg(Xtr, y_tr, Xva, y_va, name)

    print("\nEvaluating on validation (lesion-level bootstrap CIs)...")
    for name, proba in probas.items():
        rep = full_report(y_va, proba)
        rep["macro_f1_ci95"] = lesion_bootstrap(y_va, proba, groups_va, macro_f1_fn)
        rep["referral_sensitivity_ci95"] = lesion_bootstrap(y_va, proba, groups_va, ref_sens_fn)
        rep["chosen_C"] = chosen_C[name]
        results[name] = rep

    # ---- resource facts for R5
    n_params = sum(p.numel() for p in model.parameters())
    resources = {
        "backbone": "EfficientNet-B0 (ImageNet weights, frozen)",
        "parameters": int(n_params),
        "weights_size_mb_fp32": round(n_params * 4 / 1e6, 1),
        "cpu_forward_seconds_per_lesion": round(time_cpu_inference(model, df["path"].iloc[0]), 4),
        "hardware_note": "fill in your CPU model and RAM for the report",
    }

    # ---- compare with baselines from script 03
    prev = json.loads((OUT / "metrics" / "baselines_val.json").read_text())
    all_results = {k: prev[k] for k in ["image_plus_metadata"]}
    all_results.update(results)
    table = headline_table(all_results)
    per_class = pd.DataFrame({n: r["per_class_recall"] for n, r in all_results.items()}).T.round(3)

    print("\n=== VALIDATION RESULTS (Design A vs best baseline) ===")
    print(table.to_string())
    print("\n=== Per-class recall ===")
    print(per_class.to_string())
    print("\n95% CIs (lesion bootstrap):")
    for n, r in results.items():
        lo, hi = r["macro_f1_ci95"]
        slo, shi = r["referral_sensitivity_ci95"]
        print(f"  {n:<28} macro-F1 [{lo:.3f}, {hi:.3f}]   referral sens [{slo:.3f}, {shi:.3f}]")
    print("\nResources:", json.dumps(resources, indent=2))

    table.to_csv(OUT / "metrics" / "design_a_val_headline.csv")
    per_class.to_csv(OUT / "metrics" / "design_a_val_per_class.csv")
    (OUT / "metrics" / "design_a_val.json").write_text(
        json.dumps({"results": results, "resources": resources}, indent=2))
    np.savez(OUT / "metrics" / "design_a_val_probs.npz", y_val=y_va, ids_val=ids_va, **probas)
    print("\nSaved to outputs/metrics/. The test set was not used.")


if __name__ == "__main__":
    main()