"""
03_baselines.py
AI4003 CEP - Part C.i / C.ii: simple baselines on the SAME locked split.

    1. metadata only   (age, sex, body location)
    2. image only      (tiny colour-thumbnail + colour-histogram features)
    3. image + metadata
    + a "always predict nv" reference, to show why accuracy alone misleads.

Everything is fitted on TRAIN and evaluated on VALIDATION. The test set is not used.

Run from the project root:  python src/03_baselines.py
Needs:  pip install pillow
"""
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (CACHE, CLASSES, CLASS_TO_IDX, OUT, SEED, full_report,
                    lesion_bootstrap, load_splits, macro_f1_fn, ref_sens_fn)

CACHE.mkdir(exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------- simple image features
def simple_image_features(path) -> np.ndarray:
    """16x16 colour thumbnail (768 numbers) + 16-bin histogram per colour channel (48)."""
    img = Image.open(path)
    img.draft("RGB", (160, 120))  # fast JPEG decode at reduced size
    img = img.convert("RGB").resize((64, 64), Image.Resampling.BOX)
    a = np.asarray(img, dtype=np.float32) / 255.0
    thumb = np.asarray(img.resize((16, 16), Image.Resampling.BOX), dtype=np.float32).ravel() / 255.0
    hist = np.concatenate([
        np.histogram(a[..., c], bins=16, range=(0, 1))[0] / (64 * 64) for c in range(3)
    ])
    return np.concatenate([thumb, hist]).astype(np.float32)


def load_or_build_image_features(df: pd.DataFrame) -> dict:
    cache_file = CACHE / "img_simple_features.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        if set(z["ids"]) == set(df["image_id"]):
            print("Loaded cached image features.")
            return dict(zip(z["ids"], z["feats"]))
    print("Extracting simple image features (first run only, 1 to 3 minutes)...")
    ids, feats = [], []
    t0 = time.time()
    for i, (iid, path) in enumerate(zip(df["image_id"], df["path"])):
        ids.append(iid)
        feats.append(simple_image_features(path))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(df)} images ({time.time() - t0:.0f}s)")
    np.savez(cache_file, ids=np.array(ids), feats=np.stack(feats))
    return dict(zip(ids, feats))


# ------------------------------------------------------------------- models
def fit_best_logreg(X_tr, y_tr, X_va, y_va, name):
    """Tune C on VALIDATION only. class_weight='balanced' handles the rare classes."""
    best = None
    for C in (0.01, 0.1, 1.0):
        model = Pipeline([
            ("scale", StandardScaler()),  # fitted on train only (inside the pipeline)
            ("clf", LogisticRegression(C=C, class_weight="balanced",
                                       max_iter=3000, random_state=SEED)),
        ])
        t0 = time.time()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)
        f1 = f1_score(y_va, proba.argmax(1), labels=list(range(7)),
                      average="macro", zero_division=0)
        print(f"  [{name}] C={C:<5} val macro-F1={f1:.3f}  ({time.time() - t0:.1f}s)")
        if best is None or f1 > best[0]:
            best = (f1, C, proba)
    print(f"  [{name}] chosen C={best[1]}")
    return best[1], best[2]


def main():
    df = load_splits()
    z = np.load(OUT / "meta_features.npz", allow_pickle=True)
    ids_tr, ids_va = z["ids_train"], z["ids_val"]
    y_tr, y_va = z["y_train"], z["y_val"]
    M_tr, M_va = z["F_train"], z["F_val"]

    # image features, aligned to the same row order as the metadata features
    feat = load_or_build_image_features(df)
    I_tr = np.stack([feat[i] for i in ids_tr])
    I_va = np.stack([feat[i] for i in ids_va])

    lesion_of = df.set_index("image_id")["lesion_id"]
    groups_va = lesion_of.loc[ids_va].to_numpy()

    results, probas = {}, {}

    # reference: always predict the majority class
    dummy = np.zeros((len(y_va), 7))
    dummy[:, CLASS_TO_IDX["nv"]] = 1.0
    probas["always_nv"] = dummy

    print("\nTuning on validation:")
    sets = {
        "metadata_only": (M_tr, M_va),
        "image_only": (I_tr, I_va),
        "image_plus_metadata": (np.hstack([I_tr, M_tr]), np.hstack([I_va, M_va])),
    }
    chosen_C = {}
    for name, (Xtr, Xva) in sets.items():
        chosen_C[name], probas[name] = fit_best_logreg(Xtr, y_tr, Xva, y_va, name)

    print("\nEvaluating on validation (lesion-level bootstrap CIs)...")
    for name, proba in probas.items():
        rep = full_report(y_va, proba)
        rep["macro_f1_ci95"] = lesion_bootstrap(y_va, proba, groups_va, macro_f1_fn)
        rep["referral_sensitivity_ci95"] = lesion_bootstrap(y_va, proba, groups_va, ref_sens_fn)
        if name in chosen_C:
            rep["chosen_C"] = chosen_C[name]
        results[name] = rep

    # ---- tables
    table = pd.DataFrame({
        n: {
            "accuracy": r["accuracy"],
            "balanced_acc": r["balanced_accuracy"],
            "macro_F1": r["macro_f1"],
            "macro_AUC": r["macro_auc"],
            "referral_rate": r["referral_rate"],
            "referral_sens": r["referral_sensitivity"],
            "sens@30%_referral": r["sens_at_30pct_referral"],
            "cost_r=5": r["cost_r5"],
            "cost_r=10": r["cost_r10"],
        } for n, r in results.items()
    }).T.round(3)
    per_class = pd.DataFrame({n: r["per_class_recall"] for n, r in results.items()}).T.round(3)

    print("\n=== VALIDATION RESULTS (headline) ===")
    print(table.to_string())
    print("\n=== Per-class recall ===")
    print(per_class.to_string())
    print("\n95% CIs (lesion bootstrap):")
    for n, r in results.items():
        lo, hi = r["macro_f1_ci95"]
        slo, shi = r["referral_sensitivity_ci95"]
        print(f"  {n:<20} macro-F1 [{lo:.3f}, {hi:.3f}]   referral sens [{slo:.3f}, {shi:.3f}]")

    # ---- save for the report and for later scripts
    table.to_csv(OUT / "metrics" / "baselines_val_headline.csv")
    per_class.to_csv(OUT / "metrics" / "baselines_val_per_class.csv")
    (OUT / "metrics" / "baselines_val.json").write_text(json.dumps(results, indent=2))
    np.savez(OUT / "metrics" / "baselines_val_probs.npz", y_val=y_va, ids_val=ids_va, **probas)
    print("\nSaved to outputs/metrics/. The test set was not used.")


if __name__ == "__main__":
    main()
