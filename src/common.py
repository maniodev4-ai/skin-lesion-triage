"""
common.py
Shared helpers for every script: constants, locked-split loader, and the
metrics the CEP cares about (referral rate, referral sensitivity, cost).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

DATA = Path("data")
OUT = Path("outputs")
MODELS = Path("models")
CACHE = Path("cache")  # big temporary files; ignored by git

SEED = 42
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
HC_NAMES = ["mel", "bcc", "akiec"]  # high-concern group
HC_IDX = [CLASS_TO_IDX[c] for c in HC_NAMES]
R_VALUES = (2, 5, 10, 20)  # cost ratios we test, since r is unknown (R2)


# ------------------------------------------------------------------ data
def load_splits() -> pd.DataFrame:
    """Load the locked split; stop if it was changed (R6)."""
    csv_path = OUT / "splits.csv"
    locked = (OUT / "splits.sha256").read_text().strip()
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == locked, \
        "splits.csv changed since it was locked!"
    df = pd.read_csv(csv_path)
    index = {}
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        for p in (DATA / part).glob("*.jpg"):
            index[p.stem] = p
    df["path"] = df["image_id"].map(index)
    assert df["path"].notna().all(), "some images are missing on disk"
    return df


# --------------------------------------------------------------- metrics
def is_hc(y) -> np.ndarray:
    return np.isin(y, HC_IDX)


def referral_stats(y_true, referred):
    """referral rate = referred / all ; referral sensitivity = referred HC / all HC."""
    hc = is_hc(y_true)
    return float(referred.mean()), float(referred[hc].mean())


def expected_cost(y_true, referred, r):
    """Cost per lesion: a missed HC lesion costs r, an unnecessary referral costs 1."""
    hc = is_hc(y_true)
    missed = (hc & ~referred).sum()
    unnecessary = (~hc & referred).sum()
    return float((r * missed + unnecessary) / len(y_true))


def sens_at_rate(y_true, p_hc, rate):
    """Referral sensitivity when exactly `rate` of lesions are referred,
    ranking lesions by P(high-concern). Gives a fair, threshold-free comparison."""
    k = int(round(rate * len(y_true)))
    referred = np.zeros(len(y_true), dtype=bool)
    referred[np.argsort(-p_hc)[:k]] = True
    return referral_stats(y_true, referred)[1]


def p_high_concern(proba) -> np.ndarray:
    return proba[:, HC_IDX].sum(axis=1)


def full_report(y_true, proba, rate_cap=0.30):
    """All headline numbers for one model on one split (no deferral yet)."""
    pred = proba.argmax(axis=1)
    referred = np.isin(pred, HC_IDX)
    rate, sens = referral_stats(y_true, referred)
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                  average="macro", labels=list(range(7))))
    except ValueError:
        auc = float("nan")
    recalls = recall_score(y_true, pred, labels=list(range(7)),
                           average=None, zero_division=0)
    rep = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(7)),
                                   average="macro", zero_division=0)),
        "macro_auc": auc,
        "referral_rate": rate,
        "referral_sensitivity": sens,
        f"sens_at_{int(rate_cap * 100)}pct_referral":
            float(sens_at_rate(y_true, p_high_concern(proba), rate_cap)),
        "per_class_recall": {c: float(v) for c, v in zip(CLASSES, recalls)},
    }
    for r in R_VALUES:
        rep[f"cost_r{r}"] = expected_cost(y_true, referred, r)
    return rep


# ------------------------------------------------------------ bootstrap
def lesion_bootstrap(y_true, proba, groups, metric_fn, n_boot=500, seed=SEED):
    """95% CI that resamples whole LESIONS (not images), because images of one
    lesion are not independent."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    idx_by_group = [np.where(inverse == g)[0] for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        vals.append(metric_fn(y_true[idx], proba[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def macro_f1_fn(y, p):
    return f1_score(y, p.argmax(1), labels=list(range(7)),
                    average="macro", zero_division=0)


def ref_sens_fn(y, p):
    return referral_stats(y, np.isin(p.argmax(1), HC_IDX))[1]
