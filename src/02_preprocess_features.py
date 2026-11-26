"""
02_preprocess_features.py
AI4003 CEP - Part C.iii: metadata preprocessing, learned from TRAIN ONLY.

Run from the project root:  python src/02_preprocess_features.py

Reads   outputs/splits.csv, outputs/splits.sha256   (from script 01)
Writes  models/meta_preprocessor.joblib   fitted imputer/encoder/scaler
        outputs/meta_features.npz         train + val features and labels
        outputs/preprocess_report.json    numbers + proofs for the report

The TEST set is deliberately not transformed here (R6: test used once).
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data")
OUT = Path("outputs")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
HIGH_CONCERN = ["mel", "bcc", "akiec"]
NUM_COLS = ["age"]
CAT_COLS = ["sex", "localization"]
MISSING = "missing"


# ----------------------------------------------------------------- loading
def load_splits() -> pd.DataFrame:
    """Load the locked split and verify it was not changed (R6)."""
    csv_path = OUT / "splits.csv"
    locked = (OUT / "splits.sha256").read_text().strip()
    raw = csv_path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")  # Git may convert Windows line endings; same data
    assert any(hashlib.sha256(x).hexdigest() == locked for x in (raw, lf, crlf)), \
        "splits.csv changed since it was locked!"

    df = pd.read_csv(csv_path)
    # rebuild image paths from data/ so no machine-specific paths are used
    index = {}
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        for p in (DATA / part).glob("*.jpg"):
            index[p.stem] = p
    df["path"] = df["image_id"].map(index)
    assert df["path"].notna().all(), "some images are missing on disk"
    return df


def clean_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Turn 'unknown' into a real missing value. No statistics are learned here."""
    X = df[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        X[c] = X[c].replace("unknown", np.nan)
    X["age"] = pd.to_numeric(X["age"], errors="coerce")
    return X


# ------------------------------------------------------------ preprocessor
def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def mask_metadata(X: pd.DataFrame, rate: float, seed: int = 0) -> pd.DataFrame:
    """For R4 stress tests: hide each field of each row with probability `rate`."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    for c in NUM_COLS + CAT_COLS:
        X.loc[rng.random(len(X)) < rate, c] = np.nan
    return X


# -------------------------------------------------------------------- main
def main():
    df = load_splits()
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    report = {"splits_hash_verified": True}

    X_train = clean_meta(parts["train"])
    X_val = clean_meta(parts["val"])
    y_train = parts["train"]["dx"].map(CLASS_TO_IDX).to_numpy()
    y_val = parts["val"]["dx"].map(CLASS_TO_IDX).to_numpy()

    # fit on TRAIN ONLY, then apply to validation
    pre = build_preprocessor()
    F_train = pre.fit_transform(X_train)
    F_val = pre.transform(X_val)

    # ---- PROOF 1: statistics came from train only
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    train_median = float(np.nanmedian(X_train["age"]))
    all_median = float(np.nanmedian(clean_meta(df)["age"]))
    assert np.isclose(imputer.statistics_[0], train_median), "median not from train"
    report["age_median_train"] = train_median
    report["age_median_all_data_(not_used)"] = all_median

    # ---- PROOF 2: categories were learned from train only
    encoder = pre.named_transformers_["cat"].named_steps["onehot"]
    report["categories_learned_from_train"] = {
        c: list(map(str, cats)) for c, cats in zip(CAT_COLS, encoder.categories_)
    }

    # ---- PROOF 3: unseen categories and fully missing rows do not crash
    odd = pd.DataFrame({"age": [np.nan], "sex": ["martian"], "localization": ["tail"]})
    assert pre.transform(odd).shape[1] == F_train.shape[1]
    empty = pd.DataFrame({"age": [np.nan], "sex": [np.nan], "localization": [np.nan]})
    assert pre.transform(empty).shape[1] == F_train.shape[1]
    assert np.isfinite(pre.transform(empty)).all()
    report["unseen_and_all_missing_rows_ok"] = True

    # ---- R4 preview: masking works and keeps the same feature width
    F_masked = pre.transform(mask_metadata(X_val, rate=0.5))
    assert F_masked.shape == F_val.shape
    report["masked_validation_shape_ok"] = True

    # ---- numbers for the report
    report["feature_names"] = list(pre.get_feature_names_out())
    report["n_features"] = int(F_train.shape[1])
    report["missing_rate_pct"] = {
        s: {
            "age": round(float(clean_meta(p)["age"].isna().mean() * 100), 2),
            "sex": round(float(clean_meta(p)["sex"].isna().mean() * 100), 2),
            "localization": round(float(clean_meta(p)["localization"].isna().mean() * 100), 2),
        }
        for s, p in parts.items()
    }

    # ---- save
    joblib.dump(pre, MODELS / "meta_preprocessor.joblib")
    np.savez(
        OUT / "meta_features.npz",
        F_train=F_train, y_train=y_train, ids_train=parts["train"]["image_id"].to_numpy(),
        F_val=F_val, y_val=y_val, ids_val=parts["val"]["image_id"].to_numpy(),
    )
    (OUT / "preprocess_report.json").write_text(json.dumps(report, indent=2))

    print("Split hash verified (locked split unchanged).")
    print(f"Age median: train={train_median} | all data={all_median} (only train is used)")
    print("Categories learned from train:", report["categories_learned_from_train"])
    print(f"Metadata features: {F_train.shape[1]} columns")
    print(f"Train {F_train.shape} | Val {F_val.shape}  (test not touched)")
    print("Unseen category, all-missing row and masking checks passed.")


if __name__ == "__main__":
    main()