"""
01_data_audit_split.py
AI4003 CEP - Part B: data audit + lesion-grouped, stratified split.

Expected layout (Kaggle copy of HAM10000):
    data/HAM10000_metadata.csv
    data/HAM10000_images_part_1/*.jpg
    data/HAM10000_images_part_2/*.jpg

Run:  python 01_data_audit_split.py
Outputs (in outputs/):
    audit_report.json   -> numbers for the report (Part B.i, B.ii)
    splits.csv          -> image_id, lesion_id, dx, split  (LOCKED after this)
    splits.sha256       -> hash of splits.csv, proof the split never changed
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

HIGH_CONCERN = {"mel", "bcc", "akiec"}
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ------------------------------------------------------------------ load
meta = pd.read_csv(DATA / "HAM10000_metadata.csv")
report = {}

# ------------------------------------------------- B.i image <-> metadata match
img_paths = {}
for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
    for p in (DATA / part).glob("*.jpg"):
        img_paths[p.stem] = str(p)

meta_ids = set(meta["image_id"])
file_ids = set(img_paths)
report["n_metadata_rows"] = len(meta)
report["n_image_files"] = len(file_ids)
report["metadata_without_image"] = sorted(meta_ids - file_ids)
report["images_without_metadata"] = sorted(file_ids - meta_ids)
report["duplicate_image_ids_in_metadata"] = int(meta["image_id"].duplicated().sum())

# keep only rows that truly match an image (and prove it)
meta = meta[meta["image_id"].isin(file_ids)].copy()
meta["path"] = meta["image_id"].map(img_paths)
assert meta["image_id"].is_unique, "image_id must be unique"

# ------------------------------------------------- B.i missing values, classes
report["missing_values"] = meta[["age", "sex", "localization"]].isna().sum().to_dict()
report["unknown_sex"] = int((meta["sex"] == "unknown").sum())
report["unknown_localization"] = int((meta["localization"] == "unknown").sum())
report["class_counts"] = meta["dx"].value_counts().to_dict()
report["class_share_pct"] = (meta["dx"].value_counts(normalize=True) * 100).round(2).to_dict()
report["high_concern_share_pct"] = round(float(meta["dx"].isin(HIGH_CONCERN).mean() * 100), 2)
report["rare_localization_counts"] = meta["localization"].value_counts().tail(4).to_dict()
report["sex_counts"] = meta["sex"].value_counts().to_dict()
report["dx_type_counts"] = meta["dx_type"].value_counts().to_dict()  # label provenance

# ------------------------------------------------- B.i lesions and images per lesion
imgs_per_lesion = meta.groupby("lesion_id").size()
report["n_unique_lesions"] = int(imgs_per_lesion.shape[0])
report["images_per_lesion_distribution"] = imgs_per_lesion.value_counts().sort_index().to_dict()
report["lesions_with_multiple_images"] = int((imgs_per_lesion > 1).sum())
report["images_in_multi_image_lesions"] = int(imgs_per_lesion[imgs_per_lesion > 1].sum())

# label consistency inside a lesion (should be 1 dx per lesion)
dx_per_lesion = meta.groupby("lesion_id")["dx"].nunique()
report["lesions_with_conflicting_dx"] = int((dx_per_lesion > 1).sum())

# provenance by class: how much of each class is histopathology-confirmed
prov = pd.crosstab(meta["dx"], meta["dx_type"])
report["provenance_by_class"] = prov.to_dict(orient="index")

# ------------------------------------------------- B.iii grouped + stratified split
# Two-stage StratifiedGroupKFold: 1/7 -> test, then 1/6 of the rest -> val.
# Result is roughly 71% train / 14% val / 14% test, grouped by lesion_id.
y = meta["dx"].values
groups = meta["lesion_id"].values

outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
trainval_idx, test_idx = next(outer.split(meta, y, groups))

tv = meta.iloc[trainval_idx]
inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
tr_rel, va_rel = next(inner.split(tv, tv["dx"].values, tv["lesion_id"].values))

meta["split"] = ""
meta.iloc[test_idx, meta.columns.get_loc("split")] = "test"
meta.iloc[trainval_idx[tr_rel], meta.columns.get_loc("split")] = "train"
meta.iloc[trainval_idx[va_rel], meta.columns.get_loc("split")] = "val"

# ------------------------------------------------- B.iii PROOF in code (R6)
parts = {s: meta[meta["split"] == s] for s in ["train", "val", "test"]}
assert sum(len(p) for p in parts.values()) == len(meta), "every image must be in a split"

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    lesion_overlap = set(parts[a]["lesion_id"]) & set(parts[b]["lesion_id"])
    image_overlap = set(parts[a]["image_id"]) & set(parts[b]["image_id"])
    assert not lesion_overlap, f"LEAK: lesions shared between {a} and {b}"
    assert not image_overlap, f"LEAK: images shared between {a} and {b}"

for s, p in parts.items():
    missing = set(CLASSES) - set(p["dx"])
    assert not missing, f"class(es) {missing} absent from {s}"

report["split_sizes_images"] = {s: int(len(p)) for s, p in parts.items()}
report["split_sizes_lesions"] = {s: int(p["lesion_id"].nunique()) for s, p in parts.items()}
report["split_class_counts"] = {s: p["dx"].value_counts().to_dict() for s, p in parts.items()}
report["split_high_concern_pct"] = {
    s: round(float(p["dx"].isin(HIGH_CONCERN).mean() * 100), 2) for s, p in parts.items()
}
report["leakage_checks_passed"] = True

# ------------------------------------------------- B.iv lock the split
cols = ["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization", "path", "split"]
splits_path = OUT / "splits.csv"
meta[cols].sort_values("image_id").to_csv(splits_path, index=False)
digest = hashlib.sha256(splits_path.read_bytes()).hexdigest()
(OUT / "splits.sha256").write_text(digest + "\n")
report["splits_sha256"] = digest

(OUT / "audit_report.json").write_text(json.dumps(report, indent=2, default=str))

# ------------------------------------------------- console summary
print(f"Images: {report['n_image_files']} | metadata rows: {report['n_metadata_rows']}")
print(f"Unique lesions: {report['n_unique_lesions']} "
      f"({report['lesions_with_multiple_images']} have >1 image)")
print("Missing:", report["missing_values"])
print("Class counts:", report["class_counts"])
print("Split sizes (images):", report["split_sizes_images"])
print("Split sizes (lesions):", report["split_sizes_lesions"])
print("High-concern % per split:", report["split_high_concern_pct"])
print("All leakage checks passed. Split locked:", digest[:16], "...")
