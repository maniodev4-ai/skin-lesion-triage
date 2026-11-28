"""
make_demo_images.py
Creates demo_images/ with inputs for testing the app and recording the demo video.

  * 5 real dermoscopy images from the VALIDATION set (mel, nv, bcc, bkl, akiec)
  * degraded copies (224x224, the same size the model sees): blurred, dark, noisy, bright
  * bad inputs: a text file renamed .jpg, an empty .jpg, and a 40x40 image

Run from the project root:   python src/make_demo_images.py
HAM10000 images are CC BY-NC 4.0 (Tschandl et al.): keep demo_images/ out of Git, and credit
the dataset if screenshots appear in the report.
"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from common import load_splits

OUT = Path("demo_images")
OUT.mkdir(exist_ok=True)

df = load_splits()
val = df[df["split"] == "val"]

picked = {}
for cls in ["mel", "nv", "bcc", "bkl", "akiec"]:
    row = val[val["dx"] == cls].iloc[0]
    dst = OUT / f"sample_{cls}_{row['image_id']}.jpg"
    shutil.copy(row["path"], dst)
    picked[cls] = dst

# degraded copies, made at 224x224 so they match the conditions measured in scripts 11 and 14
base = Image.open(picked["nv"]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
rng = np.random.default_rng(0)
base.filter(ImageFilter.GaussianBlur(2)).save(OUT / "degraded_blurred.png")
ImageEnhance.Brightness(base).enhance(0.6).save(OUT / "degraded_dark.png")
ImageEnhance.Brightness(base).enhance(1.6).save(OUT / "degraded_bright.png")
noisy = np.clip(np.asarray(base, dtype=np.float32) + rng.normal(0, 13, (224, 224, 3)), 0, 255).astype(np.uint8)
Image.fromarray(noisy).save(OUT / "degraded_noisy.png")

# bad inputs
(OUT / "bad_not_an_image.jpg").write_text("this is a text file pretending to be a photo")
(OUT / "bad_empty.jpg").write_bytes(b"")
Image.new("RGB", (40, 40), (120, 80, 60)).save(OUT / "bad_tiny_40x40.png")

print(f"Created {len(list(OUT.iterdir()))} files in {OUT.resolve()}:\n")
for p in sorted(OUT.iterdir()):
    print("  ", p.name)
print("""
What to expect in the app:
  sample_*.jpg         normal result with probabilities, decision, heatmap and timing
  degraded_blurred     quality warning (blurry); DEFER unless the model already says REFER
  degraded_noisy       quality warning (noisy); DEFER unless REFER
  degraded_dark/bright quality warning (too dark / too bright)
  bad_not_an_image     clear error message, no crash
  bad_empty            clear error message, no crash (the uploader may refuse it before the app sees it)
  bad_tiny_40x40       error: image too small
Also try leaving the age, sex and location boxes blank, and filling them in (the result must not change).
""")
