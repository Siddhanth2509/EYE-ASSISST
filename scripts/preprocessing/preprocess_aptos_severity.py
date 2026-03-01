"""
E5 — APTOS 5-Class Severity Preprocessing
==========================================
Preprocesses APTOS train images for DR severity grading (5 classes: 0-4).
- Resize: 224x224 (INTER_AREA)
- CLAHE: Green channel only
- Saves into binary folders (NORMAL/DR) matching EyePACS structure.
- 5-class severity labels come from CSV at runtime.

Input:  Data/raw/Aptos/train_images/ (3,662 .png files)
Output: Data/processed/fundus/aptos_severity/{NORMAL,DR}/
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

# ===== PATHS =====
CSV_PATH = r"Data\labels\aptos_train.csv"
IMG_DIR = r"Data\raw\Aptos\train_images"
DST_ROOT = r"Data\processed\fundus\aptos_severity"
TARGET_SIZE = (224, 224)

os.makedirs(os.path.join(DST_ROOT, "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(DST_ROOT, "DR"), exist_ok=True)

df = pd.read_csv(CSV_PATH)


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # CLAHE on green channel
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    img = cv2.merge((r, g, b))

    # Normalize to [0,1]
    img = img.astype("float32") / 255.0
    return img


processed = 0
skipped = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="APTOS severity preprocessing"):
    img_id = row["id_code"]
    severity = int(row["diagnosis"])

    src = os.path.join(IMG_DIR, img_id + ".png")
    if not os.path.exists(src):
        skipped += 1
        continue

    # Binary folder: NORMAL (severity 0) or DR (severity 1-4)
    label = "NORMAL" if severity == 0 else "DR"
    dst = os.path.join(DST_ROOT, label, img_id + ".png")
    if os.path.exists(dst):
        continue

    img01 = preprocess(src)
    if img01 is None:
        skipped += 1
        continue

    out = (img01 * 255).astype("uint8")
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst, out)
    processed += 1

print(f"✅ APTOS severity preprocessing complete.")
print(f"   Processed: {processed}, Skipped: {skipped}")
print(f"   Output: {DST_ROOT}")
