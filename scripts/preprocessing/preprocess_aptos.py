"""
Phase 1C — APTOS Preprocessing
- Resize: 224x224
- CLAHE: Green channel
- Normalize: [0,1]
- Labels: 0 -> NORMAL, 1-4 -> DR
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

CSV_PATH = r"Data/raw_aptos/train.csv"
IMG_DIR  = r"Data/raw_aptos/images"
DST_ROOT = r"Data/processed_aptos/fundus"
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

    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    img = cv2.merge((r, g, b))

    img = img.astype("float32") / 255.0
    return img

for _, row in tqdm(df.iterrows(), total=len(df), desc="APTOS preprocessing"):
    img_id = row["id_code"]
    level = row["diagnosis"]

    src = os.path.join(IMG_DIR, img_id + ".png")  # APTOS is usually .png
    if not os.path.exists(src):
        continue

    label = "NORMAL" if level == 0 else "DR"
    dst = os.path.join(DST_ROOT, label, img_id + ".png")

    if os.path.exists(dst):
        continue

    img01 = preprocess(src)
    if img01 is None:
        continue

    out = (img01 * 255).astype("uint8")
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst, out)

print("✅ APTOS preprocessing complete.")
