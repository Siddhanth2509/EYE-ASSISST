"""
Phase 1B — EyePACS Preprocessing
--------------------------------
- Resize: 224x224 (INTER_AREA)
- CLAHE: Green channel only
- Normalize: [0, 1]
- Input : Data/cleaned/fundus/{NORMAL, DR}
- Output: Data/processed/fundus/{NORMAL, DR}

Safe:
- Non-destructive (reads cleaned/, writes processed/)
- Skips already-processed files
"""

import os
import cv2
from tqdm import tqdm

# ===== PATHS =====
SRC_ROOT = r"Data\cleaned\fundus"
DST_ROOT = r"Data\processed\fundus"
CLASSES = ["NORMAL", "DR"]
TARGET_SIZE = (224, 224)

os.makedirs(DST_ROOT, exist_ok=True)

def preprocess_image(src_path):
    # Load as RGB
    img = cv2.imread(src_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize (safe downscale)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # CLAHE on green channel
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    img = cv2.merge((r, g, b))

    # Normalize to [0,1]
    img = img.astype("float32") / 255.0
    return img

def save_image(dst_path, img01):
    # Convert back to 8-bit for storage
    img8 = (img01 * 255).clip(0, 255).astype("uint8")
    img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst_path, img8)

for cls in CLASSES:
    src_dir = os.path.join(SRC_ROOT, cls)
    dst_dir = os.path.join(DST_ROOT, cls)
    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fname in tqdm(files, desc=f"Processing {cls}", unit="img"):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        # Skip if already processed
        if os.path.exists(dst_path):
            continue

        img01 = preprocess_image(src_path)
        if img01 is None:
            continue

        save_image(dst_path, img01)

print("✅ Preprocessing complete.")
