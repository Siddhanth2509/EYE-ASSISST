"""
E5 — Build Combined EyePACS + APTOS Splits
============================================
Creates Data/splits/fundus/combined/ by:
1. Copying existing EyePACS train/val/test splits
2. Adding preprocessed APTOS images to train/val (80/20 stratified)
3. Test set remains EyePACS-only for fair comparison with prior experiments

Prerequisites:
- Run preprocess_aptos_severity.py first
- EyePACS splits must already exist in Data/splits/fundus/eyepacs/

Output:
- Data/splits/fundus/combined/train/{NORMAL,DR}/
- Data/splits/fundus/combined/val/{NORMAL,DR}/
- Data/splits/fundus/combined/test/{NORMAL,DR}/

Label mapping (5-class severity) is handled via:
- Data/labels/combined_trainLabels.csv
"""

import os
import shutil
import random
from collections import Counter
from tqdm import tqdm

# ===== CONFIG =====
EYEPACS_SPLITS = r"Data\splits\fundus\eyepacs"
APTOS_PROCESSED = r"Data\processed\fundus\aptos_severity"
DST_ROOT = r"Data\splits\fundus\combined"
LABELS_CSV = r"Data\labels\aptos_train.csv"

CLASSES = ["NORMAL", "DR"]
APTOS_TRAIN_RATIO = 0.80  # 80% train, 20% val for APTOS
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ===== CREATE DIRECTORIES =====
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

# ===== STEP 1: Copy EyePACS splits =====
print("=" * 60)
print("Step 1: Copying EyePACS splits → combined/")
print("=" * 60)

eyepacs_counts = {}
for split in ["train", "val", "test"]:
    split_count = 0
    for cls in CLASSES:
        src_dir = os.path.join(EYEPACS_SPLITS, split, cls)
        dst_dir = os.path.join(DST_ROOT, split, cls)
        if not os.path.exists(src_dir):
            print(f"  ⚠ Skipping {src_dir} (not found)")
            continue
        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in tqdm(files, desc=f"EyePACS {split}/{cls}"):
            dst_path = os.path.join(dst_dir, f)
            if not os.path.exists(dst_path):
                shutil.copy2(os.path.join(src_dir, f), dst_path)
        split_count += len(files)
    eyepacs_counts[split] = split_count

print(f"\nEyePACS copied: train={eyepacs_counts.get('train',0)}, "
      f"val={eyepacs_counts.get('val',0)}, test={eyepacs_counts.get('test',0)}")

# ===== STEP 2: Split & copy APTOS to train/val =====
print("\n" + "=" * 60)
print("Step 2: Adding APTOS images → combined/train + combined/val")
print("=" * 60)

# Read APTOS labels for stratified split
import pandas as pd
aptos_df = pd.read_csv(LABELS_CSV)

# Group by severity for stratified split
aptos_by_severity = {}
for _, row in aptos_df.iterrows():
    sev = int(row["diagnosis"])
    img_id = row["id_code"]
    binary_cls = "NORMAL" if sev == 0 else "DR"
    if sev not in aptos_by_severity:
        aptos_by_severity[sev] = []
    aptos_by_severity[sev].append((img_id, binary_cls))

aptos_train_count = 0
aptos_val_count = 0

for sev in sorted(aptos_by_severity.keys()):
    items = aptos_by_severity[sev]
    random.shuffle(items)
    n_train = int(len(items) * APTOS_TRAIN_RATIO)
    train_items = items[:n_train]
    val_items = items[n_train:]

    for img_id, cls in tqdm(train_items, desc=f"APTOS sev={sev} → train"):
        src = os.path.join(APTOS_PROCESSED, cls, img_id + ".png")
        dst = os.path.join(DST_ROOT, "train", cls, img_id + ".png")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
        aptos_train_count += 1

    for img_id, cls in tqdm(val_items, desc=f"APTOS sev={sev} → val"):
        src = os.path.join(APTOS_PROCESSED, cls, img_id + ".png")
        dst = os.path.join(DST_ROOT, "val", cls, img_id + ".png")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
        aptos_val_count += 1

print(f"\nAPTOS added: train={aptos_train_count}, val={aptos_val_count}")

# ===== SUMMARY =====
print("\n" + "=" * 60)
print("Combined Dataset Summary")
print("=" * 60)

for split in ["train", "val", "test"]:
    total = 0
    for cls in CLASSES:
        path = os.path.join(DST_ROOT, split, cls)
        if os.path.exists(path):
            n = len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            total += n
            print(f"  {split}/{cls}: {n}")
    print(f"  {split} total: {total}")
    print()

print("✅ Combined splits created successfully!")
print(f"   Output: {DST_ROOT}")
print(f"   Labels: Data/labels/combined_trainLabels.csv")
