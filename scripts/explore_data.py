import pandas as pd
import os
from pathlib import Path

base = r"D:\TAB\Mine\College\SEM 8\GR\eye-realtime-inference\Data"

# 1. ODIR Dataset
print("=" * 60)
print("ODIR DATASET (Multi-label: 8 diseases)")
print("=" * 60)
odir_csv = os.path.join(base, "raw", "ODIR Dataset", "full_df.csv")
df = pd.read_csv(odir_csv)
print(f"Total samples: {len(df)}")
labels = {'N': 'Normal', 'D': 'Diabetes/DR', 'G': 'Glaucoma', 'C': 'Cataract',
          'A': 'AMD', 'H': 'Hypertension', 'M': 'Myopia', 'O': 'Other'}
for col, name in labels.items():
    print(f"  {col} ({name}): {df[col].sum()}")

# Check ODIR images
odir_train = os.path.join(base, "raw", "ODIR Dataset", "ODIR-5K", "Training Images")
odir_test = os.path.join(base, "raw", "ODIR Dataset", "ODIR-5K", "Testing Images")
if os.path.exists(odir_train):
    print(f"  Training images: {len(os.listdir(odir_train))}")
if os.path.exists(odir_test):
    print(f"  Testing images: {len(os.listdir(odir_test))}")

# 2. Cataract dataset
print("\n" + "=" * 60)
print("CATARACT DATASET (External Eye Images)")
print("=" * 60)
cat_path = os.path.join(base, "raw", "Cataract (External Eye Images)")
for d in os.listdir(cat_path):
    full = os.path.join(cat_path, d)
    if os.path.isdir(full):
        count = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))])
        print(f"  {d}: {count} images")

# 3. AMD dataset
print("\n" + "=" * 60)
print("AMD DATASET")
print("=" * 60)
amd_path = os.path.join(base, "raw", "AMD (Age-related Macular Degeneration)", "1000images")
if os.path.exists(amd_path):
    subdirs = os.listdir(amd_path)
    for d in subdirs:
        full = os.path.join(amd_path, d)
        if os.path.isdir(full):
            count = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))])
            print(f"  {d}: {count} images")
        else:
            print(f"  File: {d}")

# 4. EyePACS splits
print("\n" + "=" * 60)
print("EYEPACS SPLITS (Current training data)")
print("=" * 60)
for split in ['train', 'val', 'test']:
    split_path = os.path.join(base, "splits", "fundus", "eyepacs", split)
    if os.path.exists(split_path):
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f"  {split}/{cls}: {count}")

# 5. APTOS splits
print("\n" + "=" * 60)
print("APTOS SPLITS")
print("=" * 60)
for split in ['train', 'test']:
    split_path = os.path.join(base, "splits", "fundus", "aptos", split)
    if os.path.exists(split_path):
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f"  {split}/{cls}: {count}")

# 6. Combined labels
print("\n" + "=" * 60)
print("COMBINED LABELS CSV")
print("=" * 60)
combined = pd.read_csv(os.path.join(base, "labels", "combined_trainLabels.csv"))
print(f"Total: {len(combined)}")
print(f"By source:\n{combined['source'].value_counts()}")
print(f"By severity:\n{combined['severity'].value_counts().sort_index()}")

# 7. EyePACS severity labels
print("\n" + "=" * 60)
print("EYEPACS SEVERITY LABELS")
print("=" * 60)
eyepacs = pd.read_csv(os.path.join(base, "labels", "eyepacs_trainLabels.csv"))
print(f"Total: {len(eyepacs)}")
print(eyepacs.head())
print(eyepacs.columns.tolist())
if 'level' in eyepacs.columns:
    print(f"By severity:\n{eyepacs['level'].value_counts().sort_index()}")
elif 'severity' in eyepacs.columns:
    print(f"By severity:\n{eyepacs['severity'].value_counts().sort_index()}")
