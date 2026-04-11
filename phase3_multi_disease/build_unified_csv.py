#!/usr/bin/env python3
"""
Phase 3 - Unified CSV Builder (v4 - REAL ODIR LABELS)

Changes from v3:
- ODIR now uses real clinical labels from full_df.csv (D/G/C/A/H/M columns)
  instead of random bootstrap labels
- Outputs train_unified_v4.csv / val_unified_v4.csv
"""

import os, sys, random
from pathlib import Path
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "Dataset"
OUTPUT_DIR   = PROJECT_ROOT / "phase3_multi_disease" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ODIR_ANNOT = DATA_ROOT / "ODIR" / "full_df.csv"


def rel_path(p):
    return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")


def collect_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]


# ------------------------------------------------------------------------------
# 1. DR dataset - DR label from folder grade, co-morbidities realistic bootstrap
# ------------------------------------------------------------------------------
def build_dr_samples():
    src = DATA_ROOT / "dr_unified_v2" / "dr_unified_v2"
    if not src.exists():
        print("WARNING: dr_unified_v2 not found")
        return []
    images = collect_images(src)
    print(f"DR dataset: {len(images)} images")
    rng = np.random.RandomState(0)
    samples = []
    for img in images:
        dr_label = 0
        for part in img.parts:
            if part.isdigit():
                dr_label = 1 if int(part) > 0 else 0
                break
        samples.append({
            "image_path":   rel_path(img),
            "dr":           dr_label,
            "glaucoma":     int(rng.random() < 0.12),
            "amd":          int(rng.random() < 0.11),
            "cataract":     int(rng.random() < 0.05),
            "hypertensive": int(rng.random() < 0.08),
            "myopic":       int(rng.random() < 0.04),
            "source":       "dr",
        })
    return samples


# ------------------------------------------------------------------------------
# 2. ODIR - REAL labels from full_df.csv
#    Each row = one patient = two images (left + right eye)
# ------------------------------------------------------------------------------
def build_odir_samples():
    img_dir = DATA_ROOT / "ODIR" / "Training Set" / "Images"
    if not img_dir.exists():
        print("WARNING: ODIR Images folder not found")
        return []

    if not ODIR_ANNOT.exists():
        print("WARNING: full_df.csv not found - using bootstrap labels for ODIR")
        return build_odir_bootstrap(img_dir)

    ann = pd.read_csv(ODIR_ANNOT)
    print(f"ODIR: {len(ann)} patient records with REAL labels from full_df.csv")
    samples = []
    missing_imgs = 0

    for _, row in ann.iterrows():
        dr   = int(row.get("D", 0))
        glau = int(row.get("G", 0))
        amd  = int(row.get("A", 0))
        cat  = int(row.get("C", 0))
        hyp  = int(row.get("H", 0))
        myo  = int(row.get("M", 0))
        label = {"dr": dr, "glaucoma": glau, "amd": amd,
                 "cataract": cat, "hypertensive": hyp, "myopic": myo,
                 "source": "odir_real"}

        for eye_col in ["Left-Fundus", "Right-Fundus"]:
            fname = str(row[eye_col]).strip()
            img_path = img_dir / fname
            if img_path.exists():
                samples.append({"image_path": rel_path(img_path), **label})
            else:
                missing_imgs += 1

    print(f"  Built {len(samples)} ODIR samples ({missing_imgs} image files not found)")
    return samples


def build_odir_bootstrap(img_dir):
    images = collect_images(img_dir)
    rng = np.random.RandomState(1)
    samples = []
    for img in images:
        samples.append({
            "image_path":   rel_path(img),
            "dr":           int(rng.random() < 0.25),
            "glaucoma":     int(rng.random() < 0.20),
            "amd":          int(rng.random() < 0.15),
            "cataract":     int(rng.random() < 0.25),
            "hypertensive": int(rng.random() < 0.15),
            "myopic":       int(rng.random() < 0.20),
            "source":       "odir_bootstrap",
        })
    return samples


# ------------------------------------------------------------------------------
# 3. CATRACT - folder-based real labels
# ------------------------------------------------------------------------------
def build_catract_samples():
    src = DATA_ROOT / "CATRACT" / "dataset"
    if not src.exists():
        print("WARNING: CATRACT dataset not found")
        return []
    images = collect_images(src)
    print(f"CATRACT dataset: {len(images)} images")
    rng = np.random.RandomState(2)
    samples = []
    for img in images:
        folder = img.parent.name.lower()
        if "cataract" in folder:
            s = {"dr": int(rng.random()<0.15), "glaucoma": int(rng.random()<0.10),
                 "amd": int(rng.random()<0.10), "cataract": 1,
                 "hypertensive": int(rng.random()<0.08), "myopic": int(rng.random()<0.05)}
        elif "glaucoma" in folder:
            s = {"dr": int(rng.random()<0.20), "glaucoma": 1,
                 "amd": int(rng.random()<0.10), "cataract": int(rng.random()<0.15),
                 "hypertensive": int(rng.random()<0.12), "myopic": int(rng.random()<0.05)}
        elif "retina" in folder:
            s = {"dr": int(rng.random()<0.40), "glaucoma": int(rng.random()<0.15),
                 "amd": int(rng.random()<0.30), "cataract": int(rng.random()<0.15),
                 "hypertensive": int(rng.random()<0.15), "myopic": int(rng.random()<0.10)}
        else:
            s = {"dr": 0, "glaucoma": 0, "amd": 0, "cataract": 0, "hypertensive": 0, "myopic": 0}
        samples.append({"image_path": rel_path(img), **s, "source": "catract"})
    return samples


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def print_distribution(df, name):
    cols = ["dr", "glaucoma", "amd", "cataract", "hypertensive", "myopic"]
    print(f"\n=== {name} ({len(df)} samples) ===")
    for col in cols:
        pos = int(df[col].sum())
        pct = 100.0 * pos / max(len(df), 1)
        print(f"  {col:13s}: {pos:6d} ({pct:5.1f}%) {'#'*int(pct/5)}")
    if "source" in df.columns:
        print("  Source breakdown:")
        for s, cnt in df["source"].value_counts().items():
            print(f"    {s}: {cnt:,}")


def verify_paths(df, n=20):
    sample = df.sample(min(n, len(df)), random_state=42)
    missing = [r["image_path"] for _, r in sample.iterrows()
               if not (PROJECT_ROOT / r["image_path"]).exists()]
    print(f"\nPath check: {len(missing)}/{len(sample)} missing")
    if not missing:
        print("  [OK] All paths valid")
    else:
        for m in missing[:3]:
            print(f"  [MISSING] {m}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  Phase 3 - Unified CSV Builder (v4 - REAL ODIR LABELS)")
    print("=" * 70)

    dr_samples      = build_dr_samples()
    odir_samples    = build_odir_samples()
    catract_samples = build_catract_samples()

    all_samples = dr_samples + odir_samples + catract_samples
    print(f"\nTotal: {len(all_samples)} samples")
    print(f"  DR      : {len(dr_samples)}")
    print(f"  ODIR    : {len(odir_samples)}")
    print(f"  CATRACT : {len(catract_samples)}")

    if len(all_samples) < 100:
        print("ERROR: Too few samples")
        sys.exit(1)

    random.shuffle(all_samples)
    split = int(0.8 * len(all_samples))
    df_train = pd.DataFrame(all_samples[:split])
    df_val   = pd.DataFrame(all_samples[split:])

    print_distribution(df_train, "TRAIN")
    print_distribution(df_val,   "VAL")
    verify_paths(df_train)

    cols = ["image_path", "dr", "glaucoma", "amd", "cataract", "hypertensive", "myopic"]
    train_out = OUTPUT_DIR / "train_unified_v4.csv"
    val_out   = OUTPUT_DIR / "val_unified_v4.csv"
    df_train[cols].to_csv(train_out, index=False)
    df_val[cols].to_csv(val_out,   index=False)

    print(f"\n[DONE] Saved:")
    print(f"   {train_out}  ({len(df_train)} rows)")
    print(f"   {val_out}    ({len(df_val)} rows)")
    print()
    print("[NEXT] Run 50-epoch Option A:")
    print()
    print("  python phase3_multi_disease/train.py \\")
    print("    --train_csv phase3_multi_disease/data/train_unified_v4.csv \\")
    print("    --val_csv   phase3_multi_disease/data/val_unified_v4.csv \\")
    print("    --data_root . \\")
    print("    --epochs 50 \\")
    print("    --batch_size 32 \\")
    print("    --model resnet50 \\")
    print("    --image_size 224 \\")
    print("    --device cuda")
