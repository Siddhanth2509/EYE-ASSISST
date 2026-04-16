#!/usr/bin/env python3
"""
Kaggle Dataset Auto-Downloader for EYE-ASSISST Phase 3 Upgrade

Run this script after placing kaggle.json in C:/Users/<you>/.kaggle/

Usage:
    python phase3_multi_disease/download_datasets.py
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "Dataset"

# ── Dataset definitions ────────────────────────────────────────────────────────
# Format: (kaggle_slug, target_folder, disease, priority)
DATASETS = [
    # Priority 1 — Critical
    ("konduri-niharika/eye-diseases-classification",  "EYENET",           "Cataract+Glaucoma+DR+Normal", 1),
    ("andrewmvd/palm-pathologic-myopia",              "PALM",             "Myopic",                      1),
    ("arnavjain1/glaucoma-datasets",                  "GLAUCOMA_DETECTION","Glaucoma",                   1),
    # Priority 2 — Important
    ("jr2ngb/ichangelyodr",                           "AMD",              "AMD",                         2),
    ("sovitrath/diabetic-retinopathy-224x224-2019-data","MESSIDOR2",      "DR+additional",               2),
    # Priority 3 — Bonus
    ("linchundan/fundusimage",                        "HYPERTENSIVE",     "Hypertensive+Mixed",          3),
]

def check_kaggle_token():
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if not token_path.exists():
        print("=" * 60)
        print("  KAGGLE API TOKEN NOT FOUND")
        print("=" * 60)
        print()
        print("  1. Go to: https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print(f"  4. Move downloaded kaggle.json to: {token_path}")
        print()
        sys.exit(1)
    print(f"[OK] Kaggle token found at {token_path}")

def download_dataset(slug: str, target_folder: str, disease: str, priority: int):
    out_dir = DATASET_DIR / target_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already has images
    existing = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
    if len(existing) > 100:
        print(f"  [SKIP] {target_folder} already has {len(existing)} images")
        return True

    print(f"\n{'='*60}")
    print(f"  Downloading: {slug}")
    print(f"  Disease: {disease} | Priority: {priority}")
    print(f"  Target: {out_dir}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug,
             "-p", str(out_dir), "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [WARN] Download failed: {result.stderr.strip()}")
            print(f"  Trying alternative slug format...")
            return False

        imgs = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
        print(f"  [OK] Downloaded {len(imgs)} images to {out_dir}")
        return True

    except FileNotFoundError:
        print("  [ERROR] kaggle command not found. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def print_summary():
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    total = 0
    for _, folder, disease, _ in DATASETS:
        d = DATASET_DIR / folder
        if d.exists():
            imgs = list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
            n = len(imgs)
            total += n
            status = "OK" if n > 100 else "EMPTY/FAILED"
            print(f"  [{status:6s}] {folder:25s} {n:6,} images  ({disease})")
        else:
            print(f"  [MISS  ] {folder:25s}       0 images  ({disease})")

    print(f"\n  Total new images available: {total:,}")
    print()
    print("  NEXT STEP:")
    print("    python phase3_multi_disease/build_unified_csv.py")
    print("    (rebuilds training CSV with all new datasets)")

def main():
    print("=" * 60)
    print("  EYE-ASSISST Dataset Downloader")
    print("=" * 60)

    check_kaggle_token()

    # Parse priority filter from args
    max_priority = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    results = []
    for slug, folder, disease, priority in DATASETS:
        if priority > max_priority:
            continue
        ok = download_dataset(slug, folder, disease, priority)
        results.append((folder, ok))

    print_summary()

if __name__ == "__main__":
    main()
