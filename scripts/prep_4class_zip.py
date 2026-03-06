"""
Prepare 4-class EyePACS data for Colab training (E5).

Merges severity classes 0+1 → "No Referable DR", remaps 2→1, 3→2, 4→3.
Creates a new labels CSV and zips it with the existing EyePACS split images.

Run locally:
    python scripts/prep_4class_zip.py

Output:
    eyepacs_4class_splits.zip  (in project root, upload to Google Drive)
"""

import pandas as pd
import os
import zipfile
import time
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "Data"
LABELS_CSV = DATA / "labels" / "eyepacs_trainLabels.csv"
SPLITS_DIR = DATA / "splits" / "fundus" / "eyepacs"
OUTPUT_ZIP = BASE / "eyepacs_4class_splits.zip"
OUTPUT_CSV = DATA / "labels" / "eyepacs_4class_trainLabels.csv"

# Remapping: merge 0+1 → 0, shift 2→1, 3→2, 4→3
REMAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
CLASS_NAMES = {0: "No Referable DR", 1: "Moderate", 2: "Severe", 3: "Proliferative"}


def main():
    # ─── 1. Remap labels ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Remapping severity labels (5-class -> 4-class)")
    print("=" * 60)

    df = pd.read_csv(LABELS_CSV)
    print(f"  Original labels: {len(df)} rows")
    print(f"  Original distribution:\n{df['level'].value_counts().sort_index().to_string()}\n")

    df["level_4class"] = df["level"].map(REMAP)

    # Safety check: ensure no labels were silently dropped during remapping
    assert df["level_4class"].isna().sum() == 0, (
        f"Remapping failed! {df['level_4class'].isna().sum()} labels have no mapping. "
        f"Unmapped values: {df.loc[df['level_4class'].isna(), 'level'].unique().tolist()}"
    )

    print(f"  Remapped distribution:")
    for cls, name in CLASS_NAMES.items():
        count = (df["level_4class"] == cls).sum()
        print(f"    {cls} ({name}): {count}")

    # Save new CSV (keep original 'level' column for reference)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved: {OUTPUT_CSV}")

    # ─── 2. Create zip with images + new labels CSV ───────────────
    print(f"\n{'=' * 60}")
    print("Step 2: Creating zip with images + labels")
    print("=" * 60)

    # Collect image files from train/val/test splits
    image_files = []
    for split in ["train", "val", "test"]:
        split_dir = SPLITS_DIR / split
        if not split_dir.exists():
            print(f"  WARNING: {split_dir} does not exist, skipping")
            continue
        for root, dirs, files in os.walk(split_dir):
            for f in files:
                full_path = Path(root) / f
                # Archive path: e.g. train/DR/12345_left.jpeg
                rel = full_path.relative_to(SPLITS_DIR)
                image_files.append((full_path, str(rel).replace("\\", "/")))

    print(f"  Found {len(image_files)} images across splits")

    start = time.time()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add labels CSV
        zf.write(OUTPUT_CSV, "eyepacs_4class_trainLabels.csv")
        print("  Added labels CSV to zip")

        # Add images
        for i, (full_path, arc_name) in enumerate(image_files):
            zf.write(full_path, arc_name)
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{len(image_files)} images added")

    elapsed = time.time() - start
    zip_size = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n  ✅ Zip created: {OUTPUT_ZIP}")
    print(f"  Size: {zip_size:.0f} MB")
    print(f"  Time: {elapsed:.0f}s")

    # ─── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("NEXT STEPS")
    print("=" * 60)
    print(f"  1. Upload {OUTPUT_ZIP.name} to Google Drive:")
    print(f"     My Drive/eye-realtime-inference/")
    print(f"  2. Open the E5 notebook in Colab and run all cells")


if __name__ == "__main__":
    main()
