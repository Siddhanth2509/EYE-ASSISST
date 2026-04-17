#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader for EYE-ASSISST Phase 3 Upgrade.

Downloads medical eye datasets from HuggingFace and saves them
as standard image files in the correct project folders.

Usage:
    python phase3_multi_disease/download_hf_datasets.py
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "Dataset"

# ── HuggingFace datasets ───────────────────────────────────────────────────────
# Format: (hf_repo_id, split, image_col, label_col, label_map, out_folder)
HF_DATASETS = [
    {
        "repo":       "ctmedtech/PALM",
        "splits":     ["train", "test"],
        "image_col":  "image",
        "label_col":  "label",           # 0=non-myopic, 1=myopic  (verify on HF page)
        "label_map":  {0: "non_myopic", 1: "myopic"},
        "out_folder": "PALM",
        "disease":    "Pathologic Myopia",
    },
    # Add more HF datasets here when needed, e.g.:
    # {
    #     "repo":      "some-org/glaucoma-dataset",
    #     "splits":     ["train"],
    #     "image_col":  "image",
    #     "label_col":  "label",
    #     "label_map":  {0: "non_glaucoma", 1: "glaucoma"},
    #     "out_folder": "GLAUCOMA_HF",
    #     "disease":    "Glaucoma",
    # },
]


def ensure_dependencies():
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("[INFO] Installing 'datasets' package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        print("[OK]  'datasets' installed\n")


def download_hf_dataset(cfg: dict):
    from datasets import load_dataset

    repo       = cfg["repo"]
    splits     = cfg["splits"]
    image_col  = cfg["image_col"]
    label_col  = cfg["label_col"]
    label_map  = cfg["label_map"]
    out_folder = cfg["out_folder"]
    disease    = cfg["disease"]

    out_root = DATASET_DIR / out_folder
    out_root.mkdir(parents=True, exist_ok=True)

    # Skip if already populated
    existing = list(out_root.rglob("*.jpg")) + list(out_root.rglob("*.png"))
    if len(existing) > 100:
        print(f"  [SKIP] {out_folder} already has {len(existing)} images")
        return True

    print(f"\n{'='*60}")
    print(f"  Dataset : {repo}")
    print(f"  Disease : {disease}")
    print(f"  Output  : {out_root}")
    print(f"{'='*60}")

    try:
        for split in splits:
            print(f"  Loading split: {split} ...")
            ds = load_dataset(repo, split=split, trust_remote_code=True)

            saved = 0
            for i, sample in enumerate(ds):
                img    = sample.get(image_col)
                label  = sample.get(label_col)

                if img is None:
                    continue

                # Determine class folder name
                if label is not None and label in label_map:
                    class_dir = out_root / label_map[label]
                elif label is not None:
                    class_dir = out_root / str(label)
                else:
                    class_dir = out_root / "unlabeled"

                class_dir.mkdir(parents=True, exist_ok=True)

                # Save image
                img_path = class_dir / f"{split}_{i:06d}.jpg"
                if not img_path.exists():
                    img.convert("RGB").save(img_path, "JPEG", quality=95)
                    saved += 1

                if (i + 1) % 200 == 0:
                    print(f"    Saved {i+1}/{len(ds)} images...", end="\r")

            print(f"  [OK] {split}: saved {saved} images to {out_root}")

        total = list(out_root.rglob("*.jpg"))
        print(f"  Total in folder: {len(total)} images")
        return True

    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg or "login" in msg.lower():
            print(f"\n  [AUTH REQUIRED] This dataset needs HuggingFace login:")
            print(f"    1. Run: huggingface-cli login")
            print(f"    2. Paste your HF token from https://huggingface.co/settings/tokens")
            print(f"    3. Re-run this script")
        else:
            print(f"  [ERROR] {e}")
        return False


def print_summary():
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    total = 0
    for cfg in HF_DATASETS:
        d = DATASET_DIR / cfg["out_folder"]
        if d.exists():
            imgs = list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
            n = len(imgs)
            total += n
            status = "OK" if n > 100 else "EMPTY/FAILED"
            print(f"  [{status:12s}] {cfg['out_folder']:20s} {n:6,} images  ({cfg['disease']})")
        else:
            print(f"  [MISSING      ] {cfg['out_folder']:20s}       0 images  ({cfg['disease']})")

    print(f"\n  Total images ready: {total:,}")
    if total > 0:
        print()
        print("  NEXT STEP:")
        print("    python phase3_multi_disease/build_unified_csv.py")
    print()


def main():
    print("=" * 60)
    print("  EYE-ASSISST HuggingFace Dataset Downloader")
    print("=" * 60)
    ensure_dependencies()

    for cfg in HF_DATASETS:
        download_hf_dataset(cfg)

    print_summary()


if __name__ == "__main__":
    main()
