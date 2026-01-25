import os
import csv
from PIL import Image

MANIFEST_PATH = r"Data/metadata/data_manifest.csv"
PREPROCESSING_VERSION = "v1_clahe_224"

DATASETS = {
    "eyepacs": {
        "base": r"Data/splits/fundus/eyepacs",
        "splits": ["train", "val", "test"],
        "modality": "fundus"
    },
    "aptos": {
        "base": r"Data/splits/fundus/aptos",
        "splits": ["train", "test"],
        "modality": "fundus"
    }
}

rows = []

for dataset, cfg in DATASETS.items():
    for split in cfg["splits"]:
        split_dir = os.path.join(cfg["base"], split)
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                except:
                    continue

                rows.append([
                    img_name,
                    dataset,
                    cfg["modality"],
                    label,
                    split,
                    w,
                    h,
                    PREPROCESSING_VERSION
                ])

# Write manifest (overwrite-safe)
with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id",
        "dataset",
        "modality",
        "final_label",
        "split",
        "width",
        "height",
        "preprocessing_version"
    ])
    writer.writerows(rows)

print(f"✅ Manifest created with {len(rows)} entries.")
