import os
import shutil
import random
from tqdm import tqdm

# ===== CONFIG =====
SRC_ROOT = r"Data\processed\fundus"
DST_ROOT = r"Data\splits\fundus"

CLASSES = ["NORMAL", "DR"]
SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Create destination folders
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

def split_class_images(class_name):
    src_dir = os.path.join(SRC_ROOT, class_name)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * SPLITS["train"])
    n_val = int(n_total * SPLITS["val"])

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    return train_imgs, val_imgs, test_imgs

# Perform split
for cls in CLASSES:
    train_imgs, val_imgs, test_imgs = split_class_images(cls)

    for fname in tqdm(train_imgs, desc=f"{cls} → train"):
        shutil.copy2(
            os.path.join(SRC_ROOT, cls, fname),
            os.path.join(DST_ROOT, "train", cls, fname)
        )

    for fname in tqdm(val_imgs, desc=f"{cls} → val"):
        shutil.copy2(
            os.path.join(SRC_ROOT, cls, fname),
            os.path.join(DST_ROOT, "val", cls, fname)
        )

    for fname in tqdm(test_imgs, desc=f"{cls} → test"):
        shutil.copy2(
            os.path.join(SRC_ROOT, cls, fname),
            os.path.join(DST_ROOT, "test", cls, fname)
        )

print("✅ Train / Validation / Test split completed.")
