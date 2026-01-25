import os, random, shutil
from tqdm import tqdm

SRC = r"Data\processed\fundus\aptos"
DST = r"Data\splits\fundus\aptos"
CLASSES = ["NORMAL", "DR"]
TRAIN_RATIO = 0.80
random.seed(42)

for split in ["train", "test"]:
    for c in CLASSES:
        os.makedirs(os.path.join(DST, split, c), exist_ok=True)

for c in CLASSES:
    imgs = [f for f in os.listdir(os.path.join(SRC, c)) if f.lower().endswith((".png",".jpg",".jpeg"))]
    random.shuffle(imgs)
    n_train = int(len(imgs) * TRAIN_RATIO)
    train, test = imgs[:n_train], imgs[n_train:]

    for f in tqdm(train, desc=f"{c} → train"):
        shutil.copy2(os.path.join(SRC, c, f), os.path.join(DST, "train", c, f))
    for f in tqdm(test, desc=f"{c} → test"):
        shutil.copy2(os.path.join(SRC, c, f), os.path.join(DST, "test", c, f))

print("✅ APTOS train/test split complete (unified).")
