import os
import shutil
import pandas as pd

# ===== CORRECT PATHS =====
CSV_PATH = r"trainLabels.csv"
IMAGE_DIR = r"Data\raw\EYE-Pacs Dataset\data\data"
DEST_NORMAL = r"Data\cleaned\fundus\NORMAL"
DEST_DR = r"Data\cleaned\fundus\DR"

# Create destination folders
os.makedirs(DEST_NORMAL, exist_ok=True)
os.makedirs(DEST_DR, exist_ok=True)

# Sanity checks
print("CSV exists:", os.path.exists(CSV_PATH))
print("Image dir exists:", os.path.exists(IMAGE_DIR))
print("Sample images:", os.listdir(IMAGE_DIR)[:5])

# Read CSV
df = pd.read_csv(CSV_PATH)

copied = 0
missing = 0

for _, row in df.iterrows():
    image_id = row["image"]
    level = row["level"]

    src = os.path.join(IMAGE_DIR, image_id + ".jpeg")

    if not os.path.exists(src):
        missing += 1
        continue

    if level == 0:
        dest = os.path.join(DEST_NORMAL, image_id + ".jpeg")
    else:
        dest = os.path.join(DEST_DR, image_id + ".jpeg")

    if not os.path.exists(dest):
        shutil.copy2(src, dest)
        copied += 1

print(f"✅ Copied images: {copied}")
print(f"⚠️ Missing images: {missing}")
