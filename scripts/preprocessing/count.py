import os
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "DR"]:
        p = f"Data/splits/fundus/{split}/{cls}"
        print(split, cls, len(os.listdir(p)))
