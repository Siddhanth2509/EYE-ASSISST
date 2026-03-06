import json

with open("notebooks/phase 3/E4_EfficientNet_Colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    ct = cell["cell_type"]
    lines = "".join(cell["source"]).split("\n")
    first_line = lines[0][:100] if lines else ""
    print(f"Cell {i:2d}: [{ct:8s}] {first_line}")
