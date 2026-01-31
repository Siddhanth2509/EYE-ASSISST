# Fix for ModuleNotFoundError in phase2_results.ipynb

## Problem
The notebook can't find the `src` module because it's running from `notebooks\phase 2\` directory, but the `src` module is at the project root.

## Solution
Add a new cell **at the very beginning** of your notebook (before Cell 1 with the imports) with this code:

```python
# Add project root to Python path
import sys
from pathlib import Path

# Get the project root directory (2 levels up from this notebook)
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

print(f"Added to path: {project_root}")
```

## Steps to Fix

1. Open `phase2_results.ipynb` in Jupyter Notebook/Lab
2. Click on the **first cell** (the one with imports that's failing)
3. Go to menu: **Insert → Insert Cell Above**
4. Paste the code above into this new cell
5. Run the new cell first
6. Then run the imports cell - it should work now!

## Alternative: Run from Project Root

Instead of the above fix, you can also run Jupyter from the project root directory:

```bash
# Navigate to project root
cd "d:\TAB\Mine\College\SEM 7\GR\eye-realtime-inference"

# Launch Jupyter from there
jupyter notebook
```

Then navigate to `notebooks/phase 2/phase2_results.ipynb` in the Jupyter interface.
