# EYE-ASSISST — Testing Scripts

All one-off API and accuracy test scripts live here.
Run them from the **project root** after starting the backend.

## Prerequisites

```powershell
# Start the backend first (from project root)
& ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Wait ~60 seconds for all 3 models to load, then run any script below
cd "d:\MAJOR_PROJECT\EYE-ASSISST-main\EYE-ASSISST-main"
```

---

## Scripts

### `test_api.py` — Quick health + single-image smoke test
Hits `/health` then sends one cataract image. Verifies HTTP 200 and prints the JSON response summary.

```powershell
& ".venv\Scripts\python.exe" scripts/testing/test_api.py
```

---

### `test_live.py` — Real disease images with Grad-CAM (timing)
Sends `cataract_001.png` and `Glaucoma_001.png` from the Dataset folder to `/api/analyze?include_gradcam=true`. Prints per-disease confidence, detection flags, Grad-CAM availability, and total elapsed time.

```powershell
& ".venv\Scripts\python.exe" scripts/testing/test_live.py
```

Expected output (CPU inference):
```
cataract_001.png  ->  Cataract: 98.3% DETECTED  |  3.3s  |  Grad-CAM: yes
Glaucoma_001.png  ->  Glaucoma: 70.3% DETECTED  |  3.2s  |  Grad-CAM: yes
```

---

### `test_accuracy.py` — Batch precision/recall (5 cataract + 5 normal)
Sends 5 cataract images and 5 normal images, measures:
- **Cataract recall** (how many cataract images flagged correctly)
- **Normal precision** (how many normal images correctly not flagged)

```powershell
& ".venv\Scripts\python.exe" scripts/testing/test_accuracy.py
```

---

### `fix_backend.py` — One-shot backend sanitizer
Re-applies all Windows encoding fixes to `backend/main.py` and resets `backend/models/threshold_config.json` to balanced thresholds (0.60). Use only if you pull a version that has regressed.

```powershell
& ".venv\Scripts\python.exe" scripts/testing/fix_backend.py
```

---

## Notes

- CPU inference takes **10–25 seconds** per image (3 models run sequentially).
- The fundus classifier rejects non-fundus images with HTTP 400 — that is correct behaviour.
- Thresholds live in `backend/models/threshold_config.json`. Edit that file to tune sensitivity without touching code.
