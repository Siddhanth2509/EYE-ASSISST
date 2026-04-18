# EYE-ASSISST — AI-Powered Multi-Disease Eye Diagnosis Platform

> **Production-scale** retinal fundus analysis system detecting **6 eye diseases** using deep learning.  
> Built with FastAPI backend, React frontend, and ResNet50 multi-label classifier.

---

## Disease Detection

| Disease | Model AUC | Status |
|---|---|---|
| Diabetic Retinopathy | 0.90 | ✅ Production-ready |
| Glaucoma | 0.78+ (v5) | 🔄 Retraining |
| AMD | 0.75+ (v5) | 🔄 Retraining |
| Cataract | 0.74+ (v5) | 🔄 Retraining |
| Hypertensive Retinopathy | 0.70+ (v5) | 🔄 Retraining |
| Pathologic Myopia | 0.78+ (v5) | 🔄 Retraining |

---

## Project Structure

```
EYE-ASSISST/
├── backend/                    # FastAPI inference server
│   └── main.py                 # API endpoints + model loading
│
├── AI Eye Screening UI/app/    # React clinical dashboard (existing)
├── front1/app/                 # React v2 (3D landing, patient portal)
│
├── phase0_fundus_classifier/   # Binary fundus quality classifier
├── phase1_pipelines/           # Data ingestion pipelines
├── phase2_dr_severity/         # DR severity grading (5-class)
├── phase3_multi_disease/       # Multi-disease classifier (6 classes)
│   ├── build_unified_csv.py    # Dataset CSV builder (v5)
│   ├── train.py                # Training script (ResNet50)
│   ├── calibrate_thresholds.py # Per-class threshold calibration
│   ├── download_datasets.py    # Kaggle dataset downloader
│   └── data/                   # Generated CSVs (gitignored)
│
├── Dataset/                    # All training datasets (gitignored)
│   ├── dr_unified_v2/          # 92,501 DR images
│   ├── augmented_resized_V2/   # 143,669 augmented DR
│   ├── ODIR/                   # 10,000 ODIR-5K images
│   ├── AMD/                    # 3,988 AMDNet23 images
│   ├── GLAUCOMA_DETECTION/     # 18,842 glaucoma images
│   ├── eye_diseases_classification/ # 4,217 multi-class
│   ├── Hypertension & Hypertensive Retinopathy Dataset/ # 1,424
│   ├── Messidor-2/             # 1,744 DR images
│   ├── Myopia images/          # 100,543 myopia images
│   └── CATRACT/                # 1,202 cataract images
│       Total: ~377,000 images
│
├── models/                     # Saved checkpoints
├── docs/datasets/DATASETS.md   # Full dataset documentation
├── requirements.txt            # Python dependencies
└── START_APP.bat               # One-click startup
```

---

## Datasets

See [`docs/datasets/DATASETS.md`](docs/datasets/DATASETS.md) for full details.

**Total training data: ~377,000 fundus images across 11 datasets.**

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Training CSVs (v5)
```bash
python phase3_multi_disease/build_unified_csv.py
```

### 3. Train Multi-Disease Model
```bash
python phase3_multi_disease/train.py \
    --train_csv phase3_multi_disease/data/train_unified_v5.csv \
    --val_csv   phase3_multi_disease/data/val_unified_v5.csv \
    --data_root . \
    --epochs 50 \
    --batch_size 32 \
    --model resnet50 \
    --image_size 224 \
    --device cuda
```

### 4. Calibrate Thresholds
```bash
python phase3_multi_disease/calibrate_thresholds.py \
    --checkpoint phase3_multi_disease/checkpoints/<run_id>/best_model.pt \
    --val_csv phase3_multi_disease/data/val_unified_v5.csv \
    --data_root .
```

### 5. Start Backend
```bash
cd backend
python main.py
```

### 6. Start Frontend
```bash
cd "AI Eye Screening UI/app"
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend status + model info |
| `POST` | `/api/analyze` | Analyze fundus image (returns all 6 disease flags) |
| `GET` | `/api/scans` | List all previous scans |
| `GET` | `/api/v1/scan/{id}` | Get full scan details with Grad-CAM |

### Example Response
```json
{
  "analysis_id": "SCAN-A1B2C3",
  "dr_binary": { "is_dr": true, "confidence": 72.3 },
  "dr_severity": { "grade": 2, "label": "Moderate", "color": "#F97316" },
  "multi_disease": {
    "dr":           { "detected": true,  "confidence": 72.3, "threshold": 0.60 },
    "glaucoma":     { "detected": false, "confidence": 28.1, "threshold": 0.35 },
    "amd":          { "detected": false, "confidence": 18.4, "threshold": 0.40 },
    "cataract":     { "detected": false, "confidence": 12.0, "threshold": 0.55 },
    "hypertensive": { "detected": false, "confidence": 9.2,  "threshold": 0.45 },
    "myopic":       { "detected": false, "confidence": 22.1, "threshold": 0.50 }
  },
  "gradcam": { "heatmap_base64": "..." }
}
```

---

## Model Architecture

```
Input (224×224 RGB)
    ↓
ResNet50 Backbone (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Linear (2048 → 6)
    ↓
Sigmoid × 6  →  [DR, Glaucoma, AMD, Cataract, Hypertensive, Myopic]
```

**Training details:**
- Loss: Binary Cross-Entropy with class-weighted pos_weight
- Optimizer: Adam, LR 1e-4 with ReduceLROnPlateau
- Augmentation: Random flip, rotation, color jitter, Gaussian blur
- Threshold calibration: Per-class F1-optimal threshold search (0.1–0.9)

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Model | PyTorch 2.7, ResNet50, Grad-CAM |
| Backend | FastAPI, Uvicorn, Python 3.10 |
| Frontend (Clinical) | React 19, TypeScript, TailwindCSS, Framer Motion |
| Frontend (3D) | React Three Fiber, Three.js, GSAP |
| Data Processing | Pandas, NumPy, Pillow, OpenCV |
| Evaluation | scikit-learn (AUC, F1, ROC) |

---

## Requirements

- Python 3.10+
- CUDA GPU (RTX 3060 or better recommended for training)
- Node.js 18+ (for frontend)
- 50GB+ disk space for datasets
