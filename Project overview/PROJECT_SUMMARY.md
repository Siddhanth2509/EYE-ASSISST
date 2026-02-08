# 📋 EYE-ASSISST - Quick Reference Summary

> **One-page overview of project structure, file purposes, and status**

---

## 🎯 Project Goal
End-to-end medical imaging system for **early eye disease detection** using multi-task deep learning.

---

## 📂 Quick Directory Map

### **Root Files**

| File | Purpose | Status |
|------|---------|--------|
| `train.py` | Phase-1 binary DR training entry point | ✅ Active |
| `evaluate.py` | Phase-1 model evaluation script | ✅ Active |
| `requirements.txt` | Python dependencies (PyTorch, CV libs) | ✅ Active |
| `README.md` | Project documentation & setup guide | ✅ Active |
| `IMPROVEMENTS.md` | Code improvement logs | ✅ Active |
| `ARCHITECTURE.md` | Complete system architecture (16KB) | ✅ Just created |
| `trainLabels.csv` | Legacy labels file | ⚠️ Backup only |

---

### **`configs/` - Configuration**

| File | Purpose | Status |
|------|---------|--------|
| `model.yaml` | Model hyperparameters | ❌ Empty |
| `paths.yaml` | Data path configs | ❌ Empty |
| `train.yaml` | Training settings | ❌ Empty |

**Usage:** Load YAML configs for reproducible experiments

---

### **`Data/` - Datasets (35k+ images)**

| Folder | Purpose | Contents |
|--------|---------|----------|
| `raw/` | Original downloads | EyePACS, APTOS, MESSIDOR, AMD, Cataract, ODIR |
| `cleaned/` | Quality-filtered images | `fundus/DR/`, `fundus/NORMAL/` |
| `processed/` | Preprocessed (resized, normalized) | `fundus/aptos/`, `fundus/eyepacs/` |
| **`splits/`** | **FROZEN train/val/test** | **Used by dataloaders** ⭐ |
| `labels/` | CSV label files | `eyepacs_trainLabels.csv` (severity 0-4) |
| `metadata/` | Dataset inventory | `data_manifest.csv` |

**Key:** `splits/fundus/` contains the actual training data structure

---

### **`models/` - Trained Checkpoints**

| Folder | Purpose | Files |
|--------|---------|-------|
| `binary_dr/` | Phase-1 binary DR model | `best.pt`, `latest.pt`, `training_history.json` |
| `stage1_dr_binary/` | Stage-1 multi-model | Same structure |

**Usage:** Load checkpoints for inference/resume training

---

### **`notebooks/` - Jupyter Analysis**

| Folder | Purpose | Notebooks |
|--------|---------|-----------|
| `phase1b/` | Data exploration | Image quality checks, class distribution |
| `phase 2/` | Severity results | DR grading evaluation |
| `phase 3/` | Multi-task debugging | Model head verification |

---

### **`scripts/` - Preprocessing Tools**

| Folder/File | Purpose | Usage |
|-------------|---------|-------|
| `preprocessing/preprocess_*.py` | Resize/normalize images | Run before training |
| `preprocessing/split_*.py` | Create train/val/test splits | One-time data prep |
| `spliteyepacs.py` | Split EyePACS dataset | Top-level script |
| `metadata_code/build_manifest.py` | Generate data inventory | Data tracking |

---

### **`src/` - Core ML Pipeline**

---

#### **`src/data/` - Dataset Handling**

| File/Folder | Purpose | Status |
|-------------|---------|--------|
| `datamodule.py` | Binary DR dataloader (Phase-1) | ✅ Active |
| `eyepacs_severity_datamodule.py` | DR severity dataloader (Phase-2) | ✅ Active |
| `transforms.py` | Custom augmentations | ✅ Active |
| **`datasets/`** | **PyTorch Dataset classes** | **See below** ▼ |

**Dataset Classes:**

| File | Purpose | Status |
|------|---------|--------|
| `eyepacs_severity.py` | EyePACS with severity labels (0-4) | ✅ Active |
| `eyepacs.py` | Binary EyePACS dataset | ❌ Empty placeholder |
| `aptos.py` | APTOS dataset | ❌ Empty placeholder |
| `amd.py` | AMD dataset | ❌ Empty placeholder |
| `cataract.py` | Cataract dataset | ❌ Empty placeholder |
| `odir.py` | ODIR 8-disease multi-label | ❌ Empty placeholder |

---

#### **`src/models/` - Neural Networks**

| File/Folder | Purpose | Details |
|-------------|---------|---------|
| `cnn_backbone.py` | Generic CNN wrapper | ResNet/EfficientNet support |
| `multi_task_models.py` | Multi-task architecture | Shared backbone + 3 heads |
| **`backbone/`** | Feature extractors | `resnet.py` (ResNet18/50/101) |
| **`heads/`** | Task-specific outputs | Binary, Severity, Multi-label |

**Model Heads:**

| File | Purpose | Output |
|------|---------|--------|
| `dr_binary.py` | Binary DR classification | 1 sigmoid (DR/Normal) |
| `dr_severity.py` | DR severity grading | 5 softmax (grades 0-4) |
| `multi_label.py` | Multi-disease detection | 8 sigmoids (AMD, Cataract, etc.) |

---

#### **`src/training/` - Training & Evaluation**

| File | Purpose | Usage |
|------|---------|-------|
| `train_binary.py` | BinaryTrainer class | Phase-1 training loop |
| `evaluate.py` | BinaryEvaluator class | Metrics computation |
| `run_train.py` | Training orchestration | CLI training script |
| `run_eval.py` | Evaluation orchestration | CLI evaluation script |

---

#### **`src/losses/` - Custom Loss Functions**

| File | Purpose | Usage |
|------|---------|-------|
| `masked_bce.py` | Masked Binary Cross-Entropy | Multi-label with missing labels |
| `severity_loss.py` | Ordinal regression loss | DR severity (penalizes by grade distance) |

---

#### **`src/metrics/` - Evaluation Metrics**

| File | Purpose | Metrics |
|------|---------|---------|
| `metrics.py` | Medical imaging metrics | Sensitivity, Specificity, AUC, QWK |

**Key Metrics:**
- **Sensitivity (Recall):** Catch all DR cases (TP / TP+FN)
- **Specificity:** Avoid false alarms (TN / TN+FP)
- **AUC-ROC:** Overall discrimination
- **QWK (Kappa):** Severity grading agreement

---

#### **`src/utils/` - Utilities**

| File | Purpose | Usage |
|------|---------|-------|
| `checkpoints.py` | Save/load model states | Checkpoint management |
| `logging.py` | Training logs | TensorBoard, CSV, console |
| `seed.py` | Set random seeds | Reproducibility (`set_seed(42)`) |

---

#### **`src/explainability/` - Interpretability**

| File | Purpose | Output |
|------|---------|--------|
| `gradcam.py` | Grad-CAM visualization | Heatmaps showing model attention |

**Usage:** Verify model looks at medically relevant regions (optic disc, hemorrhages)

---

#### **`src/training phase 3(multi-model)/` - Multi-Task Training**

| File | Purpose | Strategy |
|------|---------|----------|
| `train_stage1.py` | Train binary head | Stage-1: Binary DR only |
| `train_stage2.py` | Train severity head | Stage-2: Freeze binary, train severity |
| `train_stage3.py` | Joint training | Stage-3: Fine-tune all heads |
| `trainer.py` | Unified trainer class | Multi-task training loop |

**Training Flow:**
```
Stage 1 (Binary) → Stage 2 (+ Severity) → Stage 3 (+ Multi-label)
```

---

#### **`src/evaluation phase 3(multi-model)/` - Multi-Task Evaluation**

| File | Purpose | Metrics |
|------|---------|---------|
| `dr_binary_meterics.py` | Binary DR evaluation | AUC, Sensitivity, Specificity |
| `severity_metrics.py` | Severity evaluation | QWK, Confusion matrix |
| `multilabel_metrics.py` | Multi-label evaluation | Per-disease AUC, F1, Hamming loss |

---

#### **`src/configs/` - Training Stage Configs**

| File | Purpose | Status |
|------|---------|--------|
| `base.yml` | Base config template | ❌ Empty |
| `stage1_dr.yaml` | Stage-1 binary config | ❌ Empty |
| `stage2_mutlilabel.yml` | Stage-2 multi-label config | ❌ Empty |
| `stage3_joint.yaml` | Stage-3 joint training config | ❌ Empty |

---

## ⚠️ Empty Files Summary

### **Total Empty Files: 12**

**Config Files (7):**
- `configs/model.yaml`
- `configs/paths.yaml`
- `configs/train.yaml`
- `src/configs/base.yml`
- `src/configs/stage1_dr.yaml`
- `src/configs/stage2_mutlilabel.yml`
- `src/configs/stage3_joint.yaml`

**Dataset Classes (5):**
- `src/data/datasets/eyepacs.py`
- `src/data/datasets/aptos.py`
- `src/data/datasets/amd.py`
- `src/data/datasets/cataract.py`
- `src/data/datasets/odir.py`

**Action Required:**
✅ Config files → Populate with hyperparameters (learning rate, batch size, etc.)  
✅ Dataset classes → Implement PyTorch Dataset for each disease type

---

## 🎯 Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                EYE-ASSISST PIPELINE                  │
└─────────────────────────────────────────────────────┘

Raw Images (Data/raw/)
    ↓
Preprocessing (scripts/preprocessing/)
    ↓
Processed Images (Data/processed/)
    ↓
Train/Val/Test Splits (Data/splits/)
    ↓
DataModule (src/data/)
    ↓
Model (src/models/)
    ├── Backbone (ResNet50)
    ├── Binary Head (DR/Normal)
    ├── Severity Head (0-4)
    └── Multi-label Head (8 diseases)
    ↓
Training (src/training/)
    ├── Loss Functions (src/losses/)
    ├── Metrics (src/metrics/)
    └── Checkpoints (models/)
    ↓
Evaluation (src/evaluation/)
    └── Grad-CAM (src/explainability/)
```

---

## 🚀 Quick Commands

### **Training**
```bash
# Phase-1: Binary DR
python train.py --data_root Data/splits/fundus --epochs 50

# Phase-2: Severity grading
python src/training\ phase\ 3\(multi-model\)/train_stage2.py
```

### **Evaluation**
```bash
# Binary DR evaluation
python evaluate.py --checkpoint models/binary_dr/best.pt

# Severity metrics
python src/evaluation\ phase\ 3\(multi-model\)/severity_metrics.py
```

### **Data Prep**
```bash
# Preprocess images
python scripts/preprocessing/preprocess_eyepacs.py

# Create splits
python scripts/spliteyepacs.py
```

---

## 📊 Dataset Statistics

| Dataset | Split | Images | Purpose |
|---------|-------|--------|---------|
| **EyePACS** | Train | ~28k | Binary + Severity training |
| **EyePACS** | Val | ~7k | Validation |
| **APTOS** | Test | ~3.6k | External test set |
| **MESSIDOR** | Analysis | ~1.2k | Data exploration |
| **AMD/Cataract/ODIR** | Future | TBD | Multi-disease extension |

---

## 🏗️ Design Principles

✅ **Modularity:** Swap components without rewriting code  
✅ **Reproducibility:** Frozen splits, seeds, checkpoints  
✅ **Medical ML Best Practices:** External validation, Sensitivity > Accuracy  
✅ **Scalability:** Multi-task architecture, staged training  
✅ **Interpretability:** Grad-CAM, confusion matrices, calibration plots

---

## 📚 Key Technologies

- **Deep Learning:** PyTorch 2.1.2, Torchvision 0.16.2
- **Computer Vision:** OpenCV, Pillow, scikit-image
- **Data Science:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Notebook:** Jupyter, ipykernel
- **Future:** Streamlit (web app), FastAPI (inference API)

---

## 📈 Project Status

| Phase | Status | Completion |
|-------|--------|------------|
| Phase-1 (Binary DR) | ✅ Complete | 100% |
| Phase-2 (Severity) | 🔄 In Progress | 80% |
| Phase-3 (Multi-task) | 🔄 In Progress | 60% |
| Deployment | ⏳ Planned | 0% |

---

## 🔗 Related Documents

- **Full Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) (16KB detailed guide)
- **Setup Guide:** [README.md](README.md)
- **Improvement Logs:** [IMPROVEMENTS.md](IMPROVEMENTS.md)

---

**Last Updated:** February 8, 2026  
**Maintained by:** EYE-ASSISST Development Team
