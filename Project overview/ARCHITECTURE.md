# 🏗️ EYE-ASSISST Project Architecture

> **Complete architectural documentation for the Eye Disease Detection & Clinical Decision Support Platform**

---

## 📐 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EYE-ASSISST PLATFORM                          │
│              AI-Powered Eye Disease Detection System                 │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Phase 1    │───▶│   Phase 2    │───▶│   Phase 3    │
│  Binary DR   │    │ DR Severity  │    │ Multi-Task   │
│ Classification│    │   Grading    │    │  Learning    │
└──────────────┘    └──────────────┘    └──────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────┐
│  Shared Components: Data, Models, Training, Utils    │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure & Component Roles

### **ROOT LEVEL**

```
eye-realtime-inference/
│
├── 📜 train.py                    [MAIN] Entry point for Phase-1 binary DR training
├── 📜 evaluate.py                 [MAIN] Entry point for Phase-1 model evaluation
├── 📄 README.md                   [DOC]  Project overview, setup instructions
├── 📄 IMPROVEMENTS.md             [DOC]  Code analysis & improvement logs
├── 📄 ARCHITECTURE.md             [DOC]  ⭐ THIS FILE - Complete architecture guide
├── 📄 requirements.txt            [CFG]  Python dependencies (PyTorch, CV libs, etc.)
└── 📄 trainLabels.csv             [DATA] Legacy/backup labels file
```

**Purpose:**
- **Entry Points:** `train.py` & `evaluate.py` orchestrate Phase-1 training/evaluation
- **Documentation:** README for users, IMPROVEMENTS for dev logs, ARCHITECTURE for system design
- **Config:** `requirements.txt` manages all dependencies

---

### **📂 `configs/` - Configuration Files**

```
configs/
├── model.yaml                     [EMPTY] ⚠️ Placeholder for model hyperparameters
├── paths.yaml                     [EMPTY] ⚠️ Placeholder for data paths config
└── train.yaml                     [EMPTY] ⚠️ Placeholder for training settings
```

**Purpose:**
- Centralized YAML configs for model architecture, data paths, training hyperparameters
- **Status:** Currently empty - ready for future structured configuration

**Usage:** Load configs in training scripts for reproducibility & easy experimentation

---

### **📂 `Data/` - All Dataset Storage**

```
Data/
├── raw/                           [DATA] Original unprocessed datasets
│   ├── all-images/                       MESSIDOR dataset (im0001.ppm - im1200.ppm)
│   ├── AMD (Age-related Macular Degeneration)/
│   ├── Aptos/                            APTOS 2019 Blindness Detection dataset
│   ├── Cataract (External Eye Images)/
│   ├── EYE-Pacs Dataset/                 EyePACS (Kaggle DR Competition)
│   └── ODIR Dataset/                     Ocular Disease Intelligent Recognition
│
├── cleaned/                       [DATA] Cleaned/filtered datasets
│   ├── external/                         External eye images (cataract, etc.)
│   └── fundus/                           Fundus images only
│       ├── DR/                           Contains DR-positive fundus images
│       └── NORMAL/                       Contains healthy fundus images
│
├── processed/                     [DATA] Preprocessed & standardized images
│   └── fundus/
│       ├── aptos/                        APTOS preprocessed (resizing, normalization)
│       └── eyepacs/                      EyePACS preprocessed
│
├── splits/                        [DATA] ⭐ FROZEN TRAIN/VAL/TEST SPLITS (Phase-1)
│   └── fundus/
│       ├── eyepacs/
│       │   ├── train/                    Training set (DR/, NORMAL/)
│       │   └── val/                      Validation set (DR/, NORMAL/)
│       └── aptos/
│           └── test/                     External test set (DR/, NORMAL/)
│
├── labels/                        [DATA] CSV label files
│   ├── aptos_test.csv                    APTOS test labels
│   ├── aptos_train.csv                   APTOS training labels
│   └── eyepacs_trainLabels.csv           EyePACS original labels (image, level)
│
└── metadata/                      [DATA] Dataset statistics & manifest
    └── data_manifest.csv                 Complete data inventory
```

**Data Flow:**
1. **Raw** → Original downloaded datasets
2. **Cleaned** → Quality filtering, format standardization
3. **Processed** → Resized, normalized, ready for model input
4. **Splits** → Final train/val/test folders (used by dataloaders)

**Key Notes:**
- `splits/` contains **FROZEN** splits for reproducibility
- `labels/` stores severity grades (0-4) for EyePACS DR staging
- `cleaned/fundus/` separates fundus vs external eye images

---

### **📂 `models/` - Trained Model Checkpoints**

```
models/
├── binary_dr/                     [MODELS] Phase-1 binary DR classifier
│   ├── best_model.pt                     Best checkpoint (highest val metric)
│   ├── best.pt                           Alternative best model
│   ├── latest.pt                         Most recent epoch checkpoint
│   └── training_history.json             Loss/metric curves
│
└── stage1_dr_binary/              [MODELS] Stage-1 multi-model training
    ├── best_model.pt
    ├── best.pt
    ├── latest.pt
    └── training_history.json
```

**Purpose:**
- Save trained model weights for inference & retraining
- Track training history (loss/accuracy curves) via JSON logs

**Checkpoint Strategy:**
- `best.pt`: Saved when validation metric improves
- `latest.pt`: Saved every N epochs for recovery
- `training_history.json`: Plotted for convergence analysis

---

### **📂 `notebooks/` - Jupyter Experimentation**

```
notebooks/
├── phase1b/                       [ANALYSIS] Phase-1 exploratory analysis
│   ├── 01_inspect_images.ipynb            EDA: Image quality, class distribution
│   └── 01_inspect_aptos.ipynb             APTOS dataset exploration
│
├── phase 2/                       [RESULTS] Phase-2 severity grading results
│   └── phase2_results.ipynb               DR severity model evaluation
│
└── phase 3/                       [DEBUG] Phase-3 multi-task debugging
    └── head_checker.ipynb                 Verify multi-head model outputs
```

**Purpose:**
- **Phase 1b:** Data quality checks, distribution analysis, baseline metrics
- **Phase 2:** Evaluate DR severity grading (0-4 scale)
- **Phase 3:** Debug multi-task model (binary + severity + multi-label heads)

**Usage:** Interactive exploration, visualization, hypothesis testing before production code

---

### **📂 `scripts/` - Data Preprocessing & Utilities**

```
scripts/
├── spliteyepacs.py                [SCRIPT] Split EyePACS into train/val/test
│
├── preprocessing/                 [SCRIPTS] Data preparation pipelines
│   ├── count.py                           Count images per class
│   ├── preprocess_aptos.py                Resize/normalize APTOS images
│   ├── preprocess_eyepacs.py              Resize/normalize EyePACS images
│   ├── split_aptos.py                     Split APTOS into test set
│   └── split_train_val_test.py            Generic train/val/test splitter
│
└── metadata_code/                 [SCRIPTS] Dataset metadata generation
    └── build_manifest.py                  Create data_manifest.csv inventory
```

**Purpose:**
- **Preprocessing:** Convert raw images → processed images (resize, normalize, crop)
- **Splitting:** Create reproducible train/val/test splits
- **Metadata:** Track data provenance, counts, quality metrics

**Run Order:**
1. `preprocess_*.py` → Clean raw images
2. `split_*.py` → Create frozen splits
3. `build_manifest.py` → Generate metadata CSV

---

### **📂 `src/` - Core ML Pipeline (Modular Design)**

```
src/
├── __init__.py                    [MODULE] Make src a Python package
│
├── configs/                       [CFG] Training stage configurations
│   ├── base.yml                           [EMPTY] ⚠️ Base config template
│   ├── stage1_dr.yaml                     [EMPTY] ⚠️ Stage-1 DR binary config
│   ├── stage2_mutlilabel.yml              [EMPTY] ⚠️ Stage-2 multi-label config
│   └── stage3_joint.yaml                  [EMPTY] ⚠️ Stage-3 joint training config
│
├── data/                          [MODULE] ⭐ Dataset handling & augmentation
├── models/                        [MODULE] ⭐ Neural network architectures
├── training/                      [MODULE] ⭐ Training & evaluation loops
├── losses/                        [MODULE] ⭐ Custom loss functions
├── metrics/                       [MODULE] ⭐ Evaluation metrics
├── utils/                         [MODULE] ⭐ Utilities (logging, checkpoints, etc.)
├── explainability/                [MODULE] ⭐ Grad-CAM & interpretability
├── training phase 3(multi-model)/ [MODULE] ⭐ Phase-3 multi-model training
└── evaluation phase 3(multi-model)/ [MODULE] ⭐ Phase-3 evaluation
```

---

#### **📂 `src/data/` - Dataset & DataModule**

```
src/data/
├── __init__.py
├── datamodule.py                  [CLASS] FundusDataModule (Phase-1 binary DR)
├── eyepacs_severity_datamodule.py [CLASS] EyePACSSeverityDataModule (Phase-2)
├── transforms.py                  [UTILS] Custom image augmentations
│
└── datasets/                      [DATASETS] PyTorch Dataset classes
    ├── eyepacs_severity.py                ✅ EyePACS with severity labels (0-4)
    ├── eyepacs.py                         [EMPTY] ⚠️ Placeholder for EyePACS binary
    ├── aptos.py                           [EMPTY] ⚠️ Placeholder for APTOS dataset
    ├── amd.py                             [EMPTY] ⚠️ Placeholder for AMD dataset
    ├── cataract.py                        [EMPTY] ⚠️ Placeholder for cataract dataset
    └── odir.py                            [EMPTY] ⚠️ Placeholder for ODIR dataset
```

**Key Components:**

| File | Purpose | Status |
|------|---------|--------|
| `datamodule.py` | Binary DR classification dataloader (EyePACS train/val, APTOS test) | ✅ Active |
| `eyepacs_severity_datamodule.py` | DR severity grading (0-4) dataloader | ✅ Active |
| `transforms.py` | Custom augmentations (rotation, crop, color jitter) | ✅ Active |
| `datasets/eyepacs_severity.py` | EyePACS Dataset returning `(image, dr_label, severity_label)` | ✅ Active |
| `datasets/*.py` (others) | Future multi-disease datasets | ⚠️ Empty placeholders |

**Data Flow:**
```
Raw Images → Dataset Class → DataModule → DataLoader → Model
              ↓                  ↓              ↓
        Augmentation      Train/Val Split  Batching
```

---

#### **📂 `src/models/` - Neural Network Architectures**

```
src/models/
├── __init__.py
├── cnn_backbone.py                [CLASS] Generic CNN feature extractor
├── multi_task_models.py           [CLASS] Multi-task model (shared backbone + multiple heads)
│
├── backbone/                      [BACKBONES] Feature extraction networks
│   └── resnet.py                          ResNetBackbone (ResNet18/50/101)
│
└── heads/                         [HEADS] Task-specific output layers
    ├── dr_binary.py                       Binary DR classification head (1 output)
    ├── dr_severity.py                     DR severity grading head (5 classes: 0-4)
    └── multi_label.py                     Multi-label disease classification head
```

**Architecture Pattern:**
```
Input Image
     ↓
┌──────────────┐
│   Backbone   │ ← Shared feature extractor (ResNet, EfficientNet)
│ (ResNet50)   │
└──────────────┘
     ↓ [Features: 2048-dim]
     ├─────────────┬─────────────┐
     ▼             ▼             ▼
┌─────────┐  ┌──────────┐  ┌─────────────┐
│ Binary  │  │ Severity │  │ Multi-Label │
│  Head   │  │   Head   │  │    Head     │
└─────────┘  └──────────┘  └─────────────┘
     │             │             │
     ▼             ▼             ▼
  DR/Normal    Grade 0-4    AMD/Cataract/etc.
```

**Key Files:**

| File | Purpose | Architecture |
|------|---------|--------------|
| `cnn_backbone.py` | Generic CNN wrapper | ResNet/EfficientNet/VGG |
| `multi_task_models.py` | Multi-task architecture | Shared backbone + 3 heads |
| `backbone/resnet.py` | ResNet variants | ResNet18/50/101/152 |
| `heads/dr_binary.py` | Binary classification | FC layer (2048 → 1 sigmoid) |
| `heads/dr_severity.py` | 5-class severity | FC layer (2048 → 5 softmax) |
| `heads/multi_label.py` | Multi-label (8 diseases) | FC layer (2048 → 8 sigmoid) |

**Design Principles:**
- **Modularity:** Swap backbones without changing heads
- **Pretrained Weights:** ImageNet initialization for transfer learning
- **Multi-Task Learning:** Share low-level features, specialize high-level heads

---

#### **📂 `src/training/` - Training & Evaluation**

```
src/training/
├── __init__.py
├── train_binary.py                [CLASS] BinaryTrainer for Phase-1 DR classification
├── evaluate.py                    [CLASS] BinaryEvaluator for Phase-1 evaluation
├── run_train.py                   [SCRIPT] Training orchestration script
└── run_eval.py                    [SCRIPT] Evaluation orchestration script
```

**Purpose:**
- **Trainer Classes:** Encapsulate training loops, validation, checkpointing
- **Evaluator Classes:** Compute metrics (accuracy, sensitivity, specificity, AUC)
- **Run Scripts:** CLI interfaces for training/evaluation

**Training Loop Structure:**
```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    metrics = evaluate(model, val_dataloader)
    
    # Checkpointing
    if metrics['auc'] > best_auc:
        save_checkpoint(model, 'best.pt')
```

**Key Metrics:**
- **Binary DR:** Accuracy, Sensitivity (Recall), Specificity, AUC-ROC
- **Severity:** Quadratic Weighted Kappa (QWK), Confusion Matrix
- **Multi-Label:** Per-disease AUC, F1-score, Hamming Loss

---

#### **📂 `src/losses/` - Custom Loss Functions**

```
src/losses/
├── masked_bce.py                  [LOSS] Masked Binary Cross-Entropy (ignore missing labels)
└── severity_loss.py               [LOSS] Ordinal regression loss for DR severity (0-4)
```

**Purpose:**
- **Masked BCE:** Handle multi-label datasets with missing annotations (ODIR, etc.)
- **Severity Loss:** Penalize severity prediction errors proportional to grade difference
  - Example: Predicting Grade 1 when truth is Grade 3 is worse than Grade 2 vs 3

**Severity Loss (Ordinal Regression):**
```python
# Traditional Cross-Entropy: Treats classes as independent
# Problem: Predicting 0 when truth is 4 is same penalty as 3 vs 4

# Ordinal Loss: Penalizes based on distance
# Penalty(pred=0, true=4) > Penalty(pred=3, true=4)
```

---

#### **📂 `src/metrics/` - Evaluation Metrics**

```
src/metrics/
├── __init__.py
└── metrics.py                     [METRICS] Medical imaging metrics (Sensitivity, Specificity, AUC, QWK)
```

**Medical Metrics:**

| Metric | Purpose | Formula |
|--------|---------|---------|
| **Sensitivity** (Recall) | Catch all DR cases | TP / (TP + FN) |
| **Specificity** | Avoid false alarms | TN / (TN + FP) |
| **AUC-ROC** | Overall discrimination | Area under ROC curve |
| **QWK** (Kappa) | Severity grading agreement | Weighted Cohen's Kappa |
| **F1-Score** | Balanced precision-recall | 2 × (P × R) / (P + R) |

**Why These Metrics?**
- In medical screening, **Sensitivity > Accuracy** (missing a DR case is worse than false alarm)
- **Specificity** balances to reduce unnecessary follow-ups
- **QWK** accounts for ordinal nature of severity grades

---

#### **📂 `src/utils/` - Shared Utilities**

```
src/utils/
├── __init__.py
├── checkpoints.py                 [UTILS] Save/load model checkpoints
├── logging.py                     [UTILS] Training logs (TensorBoard, CSV, console)
└── seed.py                        [UTILS] Set random seeds for reproducibility
```

**Purpose:**
- **Checkpointing:** Save model + optimizer state, resume training
- **Logging:** Track loss/metrics over time, visualize with TensorBoard
- **Seeding:** Ensure reproducible experiments (`torch.manual_seed`, `np.random.seed`)

**Reproducibility Setup:**
```python
from src.utils.seed import set_seed
set_seed(42)  # Sets seeds for Python, NumPy, PyTorch, CUDA
```

---

#### **📂 `src/explainability/` - Model Interpretability**

```
src/explainability/
└── gradcam.py                     [VIZ] Grad-CAM visualization for CNN models
```

**Purpose:**
- **Grad-CAM:** Generate heatmaps showing which image regions influence predictions
- **Clinical Trust:** Verify model looks at optic disc, retina (not image borders, artifacts)

**Example Output:**
```
Original Image → Model → Grad-CAM Heatmap (red = high attention)
[Shows model focuses on hemorrhages, exudates in DR images]
```

---

#### **📂 `src/training phase 3(multi-model)/` - Phase-3 Multi-Model Training**

```
src/training phase 3(multi-model)/
├── train_stage1.py                [TRAINER] Stage-1: Train DR binary head only
├── train_stage2.py                [TRAINER] Stage-2: Train DR severity head (freeze binary)
├── train_stage3.py                [TRAINER] Stage-3: Joint training (all heads)
└── trainer.py                     [CLASS] Unified trainer for multi-task models
```

**Multi-Stage Training Strategy:**

| Stage | Goal | Frozen Layers | Active Heads |
|-------|------|---------------|--------------|
| **Stage 1** | Binary DR detection | None | Binary head |
| **Stage 2** | DR severity grading | Backbone + Binary head | Severity head |
| **Stage 3** | Joint optimization | None (fine-tune all) | All heads |

**Why Staged Training?**
1. **Stage 1:** Learn discriminative features for DR vs Normal
2. **Stage 2:** Learn severity features without forgetting binary task
3. **Stage 3:** Fine-tune end-to-end for optimal multi-task performance

---

#### **📂 `src/evaluation phase 3(multi-model)/` - Phase-3 Evaluation**

```
src/evaluation phase 3(multi-model)/
├── dr_binary_meterics.py          [EVAL] Binary DR metrics (Sensitivity, Specificity, AUC)
├── severity_metrics.py            [EVAL] Severity grading metrics (QWK, confusion matrix)
└── multilabel_metrics.py          [EVAL] Multi-label disease metrics (per-class AUC, F1)
```

**Purpose:**
- Compute task-specific metrics for multi-task model
- Generate confusion matrices, ROC curves, calibration plots
- Ensure no catastrophic forgetting (binary performance drops when adding severity head)

---

## 🔄 Complete Training Pipeline Flow

### **Phase 1: Binary DR Classification**

```
┌─────────────────────────────────────────────────────┐
│              PHASE 1: BINARY DR DETECTION           │
└─────────────────────────────────────────────────────┘

1. Data Preparation
   └─ Run: scripts/preprocessing/preprocess_eyepacs.py
   └─ Run: scripts/preprocessing/split_train_val_test.py
   └─ Output: Data/splits/fundus/eyepacs/{train,val}
             Data/splits/fundus/aptos/test

2. Training
   └─ Run: python train.py --data_root Data/splits/fundus --epochs 50
   └─ Uses: src/data/datamodule.py (FundusDataModule)
   └─ Model: ResNet18/50 (from torchvision)
   └─ Output: models/binary_dr/best.pt

3. Evaluation
   └─ Run: python evaluate.py --checkpoint models/binary_dr/best.pt
   └─ Metrics: Accuracy, Sensitivity, Specificity, AUC
   └─ Output: Confusion matrix, ROC curve
```

### **Phase 2: DR Severity Grading**

```
┌─────────────────────────────────────────────────────┐
│            PHASE 2: DR SEVERITY GRADING             │
└─────────────────────────────────────────────────────┘

1. Data Preparation
   └─ Use: Data/labels/eyepacs_trainLabels.csv (severity 0-4)
   └─ DataModule: src/data/eyepacs_severity_datamodule.py

2. Training
   └─ Run: src/training phase 3(multi-model)/train_stage2.py
   └─ Model: Multi-task model (binary head + severity head)
   └─ Strategy: Freeze binary head, train severity head only

3. Evaluation
   └─ Run: src/evaluation phase 3(multi-model)/severity_metrics.py
   └─ Metrics: Quadratic Weighted Kappa (QWK), Per-class accuracy
```

### **Phase 3: Multi-Task Learning**

```
┌─────────────────────────────────────────────────────┐
│         PHASE 3: MULTI-TASK JOINT TRAINING          │
└─────────────────────────────────────────────────────┘

1. Architecture
   └─ Backbone: ResNet50 (shared)
   └─ Heads: 
      ├─ DR Binary (2 classes)
      ├─ DR Severity (5 classes: 0-4)
      └─ Multi-label (8 diseases: AMD, Cataract, etc.)

2. Training Strategy
   └─ Stage 1: Train binary head
   └─ Stage 2: Add severity head (freeze binary)
   └─ Stage 3: Add multi-label head, fine-tune all

3. Loss Function
   └─ Total Loss = λ₁·BinaryLoss + λ₂·SeverityLoss + λ₃·MultiLabelLoss
   └─ Weights: λ₁=1.0, λ₂=0.5, λ₃=0.3 (tunable)

4. Evaluation
   └─ Binary: AUC, Sensitivity, Specificity
   └─ Severity: QWK, Confusion Matrix
   └─ Multi-label: Per-disease AUC, Hamming Loss
```

---

## ⚠️ Empty Files & Placeholders (Ready for Implementation)

### **Configuration Files (Empty)**
- `configs/model.yaml` ❌
- `configs/paths.yaml` ❌
- `configs/train.yaml` ❌
- `src/configs/base.yml` ❌
- `src/configs/stage1_dr.yaml` ❌
- `src/configs/stage2_mutlilabel.yml` ❌
- `src/configs/stage3_joint.yaml` ❌

**Next Steps:**
- Populate YAML configs with hyperparameters (learning rate, batch size, etc.)
- Enable easy experiment tracking via config files

### **Dataset Classes (Empty Placeholders)**
- `src/data/datasets/eyepacs.py` ❌
- `src/data/datasets/aptos.py` ❌
- `src/data/datasets/amd.py` ❌
- `src/data/datasets/cataract.py` ❌
- `src/data/datasets/odir.py` ❌

**Next Steps:**
- Implement PyTorch Dataset classes for each disease type
- Enable multi-dataset training for robust generalization

---

## 🎯 Key Design Principles

### **1. Modularity**
```
✅ Each component is self-contained (datasets, models, trainers)
✅ Swap backbones without changing training code
✅ Add new datasets by implementing Dataset class
```

### **2. Reproducibility**
```
✅ Frozen train/val/test splits (Data/splits/)
✅ Random seed control (src/utils/seed.py)
✅ Model checkpoints saved with optimizer state
✅ Training history logged in JSON
```

### **3. Medical ML Best Practices**
```
✅ External validation (APTOS test set, different source)
✅ Sensitivity > Accuracy (catch DR cases is priority)
✅ Grad-CAM for model interpretability
✅ Ordinal loss for severity grading (respects class ordering)
```

### **4. Scalability**
```
✅ Multi-task learning architecture (shared backbone)
✅ Staged training for incremental learning
✅ Placeholder datasets for future diseases (AMD, Cataract, ODIR)
✅ CLI scripts for easy automation
```

---

## 📊 Dataset Summary

| Dataset | Purpose | Images | Classes | Notes |
|---------|---------|--------|---------|-------|
| **EyePACS** | Train/Val | ~35k | Binary (DR/Normal) + Severity (0-4) | Kaggle DR Competition |
| **APTOS** | External Test | ~3.6k | Binary + Severity | APTOS 2019 Blindness Detection |
| **MESSIDOR** | Analysis | ~1.2k | Multi-grade | `Data/raw/all-images/` |
| **AMD** | Future | TBD | AMD detection | Placeholder |
| **Cataract** | Future | TBD | Cataract detection | External eye images |
| **ODIR** | Future | ~8k | 8 diseases (multi-label) | ODIR-5K Challenge |

---

## 🚀 Quick Start Guide

### **1. Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

### **2. Phase-1 Training (Binary DR)**
```bash
# Prepare data (if not already split)
python scripts/preprocessing/preprocess_eyepacs.py
python scripts/spliteyepacs.py

# Train binary DR classifier
python train.py --data_root Data/splits/fundus \
                --model resnet18 \
                --epochs 50 \
                --batch_size 32

# Evaluate on APTOS test set
python evaluate.py --checkpoint models/binary_dr/best.pt \
                   --data_root Data/splits/fundus
```

### **3. Phase-2 Training (Severity Grading)**
```bash
# Train severity grading model
python src/training\ phase\ 3\(multi-model\)/train_stage2.py

# Evaluate severity predictions
python src/evaluation\ phase\ 3\(multi-model\)/severity_metrics.py
```

### **4. Grad-CAM Visualization**
```python
from src.explainability.gradcam import GradCAM
import torch

model = torch.load('models/binary_dr/best.pt')
gradcam = GradCAM(model)
heatmap = gradcam.generate(image)
gradcam.visualize(image, heatmap)
```

---

## 📈 Future Roadmap

### **Immediate Next Steps**
1. ✅ Populate YAML config files
2. ✅ Implement missing dataset classes (APTOS, AMD, Cataract, ODIR)
3. ✅ Complete Phase-3 multi-task training pipeline
4. ✅ Add ensemble methods (model averaging)

### **Advanced Features**
- ⏳ Real-time inference API (FastAPI/Streamlit)
- ⏳ Model compression (pruning, quantization for mobile deployment)
- ⏳ Active learning for labeling efficiency
- ⏳ Federated learning for multi-hospital deployment

---

## 📚 References & Standards

- **Medical AI Guidelines:** FDA Software as Medical Device (SaMD)
- **DR Grading Standard:** ICDR (International Clinical Diabetic Retinopathy) Scale
- **Explainability:** Grad-CAM (Selvaraju et al., 2017)
- **Ordinal Regression:** CORAL Loss (Cao et al., 2020)

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | Feb 2026 | Initial architecture documentation |

---

**Maintained by:** EYE-ASSISST Development Team  
**Last Updated:** February 8, 2026
