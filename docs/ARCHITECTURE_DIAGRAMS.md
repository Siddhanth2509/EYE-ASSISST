# 📊 EYE-ASSISST - Visual Architecture Diagrams

> **Visual representation of system architecture, data flow, and model design**

---

## 🎨 System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         EYE-ASSISST PLATFORM                                      │
│           AI-Powered Eye Disease Detection & Clinical Decision Support            │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PROJECT ROOT                                        │
│                         eye-realtime-inference/                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐           ┌───────────────────┐          ┌──────────────────┐
│   SCRIPTS     │           │       DATA        │          │      MODELS      │
│ Preprocessing │           │   Raw/Processed   │          │   Checkpoints    │
│ & Data Prep   │           │   Splits/Labels   │          │  Training Logs   │
└───────────────┘           └───────────────────┘          └──────────────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SRC/ (CORE PIPELINE)                                │
│                                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   DATA   │→ │  MODELS  │→ │ TRAINING │→ │  LOSSES  │→ │ METRICS  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│       ↓             ↓              ↓             ↓             ↓               │
│  Datasets    Backbones      Trainers      Custom Loss   Medical Metrics        │
│  Transforms   Heads       Evaluators     Functions         (AUC, QWK)         │
│                                                                                  │
│  ┌──────────────────┐        ┌──────────────────┐                              │
│  │ EXPLAINABILITY   │        │     UTILS        │                              │
│  │   (Grad-CAM)     │        │  Checkpoints     │                              │
│  └──────────────────┘        │  Logging, Seeds  │                              │
│                               └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
           ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
           │   PHASE 1      │  │   PHASE 2      │  │   PHASE 3      │
           │   Binary DR    │  │  DR Severity   │  │  Multi-Task    │
           │ Classification │  │   Grading      │  │   Learning     │
           └────────────────┘  └────────────────┘  └────────────────┘
```

---

## 📦 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: RAW DATA (Data/raw/)                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
│
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  │  EyePACS   │  │   APTOS    │  │ MESSIDOR   │  │ AMD/Others │
│  │   ~35k     │  │   ~3.6k    │  │   ~1.2k    │  │    TBD     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘
│
└──────────────────────┬────────────────────────────────────────────────────────
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: PREPROCESSING (scripts/preprocessing/)                               │
│                                                                               │
│  ✓ Image quality filtering                                                   │
│  ✓ Resize to 224×224 or 512×512                                             │
│  ✓ Normalize pixel values                                                    │
│  ✓ Remove duplicates/corrupted files                                         │
│  ✓ Crop black borders                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CLEANED DATA (Data/cleaned/)                                         │
└──────────────────────────────────────────────────────────────────────────────┘
│
│  fundus/                           external/
│  ├── DR/       (DR-positive)       ├── cataract/
│  └── NORMAL/   (Healthy)           └── others/
│
└──────────────────────┬────────────────────────────────────────────────────────
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: PROCESSED DATA (Data/processed/)                                     │
│                                                                               │
│  fundus/                                                                      │
│  ├── eyepacs/  (Standardized format)                                        │
│  └── aptos/    (Ready for model input)                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: TRAIN/VAL/TEST SPLITS (Data/splits/)  ⭐ FROZEN                     │
│                                                                               │
│  fundus/                                                                      │
│  ├── eyepacs/                                                                │
│  │   ├── train/  (70%)  → DR/, NORMAL/                                     │
│  │   └── val/    (15%)  → DR/, NORMAL/                                     │
│  └── aptos/                                                                  │
│      └── test/   (15%)  → DR/, NORMAL/  (External validation)              │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: DATALOADERS (src/data/)                                              │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ DataModule → Dataset → Transforms → DataLoader              │            │
│  │      ↓          ↓          ↓             ↓                   │            │
│  │   Setup    Load Image  Augment      Batch + Shuffle         │            │
│  └─────────────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
                 [READY FOR TRAINING]
```

---

## 🧠 Model Architecture (Multi-Task Learning)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MULTI-TASK MODEL ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT: Fundus Image (3 × 224 × 224)
         │
         │  [Batch of 32 images]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SHARED BACKBONE                                   │
│                 (Feature Extractor - ResNet50)                           │
│                                                                           │
│  Conv1 (7×7, 64) → BatchNorm → ReLU → MaxPool                           │
│         ↓                                                                │
│  ResBlock (Layer 1) - 64 channels                                       │
│         ↓                                                                │
│  ResBlock (Layer 2) - 128 channels                                      │
│         ↓                                                                │
│  ResBlock (Layer 3) - 256 channels                                      │
│         ↓                                                                │
│  ResBlock (Layer 4) - 512 channels                                      │
│         ↓                                                                │
│  Global Average Pooling                                                  │
│         ↓                                                                │
│  Feature Vector: 2048-dimensional                                        │
└─────────────────────────────────────────────────────────────────────────┘
         │
         │  [Shared features for all tasks]
         │
         ├──────────────────────┬──────────────────────┬─────────────────────┐
         │                      │                      │                     │
         ▼                      ▼                      ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐
│  BINARY HEAD    │  │  SEVERITY HEAD   │  │ MULTI-LABEL HEAD │  │   FUTURE    │
│  (DR/Normal)    │  │   (Grades 0-4)   │  │  (8 Diseases)    │  │    HEADS    │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └─────────────┘
│                      │                      │
│  FC(2048 → 512)     │  FC(2048 → 512)     │  FC(2048 → 512)
│  ReLU                │  ReLU                │  ReLU
│  Dropout(0.5)        │  Dropout(0.5)        │  Dropout(0.3)
│  FC(512 → 1)         │  FC(512 → 5)         │  FC(512 → 8)
│  Sigmoid             │  Softmax             │  Sigmoid (per class)
│                      │                      │
▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   PREDICTION     │  │   PREDICTION     │  │   PREDICTION     │
│                  │  │                  │  │                  │
│  P(DR) ∈ [0,1]  │  │ Grade ∈ {0,1,2,3,4} │  │ [AMD: 0.8,      │
│                  │  │                  │  │  Cataract: 0.1,  │
│  Threshold: 0.5  │  │  Ordinal scale   │  │  Glaucoma: 0.3,  │
│  ↓               │  │  ↓               │  │  ...etc.]        │
│  "DR Detected"   │  │  "Grade 2 (Mild)"│  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘

LOSS:
Total Loss = λ₁·BCE(Binary) + λ₂·CE(Severity) + λ₃·BCE(MultiLabel)
             ↑                ↑                 ↑
        Weights: 1.0          0.5               0.3 (tunable)
```

---

## 🔄 Training Pipeline (3-Stage Strategy)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STAGED TRAINING STRATEGY                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Binary DR Detection                                                  │
└───────────────────────────────────────────────────────────────────────────────┘
│
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  │   Backbone   │ → → │ Binary Head  │ → → │   Output     │
│  │  (ResNet50)  │     │  (Trainable) │     │  DR/Normal   │
│  │  Pretrained  │     └──────────────┘     └──────────────┘
│  └──────────────┘
│        ↓
│  Loss: BCE(Binary)
│  Metric: AUC, Sensitivity, Specificity
│  Duration: 30 epochs
│
└───────────────────────┬───────────────────────────────────────────────────────
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DR Severity Grading                                                  │
└───────────────────────────────────────────────────────────────────────────────┘
│
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  │   Backbone   │ → → │ Binary Head  │     │   Output     │
│  │  (Frozen)    │  ┐  │  (Frozen)    │     │  DR/Normal   │
│  └──────────────┘  │  └──────────────┘     └──────────────┘
│                    │
│                    └─▶ ┌──────────────┐     ┌──────────────┐
│                        │ Severity Head│ → → │   Output     │
│                        │ (Trainable)  │     │ Grades 0-4   │
│                        └──────────────┘     └──────────────┘
│        ↓
│  Loss: Ordinal Regression Loss (Grade-aware)
│  Metric: Quadratic Weighted Kappa (QWK)
│  Duration: 25 epochs
│
│  Stage-2 Training Variants (as of Mar 2026):
│    • ResNet18 baseline          ← original
│    • ResNet18 fine-tune (E3+)   ← extended run with lower LR
│    • EfficientNet-B3 @384px     ← high-resolution DR severity experiment
│
└───────────────────────┬───────────────────────────────────────────────────────
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Joint Multi-Task Fine-Tuning                                         │
└───────────────────────────────────────────────────────────────────────────────┘
│
│  ┌──────────────┐
│  │   Backbone   │ ────┬──▶ Binary Head    → DR/Normal
│  │  (Fine-tune) │     │
│  └──────────────┘     ├──▶ Severity Head  → Grades 0-4
│                       │
│                       └──▶ Multi-label Head → 8 Diseases
│        ↓
│  Loss: λ₁·BCE + λ₂·Severity + λ₃·MultiLabel
│  Metric: Combined (AUC, QWK, Per-disease F1)
│  Duration: 20 epochs (early stopping)
│
└───────────────────────────────────────────────────────────────────────────────
```

---

## 📊 Evaluation Metrics Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION FRAMEWORK                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

Model Predictions
       │
       ├──────────────────────────────────────────────────────────────┐
       │                                │                              │
       ▼                                ▼                              ▼
┌──────────────────┐         ┌──────────────────┐          ┌──────────────────┐
│  BINARY TASK     │         │  SEVERITY TASK   │          │ MULTI-LABEL TASK │
│  (DR Detection)  │         │  (Grades 0-4)    │          │ (8 Diseases)     │
└──────────────────┘         └──────────────────┘          └──────────────────┘
       │                                │                              │
       ▼                                ▼                              ▼
┌──────────────────┐         ┌──────────────────┐          ┌──────────────────┐
│ Binary Metrics   │         │ Ordinal Metrics  │          │ Multi-label      │
│                  │         │                  │          │ Metrics          │
│ • Accuracy       │         │ • QWK (Kappa)    │          │ • Per-class AUC  │
│ • Sensitivity    │         │ • MAE            │          │ • Hamming Loss   │
│   (Recall)       │         │ • Confusion      │          │ • Subset Accuracy│
│ • Specificity    │         │   Matrix         │          │ • Micro/Macro F1 │
│ • AUC-ROC        │         │ • Per-class Acc  │          │                  │
│ • F1-Score       │         │                  │          │                  │
│ • Confusion      │         │                  │          │                  │
│   Matrix         │         │                  │          │                  │
└──────────────────┘         └──────────────────┘          └──────────────────┘
       │                                │                              │
       └────────────────────────────────┼──────────────────────────────┘
                                        ▼
                           ┌──────────────────────────┐
                           │  VISUALIZATION           │
                           │                          │
                           │ • ROC Curves             │
                           │ • Confusion Matrices     │
                           │ • Calibration Plots      │
                           │ • Grad-CAM Heatmaps      │
                           │ • Per-class Performance  │
                           └──────────────────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │  CLINICAL INTERPRETATION │
                           │                          │
                           │ ✓ High Sensitivity:      │
                           │   Catch DR cases         │
                           │ ✓ Balanced Specificity:  │
                           │   Reduce false alarms    │
                           │ ✓ Grad-CAM:              │
                           │   Verify medical focus   │
                           └──────────────────────────┘
```

---

## 🔍 Grad-CAM Explainability Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      GRAD-CAM INTERPRETABILITY                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

Original Image                  Model Forward Pass              Class Score
┌──────────────┐                                               ┌──────────┐
│              │                                               │  P(DR)   │
│   [Fundus]   │ ─────────────────────────────────────────▶  │  = 0.92  │
│              │               CNN Layers                      └──────────┘
└──────────────┘                                                     │
                                                                     │
                                                              Backward Pass
                                                                     ▼
                                                        ┌──────────────────────┐
                                                        │  Gradients w.r.t.    │
                                                        │  Feature Maps        │
                                                        │  (Last Conv Layer)   │
                                                        └──────────────────────┘
                                                                     │
                                                                     ▼
                                                        ┌──────────────────────┐
                                                        │  Weighted Sum        │
                                                        │  (Importance Map)    │
                                                        └──────────────────────┘
                                                                     │
                                                                     ▼
                                                        ┌──────────────────────┐
                                                        │  Upsample to         │
                                                        │  Input Size          │
                                                        └──────────────────────┘
                                                                     │
                                                                     ▼
Heatmap Overlay                                         ┌──────────────────────┐
┌──────────────┐                                       │  Grad-CAM Heatmap    │
│ 🔴🔴🔴       │ ◀──────────────────────────────────── │  (Red = Important)   │
│  🔵🔵🔵 🔴🔴  │                Apply Colormap          └──────────────────────┘
│   🔵🔵  🔴   │                (Jet/Viridis)
│    🔵🔵🔴🔴   │                       Clinical Interpretation:
└──────────────┘                       ✓ Red regions → Hemorrhages, Exudates
                                       ✓ Focus on retina (not borders)
                                       ✓ Medically plausible attention
```

---

## 🗂️ Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCIES                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

train.py / evaluate.py  (Entry Points)
    │
    ├───▶ src.data.datamodule
    │          │
    │          ├───▶ src.data.datasets.eyepacs_severity
    │          └───▶ src.data.transforms
    │
    ├───▶ src.models.multi_task_models
    │          │
    │          ├───▶ src.models.backbone.resnet
    │          ├───▶ src.models.heads.dr_binary
    │          ├───▶ src.models.heads.dr_severity
    │          └───▶ src.models.heads.multi_label
    │
    ├───▶ src.training.train_binary
    │          │
    │          ├───▶ src.losses.masked_bce
    │          ├───▶ src.losses.severity_loss
    │          └───▶ src.metrics.metrics
    │
    └───▶ src.utils
               ├───▶ checkpoints
               ├───▶ logging
               └───▶ seed

External Dependencies:
    ├───▶ PyTorch (torch, torchvision)
    ├───▶ NumPy, Pandas
    ├───▶ OpenCV, Pillow
    └───▶ Matplotlib, Seaborn
```

---

## 📁 File Status Legend

```
✅ Active & Complete    - Fully implemented and tested
🔄 Work in Progress    - Partially implemented
⚠️  Placeholder/Empty  - Ready for implementation
❌ Deprecated          - No longer used
📝 Documentation       - MD/text files
```

---

## 🎯 Key Architectural Decisions

### **1. Multi-Task Learning**
```
Why: Share low-level features (edges, textures) across tasks
Benefit: 
  • Reduces total parameters
  • Improves generalization
  • Enables transfer learning between related tasks
```

### **2. Staged Training**
```
Why: Prevent catastrophic forgetting
Benefit:
  • Stage 1: Learn robust binary features
  • Stage 2: Add severity without forgetting binary
  • Stage 3: Fine-tune everything jointly
```

### **3. External Validation (APTOS)**
```
Why: Avoid dataset-specific overfitting
Benefit:
  • APTOS has different patient demographics
  • Different image acquisition (cameras, lighting)
  • Real-world generalization test
```

### **4. Frozen Splits**
```
Why: Reproducibility & fair comparison
Benefit:
  • Same train/val/test across all experiments
  • Can compare model improvements reliably
  • Prevents data leakage
```

### **5. Medical-First Metrics**
```
Why: Clinical relevance > Accuracy
Metric Priority:
  1. Sensitivity (catch all DR cases)
  2. AUC (overall discrimination)
  3. Specificity (reduce false alarms)
  4. Accuracy (least important)
```

---

**Created:** February 8, 2026  
**Last Updated:** March 7, 2026  
**Purpose:** Visual reference for system architecture  
**Maintained by:** EYE-ASSISST Development Team
