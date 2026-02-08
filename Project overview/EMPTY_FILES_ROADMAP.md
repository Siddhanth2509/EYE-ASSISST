# ⚠️ Empty Files & Implementation Roadmap

> **Complete list of empty/placeholder files and their implementation priority**

---

## 📋 Summary

**Total Empty Files Found: 12**

| Category | Count | Priority |
|----------|-------|----------|
| Config Files | 7 | 🔴 High |
| Dataset Classes | 5 | 🟡 Medium |

---

## 🔴 HIGH PRIORITY: Configuration Files (7 files)

### **Root Level Configs (3 files)**

#### 1. `configs/model.yaml` ❌ EMPTY
**Purpose:** Model architecture hyperparameters  
**Should contain:**
```yaml
# Model Architecture
backbone: "resnet50"
pretrained: true
freeze_backbone: false

# Head configurations
binary_head:
  hidden_dim: 512
  dropout: 0.5

severity_head:
  hidden_dim: 512
  dropout: 0.5
  num_classes: 5

multilabel_head:
  hidden_dim: 512
  dropout: 0.3
  num_classes: 8
  diseases: ["AMD", "Cataract", "Glaucoma", "DR", "Hypertension", "Myopia", "Others", "Normal"]
```

**Usage:**
```python
import yaml
config = yaml.safe_load(open('configs/model.yaml'))
model = MultiTaskModel(backbone=config['backbone'])
```

---

#### 2. `configs/paths.yaml` ❌ EMPTY
**Purpose:** Data paths configuration  
**Should contain:**
```yaml
# Data directories
data_root: "Data"
raw_data: "Data/raw"
processed_data: "Data/processed"
splits: "Data/splits/fundus"

# Dataset paths
eyepacs:
  train: "Data/splits/fundus/eyepacs/train"
  val: "Data/splits/fundus/eyepacs/val"
  labels: "Data/labels/eyepacs_trainLabels.csv"

aptos:
  test: "Data/splits/fundus/aptos/test"
  labels: "Data/labels/aptos_test.csv"

# Model checkpoints
checkpoint_dir: "models"
logs_dir: "logs"
```

**Usage:**
```python
import yaml
paths = yaml.safe_load(open('configs/paths.yaml'))
data_root = paths['splits']
```

---

#### 3. `configs/train.yaml` ❌ EMPTY
**Purpose:** Training hyperparameters  
**Should contain:**
```yaml
# Training settings
epochs: 50
batch_size: 32
num_workers: 4
pin_memory: true

# Optimizer
optimizer: "Adam"
learning_rate: 0.0001
weight_decay: 0.0001

# Scheduler
scheduler: "ReduceLROnPlateau"
patience: 5
factor: 0.5
min_lr: 0.000001

# Early stopping
early_stopping: true
early_stopping_patience: 10

# Image settings
image_size: 224

# Loss weights (multi-task)
loss_weights:
  binary: 1.0
  severity: 0.5
  multilabel: 0.3

# Augmentation
augmentation:
  random_rotation: 15
  random_flip: 0.5
  color_jitter:
    brightness: 0.2
    contrast: 0.2
```

**Usage:**
```python
import yaml
train_cfg = yaml.safe_load(open('configs/train.yaml'))
lr = train_cfg['learning_rate']
```

---

### **Stage-Specific Configs (4 files)**

#### 4. `src/configs/base.yml` ❌ EMPTY
**Purpose:** Base configuration inherited by all stages  
**Should contain:**
```yaml
# Base configuration for all training stages
seed: 42

# Device
device: "cuda"
mixed_precision: true

# Logging
log_interval: 10
save_interval: 5

# Checkpointing
save_best: true
save_latest: true

# Metrics
primary_metric: "auc"
maximize_metric: true
```

---

#### 5. `src/configs/stage1_dr.yaml` ❌ EMPTY
**Purpose:** Stage-1 binary DR training config  
**Should contain:**
```yaml
# Inherits from base.yml
base_config: "src/configs/base.yml"

# Stage-1 specific
stage: 1
task: "binary_dr"

# Model
freeze_layers: []
active_heads: ["binary"]

# Training
epochs: 30
learning_rate: 0.0001

# Data
datasets:
  train: "Data/splits/fundus/eyepacs/train"
  val: "Data/splits/fundus/eyepacs/val"

# Loss
loss_function: "BCEWithLogitsLoss"
class_weights: true
```

---

#### 6. `src/configs/stage2_mutlilabel.yml` ❌ EMPTY
**Purpose:** Stage-2 severity + multi-label training config  
**Should contain:**
```yaml
# Inherits from base.yml
base_config: "src/configs/base.yml"

# Stage-2 specific
stage: 2
task: "severity_multilabel"

# Model
freeze_layers: ["backbone", "binary_head"]
active_heads: ["severity", "multilabel"]

# Training
epochs: 25
learning_rate: 0.00005

# Load pretrained
load_checkpoint: "models/stage1_dr_binary/best.pt"
freeze_pretrained: true

# Data
datasets:
  train: "Data/splits/fundus/eyepacs/train"
  val: "Data/splits/fundus/eyepacs/val"
  labels: "Data/labels/eyepacs_trainLabels.csv"

# Loss
severity_loss: "OrdinalRegressionLoss"
multilabel_loss: "MaskedBCELoss"
```

---

#### 7. `src/configs/stage3_joint.yaml` ❌ EMPTY
**Purpose:** Stage-3 joint fine-tuning config  
**Should contain:**
```yaml
# Inherits from base.yml
base_config: "src/configs/base.yml"

# Stage-3 specific
stage: 3
task: "joint_multitask"

# Model
freeze_layers: []  # Fine-tune all
active_heads: ["binary", "severity", "multilabel"]

# Training
epochs: 20
learning_rate: 0.00001  # Lower LR for fine-tuning

# Load pretrained
load_checkpoint: "models/stage2_multilabel/best.pt"

# Early stopping (prevent overfitting)
early_stopping: true
patience: 8

# Loss weights
loss_weights:
  binary: 1.0
  severity: 0.5
  multilabel: 0.3
```

---

## 🟡 MEDIUM PRIORITY: Dataset Classes (5 files)

### **Dataset Implementation Roadmap**

---

#### 8. `src/data/datasets/eyepacs.py` ❌ EMPTY
**Purpose:** Binary EyePACS dataset (DR/Normal)  
**Priority:** 🟡 Medium (currently using ImageFolder)  
**Should implement:**
```python
class EyePACSDataset(Dataset):
    """
    Binary DR classification dataset.
    
    Returns:
        image: Tensor (3, H, W)
        label: 0 (Normal) or 1 (DR)
    """
    def __init__(self, images_dir, transform=None):
        pass
    
    def __getitem__(self, idx):
        pass
```

**Why implement:**
- Custom preprocessing specific to EyePACS
- Better control over label mapping
- Add metadata (patient ID, image quality scores)

---

#### 9. `src/data/datasets/aptos.py` ❌ EMPTY
**Purpose:** APTOS 2019 dataset (DR severity 0-4)  
**Priority:** 🟡 Medium (used for test set)  
**Should implement:**
```python
class APTOSDataset(Dataset):
    """
    APTOS 2019 Blindness Detection dataset.
    
    Returns:
        image: Tensor (3, H, W)
        dr_label: 0 (Normal) or 1 (DR)
        severity_label: 0-4 (DR grade)
    """
    def __init__(self, images_dir, labels_csv, transform=None):
        pass
    
    def __getitem__(self, idx):
        pass
```

**Why implement:**
- Different preprocessing than EyePACS (different image sources)
- Add test-time augmentation (TTA) support
- Handle APTOS-specific quality issues

---

#### 10. `src/data/datasets/amd.py` ❌ EMPTY
**Purpose:** Age-related Macular Degeneration dataset  
**Priority:** 🟢 Low (future multi-disease extension)  
**Should implement:**
```python
class AMDDataset(Dataset):
    """
    AMD detection dataset.
    
    Returns:
        image: Tensor (3, H, W)
        amd_label: 0 (Normal) or 1 (AMD)
        severity: 0 (None), 1 (Early), 2 (Intermediate), 3 (Advanced)
    """
```

**Why implement later:**
- Expand beyond DR to other retinal diseases
- Multi-label classification (can have DR + AMD)
- Requires separate AMD annotations

---

#### 11. `src/data/datasets/cataract.py` ❌ EMPTY
**Purpose:** Cataract detection dataset (external eye images)  
**Priority:** 🟢 Low (future external eye disease extension)  
**Should implement:**
```python
class CataractDataset(Dataset):
    """
    Cataract detection from external eye images.
    
    Note: Different preprocessing than fundus images.
    
    Returns:
        image: Tensor (3, H, W)
        cataract_label: 0 (Normal) or 1 (Cataract)
        severity: 0-3 (Cataract grade)
    """
```

**Why implement later:**
- Different image modality (external vs fundus)
- Different preprocessing pipeline
- Requires separate model branch or new model

---

#### 12. `src/data/datasets/odir.py` ❌ EMPTY
**Purpose:** ODIR-5K multi-label dataset (8 diseases)  
**Priority:** 🟡 Medium (important for multi-label head)  
**Should implement:**
```python
class ODIRDataset(Dataset):
    """
    ODIR-5K multi-label dataset.
    
    8 diseases: Normal, DR, Glaucoma, Cataract, AMD, 
                Hypertension, Myopia, Others
    
    Returns:
        image: Tensor (3, H, W)
        labels: Tensor (8,) - multi-hot encoding
        masks: Tensor (8,) - 1 if annotated, 0 if missing
    """
    def __init__(self, images_dir, labels_csv, transform=None):
        pass
    
    def __getitem__(self, idx):
        # Handle missing labels with masks
        pass
```

**Why implement:**
- Critical for multi-label training
- Teaches model to recognize multiple diseases simultaneously
- Uses masked BCE loss (some annotations missing)

---

## 📊 Implementation Priority Matrix

| File | Priority | Effort | Impact | Deadline |
|------|----------|--------|--------|----------|
| `configs/train.yaml` | 🔴 High | Low | High | Week 1 |
| `configs/paths.yaml` | 🔴 High | Low | High | Week 1 |
| `configs/model.yaml` | 🔴 High | Low | Medium | Week 1 |
| `src/configs/base.yml` | 🔴 High | Low | High | Week 1 |
| `src/configs/stage1_dr.yaml` | 🔴 High | Low | High | Week 2 |
| `src/configs/stage2_mutlilabel.yml` | 🔴 High | Medium | High | Week 2 |
| `src/configs/stage3_joint.yaml` | 🔴 High | Medium | High | Week 3 |
| `src/data/datasets/odir.py` | 🟡 Medium | High | High | Week 4 |
| `src/data/datasets/aptos.py` | 🟡 Medium | Medium | Medium | Week 4 |
| `src/data/datasets/eyepacs.py` | 🟡 Medium | Medium | Low | Week 5 |
| `src/data/datasets/amd.py` | 🟢 Low | High | Low | Future |
| `src/data/datasets/cataract.py` | 🟢 Low | High | Low | Future |

---

## 🚀 Quick Implementation Steps

### **Week 1: Core Configs**
```bash
# Create all YAML configs with basic hyperparameters
1. Copy templates from this document
2. Adjust paths to match your system
3. Test loading configs in Python
```

### **Week 2-3: Stage Configs**
```bash
# Implement stage-specific configs
1. Start with stage1_dr.yaml (simplest)
2. Add stage2/stage3 configs
3. Test staged training pipeline
```

### **Week 4: Dataset Classes**
```bash
# Implement critical datasets
1. ODIR (multi-label head needs this)
2. APTOS (external validation)
3. Test dataloaders work correctly
```

### **Week 5+: Optional Datasets**
```bash
# Implement remaining datasets as needed
1. EyePACS custom class (if ImageFolder insufficient)
2. AMD (multi-disease expansion)
3. Cataract (external eye images)
```

---

## ✅ Validation Checklist

### **Config Files**
- [ ] All YAML files parse without errors
- [ ] Paths exist on disk
- [ ] Hyperparameters are reasonable (LR not too high/low)
- [ ] Loss weights sum to reasonable value
- [ ] Config inheritance works (base.yml → stage configs)

### **Dataset Classes**
- [ ] `__len__` returns correct count
- [ ] `__getitem__` returns tensors with correct shapes
- [ ] Labels match expected format (int for classification, float for regression)
- [ ] Transforms applied correctly
- [ ] No data leakage (train/val/test separation)
- [ ] Handle missing files gracefully (skip with warning)

---

## 🔧 Testing Commands

### **Test Config Loading**
```python
import yaml

# Test individual configs
config = yaml.safe_load(open('configs/train.yaml'))
print(config)

# Test config merging (base + stage)
base = yaml.safe_load(open('src/configs/base.yml'))
stage1 = yaml.safe_load(open('src/configs/stage1_dr.yaml'))
# Merge logic here
```

### **Test Dataset Classes**
```python
from src.data.datasets.odir import ODIRDataset
from torch.utils.data import DataLoader

# Test dataset
dataset = ODIRDataset(
    images_dir='Data/raw/ODIR Dataset',
    labels_csv='Data/labels/odir_labels.csv'
)

print(f"Dataset size: {len(dataset)}")
image, labels, masks = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Labels: {labels}")
print(f"Masks: {masks}")

# Test dataloader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
print(f"Batch shapes: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")
```

---

## 📝 Notes

### **Why Empty Files Exist**
- **Placeholder Pattern:** Common in ML projects to define structure early
- **Iterative Development:** Implement core functionality first, add datasets later
- **Modularity:** Each dataset/config is independent, can be implemented in parallel

### **Migration Strategy**
Currently using:
- `torchvision.datasets.ImageFolder` for binary classification
- Hardcoded hyperparameters in training scripts

After implementation:
- Custom Dataset classes with better control
- YAML configs for easy experimentation
- Cleaner, more maintainable code

---

**Last Updated:** February 8, 2026  
**Status:** 12 files identified, 7 high priority, 5 medium/low  
**Next Steps:** Implement configs (Week 1), then critical datasets (Week 4)
