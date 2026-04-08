# ⚠️ Empty Files & Implementation Roadmap

> **Complete list of empty/placeholder files and their implementation priority**  
> **Last Audited:** March 7, 2026

---

## 📋 Summary

**Total Empty Files Found: 28**

| Category | Count | Priority |
|----------|-------|----------|
| Config Files (YAML) | 7 | 🔴 High |
| Dataset Classes | 5 | 🟡 Medium |
| Evaluation Scripts | 3 | 🔴 High |
| Loss Functions | 1 | 🔴 High |
| Training Scripts | 2 | 🟡 Medium |
| Explainability | 1 | 🟡 Medium |
| Data Utilities | 1 | 🟢 Low |
| Package `__init__.py` files | 6 | 🟢 Low |
| Model Files | 2 | 🟡 Medium |

---

## 🔴 CRITICAL PRIORITY: Core Pipeline Gaps (4 files)

These files are referenced by active code or documentation but contain nothing.

---

### 1. `src/losses/masked_bce.py` ❌ EMPTY
**Purpose:** Masked Binary Cross-Entropy loss for multi-label training with missing annotations  
**Impact:** Blocks Stage-3 multi-label training entirely  
**Should implement:**
```python
class MaskedBCELoss(nn.Module):
    """
    BCE loss that ignores missing labels using a mask tensor.
    Essential for ODIR dataset where not all diseases are annotated.
    """
    def forward(self, logits, targets, masks):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = (loss * masks).sum() / masks.sum()
        return loss
```

---

### 2. `src/explainability/gradcam.py` ❌ EMPTY
**Purpose:** Grad-CAM visualization for model interpretability  
**Impact:** Required for research papers (explainability section) and clinical trust  
**Should implement:**
```python
class GradCAM:
    """
    Generate class-discriminative heatmaps showing which
    image regions the model focuses on for its predictions.
    """
    def __init__(self, model, target_layer): ...
    def generate(self, input_image, target_class): ...
```

---

### 3. `src/evaluation_phase3_multi-model/dr_binary_meterics.py` ❌ EMPTY
**Purpose:** Binary DR evaluation (AUC, Sensitivity, Specificity)  
**Impact:** Stage-1 model has no standalone evaluation script in the Phase-3 pipeline  

---

### 4. `src/evaluation_phase3_multi-model/multilabel_metrics.py` ❌ EMPTY
**Purpose:** Multi-label disease evaluation (per-disease AUC, Hamming loss)  
**Impact:** Blocks Stage-3 evaluation  

---

## 🔴 HIGH PRIORITY: Configuration Files (7 files)

All config YAML files are 0 bytes. Training currently uses hardcoded hyperparameters.

### Root Level Configs (3 files)

| # | File | Purpose |
|---|------|---------|
| 5 | `configs/model.yaml` | Model architecture (backbone, heads, dropout) |
| 6 | `configs/paths.yaml` | Data directories, checkpoint paths |
| 7 | `configs/train.yaml` | LR, batch size, epochs, augmentation, loss weights |

### Stage-Specific Configs (4 files)

| # | File | Purpose |
|---|------|---------|
| 8 | `src/configs/base.yml` | Base config inherited by all stages (seed, device, logging) |
| 9 | `src/configs/stage1_dr.yaml` | Stage-1 binary DR training settings |
| 10 | `src/configs/stage2_mutlilabel.yml` | Stage-2 severity + multi-label settings |
| 11 | `src/configs/stage3_joint.yaml` | Stage-3 joint fine-tuning settings |

**Why this matters:** Without configs, experiments aren't reproducible. Every training run has parameters scattered across Python scripts.

---

## 🟡 MEDIUM PRIORITY: Dataset Classes (5 files)

Only `src/data/datasets/eyepacs_severity.py` is implemented. All others are 0 bytes.

| # | File | Purpose | Needed For |
|---|------|---------|------------|
| 12 | `src/data/datasets/eyepacs.py` | Binary EyePACS (DR/Normal) | Custom preprocessing beyond ImageFolder |
| 13 | `src/data/datasets/aptos.py` | APTOS 2019 (severity 0-4) | External test set evaluation |
| 14 | `src/data/datasets/odir.py` | ODIR-5K (8-disease multi-label) | Stage-3 multi-label head |
| 15 | `src/data/datasets/amd.py` | AMD detection | Multi-disease extension |
| 16 | `src/data/datasets/cataract.py` | Cataract detection (external eye) | Multi-disease extension |

**Priority order:** `odir.py` > `aptos.py` > `eyepacs.py` > `amd.py` = `cataract.py`

---

## 🟡 MEDIUM PRIORITY: Training & Evaluation Scripts (5 files)

| # | File | Purpose | Status |
|---|------|---------|--------|
| 17 | `src/evaluation_phase3_multi-model/severity_metrics.py` | Standalone severity metrics (QWK, CM) | Empty — `evaluate_stage2_finetuned.py` exists as workaround |
| 18 | `src/training_phase3_multimodel/train_stage3.py` | Stage-3 joint multi-task training | Reserved for Phase-3 multi-disease pipeline. Currently unused but intentionally reserved. |
| 19 | `src/training_phase3_multimodel/trainer.py` | Unified multi-task trainer class | Reserved for Phase-3 multi-disease pipeline. Currently unused but intentionally reserved. |
| 20 | `src/models/heads/multi_label.py` | Multi-label classification head (8 diseases) | Empty — reserved for Stage-3 |
| 21 | `src/models/__init__.py` | Models package init | Empty (functional but no exports) |

> **Note:** `train_stage3.py` and `trainer.py` are intentional placeholders.
> They are **not** current blockers — Phase-3 M2 experiments (ResNet18 fine-tune, EfficientNet-B3) are active via `train_stage2_finetune.py` and `train_stage2_efficientnet.py`.

---

## 🟢 LOW PRIORITY: Utility & Package Files (7 files)

These are either intentionally minimal or have low impact.

| # | File | Purpose | Notes |
|---|------|---------|-------|
| 22 | `src/data/transforms.py` | Custom augmentations | Transforms are inline in DataModules currently |
| 23 | `src/utils/logging.py` | Training log utilities | Using print statements currently |
| 24 | `src/__init__.py` | Root package init | Intentionally empty (standard Python) |
| 25 | `src/data/__init__.py` | Data package init | Intentionally empty |
| 26 | `src/metrics/__init__.py` | Metrics package init | Intentionally empty |
| 27 | `src/training/__init__.py` | Training package init | Intentionally empty |
| 28 | `src/utils/__init__.py` | Utils package init | Intentionally empty |

**Note:** `__init__.py` files (items 24–28) are standard Python package markers and are typically empty. They are listed for completeness but do not need content unless you want to expose specific imports.

---

## 📊 Implementation Priority Matrix

| File | Priority | Effort | Impact | Blocks |
|------|----------|--------|--------|--------|
| `src/losses/masked_bce.py` | 🔴 Critical | Low | High | Stage-3 training |
| `src/explainability/gradcam.py` | 🔴 Critical | Medium | High | Paper explainability |
| `src/evaluation_phase3_multi-model/dr_binary_meterics.py` | 🔴 Critical | Low | Medium | Stage-1 eval in Phase-3 |
| `src/evaluation_phase3_multi-model/multilabel_metrics.py` | 🔴 Critical | Medium | High | Stage-3 eval |
| `configs/model.yaml` | 🔴 High | Low | Medium | Reproducibility |
| `configs/paths.yaml` | 🔴 High | Low | Medium | Reproducibility |
| `configs/train.yaml` | 🔴 High | Low | High | Reproducibility |
| `src/configs/base.yml` | 🔴 High | Low | Medium | Config inheritance |
| `src/configs/stage1_dr.yaml` | 🔴 High | Low | Medium | Experiment tracking |
| `src/configs/stage2_mutlilabel.yml` | 🔴 High | Low | Medium | Experiment tracking |
| `src/configs/stage3_joint.yaml` | 🔴 High | Low | Medium | Experiment tracking |
| `src/models/heads/multi_label.py` | 🟡 Medium | Medium | High | Stage-3 model |
| `src/training_phase3_multimodel/train_stage3.py` | 🟡 Medium | High | High | Stage-3 training |
| `src/training_phase3_multimodel/trainer.py` | 🟡 Medium | High | High | Stage-3 training |
| `src/data/datasets/odir.py` | 🟡 Medium | High | High | Multi-label data |
| `src/data/datasets/aptos.py` | 🟡 Medium | Medium | Medium | External test eval |
| `src/data/datasets/eyepacs.py` | 🟡 Medium | Medium | Low | Custom preprocessing |
| `src/evaluation_phase3_multi-model/severity_metrics.py` | 🟡 Medium | Low | Low | Has workaround |
| `src/data/transforms.py` | 🟢 Low | Low | Low | Has inline alternative |
| `src/utils/logging.py` | 🟢 Low | Medium | Low | Print works for now |
| `src/data/datasets/amd.py` | 🟢 Low | High | Low | Future extension |
| `src/data/datasets/cataract.py` | 🟢 Low | High | Low | Future extension |
| `__init__.py` files (6) | 🟢 Low | None | None | Intentionally empty |

---

## 🚀 Recommended Implementation Order

### **Phase A — Unblock Stage-3 (Week 1–2)**
1. `src/losses/masked_bce.py` — implement MaskedBCELoss
2. `src/models/heads/multi_label.py` — implement MultiLabelHead
3. `src/data/datasets/odir.py` — implement ODIRDataset
4. `src/training_phase3_multimodel/trainer.py` — unified trainer
5. `src/training_phase3_multimodel/train_stage3.py` — joint training script

### **Phase B — Configs & Reproducibility (Week 2–3)**
6. All 7 YAML config files — populate with current hyperparameters
7. Wire config loading into training scripts

### **Phase C — Evaluation & Explainability (Week 3–4)**
8. `src/evaluation_phase3_multi-model/dr_binary_meterics.py`
9. `src/evaluation_phase3_multi-model/multilabel_metrics.py`
10. `src/explainability/gradcam.py`

### **Phase D — Remaining Datasets (Week 4+)**
11. `src/data/datasets/aptos.py`
12. `src/data/datasets/eyepacs.py`
13. `src/data/datasets/amd.py` (future)
14. `src/data/datasets/cataract.py` (future)

### **Phase E — Polish (Optional)**
15. `src/data/transforms.py` — centralize augmentations
16. `src/utils/logging.py` — structured logging

---

## ✅ Validation Checklist

### **Config Files**
- [ ] All YAML files parse without errors
- [ ] Paths exist on disk
- [ ] Hyperparameters match values used in successful training runs
- [ ] Config inheritance works (base.yml → stage configs)

### **Dataset Classes**
- [ ] `__len__` returns correct count
- [ ] `__getitem__` returns tensors with correct shapes
- [ ] Labels match expected format
- [ ] Transforms applied correctly
- [ ] No data leakage (train/val/test separation)

### **Loss Functions**
- [ ] MaskedBCELoss handles missing labels correctly
- [ ] Gradient flows properly through masked positions

### **Evaluation Scripts**
- [ ] Metrics match sklearn reference implementations
- [ ] Reports are saved to disk (JSON/CSV)

### **Explainability**
- [ ] Grad-CAM heatmaps align with clinical features (optic disc, hemorrhages)
- [ ] Works with multi-task model (can target any head)

---

**Last Updated:** March 7, 2026  
**Status:** 28 empty files identified (22 actionable, 6 intentionally empty `__init__.py`)  
**Critical Blockers:** 4 files block Stage-3 and paper-readiness  
**Phase A Progress:** Partially addressed — `train_stage2_finetune.py` and `train_stage2_efficientnet.py` added; `train_stage3.py` and `trainer.py` reserved for future multi-disease pipeline.
