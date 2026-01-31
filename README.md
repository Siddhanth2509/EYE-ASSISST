# 👁️ **EYE-ASSISST**

### AI-Powered Eye Disease Detection & Clinical Decision Support Platform

> **An end-to-end medical imaging ML system for early eye-disease screening — designed with clinical rigor, external validation, and human-in-the-loop safety.**

⚠️ **Medical Disclaimer**
This project is strictly for **educational and research purposes**.
It **does not provide medical diagnosis or prescriptions**.
All final decisions must be made by **licensed ophthalmologists**.

---

## 🌍 Why This Project Exists

Eye diseases like **Diabetic Retinopathy (DR)** often progress silently.
Delayed detection leads to irreversible vision loss.

**EYE-ASSISST** is built to:

* 🧠 Enable **early AI-assisted screening**
* 👨‍⚕️ Support clinicians with **data-driven insights**
* 🔍 Prioritize **generalization over inflated metrics**
* ⚖️ Follow **ethical & explainable AI principles**

This is **not a demo CNN** — it is a **research-grade medical ML system**.

---

## 🚀 Project Roadmap & Status

| Phase        | Description                        | Status               |
| ------------ | ---------------------------------- | -------------------- |
| **Phase 1A** | Data Engineering                   | ✅ Completed          |
| **Phase 1B** | Medical Image Preprocessing        | ✅ Completed          |
| **Phase 2A** | CNN Strategy & Clinical Design     | ✅ Completed          |
| **Phase 2B** | CNN Training & External Validation | ✅ Completed & Frozen |
| **Phase 3**  | Multi-Disease AI System            | 🟡 Planning          |

---

## 🧠 Core Features

### ✅ Implemented (Phase 2)

* Binary DR screening (NORMAL vs DR)
* CNN-based retinal image classification
* External dataset validation (APTOS)
* Clinically prioritized metrics
* Strict data-leakage prevention
* Reproducible ML pipeline

### 🔜 Planned (Phase 3+)

* Multi-disease classification
* Explainability (Grad-CAM)
* Doctor approval workflow
* NLP symptom assistant
* Real-time inference & deployment

---

## 🗂️ Repository Structure (Phase 2)

```
eye-assisst/
├── src/
│   ├── data/        # Frozen DataModule & splits
│   ├── models/      # CNN backbone (ResNet-18)
│   ├── training/    # Training & evaluation logic
│   ├── metrics/     # Medical metrics (Sensitivity, AUC)
│   └── utils/       # Reproducibility helpers
├── notebooks/       # Phase results & analysis
├── models/          # Checkpoints (Git LFS)
├── requirements.txt
├── README.md
└── .gitignore
```

🔒 **Medical datasets are intentionally excluded from GitHub**.

---

## 📊 Datasets Used

### Primary Dataset

* **EyePACS**
  Large-scale retinal fundus dataset used for **training & validation**.

### External Test Dataset

* **APTOS**
  Used **only for final evaluation** to measure real-world generalization.

> No image from APTOS was ever seen during training or tuning.

---

## 🧪 How to Run (Development)

```bash
# 1️⃣ Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2️⃣ Create environment
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

---

# 🧠 Phase 2 — The Heart of This Project

## 🔒 Phase Freeze Guarantee

Before Phase 2 began, **Phase 1 was permanently frozen**:

* ✅ Data ingestion finalized
* ✅ Preprocessing finalized
* ✅ Train / Val / Test splits finalized
* ✅ Manifest CSV locked

🚫 **No changes allowed** during Phase 2
This ensures **zero data leakage** and **reproducible experiments**.

---

## 🧠 Phase 2A — CNN Strategy & Clinical Design

> *“Think like an ML engineer before writing code.”*

Phase 2A focuses on **decision-making, not training**.

### 🎯 Objective

Design a **clinically meaningful and generalizable DR screening system**, not just a high-accuracy model.

### 🧩 Key Decisions (Locked)

#### 1️⃣ Binary vs Multi-Class Classification

* **Chosen:** NORMAL vs DR
* **Why:**

  * Screening relevance
  * Severity labels are noisy
  * Better external generalization

---

#### 2️⃣ Loss Functions for Medical AI

Options evaluated:

* Binary Cross Entropy
* BCE with Class Weights ✅
* Focal Loss

**Clinical logic:**
False negatives (missing DR) are more dangerous than false positives.

---

#### 3️⃣ Metrics Beyond Accuracy

Primary metric:

* ⭐ **Sensitivity (Recall for DR)**

Supporting metrics:

* Specificity
* AUC-ROC
* Precision–Recall trade-off

Accuracy alone is misleading in medical datasets.

---

#### 4️⃣ Class Imbalance Handling

Strategies compared:

* Class weighting ✅
* Over/Under-sampling

**Why no resampling at split time?**
To preserve **real-world disease prevalence**.

---

#### 5️⃣ Training Protocol: EyePACS → APTOS

* Train + validate on EyePACS
* Test only on APTOS

This elevates the project from:

> “I trained a CNN”
> to
> **“I evaluated real generalization.”**

---

## 🤖 AI-Augmented ML Workflow

Used responsibly:

* **ChatGPT** → Strategy & reasoning
* **Perplexity AI** → Evidence validation
* **Notion / Markdown** → Decision logs
* **Cursor** → Implementation
* **Weights & Biases** → Experiment tracking

AI enhanced thinking — it never replaced fundamentals.

---

## 🧠 Phase 2B — Implementation & Training

### 🧩 Model Architecture

* **CNN Backbone:** ResNet-18
* ImageNet pretrained
* Single backbone enforced for Phase 2

### 🧩 Training Setup

* Optimizer: AdamW
* LR Scheduler
* Early stopping on **validation sensitivity**
* Best model saved by **clinical priority**, not accuracy

---

## 📊 Phase 2 Results — External Validation (APTOS)

⭐ **This is the most important result of the project**

| Metric                  | Value        |
| ----------------------- | ------------ |
| Accuracy                | ~95.9%       |
| Sensitivity (DR Recall) | **~96.4%** ⭐ |
| Specificity             | ~95.4%       |
| AUC-ROC                 | ~0.988       |

### 🧪 Confusion Matrix (APTOS)

* True Positives: **348**
* False Negatives: **13**
* False Positives: **17**
* True Negatives: **355**

---

## 🩺 Clinical Interpretation

* 🔥 Very low false-negative rate
* ⚖️ Balanced performance across classes
* 🌍 Strong generalization to unseen data

The model learned **disease-relevant features**, not dataset shortcuts.

---

## 📈 Why Training Curves Are Not Emphasized

* External generalization > fitting dynamics
* Early stopping occurred naturally
* Final metrics provide stronger clinical evidence

This aligns with **research-grade medical ML practice**.

---

## 🔒 Phase 2 Closure Statement

Phase 2 is **officially complete and frozen**.

✔ External validation achieved
✔ No test-set tuning
✔ Clinically meaningful metrics
✔ Clean experiment discipline

---

## 🔜 Phase 3 — Multi-Disease Medical AI (Planning)

Planned extensions:

* Multi-label disease detection
* Shared backbone + disease heads
* Grad-CAM explainability
* Real-time inference

📌 **Phase 3 has not started yet**

---

## 🏁 Final Note

> In medical AI,
> **honest generalization beats perfect numbers.**

This project prioritizes **trustworthy ML** over inflated benchmarks.

---

## 👤 Author

**Siddhanth Sharma**
B.Tech — Machine Learning & AI
Focused on **applied medical AI**, **ML engineering**, and **real-world systems**

---
