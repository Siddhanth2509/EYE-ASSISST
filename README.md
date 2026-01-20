# 👁️ AI-Powered Eye Disease Detection & Assistance Platform

An end-to-end **Deep Learning–based Eye Health Assistant** designed to support **early disease screening, data-driven insights, and doctor-in-the-loop decision support** — without replacing medical professionals.

> ⚠️ **Disclaimer**:
> This project is for **educational and research purposes only**.
> It is **not a medical diagnosis or prescription system**.
> Final decisions must always be made by certified ophthalmologists.

---

## 🚀 Project Vision

Eye diseases often go undetected until they become severe.
This project aims to:

* Enable **early screening** using deep learning
* Assist **patients** with awareness & triage
* Support **doctors** via AI-generated reports (human-in-the-loop)
* Maintain **ethical, safe, and explainable AI practices**

---

## 🧠 Core Features (Planned & In Progress)

### ✅ Phase 1A — Data Engineering (COMPLETED)

* EyePACS dataset ingestion (33k+ retinal images)
* Automated label mapping (NORMAL vs DR)
* Clean folder architecture
* Reproducible data pipeline
* Dataset excluded from GitHub via `.gitignore`

### 🟡 Phase 1B — Image Preprocessing (NEXT)

* Medical-safe resizing
* Fundus-specific normalization
* CLAHE contrast enhancement
* Train / validation / test split
* Class imbalance handling

### 🔜 Future Phases

* CNN-based deep learning models
* Explainability (Grad-CAM)
* NLP-based symptom chatbot
* Doctor approval workflow
* Web & mobile app with advanced UI/UX

---

## 🗂️ Project Structure

```text
eye-realtime-inference/
├── Data/
│   ├── raw/            # Raw datasets (ignored in Git)
│   ├── cleaned/        # Processed datasets (ignored in Git)
│   └── metadata/
├── scripts/            # Data processing scripts
├── models/             # Trained models & checkpoints
├── notebooks/          # Experiments & analysis
├── requirements.txt
├── README.md
└── .gitignore
```

> 🔒 **Note**:
> Medical images and datasets are intentionally excluded from version control.

---

## 📊 Dataset Used

* **EyePACS** – Diabetic Retinopathy retinal fundus images
  Used for large-scale data engineering and preprocessing pipeline validation.

Additional datasets (ODIR, Cataract, AMD) will be integrated in later phases using the same pipeline.

---

## 🧪 How to Run (Development)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run data scripts (example)

```bash
python scripts/split_eyepacs.py
```

---

## 🧠 Ethical AI Principles

* No automated diagnosis or prescriptions
* Human-in-the-loop decision making
* Dataset privacy & exclusion from Git
* Transparent model limitations
* Explainability-first design

---

## 👤 Author

**Siddhanth Sharma**
B.Tech | Machine Learning & AI
Focused on **applied AI, healthcare systems, and real-world ML engineering**

---

## 📌 Status

🟢 Phase 1A — Completed
## 🧠 Phase 2A — CNN Strategy & Training Design (Conceptual Phase)

> **Status:** ✅ Active
> **Nature:** Decision-making & justification only
> **Code Level:** ❌ No heavy model training yet

Phase 2A focuses on **thinking like a Machine Learning engineer before writing code**.
This phase is intentionally designed to lock **critical design decisions** for a medical AI system **before any CNN training begins**.

---

## 🔒 Phase 1 Freeze (Precondition)

Phase 2A operates under a strict data freeze:

* ✅ Data engineering — **finalized**
* ✅ Preprocessing — **finalized**
* ✅ Train / Validation / Test splits — **finalized**
* ✅ Manifest CSV — **finalized**

🚫 During Phase 2A:

* No data modification
* No preprocessing changes
* No reshuffling of splits

This ensures **clean, reproducible ML experiments** and prevents data leakage.

---

## 🎯 Objective of Phase 2A

The goal is to design a **clinically meaningful, generalizable CNN training strategy** for diabetic retinopathy screening — not just to maximize accuracy.

Phase 2A emphasizes:

* Clinical relevance
* Dataset realism
* External generalization
* Interview-ready justification

---

## 🧩 Decisions Covered in Phase 2A (In Order)

### 1️⃣ Binary vs Multi-Class Classification

**Decision Scope**

* NORMAL vs DR (screening-oriented)
* Why multi-class severity prediction is postponed

**Focus**

* Clinical screening relevance
* Label noise in DR severity grades
* Improved generalization to external datasets (APTOS)

---

### 2️⃣ Loss Functions for Medical AI

**Options Considered**

* Binary Cross Entropy
* BCE with class weights
* Focal Loss

**Focus**

* Why false negatives are more dangerous than false positives
* How loss functions encode clinical risk
* Practical trade-offs (calibration vs recall)

---

### 3️⃣ Metrics Beyond Accuracy

**Metrics Evaluated**

* Sensitivity (Recall for DR)
* Specificity
* AUC-ROC
* Precision–Recall trade-off

**Focus**

* Why accuracy is misleading in imbalanced medical datasets
* Selecting a **primary metric** aligned with clinical goals
* Supporting metrics for diagnostic insight

---

### 4️⃣ Class Imbalance Handling

**Strategies Compared**

* Class weighting
* Oversampling
* Undersampling

**Focus**

* Why imbalance was **not corrected at split time**
* Maintaining real-world data distribution
* Choosing one strategy for controlled experimentation

---

### 5️⃣ Training Protocol: EyePACS → APTOS

**Experimental Design**

* Train on EyePACS
* Validate on EyePACS
* Test on APTOS (external dataset)

**Focus**

* Measuring real generalization, not memorization
* Interpreting good vs bad cross-dataset results
* Elevating the project from “CNN training” to **generalization evaluation**

---

## 🤖 AI-Augmented Workflow (Vibe Coding, Done Right)

This project follows a **modern AI-assisted ML workflow**:

* **ChatGPT** → strategy, reasoning, and justification
* **Perplexity AI** → evidence checks & best-practice validation
* **Notion / Markdown** → decision logs
* **Cursor** → implementation (Phase 2B onward)
* **W&B** → experiment tracking (Phase 2B onward)

> AI tools are used to **enhance thinking**, not replace ML fundamentals.

---

## 🚦 Exit Criteria for Phase 2A

Phase 2A is considered complete when:

* All five decisions are **locked and documented**
* Each decision has **clear clinical + ML justification**
* The training strategy is **interview-defensible**
* No code shortcuts are taken

Only after this will the project move to **Phase 2B — CNN Implementation & Training**.

---

## 🧠 Why This Phase Matters

Phase 2A ensures the model is:

* Clinically meaningful
* Scientifically valid
* Reproducible
* Generalization-focused

This transforms the project from:

> *“I trained a CNN”*

to:

> *“I designed and evaluated a medical AI system with external validation.”*

---

📌 **Next Phase:** Phase 2B — CNN Architecture & Training (Implementation Begins)


---

## ⭐ If you like this project

Give it a star ⭐ and follow the development!
