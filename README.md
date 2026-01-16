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
🟡 Phase 1B — In Progress

---

## ⭐ If you like this project

Give it a star ⭐ and follow the development!
