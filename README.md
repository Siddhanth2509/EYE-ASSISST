# EYE-ASSISST — AI-Powered Retinal Screening Platform

<div align="center">

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)
![Frontend](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite%207-61DAFB?style=for-the-badge&logo=react)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![AI](https://img.shields.io/badge/AI-PyTorch%20%2B%20timm-EE4C2C?style=for-the-badge&logo=pytorch)
![Tested](https://img.shields.io/badge/Tested%20with-TestSprite-6C47FF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**Advanced AI-driven clinical platform for early detection of retinal diseases.**  
Built for ophthalmologists, technicians, and patients — powered by deep learning.

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [AI Models](#-ai-models) · [Testing](#-testing)

</div>

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-Disease AI** | Detects Diabetic Retinopathy, AMD, Glaucoma, Cataract, Hypertensive Retinopathy, and Myopic Degeneration |
| **DR Severity Grading** | ResNet50 multi-task model grades DR 0–4 (QWK = 0.82) |
| **Explainable AI (XAI)** | Grad-CAM heatmaps highlight the retinal regions driving each decision |
| **Fundus Gating** | MobileNetV2 classifier rejects non-fundus images before analysis (100% val accuracy) |
| **Role-Based Portals** | Separate dashboards for Patients, Doctors, Technicians, and Admins |
| **Doctor Review Portal** | Heatmap/vessel overlay toggles, zoom, severity override, clinical notes |
| **Patient Portal** | Full profile, scan history timeline, PDF report download, appointment booking |
| **EyeBot Chatbot** | Floating AI assistant with clinical knowledge base and scrollable history |
| **Admin Dashboard** | Analytics, user management, admin key rotation, system health |
| **PDF Reports** | Auto-generated clinical reports via jsPDF |

---

## Project Structure

```text
EYE-ASSISST/
├── AI Eye Screening UI/           # React + Vite Frontend (v2 — active)
│   └── app/
│       ├── src/
│       │   ├── App.tsx            # Router, role dashboards, auth state
│       │   ├── index.css          # Design tokens + Tailwind config
│       │   ├── components/        # Reusable UI components
│       │   └── pages/             # Route-level page components
│       ├── package.json
│       └── vite.config.ts
│
├── backend/                       # FastAPI Backend (single source of truth)
│   ├── main.py                    # API endpoints, scan DB, CORS, inference
│   ├── fundus_classifier.py       # Binary fundus / non-fundus gating
│   ├── gradcam_module.py          # Grad-CAM heatmap generation
│   └── models/
│       ├── fundus_best.pt         # MobileNetV2 fundus classifier
│       ├── severity_qwk_best.pt   # ResNet50 DR severity model (QWK=0.82)
│       ├── multidisease_v1.pt     # ResNet50 multi-label classifier (F1=0.36)
│       └── threshold_config.json  # Per-disease detection thresholds
│
├── phase0_fundus_classifier/      # Phase 0: fundus gating model training
├── phase1_pipelines/              # Phase 1: data preprocessing pipelines
├── phase2_dr_severity/            # Phase 2: DR severity model training
├── phase3_multi_disease/          # Phase 3: multi-label classifier training
│
├── notebooks/                     # Research Jupyter notebooks
├── configs/                       # YAML training configurations
├── docs/                          # Extended architecture documentation
├── scripts/
│   └── testing/                   # API test + accuracy scripts (see README inside)
├── testsprite_tests/              # Automated E2E test suite
├── storage/                       # Runtime scan images (gitignored)
├── requirements.txt               # Python dependencies
└── Dockerfile                     # Container deployment
```

---

## Getting Started

### Prerequisites

- **Python** 3.10 or 3.11
- **Node.js** v18+ and **npm** v9+
- CPU inference works fine — no GPU required

---

### 1. Clone

```bash
git clone https://github.com/Siddhanth2509/EYE-ASSISST.git
cd EYE-ASSISST
```

---

### 2. Backend

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server (stable — no auto-reload)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

> **Important:** Models take ~60 seconds to load on first startup.  
> Wait for all three `[OK] ... model loaded` log lines before testing.

Verify the backend is ready:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json
# Expected: "status": "healthy", "model_loaded": true
```

---

### 3. Frontend

```powershell
cd "AI Eye Screening UI\app"
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## Authentication & Access

| Role | How to Access |
|---|---|
| **Patient** | Register at `/login` → select Patient |
| **Doctor / Technician** | Account created by an Administrator |
| **Admin** | Go to `/login`, press `Ctrl+Shift+A`, enter key `EYEADMIN2026` |

> Admin key can be rotated from Admin → Settings. Stored in browser `localStorage` only.

---

## Architecture

```
Browser (React SPA — localhost:5173)
    |
    +-- Public Routes:  /  /demo  /about  /blog  /research  /contact
    +-- Auth Routes:    /login  -->  role dashboard
    |
    +-- Dashboard Shell (role-based tabs)
            +-- Technician : upload image --> POST /api/analyze --> results + heatmap
            +-- Doctor     : review queue --> heatmap overlay --> submit decision
            +-- Admin      : analytics, user management, settings
            +-- Patient    : scan history, reports, appointments

FastAPI Backend (localhost:8000)
    +-- POST /api/analyze           AI inference + Grad-CAM (primary endpoint)
    +-- POST /api/v1/predict        Compatibility alias
    +-- GET  /api/v1/scan/{id}      Fetch scan details
    +-- GET  /api/scans             Pending review queue
    +-- POST /api/v1/review/{id}    Submit doctor review
    +-- GET  /health                System health + model status

AI Pipeline (sequential, CPU)
    +-- Phase 0: Fundus Classifier (MobileNetV2)   -- reject non-fundus images
    +-- Phase 2: DR Severity Grading (ResNet50)    -- 5-class severity + confidence
    +-- Phase 3: Multi-Disease Detection (ResNet50) -- 6 disease flags + thresholds
    +-- Grad-CAM heatmap generated on the DR model layer
```

---

## AI Models

| Model | Architecture | Metric | Notes |
|---|---|---|---|
| Fundus Classifier | MobileNetV2 | 100% val accuracy | Rejects non-retinal images |
| DR Severity | ResNet50 Multi-Task | QWK = 0.82 | Grades DR 0–4 |
| Multi-Disease | ResNet50 6-class | F1 = 0.36 | Detects 6 disease classes |

**Disease classes (Phase 3):** Diabetic Retinopathy · Glaucoma · AMD · Cataract · Hypertensive Retinopathy · Myopic Degeneration

**Thresholds** are in `backend/models/threshold_config.json` (default: 0.60 for all classes). Tune without touching code.

**Inference time on CPU:** 10–25 seconds per image (3 models run sequentially + Grad-CAM).

---

## Testing

### Automated E2E — TestSprite

| Requirement | Tests | Passed | Failed |
|---|---|---|---|
| Landing Page & Navigation | 5 | 5 | 0 |
| EyeBot Chatbot | 2 | 2 | 0 |
| Contact Form | 1 | 1 | 0 |
| Access Control | 2 | 1 | 1 (fixed) |
| 3D Eye Model | 1 | 1 | 0 |
| **Total** | **11** | **10** | **0 (all fixed)** |

Full report: [`testsprite_tests/`](./testsprite_tests/)

### API & Accuracy Scripts

All testing utilities live in [`scripts/testing/`](./scripts/testing/) — see the README inside for exact commands.

```powershell
# Quick smoke test
& ".venv\Scripts\python.exe" scripts/testing/test_api.py

# Live disease detection test with Grad-CAM + timing
& ".venv\Scripts\python.exe" scripts/testing/test_live.py

# Batch precision/recall (5 cataract + 5 normal images)
& ".venv\Scripts\python.exe" scripts/testing/test_accuracy.py
```

### Test Accounts

Navigate to `/login?testmode=1` to auto-seed demo accounts:

| Role | Email | Password |
|---|---|---|
| Admin | `admin@eyeassist.test` | `Test@1234` |
| Doctor | `doctor@eyeassist.test` | `Test@1234` |
| Technician | `tech@eyeassist.test` | `Test@1234` |
| Patient | `patient@eyeassist.test` | `Test@1234` |

---

## UI/UX Design System

- **Theme:** Dark mode with cyan/blue primary palette
- **Typography:** Inter (system font stack)
- **Animations:** Framer Motion (page transitions, micro-interactions)
- **Components:** shadcn/ui on Radix UI primitives
- **Charts:** Recharts (area, bar, pie)
- **3D:** Three.js + React Three Fiber (landing page eye model)

---

## Medical Disclaimer

EYE-ASSISST is a **clinical decision support system**. All AI findings are preliminary and must be verified by a qualified medical professional. Not intended as a standalone diagnostic tool.

---

## Contributing

See [`docs/`](./docs/) for architecture diagrams and contribution guidelines.

---

*Built with React, FastAPI, and PyTorch.*
