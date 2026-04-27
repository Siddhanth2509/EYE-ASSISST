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

[Live Demo](#-demo) · [Getting Started](#-getting-started) · [Architecture](#-architecture) · [Test Report](#-testing)

</div>

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| **Multi-Disease AI** | CNN detects Diabetic Retinopathy, AMD, Glaucoma, Cataract, and more with severity grading |
| **Explainable AI (XAI)** | Grad-CAM heatmaps show *why* the AI made each decision |
| **Fundus Validation** | Heuristic + ML-based rejection of non-fundus images before analysis |
| **Role-Based Portals** | Separate workflows for Patients, Doctors, Technicians, and Admins |
| **Doctor Review Portal** | Heatmap/vessel overlay toggles, zoom, and clinical note submission |
| **Patient Portal** | Full profile management, scan history, PDF reports, appointment booking |
| **EyeBot Chatbot** | Floating AI assistant with clinical knowledge base, scrollable history |
| **Admin Dashboard** | Analytics, user management, admin key rotation, system health |
| **Interactive Demo** | Public `/demo` route — no account required |
| **PDF Reports** | Auto-generated clinical reports via jsPDF |

---

## 📁 Project Structure

```text
EYE-ASSISST/
├── AI Eye Screening UI/           # React + Vite Frontend
│   └── app/
│       ├── src/
│       │   ├── App.tsx            # Router, role dashboards, auth, state
│       │   ├── index.css          # Global design tokens & Tailwind config
│       │   ├── components/        # Reusable components
│       │   │   ├── EyeBot.tsx         # Floating AI chatbot (scrollable)
│       │   │   ├── PaymentGateway.tsx # Appointment payment flow
│       │   │   ├── ReportGenerator.tsx# jsPDF clinical report
│       │   │   ├── UserManagementPanel.tsx
│       │   │   ├── AdminSettingsPanel.tsx
│       │   │   ├── ScrollToTop.tsx
│       │   │   └── three/             # Three.js 3D eye model
│       │   └── pages/             # Route pages
│       │       ├── LandingPage.tsx
│       │       ├── PatientPortal.tsx  # Auth-guarded patient dashboard
│       │       ├── DemoPage.tsx       # Public AI demo (no login needed)
│       │       ├── AboutPage.tsx
│       │       ├── BlogPage.tsx
│       │       ├── CareersPage.tsx
│       │       ├── CaseStudiesPage.tsx
│       │       ├── ContactPage.tsx
│       │       ├── DocsPage.tsx
│       │       ├── PrivacyPage.tsx
│       │       └── ResearchPage.tsx
│       ├── package.json
│       └── vite.config.ts
│
├── backend/                       # FastAPI Backend
│   ├── main.py                    # API endpoints, scan DB, CORS config
│   ├── fundus_classifier.py       # Binary fundus vs non-fundus classifier
│   └── gradcam_module.py          # Grad-CAM heatmap generation
│
├── phase0_fundus_classifier/      # Fundus gating model training
├── phase1_pipelines/              # Data preprocessing pipelines
├── phase2_dr_severity/            # DR severity model training
├── phase3_multi_disease/          # Multi-label disease classifier training
├── notebooks/                     # Jupyter notebooks for research
├── configs/                       # YAML training configurations
├── scripts/                       # Utility scripts
├── docs/                          # Extended documentation
├── testsprite_tests/              # Automated test suite & reports
│   ├── testsprite-mcp-test-report.md  # Latest TestSprite test report
│   └── tmp/                       # Raw test outputs & generated test scripts
├── requirements.txt               # Python dependencies
└── START_APP.bat                  # One-click Windows launcher
```

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** v18+ and **npm** v9+
- **Python** 3.10 – 3.11
- GPU with CUDA (optional — CPU inference works for demos)

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/EYE-ASSISST.git
cd EYE-ASSISST
```

### 2. Start the Backend
```bash
cd backend

# Install Python dependencies
pip install -r ../requirements.txt

# Start the FastAPI server (runs on http://localhost:8000)
python main.py
```

### 3. Start the Frontend
```bash
cd "AI Eye Screening UI/app"

# Install Node dependencies
npm install

# Development server (http://localhost:5173)
npm run dev

# OR production build + serve
npm run build && npm run preview
```

### 4. One-Click Launch (Windows)
```bash
# From the project root
START_APP.bat
```

---

## 🔐 Authentication & Access

| Role | How to Access |
|---|---|
| **Patient** | Register via `/login` → select Patient |
| **Doctor / Technician** | Account must be created by Administrator |
| **Admin** | Go to `/login`, press `Ctrl+Shift+A`, enter key `EYEADMIN2026` |

> **Admin key** can be rotated from the Admin → Settings panel.  
> The key is stored only in the browser's `localStorage`.

---

## 🏗️ Architecture

```
Browser (React SPA)
    │
    ├── Public Routes: /, /demo, /about, /blog, /research, /contact ...
    ├── Auth Routes:   /login  → role-select → dashboard
    │                 /patient-portal  (auth-guarded → redirects to /login)
    │
    └── Dashboard Shell (role-based tabs)
            ├── Technician: image upload → AI inference (POST /api/v1/predict)
            ├── Doctor:     review queue → heatmap/overlay → submit review
            ├── Admin:      analytics, user management, settings
            └── History:    patient scan timeline

FastAPI Backend (localhost:8000)
    ├── POST /api/v1/predict        → AI inference + Grad-CAM
    ├── GET  /api/v1/scan/{id}      → fetch scan details
    ├── GET  /api/scans             → pending review queue
    ├── POST /api/v1/review/{id}    → submit doctor review
    └── GET  /health                → system health check

AI Pipeline
    ├── Phase 0: Fundus Classifier (MobileNetV2) — gate non-fundus images
    ├── Phase 1: Data Preprocessing (CLAHE, normalization)
    ├── Phase 2: DR Severity Grading (EfficientNet-B4, 5-class)
    └── Phase 3: Multi-Disease Detection (multi-label classifier)
```

---

## 🧪 Testing

### Automated E2E Testing with TestSprite

This project has been validated using **[TestSprite](https://www.testsprite.com)** — an AI-powered automated frontend testing platform.

**Test Run Summary (Run 2 — Production Build)**

| Requirement | Tests | ✅ Passed | ❌ Failed | Notes |
|---|---|---|---|---|
| Landing Page & Navigation | 5 | 5 | 0 | All navigation flows verified |
| EyeBot Chatbot | 2 | 2 | 0 | Scroll, persistence across routes |
| Contact Form | 1 | 1 | 0 | Submit + success state |
| Access Control | 2 | 1 | 1 | Fixed: patient portal now auth-guarded |
| 3D Eye Model | 1 | 1 | 0 | Renders without blocking navigation |
| **Total (actionable)** | **11** | **10** | **1** | **91% pass rate** |

> The remaining 1 failure (unauthenticated patient portal access) has been **fixed** — the portal now redirects to `/login` if no session is present.

**View full report:** [`testsprite_tests/testsprite-mcp-test-report.md`](./testsprite_tests/testsprite-mcp-test-report.md)

**Test Dashboard:** https://www.testsprite.com/dashboard/mcp/tests/6b8b9c2e-1b90-4ac1-99c4-80448c29f1fa

### Running Tests Locally

```bash
# 1. Build production bundle
cd "AI Eye Screening UI/app"
npm run build && npm run preview -- --port 5173

# 2. Start backend
cd ../../backend && python main.py

# 3. Run TestSprite via MCP
# (Requires TestSprite MCP server configured in your IDE)
```

### Test Accounts (Test Mode)
Navigate to `/login?testmode=1` to auto-seed demo accounts:

| Role | Email | Password |
|---|---|---|
| Admin | `admin@eyeassist.test` | `Test@1234` |
| Doctor | `doctor@eyeassist.test` | `Test@1234` |
| Technician | `tech@eyeassist.test` | `Test@1234` |
| Patient | `patient@eyeassist.test` | `Test@1234` |

---

## 🎨 UI/UX Design System

- **Theme:** Dark mode with cyan/blue primary palette
- **Typography:** Inter (system font stack)
- **Animations:** Framer Motion (page transitions, micro-interactions)
- **Components:** shadcn/ui on Radix UI primitives
- **Charts:** Recharts (area, bar, pie)
- **3D:** Three.js + React Three Fiber (landing page eye model)

---

## ⚠️ Medical Disclaimer

EYE-ASSISST is a **clinical decision support system**. All AI findings are preliminary and must be verified by a qualified medical professional. Not intended as a standalone diagnostic tool.

---

## 🤝 Contributing

See [`docs/`](./docs/) for architecture diagrams and contribution guidelines.

---

*Built with ❤️ using React, FastAPI, and PyTorch.*
