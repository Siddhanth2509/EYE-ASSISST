# EYE-ASSISST — Product Specification Document
**Version:** 2.0  
**Date:** April 2026  
**Classification:** Internal / Academic Major Project  
**Authors:** Development Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Purpose & Vision](#2-product-purpose--vision)
3. [Target Users & Roles](#3-target-users--roles)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Frontend Features](#5-frontend-features)
6. [AI Model Specifications](#6-ai-model-specifications)
7. [Backend API Specification](#7-backend-api-specification)
8. [Data Flow & Workflows](#8-data-flow--workflows)
9. [Security & Compliance](#9-security--compliance)
10. [Non-Functional Requirements](#10-non-functional-requirements)
11. [Known Limitations & Roadmap](#11-known-limitations--roadmap)

---

## 1. Executive Summary

**EYE-ASSISST** is an AI-powered, web-based ophthalmology screening platform designed to assist clinicians and technicians in detecting retinal diseases from fundus (eye fundus) photographs. The system uses a deep-learning pipeline (ResNet-50 multi-label classifier + Grad-CAM explainability) to screen for six major eye diseases simultaneously, reducing diagnostic time and enabling early detection at scale, particularly in resource-limited healthcare settings in India.

### Key Value Propositions
- **Automated screening** of fundus images with sub-second inference
- **Multi-disease detection** in a single pass (6 diseases)
- **Explainable AI** via Grad-CAM heatmaps shown to reviewing doctors
- **Role-based access control** for patients, doctors, technicians, and admins
- **Affordable, web-based** — no proprietary hardware required

---

## 2. Product Purpose & Vision

### Problem Statement
Diabetic Retinopathy (DR), Glaucoma, Age-related Macular Degeneration (AMD), Cataract, Hypertensive Retinopathy, and Myopic Maculopathy collectively affect hundreds of millions worldwide. In India alone, early screening reaches fewer than 15% of at-risk patients due to specialist scarcity.

### Solution
EYE-ASSISST provides a three-tier workflow:
1. **Technician** captures/uploads a fundus photograph
2. **AI Model** analyzes the image and outputs disease probabilities + a Grad-CAM heatmap
3. **Doctor** reviews the AI result with visual explanation and approves, modifies, or overrides the diagnosis

Patients can then access their scan history, book follow-up appointments, and download medical reports through a dedicated Patient Portal.

### Vision
To become a scalable, HIPAA/DPDPA-compliant AI screening layer that can plug into any hospital or clinic's workflow in India and beyond, reducing the burden on specialist ophthalmologists.

---

## 3. Target Users & Roles

| Role | Description | Primary Actions |
|------|-------------|-----------------|
| **Patient** | End-user with a registered account | View scan reports, book appointments, download PDF reports, access education content |
| **Technician** | Clinic staff who operate the screening device | Upload fundus images, enter patient metadata, submit images for AI analysis |
| **Doctor** | Licensed ophthalmologist | Review AI predictions + Grad-CAM heatmap, approve/modify/reject diagnoses, add clinical notes |
| **Admin** | Platform administrator | Monitor analytics, manage user accounts (add/remove/reset passwords), view system health |

### Authentication Model
- **4-step role selection** → credential entry flow at `/login`
- **Sign-up fields:** Full Name, Phone Number, Email, Password, Confirm Password
- **Field validation:** Email format regex, phone format, minimum 8-character password, confirm password match
- **Role enforcement:** An email registered under one role cannot log in under a different role
- **Session storage:** `localStorage` key `eye_session` stores `{ role, email, name, phone, patientId, ts }`
- **Patient portal:** Directly navigates to `/patient-portal` after login; other roles go to `/dashboard`

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────┐
│             React Frontend (Vite)            │
│  TypeScript · Tailwind CSS · Framer Motion  │
│  react-router-dom v7 · Recharts · jsPDF     │
│  Three.js / React Three Fiber               │
└──────────────────┬──────────────────────────┘
                   │ HTTP REST (fetch)
                   │ localhost:8000
┌──────────────────▼──────────────────────────┐
│           FastAPI Backend (Python)           │
│  Uvicorn · PyTorch · OpenCV · Pillow        │
│  ResNet-50 multi-label model (phase3)       │
│  Grad-CAM explainability module             │
└──────────────────┬──────────────────────────┘
                   │
       ┌───────────▼────────────┐
       │  Model Artifacts        │
       │  best_model.pth        │
       │  (phase3_multi_disease)│
       └────────────────────────┘
```

### Technology Stack

**Frontend**
| Package | Version | Purpose |
|---------|---------|---------|
| React | 19.2 | UI framework |
| TypeScript | 5.9 | Type safety |
| Vite | 7.x | Build tool |
| Tailwind CSS | 3.4 | Utility-first styling |
| Framer Motion | 12.x | Animations |
| react-router-dom | 7.14 | Client-side routing |
| Recharts | 2.15 | Analytics charts |
| Three.js / R3F | latest | 3D eye model on landing page |
| jsPDF | 4.x | PDF report generation |
| Sonner | 2.x | Toast notifications |
| Lucide React | 0.562 | Icon system |

**Backend**
| Package | Purpose |
|---------|---------|
| FastAPI | REST API framework |
| PyTorch | Deep learning inference |
| torchvision | Image preprocessing |
| OpenCV / Pillow | Image processing |
| NumPy | Array math |
| Grad-CAM (custom) | Explainability heatmap |

---

## 5. Frontend Features

### 5.1 Public Landing Page (`/`)

- **Hero section** with animated 3D eye model (Three.js / React Three Fiber)
  - Realistic anatomical eye with: cornea, iris, pupil, sclera, optic nerve, retinal vessels, lens
  - Auto-rotating, interactive on mouse hover
  - Dark radial gradient background (`#0d1a2e → #0B0F19`) for contrast
- **How It Works** — 3-step explainer with animated cards
- **AI Disease Detection** — 6-disease coverage cards with statistics
- **Research Metrics** — AUC scores, dataset statistics, publication references
- **3-tier Pricing** with monthly/annual billing toggle (Annual saves 20%)
  - Free | Professional (₹999/mo) | Enterprise (custom)
- **Footer** with full navigation links

### 5.2 Role-Based Login (`/login`)

- Step 1: Role card selection grid (Patient / Doctor / Technician / Admin) with icons
- Step 2: Sign-In or Sign-Up form for selected role
  - **Sign-Up fields:** Full Name · Phone Number · Email · Password · Confirm Password
  - **Sign-In fields:** Email · Password
- "← Back to main site" link at top
- Email format validation, phone format validation, password match + length check
- Role mismatch error if credentials belong to a different role
- Loading state with spinner while authenticating

### 5.3 Patient Portal (`/patient-portal`)

**Header**
- Logo + "Back" → navigates to `/login`
- "Patient Portal" badge
- User icon → click opens profile dropdown:
  - Editable Name + Phone fields
  - Email display (read-only)
  - "Save Changes" button (persists to `localStorage`)
  - "Sign Out" button (clears session, navigates to `/login`)

**Tab 1 — My Reports**
- Expandable scan history cards showing:
  - Scan type, laterality (Right/Left Eye), date, doctor
  - AI diagnosis badge + confidence %
  - Severity progress bar
  - Doctor's notes
  - **Download Report button** → generates formatted A4 PDF via jsPDF with:
    - Cyan header bar with EYE-ASSISST branding
    - Patient info table (name, ID, scan date, doctor, eye, diagnosis, severity, confidence)
    - Doctor's notes section
    - Footer with HIPAA/DPDPA disclaimer
  - **Print button** → `window.print()`
- Sidebar: Data Security card, Quick Actions (Book Follow-up / View Timeline / Learn More), Contact Info

**Tab 2 — Book Appointment**
- Doctor cards (Retina Specialist, Glaucoma Specialist, General Ophthalmology)
  - Name, specialty, star rating, review count, initials avatar, fee
  - Time slot selector
- Payment Gateway integration (UPI QR + Card/Net Banking modal)
- Booking confirmation animation

**Tab 3 — Health Timeline**
- Recharts AreaChart of severity over time across all scans

**Tab 4 — Learn**
- Educational content cards (articles + YouTube videos)
- Article modal with full text
- YouTube video modal with iframe embed

### 5.4 Technician Dashboard

- **Drag-and-drop upload zone** with file browser fallback
- **File validation:**
  - Accepted MIME types: `image/jpeg`, `image/png`, `image/webp`, `image/bmp`, `image/tiff`
  - File size warning if < 8KB
  - Image decode validation via `new Image()` — rejects corrupted files
  - Clear error toast for invalid file types
- **Patient metadata form:** Patient ID, Name, Age, Laterality (OD/OS)
- **Analyze button** → sends multipart POST to `/api/v1/analyze`
- **Results panel:**
  - Binary result (DR / No DR) with severity label + color
  - Confidence % with animated progress bar
  - Severity level (0–4) with severity-colored badge
  - Grad-CAM heatmap image displayed alongside original
  - Multi-disease flags panel (6 diseases with per-disease confidence bars)
  - "Disease Risk Radar" visual with color-coded rows

### 5.5 Doctor Review Portal

- **Left panel:** Pending scans list with patient ID + severity badge
- **Center viewport:**
  - **Original Image panel** — with graceful "No image stored" placeholder if image unavailable
  - **Heatmap/Original panel** — toggle between Grad-CAM heatmap and original; graceful "No heatmap available" placeholder
  - **Heatmap toggle** (Switch component)
  - **Vessel Overlay toggle** (Switch component) — applies CSS filter `saturate(4) hue-rotate(300deg) contrast(1.4) brightness(0.85)` + red border glow to highlight retinal vessels
  - Zoom controls (50%–200%)
- **Right panel:**
  - AI Assessment card (result + confidence)
  - Doctor Review: Radio group (Approve / Modify / Reject+Re-scan) with colored selection states
  - Clinical Notes textarea
  - Submit Review button → POST to `/api/v1/review/{scan_id}`
  - **After review:** Scan immediately removed from pending list (optimistic update)

### 5.6 Admin Panel

**Analytics Tab**
- Key Metrics: Total Scans, Model Accuracy (AUC), Override Rate, Active Users
- Daily Screening Volume BarChart
- Severity Distribution PieChart
- Model Performance Metrics: Sensitivity, Specificity, AUC-ROC, Accuracy (with progress bars)
- System Health: GPU Status (CUDA 8GB VRAM), API Latency, Security Status

**User Management Tab** *(new)*
- Stats bar: count per role (patient/doctor/technician/admin)
- Filter by role + name/email search
- User table columns: User (avatar initials + name + email), Role badge, Phone, Joined date, Active/Inactive toggle, Actions
- **Add User** form: Name, Email, Phone, Role dropdown → creates user in `localStorage`
- **Reset Password** → shows toast with confirmation message
- **Remove User** → confirmation dialog, removes from list
- **Toggle Active/Inactive** → click status badge to toggle
- All changes persisted to `localStorage` key `eye_users`

### 5.7 Static Info Pages

| Route | Page | Content |
|-------|------|---------|
| `/about` | About Us | Mission statement, team grid, company timeline |
| `/blog` | Blog | 4 articles with category badges, "Read More" modal with full article text |
| `/careers` | Careers | 3 job listings (accordion expand/collapse), Apply modal with email client integration |
| `/case-studies` | Case Studies | 3 Indian hospital case studies with outcome quotes |
| `/contact` | Contact | Contact form with success state |
| `/privacy` | Privacy Policy | HIPAA policy, DPDPA 2023 rights |
| `/docs` | API Docs | REST endpoint reference with JSON examples |
| `/research` | Research | AUC scores, publication list, dataset statistics |

### 5.8 Global UX Features

- **CursorEffect** — animated custom cursor rendered globally on all pages
- **ScrollToTop** — `useEffect` fires `window.scrollTo(0,0)` on every route change (via `useLocation`)
- **Dark theme** throughout — `bg-[#070A12]` / `bg-background` / card-based layout
- **Toast notifications** (Sonner) for all user actions
- **Responsive layout** — all pages use Tailwind responsive breakpoints (`sm:` `md:` `lg:`)

---

## 6. AI Model Specifications

### 6.1 Model Architecture

| Property | Value |
|----------|-------|
| Base model | ResNet-50 (ImageNet pretrained) |
| Task | Multi-label binary classification |
| Output | 6 independent sigmoid probabilities |
| Input size | 224×224 RGB |
| Loss function | BCEWithLogitsLoss with class positive weights |
| Optimizer | (configured in training script) |
| Training epochs | Up to 50 (early stopping on val F1 macro) |

### 6.2 Disease Classes

| Index | Disease | Abbreviation |
|-------|---------|--------------|
| 0 | Diabetic Retinopathy | DR |
| 1 | Glaucoma | Glaucoma |
| 2 | Age-related Macular Degeneration | AMD |
| 3 | Cataract | Cataract |
| 4 | Hypertensive Retinopathy | Hypertensive |
| 5 | Myopic Maculopathy | Myopic |

### 6.3 Class Imbalance Correction

Positive class weights applied during training to handle severe dataset imbalance:

| Disease | Pos Weight |
|---------|-----------|
| DR | 2.42× |
| Glaucoma | 14.69× |
| AMD | 33.78× |
| Cataract | 50.00× |
| Hypertensive | 21.31× |
| Myopic | 5.87× |

### 6.4 Training Results (Epoch 1 baseline)

| Metric | Value |
|--------|-------|
| Train Loss | 0.3657 |
| Train F1 (micro/macro) | 0.4654 / 0.5167 |
| Val Loss | 0.2940 |
| Val F1 (micro/macro) | 0.2499 / 0.3382 |
| Val AUC (macro) | 0.7975 |

Per-class (after epoch 2 best model):

| Disease | F1 | AUC | Status |
|---------|----|-----|--------|
| DR | 0.7224 | 0.9016 | ✅ Strong |
| Glaucoma | 0.1419 | 0.7328 | ⚠️ Lagging |
| AMD | 0.0805 | 0.6942 | ⚠️ Lagging |
| Cataract | 0.0690 | 0.9224 | ⚠️ Low F1 but high AUC |
| Hypertensive | 0.1161 | 0.6593 | ⚠️ Lagging |
| Myopic | 0.9262 | 0.9512 | ✅ Strong |

### 6.5 Explainability — Grad-CAM

- **Method:** Gradient-weighted Class Activation Mapping on final convolutional layer
- **Output:** Heatmap image overlaid on original fundus in false-color (blue → red = low → high attention)
- **Displayed:** Side-by-side with original image in Doctor Review Portal
- **Vessel Overlay:** CSS filter applied client-side (`saturate(4) hue-rotate(300deg)`) to highlight vessel structures

### 6.6 Model Files

| File | Location |
|------|----------|
| Training script | `phase3_multi_disease/train.py` |
| Inference pipeline | `backend/main.py` |
| Best model checkpoint | `phase3_multi_disease/checkpoints/best_model.pth` |
| Previous model backup | Preserved separately before each retraining run |

---

## 7. Backend API Specification

**Base URL:** `http://localhost:8000`

### POST `/api/v1/analyze`

Upload a fundus image for AI analysis.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Fundus image (JPEG/PNG/WEBP) |
| `patient_id` | string | ✅ | Patient identifier |
| `laterality` | string | ✅ | `OD` (right) or `OS` (left) |

**Response:** `application/json`
```json
{
  "scan_id": "SCAN-20260425-001",
  "patient_id": "P-2026-0041",
  "timestamp": "2026-04-25T12:00:00Z",
  "binary_result": "DR Detected",
  "confidence": 87.5,
  "severity_level": 2,
  "severity_label": "Moderate DR",
  "multi_disease": {
    "dr": { "probability": 0.875, "detected": true },
    "glaucoma": { "probability": 0.12, "detected": false },
    "amd": { "probability": 0.05, "detected": false },
    "cataract": { "probability": 0.03, "detected": false },
    "hypertensive": { "probability": 0.08, "detected": false },
    "myopic": { "probability": 0.92, "detected": true }
  },
  "gradcam_available": true,
  "original_image": "data:image/jpeg;base64,...",
  "heatmap_image": "data:image/jpeg;base64,..."
}
```

### GET `/api/scans`

List all scan records.

**Response:**
```json
{
  "scans": [
    {
      "scan_id": "SCAN-001",
      "patient_id": "P-2026-0041",
      "timestamp": "2026-04-25T12:00:00Z",
      "binary_result": "DR Detected",
      "confidence": 87.5,
      "severity_level": 2,
      "severity_label": "Moderate DR"
    }
  ]
}
```

### GET `/api/v1/scan/{scan_id}`

Get full details of a specific scan including images.

### POST `/api/v1/review/{scan_id}`

Submit doctor review for a scan.

**Request body:**
```json
{
  "action": "approve",
  "notes": "Clinical observations here",
  "doctor_id": "DOC-001"
}
```
`action` values: `approve` | `modify` | `override`

### GET `/api/v1/analytics/statistics`

Get platform-level statistics (Admin only).

**Response:**
```json
{
  "total_scans": 142,
  "severity_distribution": { "0": 60, "1": 40, "2": 25, "3": 12, "4": 5 },
  "reviews_submitted": 38,
  "override_rate": 7.9
}
```

---

## 8. Data Flow & Workflows

### 8.1 Primary Screening Workflow

```
Technician uploads fundus image
    │
    ▼
Frontend validates: MIME type + file size + image decode
    │  Invalid → toast error, reject
    ▼
POST /api/v1/analyze (multipart)
    │
    ▼
Backend: OpenCV resize → 224×224 → normalize
    │
    ▼
ResNet-50 forward pass → 6 sigmoid outputs
    │
    ▼
Grad-CAM on final conv layer → heatmap image
    │
    ▼
Response: probabilities + severity + base64 images
    │
    ▼
Frontend renders: result card + heatmap + multi-disease bars
    │
    ▼
Scan appears in Doctor Review Portal pending list
```

### 8.2 Doctor Review Workflow

```
Doctor selects scan from pending list
    │
    ▼
Backend fetches scan images (GET /api/v1/scan/{id})
    │  No image stored → graceful placeholder shown
    ▼
Doctor reviews: original image / Grad-CAM / vessel overlay
    │
    ▼
Doctor selects action (Approve / Modify / Override)
    │
    ▼
POST /api/v1/review/{scan_id}
    │
    ▼
Scan removed from pending list immediately (optimistic)
    │
    ▼
Backend refreshes scan list
```

### 8.3 Patient Report Download Workflow

```
Patient clicks "Download Report" on a scan card
    │
    ▼
jsPDF instantiated (A4 portrait)
    │
    ▼
Cyan header bar + EYE-ASSISST branding rendered
    │
    ▼
Patient info table (name, ID, scan date, doctor, eye, diagnosis, severity, confidence)
    │
    ▼
Doctor's notes section with auto-wrap
    │
    ▼
HIPAA/DPDPA footer rendered
    │
    ▼
doc.save("EyeReport-SCAN-XXX-P-XXXX.pdf") → browser download
```

---

## 9. Security & Compliance

### Authentication
- Role-based credential enforcement (email tied to one role)
- Session stored in `localStorage` with role + timestamp
- No JWT/server session in current version (planned)

### Data Privacy
- **HIPAA:** No PHI stored server-side in current demo; designed for HIPAA-compliant deployment
- **DPDPA 2023 (India):** Privacy policy page discloses data rights; user data deletion supported via account removal in Admin panel
- **GDPR Note:** Not primary target market but EU rights respected in design

### Image Security
- Images accepted only as JPEG/PNG/WEBP/BMP/TIFF
- File validated on both MIME type and actual image decode
- Images stored as base64 in scan records (in-memory, backend session)

### API Security (Planned)
- Bearer token authentication
- HTTPS in production
- Rate limiting on `/api/v1/analyze`

---

## 10. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| **Inference time** | < 3 seconds per image (GPU) |
| **UI responsiveness** | Mobile, tablet, desktop (320px–4K) |
| **Page load time** | < 2 seconds (Vite production build) |
| **Build bundle size** | < 2.5MB JS gzip |
| **Uptime (prod)** | 99.5% |
| **Browser support** | Chrome 120+, Firefox 115+, Edge 120+, Safari 17+ |
| **Screen sizes** | Full responsive from 320px width |
| **Accessibility** | WCAG 2.1 AA (labels, ARIA, keyboard navigation) |
| **GPU requirement** | NVIDIA RTX 3050 8GB VRAM (or equivalent CUDA-capable) |

---

## 11. Known Limitations & Roadmap

### Current Limitations
| Issue | Status |
|-------|--------|
| Backend auth is simulated (localStorage) | Known; JWT planned |
| Glaucoma/AMD/Hypertensive F1 scores low (< 0.15) | Training ongoing (epoch 2+ underway) |
| Vessel overlay is CSS filter approximation, not true segmentation | By design for v1 |
| PDF uses placeholder scan data (not live from backend) | Needs backend integration |
| Three.js EyeModel3D has TypeScript type errors (R3F JSX types not registered) | Pre-existing; does not affect runtime |
| `gsap` not installed (LandingPage uses it) | Pre-existing; GSAP ScrollTrigger effects disabled |
| User management is localStorage-only (no server persistence) | Planned |

### Roadmap

#### Phase 1 (Current — v2.0)
- [x] Multi-disease detection model (6 diseases, ResNet-50)
- [x] Grad-CAM heatmap explainability
- [x] Role-based login with validation
- [x] Patient Portal with PDF reports (jsPDF)
- [x] Admin user management panel
- [x] Doctor vessel overlay toggle
- [x] Image file validation with decode check
- [x] ScrollToTop on page navigation
- [x] Careers email client fix

#### Phase 2 (Planned)
- [ ] JWT / server-side authentication
- [ ] Real vessel segmentation model (separate endpoint)
- [ ] Improve Glaucoma, AMD, Hypertensive F1 scores (continued training)
- [ ] GSAP scroll animations on landing page
- [ ] Live patient data → PDF (pull from backend scan record)
- [ ] Telemedicine video consultation integration
- [ ] Mobile app (React Native)
- [ ] Multi-language support (Hindi, Tamil, Telugu)

#### Phase 3 (Future)
- [ ] EHR/EMR integration (HL7 FHIR)
- [ ] Government health portal API integration (Ayushman Bharat)
- [ ] NABH/NABL certification pathway
- [ ] Federated learning across hospitals (privacy-preserving)
- [ ] OCT (Optical Coherence Tomography) scan support

---

## Appendix A — Dataset

| Source | Images | Labels |
|--------|--------|--------|
| APTOS 2019 (Kaggle) | ~3,662 | DR severity (0–4) |
| ORIGA | ~650 | Glaucoma |
| MESSIDOR-2 | ~1,748 | DR grading |
| EyePACS | ~88,702 | DR (binary + grade) |
| Custom aggregation | ~11,789 | 6-class multi-label |

**Training split:** 80% train / 20% validation  
**Batch size:** 1 (single sample per step — memory constrained)  
**Total training steps per epoch:** ~9,431

---

## Appendix B — File Structure

```
EYE-ASSISST-main/
├── AI Eye Screening UI/
│   └── app/
│       ├── src/
│       │   ├── App.tsx                    ← Main app + all dashboards
│       │   ├── components/
│       │   │   ├── CursorEffect.tsx
│       │   │   ├── ScrollToTop.tsx
│       │   │   ├── UserManagementPanel.tsx
│       │   │   ├── PaymentGateway.tsx
│       │   │   └── three/
│       │   │       └── EyeModel3D.tsx
│       │   └── pages/
│       │       ├── LandingPage.tsx
│       │       ├── PatientPortal.tsx
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
├── backend/
│   └── main.py                           ← FastAPI inference server
└── phase3_multi_disease/
    ├── train.py                          ← Multi-disease training script
    └── checkpoints/
        └── best_model.pth               ← Saved best model
```

---

*Document generated by EYE-ASSISST Development Team · April 2026*  
*For internal use and academic submission only*
