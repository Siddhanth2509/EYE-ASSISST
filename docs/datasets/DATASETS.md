# Dataset Documentation

All datasets used in EYE-ASSISST Phase 3 Multi-Disease Training.

---

## Quick Summary

| # | Dataset | Images | Disease | Label Type |
|---|---|---|---|---|
| 1 | DR Unified v2 | 92,501 | Diabetic Retinopathy | Folder grade (0–4) |
| 2 | Augmented Resized V2 | 143,669 | Diabetic Retinopathy | Folder grade (0–4) |
| 3 | ODIR-5K | 10,000 | All 6 diseases | Real clinical CSV |
| 4 | AMDNet23 | ~3,988 | AMD, Cataract, DR | Class folders |
| 5 | Glaucoma Fundus Detection | 18,842 | Glaucoma | Binary folders (0/1) |
| 6 | REFUGE2 | ~1,600 | Glaucoma | All positive |
| 7 | Eye Diseases Classification | 4,217 | Cataract, DR, Glaucoma | Class folders |
| 8 | HRDC Hypertension | 1,424 | Hypertensive Retinopathy | CSV (Image, Hypertensive) |
| 9 | Messidor-2 | 1,744 | Diabetic Retinopathy | CSV (id_code, diagnosis) |
| 10 | Myopia Images | 100,543 | Pathologic Myopia | Myopia_images / Normal_images |
| 11 | CATRACT | 1,202 | Cataract, Glaucoma | Folder name |
| **Total** | | **~379,730** | **6 diseases** | |

---

## Disease Coverage

| Disease | Primary Dataset | Supporting | Total Positives (est.) |
|---|---|---|---|
| Diabetic Retinopathy | DR Unified v2, Augmented V2 | ODIR, Messidor-2, EyeDiseases | ~120,000 |
| Glaucoma | Glaucoma Fundus Detection | REFUGE2, ODIR, EyeDiseases | ~15,000 |
| AMD | AMDNet23 | ODIR | ~2,800 |
| Cataract | AMDNet23, CATRACT | ODIR, EyeDiseases | ~4,000 |
| Hypertensive Ret. | HRDC | ODIR | ~900 |
| Pathologic Myopia | Myopia Images | ODIR | ~48,000 |

---

## Individual Dataset Details

### 1. DR Unified v2
- **Source:** Kaggle EyePACS + clinical aggregation
- **Structure:** `dr_unified_v2/{grade}/image.jpg` (grade = 0,1,2,3,4)
- **Labels:** Grade 0 = No DR, Grade 1–4 = DR Positive
- **Used for:** DR severity classification

### 2. Augmented Resized V2
- **Source:** Augmented EyePACS (resized + color-jittered)
- **Structure:** `augmented_resized_V2/train/{grade}/`, val/, test/
- **Labels:** Same grade-based as DR Unified

### 3. ODIR-5K (Ocular Disease Intelligent Recognition)
- **Source:** Peking University / iChallenge 2019
- **Structure:** `ODIR/Training Set/Images/` + `full_df.csv`
- **CSV columns:** `Left-Fundus, Right-Fundus, D, G, C, A, H, M, N, label`
  - D=Diabetic, G=Glaucoma, C=Cataract, A=AMD, H=Hypertensive, M=Myopic
- **Patients:** 5,000 × 2 eyes = 10,000 images
- **Label quality:** ⭐⭐⭐⭐⭐ Real clinical diagnoses

### 4. AMDNet23 Fundus Dataset
- **Source:** AMDNet23 Challenge
- **Structure:** `AMD/AMD1/.../AMDNet23 Dataset/train/{amd,cataract,diabetes,normal}/`
- **Classes:** amd, cataract, diabetes (=DR), normal
- **Label quality:** ⭐⭐⭐⭐ Expert-annotated

### 5. Fundus Glaucoma Detection
- **Source:** Kaggle (Fundus Glaucoma Detection Dataset)
- **Structure:** `GLAUCOMA_DETECTION/Fundus Glaucoma Detection Data/{train,val,test}/{0,1}/`
  - 0 = Non-Glaucoma, 1 = Glaucoma
- **Split:** train 8,621 / val 1,287 / test 1,288 (approx)
- **Label quality:** ⭐⭐⭐⭐

### 6. REFUGE2 (Retinal Fundus Glaucoma Challenge)
- **Source:** MICCAI 2020 Challenge
- **Structure:** `REFUGE2/{train,val,test}/images/` + `masks/`
- **Note:** All images are glaucoma-challenge images → treated as glaucoma positive
- **Label quality:** ⭐⭐⭐⭐⭐ Expert annotated with optic disc masks

### 7. Eye Diseases Classification
- **Source:** Kaggle (konduri-niharika)
- **Structure:** `eye_diseases_classification/dataset/{cataract,diabetic_retinopathy,glaucoma,normal}/`
- **Images:** 4,217 total
- **Label quality:** ⭐⭐⭐

### 8. HRDC Hypertension Dataset
- **Source:** Hypertensive Retinopathy Detection Challenge (ISBI)
- **Structure:** `Hypertension.../1-Hypertensive Classification/.../1-Images/1-Training Set/`
- **CSV:** `HRDC Hypertensive Classification Training Labels.csv` → `Image, Hypertensive`
- **Label quality:** ⭐⭐⭐⭐⭐ Clinical ground truth

### 9. Messidor-2
- **Source:** ADCIS / Messidor Program (France)
- **CSV:** `messidor_data.csv` → `id_code, diagnosis, adjudicated_dme, adjudicated_gradable`
- **Labels:** diagnosis 0=No DR, 1=DR
- **Images:** 1,744 high-quality retinal photographs
- **Label quality:** ⭐⭐⭐⭐⭐ Expert ophthalmologist grading

### 10. Myopia Images
- **Source:** Clinical fundus collection
- **Structure:** `Myopia images/Images/Myopia_images/` (47,025) and `Normal_images/` (53,518)
- **Labels:** Folder-based binary (myopic vs normal)
- **Label quality:** ⭐⭐⭐⭐

### 11. CATRACT
- **Source:** Kaggle
- **Structure:** `CATRACT/dataset/{cataract,glaucoma,retina,normal}/`
- **Label quality:** ⭐⭐⭐

---

## How Labels Are Generated

For datasets with **real clinical labels** (ODIR, Messidor-2, HRDC, REFUGE2): labels come directly from the provided CSV / challenge annotation.

For datasets with **single-disease folder labels**: the primary disease is set to 1. Co-morbidities for other diseases are set to 0 (or small realistic bootstrap probabilities where clinically plausible, seeded for reproducibility).

> Bootstrap prevalence rates used:
> - DR: 8–15% co-morbidity
> - Glaucoma: 4–5%
> - AMD: 3–4%
> - Hypertensive: 5–8%
> - Myopic: 3%

---

## CSV Format

Both `train_unified_v5.csv` and `val_unified_v5.csv` share this schema:

```
image_path, dr, glaucoma, amd, cataract, hypertensive, myopic
Dataset/ODIR/Training Set/Images/0_left.jpg, 1, 0, 0, 0, 0, 0
Dataset/AMD/.../.../train/amd/image001.png, 0, 0, 1, 0, 0, 0
```

- **image_path**: relative to project root, forward slashes
- All disease columns: **0** or **1** (binary multi-label)
- Split: **80% train / 20% val**, stratified shuffle (seed=42)
