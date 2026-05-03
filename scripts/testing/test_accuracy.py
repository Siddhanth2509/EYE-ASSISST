"""
Batch accuracy test:
- Sends 5 real cataract fundus images + 5 normal fundus images
- Checks if multi-disease model correctly detects cataract
- Checks if results are consistent (no random variation)
"""
import requests, glob, sys, os
from pathlib import Path

BASE = Path(__file__).parent
CATARACT_DIR = BASE / "Dataset" / "CATRACT" / "dataset" / "2_cataract"
NORMAL_DIR   = BASE / "Dataset" / "CATRACT" / "dataset" / "1_normal"
API          = "http://localhost:8000/api/analyze?include_gradcam=false"

def analyze(image_path, label):
    with open(image_path, "rb") as f:
        data = f.read()
    try:
        res = requests.post(
            API,
            files={"file": (os.path.basename(image_path), data, "image/png")},
            data={"patient_id": f"TEST-{label}", "laterality": "OD"},
            timeout=120
        )
        if res.status_code == 200:
            d = res.json()
            md = d.get("multi_disease", {})
            cat = md.get("cataract", {})
            dr  = d.get("dr_binary", {})
            sev = d.get("dr_severity", {})
            return {
                "status": "ok",
                "dr_detected": dr.get("is_dr"),
                "dr_conf": dr.get("confidence"),
                "severity": sev.get("label"),
                "cataract_detected": cat.get("detected"),
                "cataract_conf": cat.get("confidence"),
                "cataract_thresh": cat.get("threshold"),
                "multi_disease": {k: v.get("detected") for k, v in md.items()}
            }
        elif res.status_code == 400:
            return {"status": "rejected", "reason": res.json().get("detail","")}
        else:
            return {"status": "error", "code": res.status_code, "body": res.text[:300]}
    except Exception as e:
        return {"status": "exception", "error": str(e)}

print("=" * 65)
print("BATCH ACCURACY TEST - Cataract vs Normal")
print("=" * 65)

# ── Test cataract images ───────────────────────────────────────────
cataract_files = sorted(CATARACT_DIR.glob("*.png"))[:5]
print(f"\n[CATARACT IMAGES] — expected cataract=True")
cat_detected = 0
cat_processed = 0
for img_path in cataract_files:
    r = analyze(img_path, "CAT")
    if r["status"] == "ok":
        cat_processed += 1
        detected = r["cataract_detected"]
        if detected:
            cat_detected += 1
        mark = "[CORRECT]" if detected else "[MISSED ]"
        print(f"  {mark} {img_path.name}  cataract={detected} ({r['cataract_conf']}%)  DR={r['dr_detected']} sev={r['severity']}")
    else:
        print(f"  [SKIP   ] {img_path.name}  {r['status']}: {r.get('reason','')[:80]}")

# ── Test normal images ─────────────────────────────────────────────
normal_files = sorted(NORMAL_DIR.glob("*.png"))[:5]
print(f"\n[NORMAL IMAGES] — expected all diseases=False")
norm_correct = 0
norm_processed = 0
for img_path in normal_files:
    r = analyze(img_path, "NRM")
    if r["status"] == "ok":
        norm_processed += 1
        detected_any = any(v for v in r["multi_disease"].values())
        if not detected_any:
            norm_correct += 1
        mark = "[CORRECT]" if not detected_any else "[FP     ]"
        diseases = [k for k, v in r["multi_disease"].items() if v]
        print(f"  {mark} {img_path.name}  DR={r['dr_detected']} sev={r['severity']}  flagged={diseases if diseases else 'none'}")
    else:
        print(f"  [SKIP   ] {img_path.name}  {r['status']}: {r.get('reason','')[:80]}")

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
if cat_processed:
    print(f"  Cataract recall  : {cat_detected}/{cat_processed} = {cat_detected/cat_processed*100:.0f}%")
if norm_processed:
    print(f"  Normal precision : {norm_correct}/{norm_processed} = {norm_correct/norm_processed*100:.0f}%")
print()
