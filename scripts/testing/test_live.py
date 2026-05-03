"""
Live test: send a real cataract image to the backend WITH Grad-CAM enabled
(same as the frontend does) and measure timing.
"""
import requests, time, sys
from pathlib import Path

BASE = Path(__file__).parent
CATARACT_IMG = BASE / "Dataset" / "CATRACT" / "dataset" / "2_cataract" / "cataract_001.png"
GLAUCOMA_DIR = BASE / "Dataset" / "CATRACT" / "dataset" / "2_glaucoma"

API = "http://localhost:8000"

def test_image(img_path, label):
    print(f"\n{'='*55}")
    print(f"  Testing: {img_path.name} [{label}]")
    print(f"{'='*55}")
    with open(img_path, "rb") as f:
        data = f.read()
    print(f"  File size : {len(data)/1024:.0f} KB")
    
    start = time.time()
    try:
        res = requests.post(
            f"{API}/api/analyze?include_gradcam=true",
            files={"file": (img_path.name, data, "image/png")},
            data={"patient_id": f"TEST-{label}", "laterality": "OD"},
            timeout=120
        )
        elapsed = time.time() - start
        print(f"  HTTP status : {res.status_code}")
        print(f"  Time taken  : {elapsed:.1f}s")

        if res.status_code == 200:
            d = res.json()
            dr  = d.get("dr_binary", {})
            sev = d.get("dr_severity", {})
            md  = d.get("multi_disease", {})
            gcam = d.get("gradcam", {})
            print(f"  DR detected : {dr.get('is_dr')}  ({dr.get('confidence')}%)")
            print(f"  Severity    : {sev.get('label')} (grade {sev.get('grade')})")
            print(f"  GradCAM     : available={gcam.get('available')}, has_data={bool(gcam.get('heatmap_base64'))}")
            print(f"  Multi-disease:")
            for k, v in md.items():
                flag = "[DETECTED]" if v.get("detected") else "[ ok ]"
                print(f"    {flag} {v.get('name',k):40s} {v.get('confidence',0):5.1f}%  (thresh={v.get('threshold')})")
            return True
        elif res.status_code == 400:
            print(f"  Rejected: {res.json().get('detail','')[:120]}")
            return False
        else:
            print(f"  ERROR {res.status_code}: {res.text[:300]}")
            return False
    except requests.exceptions.Timeout:
        print(f"  TIMEOUT after {time.time()-start:.0f}s - backend too slow!")
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False

# Run tests
print("LIVE BACKEND TEST - Real Disease Images with Grad-CAM")
print("="*55)

# Test 1: Cataract
ok1 = test_image(CATARACT_IMG, "CATARACT")

# Test 2: Glaucoma (first available)
glaucoma_files = sorted(GLAUCOMA_DIR.glob("*.png"))
if glaucoma_files:
    ok2 = test_image(glaucoma_files[0], "GLAUCOMA")
else:
    print("\nNo glaucoma images found")
    ok2 = False

print(f"\n{'='*55}")
print(f"  Cataract test : {'PASS' if ok1 else 'FAIL'}")
print(f"  Glaucoma test : {'PASS' if ok2 else 'FAIL'}")
print(f"{'='*55}\n")
