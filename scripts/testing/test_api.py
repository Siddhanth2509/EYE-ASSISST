"""
End-to-end test of the backend API - tests health, DR detection, and multi-disease
"""
import requests, io, sys
from PIL import Image, ImageDraw

print("=== BACKEND END-TO-END TEST ===\n")

# ── 1. Health check ───────────────────────────────────────────────────────────
try:
    h = requests.get("http://localhost:8000/health", timeout=5).json()
    status = h["status"]
    loaded = h["model_loaded"]
    print(f"[OK] Health: {status} | model_loaded={loaded}")
    if not loaded:
        print("[WARN] Model not loaded - results will be wrong!")
except Exception as e:
    print(f"[FAIL] Health check failed: {e}")
    sys.exit(1)

# ── 2. Create a realistic red fundus-like image ───────────────────────────────
print("\nCreating test fundus image...")
img = Image.new("RGB", (512, 512), color=(30, 10, 10))
draw = ImageDraw.Draw(img)
# Draw a large red circle (simulates fundus)
draw.ellipse([30, 30, 482, 482], fill=(160, 50, 40), outline=(200, 60, 50))
# Draw a bright optic disc
draw.ellipse([180, 200, 240, 260], fill=(240, 200, 160))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=95)
buf.seek(0)

# ── 3. Call analyze endpoint ──────────────────────────────────────────────────
print("Calling /api/analyze ...")
try:
    res = requests.post(
        "http://localhost:8000/api/analyze?include_gradcam=false",
        files={"file": ("fundus_test.jpg", buf, "image/jpeg")},
        data={"patient_id": "TEST-001", "laterality": "OD"},
        timeout=90
    )
    print(f"HTTP Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        dr  = data.get("dr_binary", {})
        sev = data.get("dr_severity", {})
        md  = data.get("multi_disease", {})
        print(f"\n  DR Detected  : {dr.get('is_dr')}")
        print(f"  Confidence   : {dr.get('confidence')}%")
        print(f"  Severity     : {sev.get('label')} (grade {sev.get('grade')})")
        print("\n  Multi-Disease Results:")
        if md:
            for k, v in md.items():
                name      = v.get("name", k)
                detected  = v.get("detected")
                conf      = v.get("confidence")
                threshold = v.get("threshold")
                flag = "[DETECTED]" if detected else "[ ok ]"
                print(f"    {flag} {name:40s} conf={conf:5.1f}%  thresh={threshold}")
        else:
            print("    (no multi-disease data returned)")
    elif res.status_code == 400:
        detail = res.json().get("detail", "")
        print(f"  Image rejected (fundus validation): {detail}")
    else:
        print(f"  ERROR: {res.text[:600]}")
except Exception as e:
    print(f"[FAIL] Request failed: {e}")
