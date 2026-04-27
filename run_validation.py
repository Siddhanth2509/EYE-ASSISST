import os
import sys
from pathlib import Path
from PIL import Image

# Add project root and backend to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Import models from backend
from backend.main import model as dr_model, multi_disease_detector, get_fundus_classifier

fundus_classifier = get_fundus_classifier()

def test_image(image_path, expected_label):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
        
    print(f"\n{'-'*60}")
    print(f"Testing Image: {os.path.basename(image_path)}")
    print(f"Expected Category: {expected_label}")
    print(f"{'-'*60}")
    
    try:
        img = Image.open(image_path)
        
        # 1. Fundus validation
        is_fundus, prob_fundus, msg = fundus_classifier.predict(img)
        print(f"\n[1] Fundus Classifier (Threshold: {fundus_classifier.threshold})")
        print(f"    Result: {'FUNDUS' if is_fundus else 'NON-FUNDUS'}")
        print(f"    Confidence: {prob_fundus*100:.1f}%")
        
        if not is_fundus:
            print(f"    Message: {msg}")
            
        # 2. DR Severity
        print(f"\n[2] DR Severity Model")
        dr_res = dr_model.predict(img)
        print(f"    Predicted: {dr_res['severity_label']} (Level {dr_res['severity_level']})")
        print(f"    Confidence: {dr_res['confidence']:.1f}%")
        print(f"    Raw Probs: {[f'{p:.3f}' for p in dr_res['severity_probs']]}")
        
        # 3. Multi-Disease
        print(f"\n[3] Multi-Disease Model")
        md_res = multi_disease_detector.predict(img)
        detected = []
        for disease, data in md_res.items():
            if data['detected']:
                detected.append(f"{disease} ({data['confidence']:.1f}%)")
        if detected:
            print(f"    Detected: {', '.join(detected)}")
        else:
            print(f"    Detected: None")
            
    except Exception as e:
        print(f"Error processing image: {e}")

test_images = [
    ("D:/MAJOR_PROJECT/EYE-ASSISST-main/EYE-ASSISST-main/Dataset/dr_unified_v2/dr_unified_v2/test/0/0212dd31f623.jpg", "Healthy (Normal DR)"),
    ("D:/MAJOR_PROJECT/EYE-ASSISST-main/EYE-ASSISST-main/Dataset/dr_unified_v2/dr_unified_v2/test/3/1b32e1d775ea.jpg", "Severe DR"),
    ("D:/MAJOR_PROJECT/EYE-ASSISST-main/EYE-ASSISST-main/Dataset/CATRACT/dataset/1_normal/NL_001.png", "Cataract Dataset (Normal)"),
    ("D:/MAJOR_PROJECT/EYE-ASSISST-main/EYE-ASSISST-main/phase0_fundus_classifier/data/non_fundus/simple/synthetic_0000.jpg", "Non-Fundus")
]

# Run tests
print(f"Starting Validation Tests for 3 Models...")
for img_path, label in test_images:
    test_image(img_path, label)
