"""
EYE-ASSISST FastAPI Backend
AI-Powered Eye Disease Screening API
"""

import os
import io
import base64
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import Grad-CAM module
from gradcam_module import GradCAMGenerator

# Initialize FastAPI app
app = FastAPI(
    title="EYE-ASSISST API",
    description="AI-Powered Eye Disease Screening Backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENUMS AND MODELS
# ============================================================================

class UserRole(str, Enum):
    DOCTOR = "doctor"
    TECHNICIAN = "technician"
    ADMIN = "admin"

class SeverityLevel(int, Enum):
    NORMAL = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    PROLIFERATIVE = 4

class DiseaseType(str, Enum):
    DR = "diabetic_retinopathy"
    AMD = "amd"
    GLAUCOMA = "glaucoma"
    CATARACT = "cataract"

class AnalysisResult(BaseModel):
    scan_id: str
    patient_id: str
    timestamp: str
    binary_result: str  # "Normal" or "DR Detected"
    confidence: float
    severity_level: int
    severity_label: str
    severity_color: str
    multi_disease_flags: Dict[str, Any]
    gradcam_available: bool

class DoctorReview(BaseModel):
    scan_id: str
    doctor_id: str
    action: str  # "approve", "edit", "override"
    final_diagnosis: Optional[str] = None
    final_severity: Optional[int] = None
    notes: Optional[str] = None
    timestamp: str

class ModelMetrics(BaseModel):
    sensitivity: float
    specificity: float
    auc: float
    accuracy: float
    total_scans: int
    avg_confidence: float

class SystemStatus(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    version: str
    uptime: str

# ============================================================================
# MOCK ML MODEL (Replace with actual model loading)
# ============================================================================

class MockEyeDiseaseModel:
    """Mock ML model for demonstration. Replace with actual model."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = True
        self.input_size = (512, 512)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load a pretrained ResNet for feature extraction (mock)
        try:
            self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            self.feature_extractor = None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Run inference on fundus image."""
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Mock prediction (replace with actual model inference)
        # Simulate realistic DR detection results
        np.random.seed(hash(image.tobytes()) % 2**32)
        
        # Generate pseudo-realistic results
        confidence = float(np.random.beta(7, 2))  # Skewed toward high confidence
        severity_probs = np.random.dirichlet([3, 2, 1.5, 1, 0.5])
        severity_level = int(np.argmax(severity_probs))
        
        # Multi-disease detection (mock)
        multi_disease = {
            "amd": {"detected": bool(np.random.random() > 0.85), "confidence": float(np.random.random())},
            "glaucoma": {"detected": bool(np.random.random() > 0.90), "confidence": float(np.random.random())},
            "cataract": {"detected": bool(np.random.random() > 0.88), "confidence": float(np.random.random())},
        }
        
        # Determine binary result
        binary_result = "DR Detected" if severity_level > 0 else "Normal"
        
        severity_labels = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
        severity_colors = ["#10B981", "#F59E0B", "#F97316", "#EF4444", "#DC2626"]
        
        return {
            "binary_result": binary_result,
            "confidence": round(confidence * 100, 2),
            "severity_level": severity_level,
            "severity_label": severity_labels[severity_level],
            "severity_color": severity_colors[severity_level],
            "severity_probs": severity_probs.tolist(),
            "multi_disease": multi_disease,
            "features": self.extract_features(input_tensor)
        }
    
    def extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract feature maps for Grad-CAM."""
        if self.feature_extractor is None:
            return torch.randn(1, 2048, 16, 16).to(self.device)
        
        with torch.no_grad():
            # Extract features from layer4
            x = self.feature_extractor.conv1(input_tensor)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)
            x = self.feature_extractor.layer1(x)
            x = self.feature_extractor.layer2(x)
            x = self.feature_extractor.layer3(x)
            x = self.feature_extractor.layer4(x)
        return x

# Initialize model and Grad-CAM
model = MockEyeDiseaseModel()
gradcam_generator = GradCAMGenerator(model)

# In-memory storage (replace with database in production)
scan_database: Dict[str, Dict] = {}
review_database: Dict[str, Dict] = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "EYE-ASSISST API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Check system health and model status."""
    return SystemStatus(
        status="healthy",
        model_loaded=model.model_loaded,
        gpu_available=torch.cuda.is_available(),
        version="1.0.0",
        uptime="active"
    )

@app.post("/api/v1/analyze", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    patient_id: str = Form("unknown"),
    laterality: str = Form("OD"),
    age: Optional[int] = Form(None)
):
    """
    Upload and analyze a fundus image.
    
    - **file**: Fundus image file (JPG, PNG)
    - **patient_id**: Patient identifier
    - **laterality**: Eye side (OD=right, OS=left)
    - **age**: Patient age (optional)
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate scan ID
        scan_id = f"SCAN-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.utcnow().isoformat()
        
        # Run AI inference
        prediction = model.predict(image)
        
        # Generate Grad-CAM heatmap
        heatmap_base64 = gradcam_generator.generate_heatmap(image, prediction)
        
        # Store scan data
        scan_data = {
            "scan_id": scan_id,
            "patient_id": patient_id,
            "timestamp": timestamp,
            "laterality": laterality,
            "age": age,
            "original_image": image_to_base64(image),
            "heatmap_image": heatmap_base64,
            "prediction": prediction,
            "review_status": "pending"
        }
        scan_database[scan_id] = scan_data
        
        return AnalysisResult(
            scan_id=scan_id,
            patient_id=patient_id,
            timestamp=timestamp,
            binary_result=prediction["binary_result"],
            confidence=prediction["confidence"],
            severity_level=prediction["severity_level"],
            severity_label=prediction["severity_label"],
            severity_color=prediction["severity_color"],
            multi_disease_flags=prediction["multi_disease"],
            gradcam_available=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/v1/heatmap/{scan_id}")
async def get_heatmap(scan_id: str):
    """Get Grad-CAM heatmap for a specific scan."""
    if scan_id not in scan_database:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    scan_data = scan_database[scan_id]
    return {
        "scan_id": scan_id,
        "heatmap_image": scan_data["heatmap_image"],
        "original_image": scan_data["original_image"]
    }

@app.post("/api/v1/review")
async def submit_review(review: DoctorReview):
    """Submit doctor review for an AI analysis."""
    if review.scan_id not in scan_database:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    review_data = {
        "review_id": f"REV-{uuid.uuid4().hex[:8].upper()}",
        "scan_id": review.scan_id,
        "doctor_id": review.doctor_id,
        "action": review.action,
        "final_diagnosis": review.final_diagnosis,
        "final_severity": review.final_severity,
        "notes": review.notes,
        "timestamp": review.timestamp
    }
    
    review_database[review.review_id] = review_data
    scan_database[review.scan_id]["review_status"] = review.action
    scan_database[review.scan_id]["doctor_review"] = review_data
    
    return {"status": "success", "review_id": review_data["review_id"]}

@app.get("/api/v1/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    """Get screening history for a patient."""
    patient_scans = [
        {
            "scan_id": scan["scan_id"],
            "timestamp": scan["timestamp"],
            "laterality": scan["laterality"],
            "severity_level": scan["prediction"]["severity_level"],
            "severity_label": scan["prediction"]["severity_label"],
            "severity_color": scan["prediction"]["severity_color"],
            "confidence": scan["prediction"]["confidence"],
            "review_status": scan["review_status"]
        }
        for scan in scan_database.values()
        if scan["patient_id"] == patient_id
    ]
    
    # Sort by timestamp descending
    patient_scans.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "patient_id": patient_id,
        "total_scans": len(patient_scans),
        "scans": patient_scans
    }

@app.get("/api/v1/patient/{patient_id}/comparison")
async def compare_scans(patient_id: str, scan1_id: str, scan2_id: str):
    """Compare two scans for a patient."""
    if scan1_id not in scan_database or scan2_id not in scan_database:
        raise HTTPException(status_code=404, detail="One or both scans not found")
    
    scan1 = scan_database[scan1_id]
    scan2 = scan_database[scan2_id]
    
    return {
        "patient_id": patient_id,
        "comparison": {
            "scan1": {
                "scan_id": scan1_id,
                "timestamp": scan1["timestamp"],
                "severity_level": scan1["prediction"]["severity_level"],
                "confidence": scan1["prediction"]["confidence"]
            },
            "scan2": {
                "scan_id": scan2_id,
                "timestamp": scan2["timestamp"],
                "severity_level": scan2["prediction"]["severity_level"],
                "confidence": scan2["prediction"]["confidence"]
            },
            "severity_change": scan2["prediction"]["severity_level"] - scan1["prediction"]["severity_level"]
        }
    }

@app.get("/api/v1/analytics/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics."""
    total_scans = len(scan_database)
    
    # Calculate mock metrics (replace with actual metrics from validation)
    if total_scans > 0:
        avg_confidence = np.mean([s["prediction"]["confidence"] for s in scan_database.values()])
    else:
        avg_confidence = 0.0
    
    return ModelMetrics(
        sensitivity=0.94,
        specificity=0.89,
        auc=0.96,
        accuracy=0.91,
        total_scans=total_scans,
        avg_confidence=round(avg_confidence, 2)
    )

@app.get("/api/v1/analytics/statistics")
async def get_screening_statistics():
    """Get screening volume statistics."""
    total_scans = len(scan_database)
    
    # Count by severity
    severity_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for scan in scan_database.values():
        severity_counts[scan["prediction"]["severity_level"]] += 1
    
    # Count reviews
    reviews_total = len(review_database)
    overrides = sum(1 for r in review_database.values() if r["action"] == "override")
    
    return {
        "total_scans": total_scans,
        "severity_distribution": severity_counts,
        "reviews_submitted": reviews_total,
        "override_rate": round(overrides / reviews_total * 100, 2) if reviews_total > 0 else 0,
        "pending_reviews": total_scans - reviews_total
    }

@app.get("/api/v1/scan/{scan_id}")
async def get_scan_details(scan_id: str):
    """Get full details of a scan including images."""
    if scan_id not in scan_database:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    scan_data = scan_database[scan_id]
    return {
        "scan_id": scan_id,
        "patient_id": scan_data["patient_id"],
        "timestamp": scan_data["timestamp"],
        "laterality": scan_data["laterality"],
        "age": scan_data["age"],
        "prediction": scan_data["prediction"],
        "original_image": scan_data["original_image"],
        "heatmap_image": scan_data["heatmap_image"],
        "review_status": scan_data["review_status"],
        "doctor_review": scan_data.get("doctor_review")
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
