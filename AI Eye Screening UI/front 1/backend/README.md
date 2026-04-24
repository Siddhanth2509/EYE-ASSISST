# EYE-ASSISST Backend API

AI-Powered Eye Disease Screening Backend - FastAPI implementation with Grad-CAM explainability.

## Features

- **Fundus Image Analysis**: REST API endpoints for uploading and analyzing retinal images
- **Grad-CAM Explainability**: Generate attention heatmaps showing model focus areas
- **Multi-Disease Detection**: Support for DR, AMD, Glaucoma, Cataract screening
- **Doctor Review System**: Submit and track physician reviews of AI analyses
- **Patient History**: Retrieve and compare historical screenings
- **Analytics Dashboard**: Model performance metrics and system statistics

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

Once running, access the interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | System health check |

### Image Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Upload and analyze fundus image |
| GET | `/api/v1/heatmap/{scan_id}` | Get Grad-CAM heatmap for scan |
| GET | `/api/v1/scan/{scan_id}` | Get full scan details |

### Doctor Review

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/review` | Submit doctor review |

### Patient History

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/patient/{patient_id}/history` | Get patient screening history |
| GET | `/api/v1/patient/{patient_id}/comparison` | Compare two scans |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analytics/metrics` | Model performance metrics |
| GET | `/api/v1/analytics/statistics` | Screening volume statistics |

## Request/Response Examples

### Analyze Image

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@fundus_image.jpg" \
  -F "patient_id=P-2026-0041" \
  -F "laterality=OD" \
  -F "age=58"
```

**Response:**
```json
{
  "scan_id": "SCAN-A7B2C9D1",
  "patient_id": "P-2026-0041",
  "timestamp": "2026-03-25T14:32:00",
  "binary_result": "DR Detected",
  "confidence": 91.5,
  "severity_level": 2,
  "severity_label": "Moderate",
  "severity_color": "#F97316",
  "multi_disease_flags": {
    "amd": {"detected": false, "confidence": 0.12},
    "glaucoma": {"detected": false, "confidence": 0.08},
    "cataract": {"detected": false, "confidence": 0.15}
  },
  "gradcam_available": true
}
```

### Submit Doctor Review

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/review" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "scan_id": "SCAN-A7B2C9D1",
    "doctor_id": "DOC-001",
    "action": "approve",
    "notes": "AI assessment confirmed. Mild microaneurysms present.",
    "timestamp": "2026-03-25T15:00:00"
  }'
```

**Response:**
```json
{
  "status": "success",
  "review_id": "REV-E4F5G6H7"
}
```

### Get Heatmap

**Request:**
```bash
curl "http://localhost:8000/api/v1/heatmap/SCAN-A7B2C9D1"
```

**Response:**
```json
{
  "scan_id": "SCAN-A7B2C9D1",
  "heatmap_image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "original_image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## Grad-CAM Module

The `gradcam_module.py` provides explainability features:

### Classes

- **GradCAMGenerator**: Main class for generating attention heatmaps
- **MultiClassGradCAM**: Extended version for multi-class visualization

### Usage

```python
from gradcam_module import GradCAMGenerator

# Initialize with your model
cam_generator = GradCAMGenerator(model)

# Generate heatmap
heatmap_base64 = cam_generator.generate_heatmap(
    image=pil_image,
    prediction=prediction_dict,
    alpha=0.5,
    colormap=cv2.COLORMAP_JET
)

# Generate vessel segmentation
vessel_overlay = cam_generator.generate_vessel_segmentation(image)
```

## Model Integration

The current implementation uses a mock model for demonstration. To integrate your actual model:

1. Replace `MockEyeDiseaseModel` in `main.py` with your model class
2. Ensure your model has:
   - `preprocess_image()` method
   - `predict()` method returning the expected format
   - `extract_features()` method for Grad-CAM

### Expected Prediction Format

```python
{
    "binary_result": "DR Detected" or "Normal",
    "confidence": 91.5,  # 0-100
    "severity_level": 2,  # 0-4
    "severity_label": "Moderate",
    "severity_color": "#F97316",
    "severity_probs": [0.1, 0.2, 0.5, 0.15, 0.05],
    "multi_disease": {
        "amd": {"detected": False, "confidence": 0.12},
        "glaucoma": {"detected": False, "confidence": 0.08},
        "cataract": {"detected": False, "confidence": 0.15}
    }
}
```

## Severity Levels

| Level | Label | Color | Description |
|-------|-------|-------|-------------|
| 0 | Normal | #10B981 | No diabetic retinopathy |
| 1 | Mild | #F59E0B | Mild NPDR |
| 2 | Moderate | #F97316 | Moderate NPDR |
| 3 | Severe | #EF4444 | Severe NPDR |
| 4 | Proliferative | #DC2626 | PDR |

## Configuration

### Environment Variables

```bash
# Optional: Configure model path
MODEL_PATH=/path/to/model.pth

# Optional: Configure device
DEVICE=cuda  # or cpu

# Optional: API settings
API_HOST=0.0.0.0
API_PORT=8000
```

## CORS Configuration

For production, update the CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA if available
- **Image Size**: Recommended input size is 512x512 pixels
- **Batch Processing**: Currently processes single images
- **Memory**: Grad-CAM requires additional GPU memory for gradient computation

## Testing

```bash
# Run health check
curl http://localhost:8000/health

# Test with sample image
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@test_image.jpg" \
  -F "patient_id=TEST-001"
```

## Frontend Integration

The API is designed to work with the EYE-ASSISST React frontend:

```javascript
// Example: Upload and analyze
const formData = new FormData();
formData.append('file', imageFile);
formData.append('patient_id', patientId);

const response = await fetch('/api/v1/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

## License

Medical Device Software - For clinical use only with appropriate regulatory approvals.

## Support

For technical support or questions about integration, contact the development team.
