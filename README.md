🧠 Overview

EYE-ASSISST is a production-ready system for real-time eye disease classification using deep learning.
It includes:

🌀 Model export to ONNX / TorchScript

⚡ FastAPI-based high-speed inference server

🎥 Webcam + image-batch client for predictions

🔥 Grad-CAM heatmaps for explainability

🧱 Modular, scalable architecture ready for:

active learning

online updates

deployment to clinics / edge devices

This repository is structured for ML deployment, not just training — making it suitable for real-world integrations and portfolio showcase.

🏗️ Project Structure
eye-realtime-inference/
│
├── export_model.py           # Convert trained model → ONNX/TorchScript
├── inference_server.py       # FastAPI server for real-time predictions
├── client_demo.py            # Webcam / folder prediction client
├── utils.py                  # Preprocessing + Grad-CAM utilities
├── requirements.txt          # Dependencies
└── README.md                 # Documentation

🧩 Key Features
🔹 Real-Time Inference

Optimized for lightning-fast predictions on CPU/GPU using ONNXRuntime.

🔹 Model Export

Convert your trained eye-disease CNN into formats suitable for deployment:

ONNX for edge devices

TorchScript for PyTorch-serving pipelines

🔹 FastAPI Inference Server

Production-style inference API supporting:

single image prediction

batch prediction

optional Grad-CAM output

🔹 Grad-CAM Explainability

Highlight regions that influence model decisions — a must for medical AI.

🔹 Clean, Modular Architecture

Designed for:

future active learning

incremental training

drift monitoring

hardware optimization (TFLite, TensorRT, etc.)

🧬 Architecture Diagram
+-----------------------------+
|   Client (Webcam / UI)      |
|  - sends image frames       |
+---------------+-------------+
                |
                v
+-----------------------------+
|    FastAPI Inference API    |
|  - loads ONNX/TorchScript   |
|  - runs inference           |
|  - optional Grad-CAM        |
+---------------+-------------+
                |
                v
+-----------------------------+
|   Model Backend (Runtime)   |
|   - ONNX Runtime / Torch    |
|   - Preprocessing pipeline  |
|   - Postprocessing          |
+-----------------------------+
                |
                v
+-----------------------------+
|  Edge Device / Cloud        |
+-----------------------------+

⚙️ Installation

Clone the repo:

git clone https://github.com/Siddhanth2509/EYE-ASSISST.git
cd EYE-ASSISST


Install dependencies:

pip install -r requirements.txt

📦 Model Export

Export trained model (TensorFlow/Keras → ONNX):

python export_model.py --input-model path/to/model.h5 --output models/xception.onnx


OR PyTorch → TorchScript:

python export_model.py --pytorch-model path/to/model.pt --torchscript models/model.ts

🚀 Run the Inference Server
uvicorn inference_server:app --host 0.0.0.0 --port 8000


API will be available at:

http://localhost:8000/docs

🎥 Run Local Client Demo
Predict a single image
python client_demo.py --single path/to/image.jpg

Use webcam for live predictions
python client_demo.py --webcam

Batch predict a folder
python client_demo.py --folder ./images/

🔥 Grad-CAM Visualization

Enable Grad-CAM via API:

POST /predict?explain=true


Returns:

class label

confidence

Grad-CAM heatmap (base64 or image file)

🛣️ Roadmap

 Integrate active-learning loop

 Add model drift detection

 TFLite / TensorRT conversion

 Deploy on Raspberry Pi / Jetson

 Add web dashboard (Streamlit / React)

 Add multi-class ensemble model

🧑‍💻 Contributing

Pull requests are welcome!
For major changes, open an issue to discuss improvements.

📄 License

This project is licensed under the MIT License.

🎉 Final Notes

This repo is designed for real-world ML deployment — not just training notebooks.
It shows practical engineering skills such as:

API development

model export

real-time inference

explainability

architecture design

Perfect for ML internships, jobs, and production applications.
