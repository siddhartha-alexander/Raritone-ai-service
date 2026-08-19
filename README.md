# Raritone AI Service

AI/ML computer-vision service for Raritone's virtual try-on pipeline.

## Overview

This service provides:

- Person pose detection
- Pose landmark extraction
- Relative body proportion analysis
- Person segmentation
- Background removal
- Try-on preparation
- FastAPI APIs

## Pipeline

User Image
→ Image Validation
→ OpenCV Preprocessing
→ Person Segmentation
→ Person Mask
→ Pose Detection
→ Landmarks
→ Body Proportions
→ FastAPI
→ JSON Response

## Technologies

- Python
- OpenCV
- MediaPipe
- rembg
- ONNX Runtime
- NumPy
- FastAPI
- Pydantic
- Uvicorn

## Project Structure

```text
raritone-ai-service/
├── app/
│   ├── main.py
│   ├── pose_detector.py
│   ├── segmentation.py
│   ├── preprocessing.py
│   ├── measurements.py
│   └── schemas.py
├── models/
├── tests/
├── sample_images/
├── outputs/
├── requirements.txt
├── API_DOCUMENTATION.md
├── segmentation_results.csv
└── README.md