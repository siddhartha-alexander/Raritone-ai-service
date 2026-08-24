# Raritone AI Service

AI/ML computer-vision service for Raritone's 2D virtual try-on pipeline.

## Features

- Person pose detection using MediaPipe
- Person segmentation and mask generation
- Garment preprocessing and garment mask generation
- Pose-aware garment alignment
- Baseline 2D virtual try-on
- FastAPI integration
- Input validation and error handling
- Processing-time measurement
- VTON quality evaluation

## Pipeline

Person Image + Garment Image  
↓  
Input Validation  
↓  
Person Preprocessing  
↓  
Pose Detection + Segmentation  
↓  
Garment Preprocessing  
↓  
Garment Mask  
↓  
Pose-Based Alignment  
↓  
Baseline 2D Try-On  
↓  
Result Image  
↓  
FastAPI JSON Response

## Technologies

- Python 3.12
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
│   ├── pose_processor.py
│   ├── person_processor.py
│   ├── garment_processor.py
│   ├── segmentation.py
│   ├── tryon_pipeline.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── measurements.py
│   └── schemas.py
├── models/
├── vton_dataset/
│   ├── persons/
│   ├── garments/
│   ├── masks/
│   ├── poses/
│   └── results/
├── outputs/
├── tests/
├── vton_evaluation.csv
├── requirements.txt
├── API_DOCUMENTATION.md
└── README.md