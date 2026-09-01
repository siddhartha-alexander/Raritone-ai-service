# Raritone 2D Virtual Try-On Engine

A modular 2D Virtual Try-On (VTON) pipeline developed for Raritone.

The system accepts a person image and garment image, performs image validation, pose estimation, segmentation, garment preprocessing and alignment, and generates a virtual try-on result.

A local pose-aware baseline pipeline is exposed through FastAPI. A GPU-based generative VTON experiment was also successfully executed using FASHN VTON v1.5 on Kaggle.

## Architecture

Person Image
→ Image Validation
→ Person Detection
→ Pose Estimation
→ Person Segmentation

Garment Image
→ Image Validation
→ Background Removal
→ Garment Segmentation
→ Crop / Resize / Normalize

Person + Pose + Garment
→ Garment Alignment
→ Try-On Generation
→ Output Validation
→ Result

## Project Structure

```text
raritone-vton/
├── app/
│   ├── main.py
│   ├── person_processor.py
│   ├── garment_processor.py
│   ├── segmentation.py
│   ├── alignment.py
│   ├── tryon_pipeline.py
│   ├── validation.py
│   └── output_validation.py
├── dataset/
│   ├── persons/
│   ├── garments/
│   ├── poses/
│   └── masks/
├── models/
├── outputs/
│   └── gpu/
├── evaluation/
│   ├── evaluation.csv
│   └── failure_analysis.md
├── kaggle/
├── tests/
├── requirements.txt
└── README.md