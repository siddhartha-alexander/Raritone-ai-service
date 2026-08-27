# Raritone AI – Virtual Try-On Service

AI/ML backend service for pose estimation, person segmentation, body analysis, garment preprocessing, quality validation, and baseline pose-aware 2D virtual try-on.

## Current Version

**Try-On Model:** `tryon-v1-baseline`

The current implementation is a baseline engineering prototype based on pose-aware garment alignment and image compositing. It should not be considered a photorealistic generative VTON model.

---

## VTON Pipeline

```text
Person Image + Garment Image
        ↓
Upload Validation
        ↓
Person Quality Validation
        ↓
Garment Quality Validation
        ↓
Quality Gate
        ↓
Pose Detection + Person Segmentation
        ↓
Garment Processing
        ↓
Pose-Aware Garment Alignment
        ↓
Baseline Try-On Inference
        ↓
Output Validation
        ↓
Result + Quality Information
```

---

## Project Structure

```text
raritone-ai-service/
│
├── app/
│   ├── main.py
│   ├── pose_detector.py
│   ├── person_processor.py
│   ├── garment_processor.py
│   ├── segmentation.py
│   ├── quality_gate.py
│   ├── tryon_pipeline.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── measurements.py
│   └── schemas.py
│
├── models/
├── outputs/
├── tests/
├── vton_dataset/
│   ├── persons/
│   ├── garments/
│   ├── masks/
│   ├── poses/
│   └── results/
│
├── vton_evaluation.csv
├── performance_v2.csv
├── requirements.txt
├── API_DOCUMENTATION.md
└── README.md
```

---

## Person Quality Validation

The person preprocessing pipeline validates the person image before VTON inference.

Checks include:

- Person detection
- Pose availability
- Landmark confidence
- Person segmentation
- Person mask quality
- Image quality
- Input suitability

Example quality output:

```json
{
  "valid": true,
  "person_count": 1,
  "pose_quality": 0.9971,
  "mask_quality": 0.719
}
```

---

## Garment Preprocessing V2

The garment preprocessing pipeline performs:

- Image validation
- Background removal
- Garment cropping
- Aspect-ratio normalization
- Resolution normalization
- Garment mask generation
- Bounding-box extraction
- Garment quality validation

Example:

```json
{
  "valid": true,
  "category": "top",
  "width": 512,
  "height": 512,
  "mask_available": true,
  "garment_quality": 1.0
}
```

---

## Quality Gate

The Quality Gate prevents unsuitable inputs from reaching try-on inference.

```text
Person Quality
      ↓
Pose Quality
      ↓
Mask Quality
      ↓
Garment Quality
      ↓
PASS → Run Try-On

FAIL → Reject Request
```

Example rejected input:

```json
{
  "detail": {
    "success": false,
    "status": "failed",
    "error_code": "POSE_NOT_DETECTED",
    "message": "Please upload a clear full-body image with the person facing the camera.",
    "quality": {
      "pose": 0,
      "mask": 0
    }
  }
}
```

This avoids unnecessary inference on invalid inputs.

---

## API Endpoints

### Health

`GET /api/ai/health`

### Pose Detection

`POST /api/ai/pose`

### Body Measurements

`POST /api/ai/measurements`

### Person Segmentation

`POST /api/ai/segment`

### Try-On Preparation

`POST /api/ai/prepare-tryon`

### Virtual Try-On

`POST /api/ai/tryon`

The try-on endpoint expects multipart form data containing:

- `person_image`
- `garment_image`

---

## Successful Try-On Response

Example:

```json
{
  "success": true,
  "status": "completed",
  "result_image": "outputs/tryon_result.png",
  "model_version": "tryon-v1-baseline",
  "quality": {
    "pose": 0.9971,
    "mask": 0.719,
    "garment": 1.0
  },
  "processing_time": 5.6675
}
```

---

## Error Handling

The service handles:

- Invalid image
- Unsupported image
- Empty upload
- Oversized image
- Person not detected
- Pose not detected
- Low-quality VTON input
- Garment preprocessing failure
- Try-on inference failure
- Invalid output
- Processing timeout

HTTP status codes include:

| Status | Meaning |
|---|---|
| 200 | Successful processing |
| 400 | Invalid input / quality validation failure |
| 404 | Person or pose not detected |
| 408 | Try-on timeout |
| 413 | File exceeds upload limit |
| 422 | Required file missing |
| 500 | Internal/inference failure |

---

## Performance Evaluation

10 consecutive VTON pipeline executions were tested.

| Metric | Result |
|---|---:|
| Requests | 10 |
| Successful | 10 |
| Failed | 0 |
| Failure Rate | 0.00% |
| Average Person Processing | 0.7755 s |
| Average Garment Processing | 0.9767 s |
| Average Try-On Inference | 0.7511 s |
| Average Total Latency | 2.5033 s |
| P95 Latency | 2.8922 s |
| Minimum Latency | 2.1104 s |
| Maximum Latency | 2.9223 s |

### Current Performance Bottleneck

Garment preprocessing is currently the largest measured stage, averaging approximately **0.98 seconds**.

---

## VTON Quality Evaluation

Current evaluation results:

| Metric | Score |
|---|---:|
| Garment Alignment | 3.67 / 5 |
| Garment Preservation | 4.50 / 5 |
| Body Alignment | 3.50 / 5 |
| Boundary Quality | 3.50 / 5 |
| Face Preservation | 5.00 / 5 |
| Overall Realism | 3.50 / 5 |

**Overall Average Quality:** `3.94 / 5`

---

## Failure Analysis

The lowest-scoring evaluated combinations were:

- `pair_001` — 3.50/5
- `pair_003` — 3.50/5
- `pair_005` — 3.50/5

Main observed limitations:

- Pose-based garment alignment
- Body alignment
- Garment boundary quality
- Segmentation/compositing artifacts
- Baseline model realism

Face preservation and garment preservation performed comparatively well.

---

## Running the Service

Activate the virtual environment and run:

```bash
uvicorn app.main:app --reload
```

Open Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## Running Tests

Examples:

```bash
python -m tests.test_person_quality
python -m tests.test_garment_v2
python -m tests.test_quality_gate
python -m tests.test_performance_v2
python -m tests.analyze_failures
```

---

## Current Limitations

The current system is a baseline 2D VTON engineering prototype.

It does not yet perform photorealistic garment generation or physically accurate cloth deformation.

Known limitations include:

- Arms may not occlude garments correctly
- Garment boundaries may appear artificial
- Difficult poses can reduce alignment accuracy
- Loose clothing and complex garments are challenging
- Garment deformation is limited
- Segmentation quality affects compositing
- Current baseline is primarily optimized for upper-body garments

A production VTON model should undergo model-license, security, hardware, latency, and commercial-use review before integration.

---

## Full Stack Integration

The frontend should call:

```text
POST /api/ai/tryon
```

Content type:

```text
multipart/form-data
```

Required fields:

```text
person_image
garment_image
```

The backend performs validation and quality gating automatically before inference.

The frontend should display the returned error `message` when a request is rejected by the quality gate.

## 3D Asset Generation Pipeline

The Raritone 3D prototype converts a product image into a candidate 3D asset and keeps it private until human review.

Pipeline:

```text
Product Image
↓
3D Generation
↓
Candidate Mesh
↓
GLB Conversion
↓
Asset Validation
↓
Human Review
↓
Optimization
↓
Approved / Rejected
↓
Raritone 3D Catalog