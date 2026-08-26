# Raritone AI API Documentation

## Service

Raritone AI Body Analysis and Virtual Try-On Service

## Try-On Model Version

`tryon-v1-baseline`

---

# Virtual Try-On

## Endpoint

`POST /api/ai/tryon`

## Content Type

`multipart/form-data`

## Request Fields

| Field | Type | Required |
|---|---|---|
| person_image | Image file | Yes |
| garment_image | Image file | Yes |

Maximum file size: **10 MB per uploaded image**.

---

## Processing Pipeline

```text
Request
↓
Image Validation
↓
Person Quality Check
↓
Garment Quality Check
↓
Quality Gate
↓
Pose + Segmentation
↓
Garment Processing
↓
Try-On Inference
↓
Output Validation
↓
Response
```

---

## Successful Response

HTTP `200`

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

## Quality Fields

| Field | Description |
|---|---|
| pose | Pose/landmark confidence quality |
| mask | Person segmentation mask quality |
| garment | Garment preprocessing quality |

---

# Quality Gate

Inputs are validated before try-on inference.

A request may be rejected when:

- Person cannot be detected
- Pose cannot be detected reliably
- Person image is unsuitable
- Segmentation quality is insufficient
- Garment cannot be processed
- Garment quality is insufficient

Example:

HTTP `404`

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

---

# Error Codes

| HTTP | Error | Meaning |
|---|---|---|
| 400 | PERSON_PREPROCESSING_FAILED | Person input could not be prepared |
| 400 | GARMENT_PREPROCESSING_FAILED | Garment input could not be prepared |
| 400 | Quality gate error | Input quality is insufficient |
| 404 | PERSON_NOT_DETECTED / POSE_NOT_DETECTED | Valid person/pose unavailable |
| 408 | TRYON_TIMEOUT | Processing exceeded timeout |
| 413 | FILE_TOO_LARGE | Image exceeds upload limit |
| 422 | VALIDATION_ERROR | Required request field missing |
| 500 | TRYON_INFERENCE_FAILED | Try-on processing failed |
| 500 | INVALID_TRYON_OUTPUT | Invalid or empty output generated |

---

# Performance

10 consecutive successful pipeline executions:

```text
Success rate:              100%
Failure rate:              0.00%

Average person processing: 0.7755 s
Average garment processing:0.9767 s
Average inference:         0.7511 s

Average total latency:     2.5033 s
P95 latency:               2.8922 s
Minimum latency:           2.1104 s
Maximum latency:           2.9223 s
```

Current measured bottleneck: **garment preprocessing**.

---

# Quality Evaluation

```text
Garment alignment:      3.67 / 5
Garment preservation:   4.50 / 5
Body alignment:         3.50 / 5
Boundary quality:       3.50 / 5
Face preservation:      5.00 / 5
Overall realism:        3.50 / 5

Overall average:        3.94 / 5
```

---

# Frontend Integration

Example JavaScript request:

```javascript
const formData = new FormData();

formData.append("person_image", personFile);
formData.append("garment_image", garmentFile);

const response = await fetch(
  "http://127.0.0.1:8000/api/ai/tryon",
  {
    method: "POST",
    body: formData
  }
);

const data = await response.json();

console.log(data);
```

For local development:

```text
http://127.0.0.1:8000
```

`127.0.0.1` is only the local development address. The MERN/full-stack application should use the deployed AI service base URL when the backend is deployed.

---

# Other Endpoints

```text
GET  /api/ai/health
POST /api/ai/pose
POST /api/ai/measurements
POST /api/ai/segment
POST /api/ai/prepare-tryon
POST /api/ai/tryon
```

---

# Production Note

`tryon-v1-baseline` is a baseline pose-aware 2D compositing prototype.

It should not be described as a photorealistic generative VTON model. A production model requires separate model, licensing, security, performance, and commercial-use validation.

## Internal Evaluation

The VTON evaluation pipeline is intended for internal development and model validation.

Current evaluation assets are stored under:

```text
evaluation