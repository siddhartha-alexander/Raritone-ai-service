## Base URL

```text
http://127.0.0.1:8000
```

## Health Check

### Endpoint

```text
GET /api/ai/health
```

### Sample Response

```json
{
  "status": "healthy",
  "service": "raritone-ai",
  "model_version": "pose-v1"
}
```

---

# Virtual Try-On API

## Endpoint

```text
POST /api/ai/tryon
```

## Request Format

Content type:

```text
multipart/form-data
```

The request requires two image files:

| Field           | Type | Description                          |
| --------------- | ---- | ------------------------------------ |
| `person_image`  | File | Image containing the person          |
| `garment_image` | File | Image of the garment to be processed |

### Supported Formats

* JPG
* JPEG
* PNG

### Maximum File Size

```text
10 MB per uploaded image
```

## Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/api/ai/tryon" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "person_image=@person.jpg;type=image/jpeg" \
  -F "garment_image=@tshirt.jpg;type=image/jpeg"
```

## Successful Response

```json
{
  "success": true,
  "status": "completed",
  "result_image": "outputs/tryon_result.png",
  "model_version": "tryon-v1-baseline",
  "processing_time": 43.8942
}
```

## Response Fields

| Field             | Type    | Description                                 |
| ----------------- | ------- | ------------------------------------------- |
| `success`         | Boolean | Indicates whether processing was successful |
| `status`          | String  | Current processing status                   |
| `result_image`    | String  | Reference to the generated try-on result    |
| `model_version`   | String  | Version of the try-on pipeline              |
| `processing_time` | Float   | Total processing time in seconds            |

## Error Responses

| Status Code | Description                                  |
| ----------- | -------------------------------------------- |
| 400         | Invalid or corrupted person or garment image |
| 404         | No person detected in the person image       |
| 408         | Try-on processing timeout                    |
| 413         | Uploaded image exceeds the maximum file size |
| 422         | Required image file was not provided         |
| 500         | Internal try-on processing failure           |

# Processing Pipeline

```text
Person Image + Garment Image
            ↓
      Input Validation
            ↓
      Person Processing
            ↓
Pose Detection + Person Segmentation
            ↓
      Garment Processing
            ↓
    Pose-Based Alignment
            ↓
    Baseline 2D Try-On
            ↓
       Result Validation
            ↓
       Output Image
```

# Performance Testing

Performance testing was conducted using 10 consecutive try-on requests.

* Total requests: 10
* Successful requests: 10
* Failed requests: 0
* Failure rate: 0%
* Average processing time: 43.8942 seconds
* Minimum processing time: 37.1062 seconds
* Maximum processing time: 81.6236 seconds

## Current Bottleneck

The primary bottleneck is the CPU-based pose detection, segmentation, garment alignment, and baseline try-on processing pipeline. Processing time may vary depending on image size and system load.

## Model Version

```text
tryon-v1-baseline
```

## Processing Expectation

The current prototype has an average processing time of approximately 44 seconds per request. Frontend integration should therefore treat the try-on operation as a long-running request and display an appropriate loading state.

## Important Limitation

The current implementation is a baseline pose-aware 2D virtual try-on prototype. It performs garment preprocessing, person segmentation, pose-based alignment, and image compositing.

It is intended for engineering integration and testing and should not be presented as a photorealistic generative virtual try-on system.
