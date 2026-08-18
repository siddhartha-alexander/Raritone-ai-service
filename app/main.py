import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.measurements import calculate_measurements
from app.pose_detector import PoseDetector
from app.preprocessing import (
    decode_image,
    preprocess_image,
    validate_image_type,
)
from app.schemas import AIResponse, HealthResponse


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Raritone AI Body Analysis Service",
    description="Computer vision API for pose detection and body proportion analysis",
    version="1.0.0",
)


# --------------------------------------------------
# AI model
# --------------------------------------------------

pose_detector = PoseDetector()

MODEL_VERSION = "pose-v1"
SERVICE_NAME = "raritone-ai"


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get(
    "/api/ai/health",
    response_model=HealthResponse,
)
def health_check():
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "model_version": MODEL_VERSION,
    }


# --------------------------------------------------
# Image processing helper
# --------------------------------------------------

async def process_image(file: UploadFile):
    start_time = time.perf_counter()

    logger.info("Request received")

    # Validate file type
    try:
        validate_image_type(file.content_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # Read image
    image_bytes = await file.read()

    try:
        image = decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    logger.info("Image validated")

    # OpenCV preprocessing
    preprocessing_start = time.perf_counter()

    try:
        image = preprocess_image(image)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    preprocessing_time = round(
        time.perf_counter() - preprocessing_start,
        4,
    )

    logger.info(
        "Image preprocessing completed: %.4fs",
        preprocessing_time,
    )

    # Pose detection
    pose_start = time.perf_counter()

    logger.info("Pose detection started")

    try:
        landmarks = pose_detector.detect(image)
    except Exception:
        logger.exception("Pose detection failed")

        raise HTTPException(
            status_code=500,
            detail="Pose detection failed.",
        )

    pose_time = round(
        time.perf_counter() - pose_start,
        4,
    )

    logger.info(
        "Pose detection completed: %.4fs",
        pose_time,
    )

    # Person not detected
    if landmarks is None:
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

    total_time = round(
        time.perf_counter() - start_time,
        4,
    )

    logger.info(
        "Processing time: %.4fs",
        total_time,
    )

    logger.info("Response returned")

    return landmarks, total_time


# --------------------------------------------------
# Pose endpoint
# --------------------------------------------------

@app.post(
    "/api/ai/pose",
    response_model=AIResponse,
    responses={
        400: {"description": "Invalid or corrupted image"},
        404: {"description": "No person detected"},
        413: {"description": "Image exceeds maximum allowed size"},
        422: {"description": "Image file was not provided"},
        500: {"description": "Internal server error"},
    },
)
async def pose_api(
    file: UploadFile = File(...)
):
    landmarks, processing_time = await process_image(file)

    return {
        "success": True,
        "person_detected": True,
        "landmarks": landmarks,
        "measurements": None,
        "model_version": MODEL_VERSION,
        "processing_time": processing_time,
    }


# --------------------------------------------------
# Measurement endpoint
# --------------------------------------------------

@app.post(
    "/api/ai/measurements",
    response_model=AIResponse,
    responses={
        400: {"description": "Invalid or corrupted image"},
        404: {"description": "No person detected"},
        413: {"description": "Image exceeds maximum allowed size"},
        422: {"description": "Image file was not provided"},
        500: {"description": "Internal server error"},
    },
)
async def measurements_api(
    file: UploadFile = File(...)
):
    landmarks, processing_time = await process_image(file)

    try:
        measurements = calculate_measurements(landmarks)
    except Exception:
        logger.exception("Measurement calculation failed")

        raise HTTPException(
            status_code=500,
            detail="Measurement calculation failed.",
        )

    return {
        "success": True,
        "person_detected": True,
        "landmarks": landmarks,
        "measurements": measurements,
        "model_version": MODEL_VERSION,
        "processing_time": processing_time,
    }