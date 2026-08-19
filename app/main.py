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
from app.segmentation import (
    person_detected,
    remove_background,
    save_outputs,
    segment_person,
)
from app.schemas import (
    AIResponse,
    HealthResponse,
    SegmentationResponse,
    TryOnPreparationResponse,
)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Raritone AI Body Analysis Service",
    description=(
        "Computer vision API for pose detection, "
        "body measurements, and person segmentation."
    ),
    version="1.0.0",
)


# =========================================================
# Configuration
# =========================================================

MODEL_VERSION = "pose-v1"
SEGMENTATION_MODEL_VERSION = "seg-v1"
SERVICE_NAME = "raritone-ai"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# =========================================================
# AI Model
# =========================================================

pose_detector = PoseDetector()


# =========================================================
# Health Check
# =========================================================

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


# =========================================================
# Common Image Validation
# =========================================================

async def read_and_preprocess_image(file: UploadFile):
    """
    Validate, read, decode, and preprocess an uploaded image.
    """

    logger.info("Request received")

    # -----------------------------------------------------
    # Validate file type
    # -----------------------------------------------------

    try:
        validate_image_type(file.content_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # -----------------------------------------------------
    # Read file
    # -----------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    # -----------------------------------------------------
    # Maximum upload size
    # -----------------------------------------------------

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the maximum allowed size of 10 MB.",
        )

    # -----------------------------------------------------
    # Decode image
    # -----------------------------------------------------

    try:
        image = decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    logger.info("Image validated")

    # -----------------------------------------------------
    # OpenCV preprocessing
    # -----------------------------------------------------

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

    return image


# =========================================================
# Pose Processing
# =========================================================

async def process_pose(file: UploadFile):
    """
    Validate image and run pose detection.
    """

    start_time = time.perf_counter()

    image = await read_and_preprocess_image(file)

    # -----------------------------------------------------
    # Pose Detection
    # -----------------------------------------------------

    logger.info("Pose detection started")

    pose_start = time.perf_counter()

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

    # -----------------------------------------------------
    # No person detected
    # -----------------------------------------------------

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
        "Total processing time: %.4fs",
        total_time,
    )

    logger.info("Response returned")

    return landmarks, total_time


# =========================================================
# Pose Endpoint
# =========================================================

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
    landmarks, processing_time = await process_pose(file)

    return {
        "success": True,
        "person_detected": True,
        "landmarks": landmarks,
        "measurements": None,
        "model_version": MODEL_VERSION,
        "processing_time": processing_time,
    }


# =========================================================
# Measurements Endpoint
# =========================================================

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
    landmarks, processing_time = await process_pose(file)

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


# =========================================================
# Segmentation Endpoint
# =========================================================

@app.post(
    "/api/ai/segment",
    response_model=SegmentationResponse,
    responses={
        400: {"description": "Invalid or corrupted image"},
        404: {"description": "No person detected"},
        413: {"description": "Image exceeds maximum allowed size"},
        422: {"description": "Image file was not provided"},
        500: {"description": "Internal server error"},
    },
)
async def segment_api(
    file: UploadFile = File(...)
):
    start_time = time.perf_counter()

    logger.info("Segmentation request received")

    image = await read_and_preprocess_image(file)

    # -----------------------------------------------------
    # Person Segmentation
    # -----------------------------------------------------

    logger.info("Person segmentation started")

    segmentation_start = time.perf_counter()

    try:
        mask = segment_person(image)
    except Exception:
        logger.exception("Segmentation failed")

        raise HTTPException(
            status_code=500,
            detail="Segmentation processing failed.",
        )

    segmentation_time = round(
        time.perf_counter() - segmentation_start,
        4,
    )

    logger.info(
        "Person segmentation completed: %.4fs",
        segmentation_time,
    )

    # -----------------------------------------------------
    # Person Detection Check
    # -----------------------------------------------------

    if not person_detected(mask):
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

    # -----------------------------------------------------
    # Background Removal
    # -----------------------------------------------------

    try:
        background_removed = remove_background(
            image,
            mask,
        )

        mask_reference, background_reference = save_outputs(
            mask,
            background_removed,
        )

    except Exception:
        logger.exception("Background removal failed")

        raise HTTPException(
            status_code=500,
            detail="Background removal failed.",
        )

    # -----------------------------------------------------
    # Total Time
    # -----------------------------------------------------

    total_time = round(
        time.perf_counter() - start_time,
        4,
    )

    logger.info(
        "Segmentation processing time: %.4fs",
        total_time,
    )

    logger.info("Segmentation response returned")

    return {
        "success": True,
        "person_detected": True,
        "mask_reference": mask_reference,
        "background_removed_reference": background_reference,
        "model_version": SEGMENTATION_MODEL_VERSION,
        "processing_time": total_time,
    }


# =========================================================
# Try-On Preparation Endpoint
# =========================================================

@app.post(
    "/api/ai/prepare-tryon",
    response_model=TryOnPreparationResponse,
    responses={
        400: {"description": "Invalid or corrupted image"},
        404: {"description": "No person detected"},
        413: {"description": "Image exceeds maximum allowed size"},
        422: {"description": "Image file was not provided"},
        500: {"description": "Internal server error"},
    },
)
async def prepare_tryon_api(
    file: UploadFile = File(...)
):
    start_time = time.perf_counter()

    logger.info("Try-on preparation request received")

    image = await read_and_preprocess_image(file)

    # -----------------------------------------------------
    # Person Segmentation
    # -----------------------------------------------------

    logger.info("Try-on segmentation started")

    try:
        mask = segment_person(image)
    except Exception:
        logger.exception("Try-on segmentation failed")

        raise HTTPException(
            status_code=500,
            detail="Segmentation processing failed.",
        )

    # -----------------------------------------------------
    # Person Detection
    # -----------------------------------------------------

    if not person_detected(mask):
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

    # -----------------------------------------------------
    # Background Removed Output
    # -----------------------------------------------------

    try:
        background_removed = remove_background(
            image,
            mask,
        )

        mask_reference, _ = save_outputs(
            mask,
            background_removed,
        )

    except Exception:
        logger.exception(
            "Try-on background preparation failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Try-on preparation failed.",
        )

    # -----------------------------------------------------
    # Pose Detection
    # -----------------------------------------------------

    logger.info("Pose detection started")

    try:
        landmarks = pose_detector.detect(image)
    except Exception:
        logger.exception(
            "Try-on pose detection failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Pose detection failed.",
        )

    pose_available = landmarks is not None

    # -----------------------------------------------------
    # Garment Region
    # -----------------------------------------------------

    # We currently do not have a dedicated clothing
    # segmentation model, so we intentionally return None.
    garment_region = None

    # -----------------------------------------------------
    # Total Processing Time
    # -----------------------------------------------------

    processing_time = round(
        time.perf_counter() - start_time,
        4,
    )

    logger.info(
        "Try-on preparation completed: %.4fs",
        processing_time,
    )

    logger.info("Try-on preparation response returned")

    return {
        "success": True,
        "person_detected": True,
        "pose_available": pose_available,
        "person_mask": mask_reference,
        "garment_region": garment_region,
        "processing_time": processing_time,
        "model_version": "pose-v1+seg-v1",
    }