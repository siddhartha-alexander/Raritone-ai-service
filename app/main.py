import asyncio
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference import MODEL_VERSION as TRYON_MODEL_VERSION
from app.inference import run_tryon
from app.measurements import calculate_measurements
from app.person_processor import prepare_tryon_input
from app.pose_detector import PoseDetector
from app.preprocessing import (
    decode_image,
    preprocess_image,
    validate_image_type,
)
from app.schemas import (
    AIResponse,
    HealthResponse,
    SegmentationResponse,
    TryOnPreparationResponse,
    TryOnResponse,
)
from app.segmentation import (
    person_detected,
    remove_background,
    save_outputs,
    segment_person,
)



# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)



# FastAPI Application


app = FastAPI(
    title="Raritone AI Body Analysis Service",
    description=(
        "Computer vision API for pose detection, "
        "body measurements, segmentation, and baseline "
        "2D virtual try-on."
    ),
    version="1.0.0",
)



# Configuration


MODEL_VERSION = "pose-v1"
SEGMENTATION_MODEL_VERSION = "seg-v1"
SERVICE_NAME = "raritone-ai"

MAX_FILE_SIZE = 10 * 1024 * 1024
TRYON_TIMEOUT_SECONDS = 120



# AI Model


pose_detector = PoseDetector()



# Health Check


@app.get(
    "/api/ai/health",
    response_model=HealthResponse,
)
def health_check():
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "model_version": (
            f"{MODEL_VERSION}+"
            f"{SEGMENTATION_MODEL_VERSION}+"
            f"{TRYON_MODEL_VERSION}"
        ),
    }



# Common Image Validation


async def read_and_preprocess_image(file: UploadFile):
    """
    Validate, read, decode, and preprocess an uploaded image.
    """

    logger.info(
        "Image request received: %s",
        file.filename,
    )

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
            detail=(
                "Image exceeds the maximum allowed "
                "size of 10 MB."
            ),
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



# Pose Processing


async def process_pose(file: UploadFile):
    """
    Validate image and run pose detection.
    """

    start_time = time.perf_counter()

    image = await read_and_preprocess_image(file)

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

    if landmarks is None:
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

    total_time = round(
        time.perf_counter() - start_time,
        4,
    )

    return landmarks, total_time



# Pose Endpoint


@app.post(
    "/api/ai/pose",
    response_model=AIResponse,
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



# Measurements Endpoint


@app.post(
    "/api/ai/measurements",
    response_model=AIResponse,
)
async def measurements_api(
    file: UploadFile = File(...)
):
    landmarks, processing_time = await process_pose(file)

    try:
        measurements = calculate_measurements(
            landmarks
        )
    except Exception:
        logger.exception(
            "Measurement calculation failed"
        )

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



# Segmentation Endpoint


@app.post(
    "/api/ai/segment",
    response_model=SegmentationResponse,
)
async def segment_api(
    file: UploadFile = File(...)
):
    start_time = time.perf_counter()

    image = await read_and_preprocess_image(file)

    try:
        mask = segment_person(image)
    except Exception:
        logger.exception("Segmentation failed")

        raise HTTPException(
            status_code=500,
            detail="Segmentation processing failed.",
        )

    if not person_detected(mask):
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

    try:
        background_removed = remove_background(
            image,
            mask,
        )

        mask_reference, background_reference = (
            save_outputs(
                mask,
                background_removed,
            )
        )

    except Exception:
        logger.exception(
            "Background removal failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Background removal failed.",
        )

    total_time = round(
        time.perf_counter() - start_time,
        4,
    )

    return {
        "success": True,
        "person_detected": True,
        "mask_reference": mask_reference,
        "background_removed_reference": (
            background_reference
        ),
        "model_version": (
            SEGMENTATION_MODEL_VERSION
        ),
        "processing_time": total_time,
    }



# Try-On Preparation Endpoint


@app.post(
    "/api/ai/prepare-tryon",
    response_model=TryOnPreparationResponse,
)
async def prepare_tryon_api(
    file: UploadFile = File(...)
):
    start_time = time.perf_counter()

    image = await read_and_preprocess_image(file)

    try:
        mask = segment_person(image)
    except Exception:
        logger.exception(
            "Try-on segmentation failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Segmentation processing failed.",
        )

    if not person_detected(mask):
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

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
            "Try-on preparation failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Try-on preparation failed.",
        )

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

    processing_time = round(
        time.perf_counter() - start_time,
        4,
    )

    return {
        "success": True,
        "person_detected": True,
        "pose_available": pose_available,
        "person_mask": mask_reference,
        "garment_region": None,
        "processing_time": processing_time,
        "model_version": "pose-v1+seg-v1",
    }



# Final Try-On Endpoint


@app.post(
    "/api/ai/tryon",
    response_model=TryOnResponse,
    responses={
        400: {
            "description": (
                "Invalid person or garment image"
            )
        },
        404: {
            "description": (
                "No person detected in person image"
            )
        },
        408: {
            "description": "Try-on processing timeout"
        },
        413: {
            "description": (
                "Image exceeds maximum allowed size"
            )
        },
        422: {
            "description": (
                "Required image file was not provided"
            )
        },
        500: {
            "description": (
                "Try-on processing failed"
            )
        },
    },
)
async def tryon_api(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
):
    """
    Baseline pose-aware 2D virtual try-on endpoint.

    Pipeline:
    Person Image
        ↓
    Validation
        ↓
    Pose Detection + Segmentation
        ↓
    Garment Processing
        ↓
    Pose-Based Alignment
        ↓
    Baseline 2D Composite
        ↓
    Result
    """

    total_start = time.perf_counter()

    logger.info(
        "Try-on request received | "
        "person=%s | garment=%s",
        person_image.filename,
        garment_image.filename,
    )

    # -----------------------------------------------------
    # Read and validate person image
    # -----------------------------------------------------

    person = await read_and_preprocess_image(
        person_image
    )

    # -----------------------------------------------------
    # Read and validate garment image
    # -----------------------------------------------------

    garment = await read_and_preprocess_image(
        garment_image
    )

    # -----------------------------------------------------
    # Prepare model inputs
    # -----------------------------------------------------

    try:
        (
            person_input,
            garment_input,
            person_mask,
            pose_data,
            preparation_metadata,
        ) = prepare_tryon_input(
            person,
            garment,
        )

    except ValueError as exc:
        logger.warning(
            "Try-on input preparation failed: %s",
            str(exc),
        )

        if "No person detected" in str(exc):
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception(
            "Unexpected try-on preparation failure"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to prepare try-on inputs."
            ),
        )

    # -----------------------------------------------------
    # Run inference with timeout
    # -----------------------------------------------------

    try:
        logger.info(
            "Try-on inference started"
        )

        result_image, inference_metadata = (
            await asyncio.wait_for(
                asyncio.to_thread(
                    run_tryon,
                    person_input,
                    garment_input,
                    pose_data,
                    person_mask,
                ),
                timeout=TRYON_TIMEOUT_SECONDS,
            )
        )

    except asyncio.TimeoutError:
        logger.error(
            "Try-on processing timed out"
        )

        raise HTTPException(
            status_code=408,
            detail=(
                "Try-on processing exceeded the "
                "maximum allowed processing time."
            ),
        )

    except Exception as exc:
        logger.exception(
            "Try-on inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Try-on inference failed: {str(exc)}"
            ),
        )

    # -----------------------------------------------------
    # Final timing
    # -----------------------------------------------------

    total_processing_time = round(
        time.perf_counter() - total_start,
        4,
    )

    logger.info(
        "Try-on completed successfully | "
        "preparation=%.4fs | "
        "inference=%.4fs | "
        "total=%.4fs",
        preparation_metadata[
            "total_preparation_time"
        ],
        inference_metadata[
            "inference_time"
        ],
        total_processing_time,
    )

    # -----------------------------------------------------
    # Return standardized response
    # -----------------------------------------------------

    return {
        "success": True,
        "status": "completed",
        "result_image": inference_metadata[
            "result_path"
        ],
        "model_version": TRYON_MODEL_VERSION,
        "processing_time": total_processing_time,
    }