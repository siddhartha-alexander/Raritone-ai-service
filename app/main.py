import asyncio
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
import sys
import shutil
import uuid
from pathlib import Path



from app.garment_processor import prepare_garment_v2
from app.inference import MODEL_VERSION as TRYON_MODEL_VERSION
from app.inference import run_tryon
from app.measurements import calculate_measurements
from app.person_processor import prepare_person_image
from app.pose_detector import PoseDetector
from app.preprocessing import (
    decode_image,
    preprocess_image,
    validate_image_type,
)
from app.quality_gate import (
    QualityGateError,
    run_quality_gate,
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
        "body measurements, segmentation, quality validation, "
        "and baseline 2D virtual try-on."
    ),
    version="1.1.0",
)



# Configuration


MODEL_VERSION = "pose-v1"
SEGMENTATION_MODEL_VERSION = "seg-v1"
SERVICE_NAME = "raritone-ai"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
TRYON_TIMEOUT_SECONDS = 120



# Models loaded once


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



# Common Upload Validation


async def read_and_decode_image(file: UploadFile):
    """
    Validate file type/size and decode uploaded image.

    This helper does NOT preprocess the image.
    It is used by the production try-on pipeline so that
    person_processor and garment_processor control their own
    preprocessing.
    """

    logger.info(
        "Image received | filename=%s | type=%s",
        file.filename,
        file.content_type,
    )

    try:
        validate_image_type(file.content_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                "Image exceeds the maximum allowed "
                "size of 10 MB."
            ),
        )

    try:
        image = decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return image


async def read_and_preprocess_image(file: UploadFile):
    """
    Used by the older pose/measurement/segmentation endpoints.
    """

    image = await read_and_decode_image(file)

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
        "model_version": SEGMENTATION_MODEL_VERSION,
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

    processing_time = round(
        time.perf_counter() - start_time,
        4,
    )

    return {
        "success": True,
        "person_detected": True,
        "pose_available": landmarks is not None,
        "person_mask": mask_reference,
        "garment_region": None,
        "processing_time": processing_time,
        "model_version": "pose-v1+seg-v1",
    }



# Production Try-On Endpoint


@app.post(
    "/api/ai/tryon",
    response_model=TryOnResponse,
    responses={
        400: {
            "description": "Input failed quality validation"
        },
        404: {
            "description": "Person not detected"
        },
        408: {
            "description": "Try-on processing timeout"
        },
        413: {
            "description": "Image exceeds maximum allowed size"
        },
        422: {
            "description": "Required image file was not provided"
        },
        500: {
            "description": "Try-on processing failed"
        },
    },
)
async def tryon_api(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
):
    """
    Controlled VTON endpoint.

    Request
        ↓
    Upload Validation
        ↓
    Person Quality Check
        ↓
    Garment Quality Check
        ↓
    Quality Gate
        ↓
    Pose + Segmentation
        ↓
    Try-On Inference
        ↓
    Output Validation
        ↓
    Result
    """

    total_start = time.perf_counter()

    logger.info(
        "Try-on request received | person=%s | garment=%s",
        person_image.filename,
        garment_image.filename,
    )

    # -----------------------------------------------------
    # 1. Decode raw uploaded images
    # -----------------------------------------------------

    person = await read_and_decode_image(
        person_image
    )

    garment = await read_and_decode_image(
        garment_image
    )

    # -----------------------------------------------------
    # 2. Person quality validation
    # -----------------------------------------------------

    logger.info(
        "Person quality validation started"
    )

    try:
        person_result = await asyncio.to_thread(
            prepare_person_image,
            person,
        )

    except Exception as exc:
        logger.exception(
            "Person preprocessing failed"
        )

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "status": "failed",
                "error_code": (
                    "PERSON_PREPROCESSING_FAILED"
                ),
                "message": str(exc),
            },
        )

    # -----------------------------------------------------
    # 3. Garment quality validation
    # -----------------------------------------------------

    logger.info(
        "Garment quality validation started"
    )

    try:
        garment_result = await asyncio.to_thread(
            prepare_garment_v2,
            garment,
        )

    except Exception as exc:
        logger.exception(
            "Garment preprocessing failed"
        )

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "status": "failed",
                "error_code": (
                    "GARMENT_PREPROCESSING_FAILED"
                ),
                "message": str(exc),
            },
        )

    # -----------------------------------------------------
    # 4. Quality Gate
    # -----------------------------------------------------

    logger.info(
        "Running VTON quality gate"
    )

    try:
        quality_gate_result = run_quality_gate(
            person_result,
            garment_result,
        )

    except QualityGateError as exc:
        logger.warning(
            "Quality gate rejected request | "
            "code=%s | message=%s",
            exc.error_code,
            exc.message,
        )

        status_code = 400

        if exc.error_code in {
            "PERSON_NOT_DETECTED",
            "POSE_NOT_DETECTED",
        }:
            status_code = 404

        raise HTTPException(
            status_code=status_code,
            detail={
                "success": False,
                "status": "failed",
                "error_code": exc.error_code,
                "message": exc.message,
                "quality": exc.quality,
            },
        )

    quality = quality_gate_result[
        "quality"
    ]

    logger.info(
        "Quality gate passed | "
        "pose=%.4f | mask=%.4f | garment=%.4f",
        quality["pose"],
        quality["mask"],
        quality["garment"],
    )

    # -----------------------------------------------------
    # 5. Validated model inputs
    # -----------------------------------------------------

    person_input = person_result[
        "person_input"
    ]

    person_mask = person_result[
        "person_mask"
    ]

    pose_data = person_result[
        "pose_data"
    ]

    garment_input = garment_result[
        "garment_input"
    ]

    # -----------------------------------------------------
    # 6. Try-On inference
    # -----------------------------------------------------

    inference_start = time.perf_counter()

    try:
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
            detail={
                "success": False,
                "status": "failed",
                "error_code": "TRYON_TIMEOUT",
                "message": (
                    "Try-on processing exceeded "
                    "the maximum allowed time."
                ),
            },
        )

    except Exception as exc:
        logger.exception(
            "Try-on inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "status": "failed",
                "error_code": (
                    "TRYON_INFERENCE_FAILED"
                ),
                "message": str(exc),
            },
        )

    inference_time = round(
        time.perf_counter()
        - inference_start,
        4,
    )

    # -----------------------------------------------------
    # 7. Output validation
    # -----------------------------------------------------

    if (
        result_image is None
        or not hasattr(
            result_image,
            "size",
        )
        or result_image.size == 0
    ):
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "status": "failed",
                "error_code": (
                    "INVALID_TRYON_OUTPUT"
                ),
                "message": (
                    "Try-on output image is invalid."
                ),
            },
        )

    # -----------------------------------------------------
    # 8. Final timing
    # -----------------------------------------------------

    total_processing_time = round(
        time.perf_counter()
        - total_start,
        4,
    )

    logger.info(
        "Try-on completed | "
        "person=%.4fs | garment=%.4fs | "
        "inference=%.4fs | total=%.4fs",
        person_result.get(
            "processing_time",
            0.0,
        ),
        garment_result.get(
            "processing_time",
            0.0,
        ),
        inference_time,
        total_processing_time,
    )

    # -----------------------------------------------------
    # 9. Successful response
    # -----------------------------------------------------

    return {
        "success": True,
        "status": "completed",
        "result_image": inference_metadata[
            "result_path"
        ],
        "model_version": TRYON_MODEL_VERSION,
        "quality": {
            "pose": quality["pose"],
            "mask": quality["mask"],
            "garment": quality["garment"],
        },
        "processing_time": (
            total_processing_time
        ),
    }
# ---------------------------------------------------------
# 3D ASSET PIPELINE SETUP
# ---------------------------------------------------------

# ---------------------------------------------------------
# 3D ASSET PIPELINE SETUP
# ---------------------------------------------------------

RARITONE_3D_DIR = (
    Path(__file__).resolve().parent.parent / "raritone-3d"
)

if str(RARITONE_3D_DIR) not in sys.path:
    sys.path.insert(0, str(RARITONE_3D_DIR))

from pipeline import process_asset


TEMP_3D_DIR = RARITONE_3D_DIR / "input"
TEMP_3D_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 3D ASSET PROCESSING ENDPOINT
# ---------------------------------------------------------

@app.post("/api/ai/process-3d")
async def process_3d_asset(
    file: UploadFile = File(...)
):
    """
    Upload and process an existing GLB/GLTF 3D asset.

    Pipeline:
    Upload
        ↓
    Validate
        ↓
    Process
        ↓
    Generate Metadata
        ↓
    PENDING_REVIEW
    """

    # -----------------------------------------------------
    # 1. Validate filename
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    # Prevent directory traversal while preserving
    # the original asset filename for metadata extraction.
    safe_filename = Path(file.filename).name

    extension = Path(
        safe_filename
    ).suffix.lower()

    if extension not in {
        ".glb",
        ".gltf",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only GLB and GLTF files are supported."
            ),
        )

    # -----------------------------------------------------
    # 2. Save uploaded asset
    # -----------------------------------------------------

    input_path = (
        TEMP_3D_DIR / safe_filename
    )

    logger.info(
        "3D asset received | filename=%s",
        safe_filename,
    )

    try:
        with open(
            input_path,
            "wb",
        ) as output:
            shutil.copyfileobj(
                file.file,
                output,
            )

        # -------------------------------------------------
        # 3. Run 3D asset pipeline
        # -------------------------------------------------

        logger.info(
            "3D asset processing started | file=%s",
            safe_filename,
        )

        result = await asyncio.to_thread(
            process_asset,
            str(input_path),
        )

        # -------------------------------------------------
        # 4. Handle rejected asset
        # -------------------------------------------------

        if not result.get(
            "success",
            False,
        ):
            logger.warning(
                "3D asset rejected | file=%s",
                safe_filename,
            )

            return {
                "success": False,
                "status": "REJECTED",
                "message": (
                    "3D asset failed validation."
                ),
                "result": result,
            }

        # -------------------------------------------------
        # 5. Successful processing
        # -------------------------------------------------

        logger.info(
            "3D asset processed successfully | "
            "file=%s | status=PENDING_REVIEW",
            safe_filename,
        )

        return {
            "success": True,
            "status": "PENDING_REVIEW",
            "message": (
                "3D asset processed successfully "
                "and is ready for review."
            ),
            "result": result,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "3D asset processing failed | file=%s",
            safe_filename,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"3D processing failed: {str(exc)}"
            ),
        )

    finally:
        await file.close()