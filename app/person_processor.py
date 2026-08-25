import time

import cv2
import numpy as np

from app.pose_detector import PoseDetector
from app.preprocessing import preprocess_image
from app.segmentation import (
    person_detected,
    segment_person,
)


pose_detector = PoseDetector()


MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 400
MIN_POSE_QUALITY = 0.60
MIN_MASK_QUALITY = 0.10


def calculate_pose_quality(pose_data):
    """
    Average landmark visibility score.
    Returns value between 0 and 1.
    """

    if not pose_data:
        return 0.0

    visibility_values = []

    for landmark in pose_data.values():
        visibility = landmark.get(
            "visibility",
            0.0,
        )

        visibility_values.append(
            float(visibility)
        )

    if not visibility_values:
        return 0.0

    return round(
        sum(visibility_values)
        / len(visibility_values),
        4,
    )


def calculate_mask_quality(mask):
    """
    Basic mask-quality heuristic based on foreground coverage.
    """

    if mask is None:
        return 0.0

    total_pixels = mask.size

    if total_pixels == 0:
        return 0.0

    foreground_pixels = np.count_nonzero(
        mask > 30
    )

    foreground_ratio = (
        foreground_pixels
        / total_pixels
    )

    # Extremely tiny or nearly full-image masks
    # are considered suspicious.
    if foreground_ratio < 0.02:
        return 0.0

    if foreground_ratio > 0.95:
        return 0.0

    return round(
        min(
            foreground_ratio * 3.0,
            1.0,
        ),
        4,
    )


def check_body_inside_frame(pose_data):
    """
    Check whether important landmarks remain inside
    the normalized image boundaries.
    """

    important_landmarks = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    for name in important_landmarks:

        landmark = pose_data.get(name)

        if landmark is None:
            return False

        x = landmark.get("x", -1)
        y = landmark.get("y", -1)

        if not (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
        ):
            return False

    return True


def prepare_person_image(image):
    """
    Robust person-quality validation pipeline.
    """

    start_time = time.perf_counter()

    if image is None:
        raise ValueError(
            "Invalid person image."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise ValueError(
            "Person image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Person image is empty."
        )

    height, width = image.shape[:2]

    # --------------------------------------------------
    # Resolution check
    # --------------------------------------------------

    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):
        return {
            "valid": False,
            "error_code": "LOW_RESOLUTION",
            "message": (
                "Please upload a higher-resolution "
                "full-body image."
            ),
        }

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    person_input = preprocess_image(
        image
    )

    # --------------------------------------------------
    # Segmentation
    # --------------------------------------------------

    person_mask = segment_person(
        person_input
    )

    if person_mask is None:
        return {
            "valid": False,
            "error_code": "MASK_GENERATION_FAILED",
            "message": (
                "Person segmentation failed."
            ),
        }

    if not person_detected(
        person_mask
    ):
        return {
            "valid": False,
            "error_code": "PERSON_NOT_DETECTED",
            "message": (
                "Please upload an image containing "
                "one clearly visible person."
            ),
        }

    # --------------------------------------------------
    # Pose detection
    # --------------------------------------------------

    pose_data = pose_detector.detect(
        person_input
    )

    if pose_data is None:
        return {
            "valid": False,
            "error_code": "POSE_NOT_DETECTED",
            "message": (
                "Please upload a clear full-body image "
                "with the person facing the camera."
            ),
        }

    # --------------------------------------------------
    # Quality scores
    # --------------------------------------------------

    pose_quality = (
        calculate_pose_quality(
            pose_data
        )
    )

    mask_quality = (
        calculate_mask_quality(
            person_mask
        )
    )

    body_inside_frame = (
        check_body_inside_frame(
            pose_data
        )
    )

    # --------------------------------------------------
    # Quality rejection
    # --------------------------------------------------

    if pose_quality < MIN_POSE_QUALITY:
        return {
            "valid": False,
            "error_code": "LOW_POSE_QUALITY",
            "message": (
                "Pose quality is too low. "
                "Please use a clear front-facing image."
            ),
            "pose_quality": pose_quality,
            "mask_quality": mask_quality,
        }

    if mask_quality < MIN_MASK_QUALITY:
        return {
            "valid": False,
            "error_code": "LOW_MASK_QUALITY",
            "message": (
                "Person segmentation quality is too low."
            ),
            "pose_quality": pose_quality,
            "mask_quality": mask_quality,
        }

    if not body_inside_frame:
        return {
            "valid": False,
            "error_code": "PARTIAL_BODY",
            "message": (
                "Please upload a full-body image "
                "with the complete body inside the frame."
            ),
            "pose_quality": pose_quality,
            "mask_quality": mask_quality,
        }

    processing_time = round(
        time.perf_counter()
        - start_time,
        4,
    )

    return {
        "valid": True,
        "person_count": 1,
        "person_input": person_input,
        "person_mask": person_mask,
        "pose_data": pose_data,
        "pose_quality": pose_quality,
        "mask_quality": mask_quality,
        "processing_time": processing_time,
    }