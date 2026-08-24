import time

import cv2
import numpy as np

from app.garment_processor import (
    prepare_garment_image,
)
from app.pose_detector import PoseDetector
from app.preprocessing import (
    preprocess_image,
)
from app.segmentation import (
    person_detected,
    segment_person,
)


# Load once when server starts
pose_detector = PoseDetector()


def resize_person_preserve_ratio(
    image,
    max_dimension=1024,
):
    """
    Resize large person images while preserving
    aspect ratio.
    """

    height, width = image.shape[:2]

    max_side = max(
        height,
        width,
    )

    if max_side <= max_dimension:
        return image

    scale = (
        max_dimension / max_side
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def prepare_person_image(image):
    """
    Robust person preprocessing pipeline.

    Image
        ↓
    Validation
        ↓
    Resize
        ↓
    OpenCV preprocessing
        ↓
    Segmentation
        ↓
    Pose detection
        ↓
    Normalization
        ↓
    Model-ready data
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
            "Person image must be "
            "a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Person image is empty."
        )

    # Preserve aspect ratio
    person_input = (
        resize_person_preserve_ratio(
            image
        )
    )

    # Existing OpenCV preprocessing
    person_input = preprocess_image(
        person_input
    )

    # Person segmentation
    mask = segment_person(
        person_input
    )

    if mask is None:
        raise ValueError(
            "Failed to generate person mask."
        )

    if not person_detected(mask):
        raise ValueError(
            "No person detected in the image."
        )

    # Pose detection
    pose_data = pose_detector.detect(
        person_input
    )

    if pose_data is None:
        raise ValueError(
            "Pose information could not "
            "be detected."
        )

    # Model-ready normalized input
    normalized_input = (
        person_input.astype(
            np.float32
        ) / 255.0
    )

    processing_time = round(
        time.perf_counter()
        - start_time,
        4,
    )

    metadata = {
        "person_shape": list(
            person_input.shape
        ),
        "person_detected": True,
        "mask_available": True,
        "pose_available": True,
        "processing_time": (
            processing_time
        ),
    }

    return {
        "person_input": person_input,
        "normalized_input": (
            normalized_input
        ),
        "person_mask": mask,
        "pose_data": pose_data,
        "metadata": metadata,
    }


def prepare_tryon_input(
    person_image,
    garment_image,
):
    """
    Maintain compatibility with the existing
    /api/ai/tryon pipeline.
    """

    start_time = time.perf_counter()

    person_result = (
        prepare_person_image(
            person_image
        )
    )

    garment_result = (
        prepare_garment_image(
            garment_image
        )
    )

    total_time = round(
        time.perf_counter()
        - start_time,
        4,
    )

    metadata = {
        "person_shape": list(
            person_result[
                "person_input"
            ].shape
        ),
        "garment_shape": list(
            garment_result[
                "garment_input"
            ].shape
        ),
        "person_detected": True,
        "mask_available": True,
        "pose_available": True,
        "preprocessing_time": (
            total_time
        ),
        "total_preparation_time": (
            total_time
        ),
    }

    return (
        person_result[
            "person_input"
        ],
        garment_result[
            "garment_input"
        ],
        person_result[
            "person_mask"
        ],
        person_result[
            "pose_data"
        ],
        metadata,
    )