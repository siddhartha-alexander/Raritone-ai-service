import time

import numpy as np

from app.pose_detector import PoseDetector
from app.preprocessing import preprocess_image
from app.segmentation import (
    person_detected,
    segment_person,
)



# Model loaded once


pose_detector = PoseDetector()



# Quality Thresholds


MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 400

MIN_POSE_QUALITY = 0.60
MIN_MASK_QUALITY = 0.10

# Improvement 2:
# Allow a small tolerance around the image boundary.
LANDMARK_BOUNDARY_TOLERANCE = 0.05



# Pose Quality


def calculate_pose_quality(pose_data):
    """
    Calculate average pose landmark visibility.

    Returns a score between 0 and 1.
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



# Mask Quality


def calculate_mask_quality(mask):
    """
    Estimate segmentation quality using foreground coverage.

    This is a heuristic quality indicator,
    not a ground-truth segmentation metric.
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

    # Very small foreground area is suspicious.
    if foreground_ratio < 0.02:
        return 0.0

    # Nearly the whole image being foreground is also suspicious.
    if foreground_ratio > 0.95:
        return 0.0

    return round(
        min(
            foreground_ratio * 3.0,
            1.0,
        ),
        4,
    )



# Landmark Boundary Check


def landmark_inside_frame(
    landmark,
    tolerance=LANDMARK_BOUNDARY_TOLERANCE,
):
    """
    Validate a landmark with a small normalized
    boundary tolerance.

    Example:
    tolerance=0.05 accepts coordinates from
    -0.05 to 1.05.
    """

    if landmark is None:
        return False

    x = float(
        landmark.get(
            "x",
            -999,
        )
    )

    y = float(
        landmark.get(
            "y",
            -999,
        )
    )

    lower_limit = -tolerance
    upper_limit = 1.0 + tolerance

    return (
        lower_limit
        <= x
        <= upper_limit
        and
        lower_limit
        <= y
        <= upper_limit
    )



# Category-Aware Body Validation


def check_body_inside_frame(
    pose_data,
    garment_category="upper_body",
):
    """
    Validate only the landmarks required for the
    requested garment category.

    Improvement 1:
    Upper-body try-on should not require ankles or knees.

    Supported categories:
    - upper_body
    - top
    - shirt
    - t-shirt
    - hoodie
    - dress
    - trousers
    - lower_body
    - full_body
    """

    if not pose_data:
        return False

    category = (
        garment_category
        or "upper_body"
    ).lower()

    # ---------------------------------------------
    # Upper-body garments
    # ---------------------------------------------

    if category in {
        "upper_body",
        "top",
        "shirt",
        "shirt/top",
        "t-shirt",
        "tshirt",
        "hoodie",
    }:

        required_landmarks = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ]

    # ---------------------------------------------
    # Lower-body garments
    # ---------------------------------------------

    elif category in {
        "trousers",
        "lower_body",
        "pants",
    }:

        required_landmarks = [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

    # ---------------------------------------------
    # Full-body garments
    # ---------------------------------------------

    elif category in {
        "dress",
        "full_body",
    }:

        required_landmarks = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
        ]

    else:

        # Conservative fallback
        required_landmarks = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ]

    # ---------------------------------------------
    # Validate required landmarks
    # ---------------------------------------------

    for name in required_landmarks:

        landmark = pose_data.get(
            name
        )

        if not landmark_inside_frame(
            landmark
        ):
            return False

    return True



# Main Person Preprocessing


def prepare_person_image(
    image,
    garment_category="upper_body",
):
    """
    Robust person-quality validation pipeline.

    Pipeline:
        Input
        ↓
        Resolution Validation
        ↓
        Preprocessing
        ↓
        Segmentation
        ↓
        Person Detection
        ↓
        Pose Detection
        ↓
        Pose + Mask Quality
        ↓
        Category-Aware Landmark Validation
        ↓
        Model Input
    """

    start_time = time.perf_counter()

    
    # Input Validation
    

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

    
    # Resolution Validation
    

    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):

        return {
            "valid": False,
            "error_code": "LOW_RESOLUTION",
            "message": (
                "Please upload a higher-resolution "
                "person image."
            ),
        }

    
    # Preprocessing
    

    person_input = preprocess_image(
        image
    )

    
    # Person Segmentation
    

    person_mask = segment_person(
        person_input
    )

    if person_mask is None:

        return {
            "valid": False,
            "error_code": (
                "MASK_GENERATION_FAILED"
            ),
            "message": (
                "Person segmentation failed."
            ),
        }

    if not person_detected(
        person_mask
    ):

        return {
            "valid": False,
            "error_code": (
                "PERSON_NOT_DETECTED"
            ),
            "message": (
                "Please upload an image "
                "containing one clearly "
                "visible person."
            ),
        }

    
    # Pose Detection
    

    pose_data = pose_detector.detect(
        person_input
    )

    if pose_data is None:

        return {
            "valid": False,
            "error_code": (
                "POSE_NOT_DETECTED"
            ),
            "message": (
                "Please upload a clear image "
                "with the person facing the camera."
            ),
        }

    
    # Quality Scores
    

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

    
    # Category-Aware Body Validation
    

    body_inside_frame = (
        check_body_inside_frame(
            pose_data,
            garment_category=(
                garment_category
            ),
        )
    )

    
    # Pose Quality Gate
    

    if (
        pose_quality
        < MIN_POSE_QUALITY
    ):

        return {
            "valid": False,
            "error_code": (
                "LOW_POSE_QUALITY"
            ),
            "message": (
                "Pose quality is too low. "
                "Please use a clear "
                "front-facing image."
            ),
            "pose_quality": (
                pose_quality
            ),
            "mask_quality": (
                mask_quality
            ),
        }

    
    # Mask Quality Gate
    

    if (
        mask_quality
        < MIN_MASK_QUALITY
    ):

        return {
            "valid": False,
            "error_code": (
                "LOW_MASK_QUALITY"
            ),
            "message": (
                "Person segmentation "
                "quality is too low."
            ),
            "pose_quality": (
                pose_quality
            ),
            "mask_quality": (
                mask_quality
            ),
        }

    
    # Body Visibility Gate
    

    if not body_inside_frame:

        return {
            "valid": False,
            "error_code": (
                "PARTIAL_BODY"
            ),
            "message": (
                "Required body landmarks "
                "for this garment category "
                "are outside the usable frame."
            ),
            "pose_quality": (
                pose_quality
            ),
            "mask_quality": (
                mask_quality
            ),
        }

    
    # Successful Preparation
    

    processing_time = round(
        time.perf_counter()
        - start_time,
        4,
    )

    return {
        "valid": True,
        "person_count": 1,
        "person_input": (
            person_input
        ),
        "person_mask": (
            person_mask
        ),
        "pose_data": (
            pose_data
        ),
        "pose_quality": (
            pose_quality
        ),
        "mask_quality": (
            mask_quality
        ),
        "garment_category": (
            garment_category
        ),
        "processing_time": (
            processing_time
        ),
    }