import cv2
import time

from app.pose_detector import PoseDetector
from app.preprocessing import preprocess_image
from app.segmentation import (
    person_detected,
    segment_person,
)


pose_detector = PoseDetector()


def prepare_tryon_input(person_image, garment_image):
    """
    Prepare validated inputs for the virtual try-on pipeline.

    Returns:
        person_input
        garment_input
        person_mask
        pose_data
        metadata
    """

    start_time = time.perf_counter()

    # -----------------------------------
    # Validate person image
    # -----------------------------------

    if person_image is None:
        raise ValueError("Invalid person image.")

    if not isinstance(person_image, type(garment_image)):
        raise ValueError("Invalid image format.")

    # -----------------------------------
    # Validate garment image
    # -----------------------------------

    if garment_image is None:
        raise ValueError("Invalid garment image.")

    preprocessing_start = time.perf_counter()

    # -----------------------------------
    # Preprocess person image
    # -----------------------------------

    person_input = preprocess_image(person_image)

    # -----------------------------------
    # Person segmentation
    # -----------------------------------

    mask = segment_person(person_input)

    if mask is None:
        raise ValueError("Failed to generate person mask.")

    if not person_detected(mask):
        raise ValueError("No person detected in the image.")

    # -----------------------------------
    # Pose detection
    # -----------------------------------

    pose_data = pose_detector.detect(person_input)

    if pose_data is None:
        raise ValueError("Pose information could not be detected.")

    # -----------------------------------
    # Normalize garment image
    # -----------------------------------

    garment_input = cv2.resize(
        garment_image,
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )

    preprocessing_time = round(
        time.perf_counter() - preprocessing_start,
        4,
    )

    total_time = round(
        time.perf_counter() - start_time,
        4,
    )

    # -----------------------------------
    # Metadata
    # -----------------------------------

    metadata = {
        "person_shape": list(person_input.shape),
        "garment_shape": list(garment_input.shape),
        "person_detected": True,
        "pose_available": True,
        "preprocessing_time": preprocessing_time,
        "total_preparation_time": total_time,
    }

    return (
        person_input,
        garment_input,
        mask,
        pose_data,
        metadata,
    )