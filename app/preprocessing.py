import cv2
import numpy as np


SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
}


def validate_image_type(content_type: str | None) -> None:
    """Validate that the uploaded file is a supported image type."""
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(
            "Unsupported file type. Only JPEG and PNG images are allowed."
        )


def decode_image(image_bytes: bytes):
    """Convert uploaded image bytes into an OpenCV image."""
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode the uploaded image.")

    return image


def preprocess_image(image):
    """
    Perform basic OpenCV preprocessing.

    The image is currently kept unchanged because MediaPipe
    performs its own pose-specific processing.
    """
    if image is None:
        raise ValueError("Invalid image.")

    return image