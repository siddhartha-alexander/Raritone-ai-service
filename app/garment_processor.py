from pathlib import Path

import cv2
import numpy as np
from rembg import remove

from app.segmentation import REM_BG_SESSION


def process_garment(image):
    """
    Remove garment background, generate mask,
    and crop unnecessary empty regions.
    """

    if image is None:
        raise ValueError("Invalid garment image.")

    if not isinstance(image, np.ndarray):
        raise ValueError("Garment image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Garment image is empty.")

    # Convert OpenCV image to PNG bytes
    success, encoded_image = cv2.imencode(
        ".png",
        image,
    )

    if not success:
        raise ValueError(
            "Could not encode garment image."
        )

    # Background removal using shared session
    output_bytes = remove(
        encoded_image.tobytes(),
        session=REM_BG_SESSION,
    )

    output_array = np.frombuffer(
        output_bytes,
        dtype=np.uint8,
    )

    result = cv2.imdecode(
        output_array,
        cv2.IMREAD_UNCHANGED,
    )

    if result is None:
        raise ValueError(
            "Garment processing failed."
        )

    if (
        len(result.shape) != 3
        or result.shape[2] != 4
    ):
        raise ValueError(
            "Garment mask could not be generated."
        )

    garment = result[:, :, :3]
    mask = result[:, :, 3]

    # Clean mask
    _, mask = cv2.threshold(
        mask,
        25,
        255,
        cv2.THRESH_BINARY,
    )

    # Find garment region
    points = cv2.findNonZero(mask)

    if points is None:
        raise ValueError(
            "No garment foreground detected."
        )

    x, y, w, h = cv2.boundingRect(points)

    garment = garment[
        y:y + h,
        x:x + w,
    ]

    mask = mask[
        y:y + h,
        x:x + w,
    ]

    return garment, mask


def resize_with_padding(
    image,
    target_size=512,
    is_mask=False,
):
    """
    Resize while preserving aspect ratio and add padding.
    Prevents garment distortion.
    """

    height, width = image.shape[:2]

    if height == 0 or width == 0:
        raise ValueError(
            "Invalid image dimensions."
        )

    scale = min(
        target_size / width,
        target_size / height,
    )

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    interpolation = (
        cv2.INTER_NEAREST
        if is_mask
        else cv2.INTER_AREA
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    if is_mask:
        canvas = np.zeros(
            (target_size, target_size),
            dtype=np.uint8,
        )
    else:
        canvas = np.zeros(
            (
                target_size,
                target_size,
                3,
            ),
            dtype=np.uint8,
        )

    x_offset = (
        target_size - new_width
    ) // 2

    y_offset = (
        target_size - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return canvas


def prepare_garment_image(image):
    """
    Robust garment preprocessing.

    Output:
        garment_input
        garment_mask
        normalized_input
        metadata
    """

    processed_garment, garment_mask = (
        process_garment(image)
    )

    original_height, original_width = (
        processed_garment.shape[:2]
    )

    garment_input = resize_with_padding(
        processed_garment,
        target_size=512,
        is_mask=False,
    )

    garment_mask = resize_with_padding(
        garment_mask,
        target_size=512,
        is_mask=True,
    )

    # Model-ready normalized representation
    normalized_input = (
        garment_input.astype(
            np.float32
        ) / 255.0
    )

    metadata = {
        "original_resolution": [
            original_width,
            original_height,
        ],
        "prepared_resolution": [
            512,
            512,
        ],
        "mask_available": True,
        "category": "unknown",
    }

    return {
        "garment_input": garment_input,
        "garment_mask": garment_mask,
        "normalized_input": normalized_input,
        "metadata": metadata,
    }


def save_processed_garment(
    garment,
    mask,
    output_dir="outputs/processed_garment",
):
    """
    Save processed garment and garment mask.
    """

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    garment_path = (
        output_path / "garment.png"
    )

    mask_path = (
        output_path / "mask.png"
    )

    transparent = cv2.cvtColor(
        garment,
        cv2.COLOR_BGR2BGRA,
    )

    transparent[:, :, 3] = mask

    cv2.imwrite(
        str(garment_path),
        transparent,
    )

    cv2.imwrite(
        str(mask_path),
        mask,
    )

    return {
        "garment_path": str(
            garment_path
        ),
        "mask_path": str(
            mask_path
        ),
    }