from pathlib import Path

import cv2
import numpy as np
from rembg import remove


def process_garment(image):
    """
    Remove garment background, generate mask,
    and crop unnecessary transparent space.
    """

    if image is None:
        raise ValueError("Invalid garment image.")

    # Convert OpenCV image to PNG bytes
    success, encoded_image = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Could not encode garment image.")

    # Remove background
    output_bytes = remove(encoded_image.tobytes())

    # Convert result back to OpenCV image
    output_array = np.frombuffer(
        output_bytes,
        dtype=np.uint8
    )

    result = cv2.imdecode(
        output_array,
        cv2.IMREAD_UNCHANGED
    )

    if result is None:
        raise ValueError("Garment processing failed.")

    # rembg should return BGRA image
    if len(result.shape) != 3 or result.shape[2] != 4:
        raise ValueError(
            "Garment mask could not be generated."
        )

    # Separate garment and alpha mask
    garment = result[:, :, :3]
    mask = result[:, :, 3]

    # Find actual garment boundaries
    points = cv2.findNonZero(mask)

    if points is None:
        raise ValueError("No garment detected.")

    # Crop unnecessary empty area
    x, y, w, h = cv2.boundingRect(points)

    garment = garment[y:y + h, x:x + w]
    mask = mask[y:y + h, x:x + w]

    return garment, mask


def save_processed_garment(
    garment,
    mask,
    output_dir="outputs/processed_garment",
):
    """
    Save processed garment with transparency
    and save its mask.
    """

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    garment_path = output_path / "garment.png"
    mask_path = output_path / "mask.png"

    # Convert BGR to BGRA
    transparent_garment = cv2.cvtColor(
        garment,
        cv2.COLOR_BGR2BGRA
    )

    # Add mask as alpha channel
    transparent_garment[:, :, 3] = mask

    # Save garment and mask
    cv2.imwrite(
        str(garment_path),
        transparent_garment
    )

    cv2.imwrite(
        str(mask_path),
        mask
    )

    return {
        "garment_path": str(garment_path),
        "mask_path": str(mask_path),
    }