from pathlib import Path

import cv2
import numpy as np
from rembg import remove


def process_garment(image):
    """
    Process a garment image.

    Steps:
    1. Validate image
    2. Remove background
    3. Extract alpha mask
    4. Clean mask
    5. Crop unnecessary transparent area
    """

    if image is None:
        raise ValueError(
            "Invalid garment image."
        )

  
    # Encode OpenCV image
  

    success, encoded_image = cv2.imencode(
        ".png",
        image,
    )

    if not success:
        raise ValueError(
            "Could not encode garment image."
        )

  
    # Remove background
  

    output_bytes = remove(
        encoded_image.tobytes()
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

  
    # Validate BGRA output
  

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

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

  
    # Find actual garment boundaries
  

    points = cv2.findNonZero(mask)

    if points is None:
        raise ValueError(
            "No garment detected."
        )

    x, y, w, h = cv2.boundingRect(
        points
    )

    # Add a very small safety margin
    padding = 5

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(
        garment.shape[1],
        x + w + padding,
    )

    y2 = min(
        garment.shape[0],
        y + h + padding,
    )

    garment = garment[
        y1:y2,
        x1:x2,
    ]

    mask = mask[
        y1:y2,
        x1:x2,
    ]

    return garment, mask


def save_processed_garment(
    garment,
    mask,
    output_dir="outputs/processed_garment",
):
    """
    Save the processed garment and mask.
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

  
    # Create transparent BGRA garment
  

    transparent_garment = cv2.cvtColor(
        garment,
        cv2.COLOR_BGR2BGRA,
    )

    transparent_garment[:, :, 3] = mask

  
    # Save files
  

    cv2.imwrite(
        str(garment_path),
        transparent_garment,
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