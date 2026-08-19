from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove


# Create the segmentation model session once when
# the application imports this module.
REM_BG_SESSION = new_session("u2net")


def segment_person(image):
    """
    Generate a foreground/person segmentation mask.
    """

    if image is None:
        raise ValueError("Invalid image.")

    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Could not encode image.")

    input_bytes = encoded.tobytes()

    output_bytes = remove(
        input_bytes,
        session=REM_BG_SESSION,
    )

    output_array = np.frombuffer(output_bytes, np.uint8)

    output_image = cv2.imdecode(
        output_array,
        cv2.IMREAD_UNCHANGED,
    )

    if output_image is None:
        raise ValueError(
            "Could not decode segmentation output."
        )

    if (
        output_image.ndim == 3
        and output_image.shape[2] == 4
    ):
        mask = output_image[:, :, 3]
    else:
        raise ValueError(
            "Segmentation model did not return an alpha mask."
        )

    return mask


def person_detected(mask, threshold=10):
    """
    Determine whether the segmentation mask contains
    a meaningful foreground region.
    """

    if mask is None:
        return False

    foreground_pixels = int(
        (mask > threshold).sum()
    )

    return foreground_pixels > 0


def remove_background(image, mask):
    """
    Create an image with the background removed.
    """

    if image is None or mask is None:
        raise ValueError(
            "Image and mask are required."
        )

    return cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )


def save_outputs(
    mask,
    background_removed,
    output_dir="outputs",
):
    """
    Save segmentation outputs to disk.
    """

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = (
        output_path / "person_mask.png"
    )

    foreground_path = (
        output_path
        / "person_background_removed.png"
    )

    if not cv2.imwrite(
        str(mask_path),
        mask,
    ):
        raise ValueError(
            "Failed to save person mask."
        )

    if not cv2.imwrite(
        str(foreground_path),
        background_removed,
    ):
        raise ValueError(
            "Failed to save background-removed image."
        )

    return (
        str(mask_path),
        str(foreground_path),
    )