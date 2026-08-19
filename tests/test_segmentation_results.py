import csv
import time
from pathlib import Path

import cv2

from app.segmentation import (
    person_detected,
    segment_person,
)


IMAGE_DIR = Path("sample_images")
CSV_PATH = Path("segmentation_results.csv")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def assess_quality(mask):
    """
    Basic automatic mask-quality assessment.

    This is only a simple heuristic, not a true segmentation
    quality metric.
    """
    if mask is None:
        return "poor"

    foreground_ratio = (mask > 10).mean()

    if foreground_ratio < 0.05:
        return "poor"

    if foreground_ratio > 0.95:
        return "poor"

    return "acceptable"


rows = []

images = [
    path
    for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in SUPPORTED_EXTENSIONS
]

if not images:
    raise FileNotFoundError(
        "No JPG, JPEG, or PNG images found in sample_images."
    )


for image_path in images:

    print(f"\nTesting: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image.")

        rows.append([
            image_path.name,
            False,
            False,
            "poor",
            0,
            "Image could not be loaded",
        ])

        continue

    start_time = time.perf_counter()

    try:
        mask = segment_person(image)

        processing_time = round(
            time.perf_counter() - start_time,
            4,
        )

        detected = person_detected(mask)
        mask_generated = mask is not None

        quality = assess_quality(mask)

        if detected:
            notes = "Segmentation successful"
        else:
            notes = "No person/foreground detected"

        print(f"Person detected: {detected}")
        print(f"Mask generated: {mask_generated}")
        print(f"Quality: {quality}")
        print(f"Processing time: {processing_time} seconds")

        rows.append([
            image_path.name,
            detected,
            mask_generated,
            quality,
            processing_time,
            notes,
        ])

    except Exception as exc:

        processing_time = round(
            time.perf_counter() - start_time,
            4,
        )

        print(f"Segmentation failed: {exc}")

        rows.append([
            image_path.name,
            False,
            False,
            "poor",
            processing_time,
            f"Error: {exc}",
        ])


with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "image",
        "person_detected",
        "mask_generated",
        "quality",
        "processing_time",
        "notes",
    ])

    writer.writerows(rows)


print("\n--------------------------------")
print("All segmentation tests completed.")
print(f"Results saved to: {CSV_PATH}")