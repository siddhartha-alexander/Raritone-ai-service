import time

import cv2

from app.segmentation import (
    save_outputs,
    segment_person,
    remove_background,
)


image_path = "sample_images/person.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load {image_path}"
    )

print("Image loaded successfully.")

start_time = time.perf_counter()

mask = segment_person(image)

background_removed = remove_background(
    image,
    mask,
)

processing_time = round(
    time.perf_counter() - start_time,
    4,
)

mask_path, foreground_path = save_outputs(
    mask,
    background_removed,
)

person_detected = bool(mask.max() > 0)
mask_generated = mask is not None

print(f"Person detected: {person_detected}")
print(f"Mask generated: {mask_generated}")
print(f"Processing time: {processing_time} seconds")
print(f"Mask saved to: {mask_path}")
print(
    f"Background removed image saved to: "
    f"{foreground_path}"
)