import cv2
import time

from app.person_processor import prepare_tryon_input
from app.inference import run_tryon



# Load person image


person_image = cv2.imread(
    "test_data/persons/person.jpg"
)

if person_image is None:
    raise ValueError(
        "Person image could not be loaded."
    )



# Load garment image


garment_image = cv2.imread(
    "test_data/garments/tshirt.jpg"
)

if garment_image is None:
    raise ValueError(
        "Garment image could not be loaded."
    )



# Prepare try-on inputs


pipeline_start = time.perf_counter()

(
    person_input,
    garment_input,
    person_mask,
    pose_data,
    preparation_metadata,
) = prepare_tryon_input(
    person_image,
    garment_image,
)



# Run try-on inference


result_image, inference_metadata = run_tryon(
    person_input,
    garment_input,
    pose_data,
    person_mask,
)



# Calculate total pipeline time


total_pipeline_time = round(
    time.perf_counter() - pipeline_start,
    4,
)



# Print results


print("\nTRY-ON INFERENCE SUCCESSFUL\n")

print("Result shape:")
print(result_image.shape)

print("\nPreparation metadata:")
print(preparation_metadata)

print("\nInference metadata:")
print(inference_metadata)

print("\nTotal pipeline time:")
print(total_pipeline_time, "seconds")

print("\nResult saved to:")
print("outputs/tryon_result.png")