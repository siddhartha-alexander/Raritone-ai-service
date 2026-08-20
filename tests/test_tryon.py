import cv2

from app.pose_detector import PoseDetector
from app.preprocessing import preprocess_image
from app.garment_processor import process_garment
from app.tryon_pipeline import (
    calculate_garment_region,
    align_garment,
    composite_garment,
)


# -------------------------
# Load person image
# -------------------------

person_image_path = "test_data/persons/person.jpg"

person_image = cv2.imread(person_image_path)

if person_image is None:
    raise ValueError("Person image could not be loaded.")

person_image = preprocess_image(person_image)


# -------------------------
# Detect pose
# -------------------------

pose_detector = PoseDetector()

landmarks = pose_detector.detect(person_image)

if landmarks is None:
    raise ValueError("No person detected.")




garment_image_path = "test_data/garments/tshirt.jpg"

garment_image = cv2.imread(garment_image_path)

if garment_image is None:
    raise ValueError("Garment image could not be loaded.")



garment, mask = process_garment(
    garment_image
)



image_height, image_width = person_image.shape[:2]

region = calculate_garment_region(
    landmarks,
    image_width,
    image_height,
)

print("Garment region:", region)




aligned_garment, aligned_mask = align_garment(
    garment,
    mask,
    region,
)




result = composite_garment(
    person_image,
    aligned_garment,
    aligned_mask,
    region,
)




output_path = "outputs/tryon_result.png"

cv2.imwrite(
    output_path,
    result,
)

print("Baseline try-on completed.")
print("Result saved to:", output_path)