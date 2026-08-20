import cv2

from app.garment_processor import (
    process_garment,
    save_processed_garment,
)


image_path = "test_data/garments/t shirt .jpg"
image = cv2.imread(image_path)

garment, mask = process_garment(image)

paths = save_processed_garment(
    garment,
    mask,
)

print("Garment processing successful")
print("Garment:", paths["garment_path"])
print("Mask:", paths["mask_path"])