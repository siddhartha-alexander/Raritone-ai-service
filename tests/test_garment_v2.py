import cv2

from app.garment_processor import (
    prepare_garment_v2,
)


IMAGE_PATH = (
    "vton_dataset/garments/tshirt.jpg"
)


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise ValueError(
        "Could not load garment image."
    )


result = prepare_garment_v2(
    image
)


print("\nGARMENT QUALITY RESULT")
print("=" * 50)

for key, value in result.items():

    if key in {
        "garment_input",
        "garment_mask",
    }:
        continue

    print(
        f"{key}: {value}"
    )

print("=" * 50)