import cv2

from app.person_processor import (
    prepare_person_image,
)


IMAGE_PATH = (
    "vton_dataset/persons/person.jpg"
)


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise ValueError(
        "Could not load person image."
    )


result = prepare_person_image(
    image
)


print("\nPERSON QUALITY RESULT")
print("=" * 50)

for key, value in result.items():

    if key in {
        "person_input",
        "person_mask",
        "pose_data",
    }:
        continue

    print(
        f"{key}: {value}"
    )

print("=" * 50)