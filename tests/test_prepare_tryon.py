import cv2

from app.person_processor import prepare_tryon_input


# -----------------------------------
# Load test images
# -----------------------------------

person_image = cv2.imread(
    "test_data/persons/person.jpg"
)

garment_image = cv2.imread(
    "test_data/garments/tshirt.jpg"
)


# -----------------------------------
# Validate images loaded
# -----------------------------------

if person_image is None:
    raise ValueError(
        "Person image could not be loaded."
    )

if garment_image is None:
    raise ValueError(
        "Garment image could not be loaded."
    )


# -----------------------------------
# Prepare try-on inputs
# -----------------------------------

(
    person_input,
    garment_input,
    person_mask,
    pose_data,
    metadata,
) = prepare_tryon_input(
    person_image,
    garment_image,
)


# -----------------------------------
# Print results
# -----------------------------------

print("\nTry-On Input Preparation Successful\n")

print("Person input shape:")
print(person_input.shape)

print("\nGarment input shape:")
print(garment_input.shape)

print("\nPerson mask shape:")
print(person_mask.shape)

print("\nPose available:")
print(pose_data is not None)

print("\nMetadata:")
print(metadata)