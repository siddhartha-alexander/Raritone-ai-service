import cv2

from app.person_processor import (
    prepare_person_image,
)
from app.garment_processor import (
    prepare_garment_v2,
)
from app.quality_gate import (
    QualityGateError,
    run_quality_gate,
)


PERSON_PATH = (
    "vton_dataset/persons/person.jpg"
)

GARMENT_PATH = (
    "vton_dataset/garments/tshirt.jpg"
)


person_image = cv2.imread(
    PERSON_PATH
)

garment_image = cv2.imread(
    GARMENT_PATH
)


if person_image is None:
    raise ValueError(
        "Person image could not be loaded."
    )

if garment_image is None:
    raise ValueError(
        "Garment image could not be loaded."
    )


print("\nPreparing person...")

person_result = prepare_person_image(
    person_image
)


print("Preparing garment...")

garment_result = prepare_garment_v2(
    garment_image
)


print("\nRunning quality gate...")


try:

    result = run_quality_gate(
        person_result,
        garment_result,
    )

    print("\nQUALITY GATE PASSED")
    print("=" * 50)

    print(
        "passed:",
        result["passed"],
    )

    print(
        "pose quality:",
        result["quality"]["pose"],
    )

    print(
        "mask quality:",
        result["quality"]["mask"],
    )

    print(
        "garment quality:",
        result["quality"]["garment"],
    )

    print("=" * 50)

except QualityGateError as exc:

    print("\nQUALITY GATE REJECTED")
    print("=" * 50)

    print(
        "error_code:",
        exc.error_code,
    )

    print(
        "message:",
        exc.message,
    )

    print(
        "quality:",
        exc.quality,
    )

    print("=" * 50)