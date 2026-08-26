import csv
from pathlib import Path


PERSON_DIR = Path("evaluation/persons")
GARMENT_DIR = Path("evaluation/garments")

OUTPUT_FILE = Path("evaluation/metadata.csv")


PERSON_IMAGES = [
    "darkbackground.jpg",
    "person.jpg",
    "person2.jpg",
]

GARMENT_IMAGES = [
    "garment.jpg",
    "tshirt.jpg",
]


# Four controlled evaluation conditions
VARIANTS = [
    "original",
    "low_resolution",
    "dark",
    "bright",
]


rows = []

test_number = 1


for person_image in PERSON_IMAGES:

    for garment_image in GARMENT_IMAGES:

        for variant in VARIANTS:

            test_id = f"test_{test_number:03d}"

            # Current garment set is primarily upper-body garments.
            if garment_image == "tshirt.jpg":
                category = "t-shirt"
            else:
                category = "shirt/top"

            if variant == "original":
                expected_behavior = (
                    "Try-on should complete successfully."
                )

            elif variant == "low_resolution":
                expected_behavior = (
                    "Quality gate should validate or reject "
                    "depending on pose/image quality."
                )

            elif variant == "dark":
                expected_behavior = (
                    "Pipeline should remain stable under "
                    "reduced lighting."
                )

            else:
                expected_behavior = (
                    "Pipeline should remain stable under "
                    "increased brightness."
                )

            rows.append(
                {
                    "test_id": test_id,
                    "person_image": person_image,
                    "garment_image": garment_image,
                    "category": category,
                    "variant": variant,
                    "expected_behavior": expected_behavior,
                }
            )

            test_number += 1


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = [
        "test_id",
        "person_image",
        "garment_image",
        "category",
        "variant",
        "expected_behavior",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print("=" * 60)
print("VTON EVALUATION DATASET CREATED")
print("=" * 60)

print(
    f"Person source images: {len(PERSON_IMAGES)}"
)

print(
    f"Garment source images: {len(GARMENT_IMAGES)}"
)

print(
    f"Input variants: {len(VARIANTS)}"
)

print(
    f"Total evaluation combinations: {len(rows)}"
)

print(
    f"Metadata saved to: {OUTPUT_FILE}"
)

print("=" * 60)