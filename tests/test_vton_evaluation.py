import csv
import time
from pathlib import Path

import cv2

from app.inference import run_tryon
from app.person_processor import prepare_tryon_input


PERSON_DIR = Path("vton_dataset/persons")
GARMENT_DIR = Path("vton_dataset/garments")
RESULT_DIR = Path("vton_dataset/results")

CSV_PATH = Path("vton_evaluation.csv")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


person_images = [
    path
    for path in PERSON_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in SUPPORTED_EXTENSIONS
]

garment_images = [
    path
    for path in GARMENT_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in SUPPORTED_EXTENSIONS
]


if not person_images:
    raise FileNotFoundError(
        "No person images found in vton_dataset/persons."
    )

if not garment_images:
    raise FileNotFoundError(
        "No garment images found in vton_dataset/garments."
    )


rows = []

combination_number = 1


for person_path in person_images:

    for garment_path in garment_images:

        pair_id = (
            f"pair_{combination_number:03d}"
        )

        print("\n" + "=" * 60)

        print(
            f"Testing {pair_id}: "
            f"{person_path.name} + "
            f"{garment_path.name}"
        )

        print("=" * 60)

        start_time = time.perf_counter()

        success = False
        failure_reason = ""

        result_reference = ""

        try:

            # -----------------------------------
            # Load images
            # -----------------------------------

            person_image = cv2.imread(
                str(person_path)
            )

            garment_image = cv2.imread(
                str(garment_path)
            )

            if person_image is None:
                raise ValueError(
                    "Person image could not be loaded."
                )

            if garment_image is None:
                raise ValueError(
                    "Garment image could not be loaded."
                )

            # -----------------------------------
            # Prepare inputs
            # -----------------------------------

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

            # -----------------------------------
            # Run try-on
            # -----------------------------------

            result_image, inference_metadata = (
                run_tryon(
                    person_input,
                    garment_input,
                    pose_data,
                    person_mask,
                )
            )

            # -----------------------------------
            # Save unique result
            # -----------------------------------

            result_path = (
                RESULT_DIR
                / f"{pair_id}.png"
            )

            saved = cv2.imwrite(
                str(result_path),
                result_image,
            )

            if not saved:
                raise RuntimeError(
                    "Could not save try-on result."
                )

            result_reference = str(
                result_path
            )

            success = True

            print(
                "Success | Result:",
                result_reference,
            )

        except Exception as exc:

            failure_reason = str(exc)

            print(
                "Failed:",
                failure_reason,
            )

        processing_time = round(
            time.perf_counter()
            - start_time,
            4,
        )

        # -----------------------------------
        # Save CSV row
        # -----------------------------------

        rows.append({
            "pair_id": pair_id,
            "person_image": person_path.name,
            "garment_image": garment_path.name,
            "success": success,
            "processing_time": processing_time,
            "garment_alignment": "",
            "garment_preservation": "",
            "body_alignment": "",
            "boundary_quality": "",
            "face_preservation": "",
            "overall_realism": "",
            "result_image": result_reference,
            "failure_reason": failure_reason,
        })

        print(
            "Processing time:",
            processing_time,
            "seconds",
        )

        combination_number += 1


# =========================================================
# Write evaluation CSV
# =========================================================

fieldnames = [
    "pair_id",
    "person_image",
    "garment_image",
    "success",
    "processing_time",
    "garment_alignment",
    "garment_preservation",
    "body_alignment",
    "boundary_quality",
    "face_preservation",
    "overall_realism",
    "result_image",
    "failure_reason",
]


with open(
    CSV_PATH,
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# =========================================================
# Summary
# =========================================================

total_tests = len(rows)

successful_tests = sum(
    1
    for row in rows
    if row["success"]
)

failed_tests = (
    total_tests
    - successful_tests
)

failure_rate = round(
    (
        failed_tests
        / total_tests
        * 100
    )
    if total_tests
    else 0,
    2,
)


successful_times = [
    row["processing_time"]
    for row in rows
    if row["success"]
]


print("\n" + "=" * 60)

print("VTON EVALUATION SUMMARY")

print("=" * 60)

print(
    "Total combinations:",
    total_tests,
)

print(
    "Successful:",
    successful_tests,
)

print(
    "Failed:",
    failed_tests,
)

print(
    "Failure rate:",
    f"{failure_rate}%",
)


if successful_times:

    average_time = round(
        sum(successful_times)
        / len(successful_times),
        4,
    )

    print(
        "Average processing time:",
        average_time,
        "seconds",
    )


print(
    "\nEvaluation CSV saved to:",
    CSV_PATH,
)

print(
    "Results saved to:",
    RESULT_DIR,
)

print("=" * 60)