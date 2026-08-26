import csv
import time
from pathlib import Path

import cv2
import numpy as np

from app.garment_processor import prepare_garment_v2
from app.inference import MODEL_VERSION, run_tryon
from app.person_processor import prepare_person_image
from app.quality_gate import QualityGateError, run_quality_gate



# Paths


BASE_DIR = Path("evaluation")

PERSON_DIR = BASE_DIR / "persons"
GARMENT_DIR = BASE_DIR / "garments"
GENERATED_DIR = BASE_DIR / "generated"

METADATA_PATH = BASE_DIR / "metadata.csv"
RESULTS_PATH = BASE_DIR / "evaluation_results.csv"

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



# Controlled Test Variants


def apply_variant(image, variant):
    """
    Apply a controlled transformation to an image for
    robustness evaluation.

    The original image is never modified.
    """

    result = image.copy()

    if variant == "original":
        return result

    if variant == "low_resolution":

        height, width = result.shape[:2]

        # Reduce resolution substantially and restore
        # original dimensions to simulate low-quality input.
        small_width = max(64, width // 4)
        small_height = max(64, height // 4)

        result = cv2.resize(
            result,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )

        result = cv2.resize(
            result,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return result

    if variant == "dark":

        result = cv2.convertScaleAbs(
            result,
            alpha=0.55,
            beta=0,
        )

        return result

    if variant == "bright":

        result = cv2.convertScaleAbs(
            result,
            alpha=1.25,
            beta=25,
        )

        return result

    raise ValueError(
        f"Unknown evaluation variant: {variant}"
    )



# Mask Coverage


def calculate_mask_coverage(mask):
    """
    Calculate the percentage of the image classified
    as person by the segmentation mask.
    """

    if mask is None or mask.size == 0:
        return 0.0

    foreground_pixels = np.count_nonzero(
        mask > 127
    )

    total_pixels = mask.size

    if total_pixels == 0:
        return 0.0

    coverage = (
        foreground_pixels / total_pixels
    )

    return round(
        float(coverage),
        4,
    )



# Alignment Indicator


def calculate_alignment_indicator(pose_data):
    """
    Basic automated alignment indicator based on the
    availability and visibility of upper-body landmarks.

    This is NOT a photorealism metric.
    """

    required = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]

    if not pose_data:
        return 0.0

    values = []

    for name in required:

        landmark = pose_data.get(name)

        if landmark is None:
            return 0.0

        visibility = landmark.get(
            "visibility",
            0.0,
        )

        values.append(
            float(visibility)
        )

    return round(
        sum(values) / len(values),
        4,
    )



# Load Metadata


if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata not found: {METADATA_PATH}"
    )


with open(
    METADATA_PATH,
    "r",
    newline="",
    encoding="utf-8",
) as file:

    reader = csv.DictReader(file)
    test_cases = list(reader)


if not test_cases:
    raise ValueError(
        "No evaluation test cases found."
    )



# Evaluation


results = []

successful = 0
failed = 0

processing_times = []


print("\n" + "=" * 70)
print("RARITONE VTON AUTOMATED EVALUATION")
print("=" * 70)

print(
    f"Tests scheduled: {len(test_cases)}"
)


for index, test in enumerate(
    test_cases,
    start=1,
):

    test_id = test["test_id"]

    person_name = test[
        "person_image"
    ]

    garment_name = test[
        "garment_image"
    ]

    category = test[
        "category"
    ]

    variant = test[
        "variant"
    ]

    print(
        f"\n[{index}/{len(test_cases)}] "
        f"{test_id} | "
        f"{person_name} + {garment_name} | "
        f"{variant}"
    )

    total_start = time.perf_counter()

    # Default values
    success = False
    failure_type = ""
    failure_reason = ""

    output_available = False
    output_path = ""

    pose_quality = 0.0
    mask_quality = 0.0
    garment_quality = 0.0

    mask_coverage = 0.0
    alignment_indicator = 0.0

    person_processing_time = 0.0
    garment_processing_time = 0.0
    inference_time = 0.0

    input_width = 0
    input_height = 0

    output_width = 0
    output_height = 0

    try:

        
        # Load Images
        

        person_path = (
            PERSON_DIR / person_name
        )

        garment_path = (
            GARMENT_DIR / garment_name
        )

        person_image = cv2.imread(
            str(person_path)
        )

        garment_image = cv2.imread(
            str(garment_path)
        )

        if person_image is None:
            raise ValueError(
                f"Person image could not be loaded: "
                f"{person_path}"
            )

        if garment_image is None:
            raise ValueError(
                f"Garment image could not be loaded: "
                f"{garment_path}"
            )

        
        # Apply Controlled Person Variant
        

        person_image = apply_variant(
            person_image,
            variant,
        )

        input_height, input_width = (
            person_image.shape[:2]
        )

        
        # Person Processing
        

        person_start = time.perf_counter()

        person_result = (
            prepare_person_image(
                person_image
            )
        )

        person_processing_time = round(
            time.perf_counter()
            - person_start,
            4,
        )

        
        # Garment Processing
        

        garment_start = time.perf_counter()

        garment_result = (
            prepare_garment_v2(
                garment_image
            )
        )

        garment_processing_time = round(
            time.perf_counter()
            - garment_start,
            4,
        )

        
        # Quality Gate
        

        gate_result = run_quality_gate(
            person_result,
            garment_result,
        )

        quality = gate_result[
            "quality"
        ]

        pose_quality = round(
            float(
                quality.get(
                    "pose",
                    0.0,
                )
            ),
            4,
        )

        mask_quality = round(
            float(
                quality.get(
                    "mask",
                    0.0,
                )
            ),
            4,
        )

        garment_quality = round(
            float(
                quality.get(
                    "garment",
                    0.0,
                )
            ),
            4,
        )

        
        # Automated Quality Indicators
        

        person_mask = person_result[
            "person_mask"
        ]

        pose_data = person_result[
            "pose_data"
        ]

        mask_coverage = (
            calculate_mask_coverage(
                person_mask
            )
        )

        alignment_indicator = (
            calculate_alignment_indicator(
                pose_data
            )
        )

        
        # VTON Inference
        

        inference_start = (
            time.perf_counter()
        )

        result_image, inference_metadata = (
            run_tryon(
                person_result[
                    "person_input"
                ],
                garment_result[
                    "garment_input"
                ],
                pose_data,
                person_mask,
            )
        )

        inference_time = round(
            time.perf_counter()
            - inference_start,
            4,
        )

        
        # Output Validation
        

        if (
            result_image is None
            or result_image.size == 0
        ):
            raise RuntimeError(
                "VTON returned an empty output."
            )

        output_height, output_width = (
            result_image.shape[:2]
        )

        generated_path = (
            GENERATED_DIR
            / f"{test_id}.png"
        )

        saved = cv2.imwrite(
            str(generated_path),
            result_image,
        )

        if not saved:
            raise RuntimeError(
                "Failed to save generated output."
            )

        output_available = True

        output_path = str(
            generated_path
        )

        success = True
        successful += 1

        print(
            "SUCCESS | "
            f"pose={pose_quality:.4f} | "
            f"mask={mask_quality:.4f} | "
            f"garment={garment_quality:.4f}"
        )

    except QualityGateError as exc:

        failed += 1

        failure_type = (
            exc.error_code
        )

        failure_reason = (
            exc.message
        )

        if exc.quality:

            pose_quality = float(
                exc.quality.get(
                    "pose",
                    0.0,
                )
            )

            mask_quality = float(
                exc.quality.get(
                    "mask",
                    0.0,
                )
            )

            garment_quality = float(
                exc.quality.get(
                    "garment",
                    0.0,
                )
            )

        print(
            "QUALITY GATE FAILURE | "
            f"{failure_type} | "
            f"{failure_reason}"
        )

    except Exception as exc:

        failed += 1

        failure_type = (
            type(exc).__name__
        )

        failure_reason = str(exc)

        print(
            f"FAILED | {failure_reason}"
        )

    
    # Timing
    

    total_time = round(
        time.perf_counter()
        - total_start,
        4,
    )

    processing_times.append(
        total_time
    )

    
    # Store Evaluation Result
    

    results.append(
        {
            "test_id": test_id,
            "person_image": person_name,
            "garment_image": garment_name,
            "category": category,
            "variant": variant,
            "expected_behavior": test[
                "expected_behavior"
            ],
            "success": success,
            "failure_type": failure_type,
            "failure_reason": failure_reason,
            "input_width": input_width,
            "input_height": input_height,
            "output_width": output_width,
            "output_height": output_height,
            "output_available": (
                output_available
            ),
            "mask_coverage": (
                mask_coverage
            ),
            "alignment_indicator": (
                alignment_indicator
            ),
            "pose_quality": pose_quality,
            "mask_quality": mask_quality,
            "garment_quality": (
                garment_quality
            ),
            "person_processing_time": (
                person_processing_time
            ),
            "garment_processing_time": (
                garment_processing_time
            ),
            "inference_time": (
                inference_time
            ),
            "total_processing_time": (
                total_time
            ),
            "model_version": MODEL_VERSION,
            "result_path": output_path,

            # Human evaluation fields.
            # Intentionally blank until reviewed.
            "visual_quality_score": "",
            "alignment_score": "",
            "garment_preservation_score": "",
            "artifact_score": "",
        }
    )



# Save Evaluation CSV


fieldnames = list(
    results[0].keys()
)


with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        results
    )



# Summary


total_tests = len(
    test_cases
)

failure_rate = (
    failed / total_tests
    if total_tests
    else 0.0
)

average_processing_time = (
    sum(processing_times)
    / len(processing_times)
    if processing_times
    else 0.0
)


successful_inference_times = [
    result["inference_time"]
    for result in results
    if result["success"]
]


average_inference_time = (
    sum(successful_inference_times)
    / len(successful_inference_times)
    if successful_inference_times
    else 0.0
)


print("\n" + "=" * 70)
print("AUTOMATED EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Model version: {MODEL_VERSION}"
)

print(
    f"Tests run: {total_tests}"
)

print(
    f"Successful: {successful}"
)

print(
    f"Failed: {failed}"
)

print(
    f"Failure rate: "
    f"{failure_rate * 100:.2f}%"
)

print(
    f"Average inference time: "
    f"{average_inference_time:.4f}s"
)

print(
    f"Average total processing time: "
    f"{average_processing_time:.4f}s"
)

print(
    f"Results CSV: {RESULTS_PATH}"
)

print(
    f"Generated outputs: {GENERATED_DIR}"
)

print("=" * 70)