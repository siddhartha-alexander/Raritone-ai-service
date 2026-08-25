import csv
import os
import statistics
import time

import cv2
import numpy as np

from app.garment_processor import prepare_garment_v2
from app.inference import run_tryon
from app.person_processor import prepare_person_image
from app.quality_gate import QualityGateError, run_quality_gate


PERSON_PATH = "vton_dataset/persons/person.jpg"
GARMENT_PATH = "vton_dataset/garments/garment.jpg"

NUMBER_OF_REQUESTS = 10

OUTPUT_CSV = "performance_v2.csv"


person_image = cv2.imread(PERSON_PATH)
garment_image = cv2.imread(GARMENT_PATH)

if person_image is None:
    raise ValueError(
        f"Person image could not be loaded: {PERSON_PATH}"
    )

if garment_image is None:
    raise ValueError(
        f"Garment image could not be loaded: {GARMENT_PATH}"
    )


results = []
latencies = []
failures = 0


print("\n" + "=" * 65)
print("VTON PERFORMANCE TEST V2")
print("=" * 65)


for request_number in range(
    1,
    NUMBER_OF_REQUESTS + 1,
):

    print(
        f"\nRunning request "
        f"{request_number}/{NUMBER_OF_REQUESTS}..."
    )

    total_start = time.perf_counter()

    try:
        # -----------------------------------------------
        # Person preprocessing
        # Includes preprocessing + segmentation + pose
        # -----------------------------------------------

        person_start = time.perf_counter()

        person_result = prepare_person_image(
            person_image.copy()
        )

        person_time = (
            time.perf_counter()
            - person_start
        )

        # -----------------------------------------------
        # Garment preprocessing
        # -----------------------------------------------

        garment_start = time.perf_counter()

        garment_result = prepare_garment_v2(
            garment_image.copy()
        )

        garment_time = (
            time.perf_counter()
            - garment_start
        )

        # -----------------------------------------------
        # Quality gate
        # -----------------------------------------------

        gate_start = time.perf_counter()

        quality_result = run_quality_gate(
            person_result,
            garment_result,
        )

        quality_gate_time = (
            time.perf_counter()
            - gate_start
        )

        # -----------------------------------------------
        # Inference
        # -----------------------------------------------

        inference_start = time.perf_counter()

        _, inference_metadata = run_tryon(
            person_result["person_input"],
            garment_result["garment_input"],
            person_result["pose_data"],
            person_result["person_mask"],
        )

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        # -----------------------------------------------
        # Total latency
        # -----------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        )

        latencies.append(total_time)

        quality = quality_result["quality"]

        results.append(
            {
                "request": request_number,
                "success": True,
                "person_processing_time": round(
                    person_time,
                    4,
                ),
                "garment_processing_time": round(
                    garment_time,
                    4,
                ),
                "quality_gate_time": round(
                    quality_gate_time,
                    4,
                ),
                "inference_time": round(
                    inference_time,
                    4,
                ),
                "total_time": round(
                    total_time,
                    4,
                ),
                "pose_quality": quality["pose"],
                "mask_quality": quality["mask"],
                "garment_quality": quality["garment"],
                "failure_reason": "",
            }
        )

        print(
            f"Success | "
            f"Person: {person_time:.4f}s | "
            f"Garment: {garment_time:.4f}s | "
            f"Inference: {inference_time:.4f}s | "
            f"Total: {total_time:.4f}s"
        )

    except QualityGateError as exc:

        failures += 1

        total_time = (
            time.perf_counter()
            - total_start
        )

        results.append(
            {
                "request": request_number,
                "success": False,
                "person_processing_time": "",
                "garment_processing_time": "",
                "quality_gate_time": "",
                "inference_time": "",
                "total_time": round(
                    total_time,
                    4,
                ),
                "pose_quality": "",
                "mask_quality": "",
                "garment_quality": "",
                "failure_reason": (
                    f"{exc.error_code}: "
                    f"{exc.message}"
                ),
            }
        )

        print(
            f"Rejected by quality gate: "
            f"{exc.error_code}"
        )

    except Exception as exc:

        failures += 1

        total_time = (
            time.perf_counter()
            - total_start
        )

        results.append(
            {
                "request": request_number,
                "success": False,
                "person_processing_time": "",
                "garment_processing_time": "",
                "quality_gate_time": "",
                "inference_time": "",
                "total_time": round(
                    total_time,
                    4,
                ),
                "pose_quality": "",
                "mask_quality": "",
                "garment_quality": "",
                "failure_reason": str(exc),
            }
        )

        print(
            f"Request failed: {exc}"
        )


# =========================================================
# Statistics
# =========================================================

successful_requests = (
    NUMBER_OF_REQUESTS - failures
)

failure_rate = (
    failures
    / NUMBER_OF_REQUESTS
    * 100
)


if latencies:

    average_latency = statistics.mean(
        latencies
    )

    minimum_latency = min(
        latencies
    )

    maximum_latency = max(
        latencies
    )

    # NumPy percentile gives us P95 latency.
    p95_latency = float(
        np.percentile(
            latencies,
            95,
        )
    )

else:

    average_latency = 0.0
    minimum_latency = 0.0
    maximum_latency = 0.0
    p95_latency = 0.0


successful_results = [
    row
    for row in results
    if row["success"]
]


if successful_results:

    average_person_time = statistics.mean(
        row["person_processing_time"]
        for row in successful_results
    )

    average_garment_time = statistics.mean(
        row["garment_processing_time"]
        for row in successful_results
    )

    average_inference_time = statistics.mean(
        row["inference_time"]
        for row in successful_results
    )

else:

    average_person_time = 0.0
    average_garment_time = 0.0
    average_inference_time = 0.0


# =========================================================
# Save CSV
# =========================================================

fieldnames = [
    "request",
    "success",
    "person_processing_time",
    "garment_processing_time",
    "quality_gate_time",
    "inference_time",
    "total_time",
    "pose_quality",
    "mask_quality",
    "garment_quality",
    "failure_reason",
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


# =========================================================
# Final Report
# =========================================================

print("\n" + "=" * 65)
print("PERFORMANCE TEST RESULTS")
print("=" * 65)

print(
    f"Total requests: {NUMBER_OF_REQUESTS}"
)

print(
    f"Successful requests: {successful_requests}"
)

print(
    f"Failed requests: {failures}"
)

print(
    f"Failure rate: {failure_rate:.2f}%"
)

print("-" * 65)

print(
    f"Average person processing: "
    f"{average_person_time:.4f}s"
)

print(
    f"Average garment processing: "
    f"{average_garment_time:.4f}s"
)

print(
    f"Average inference time: "
    f"{average_inference_time:.4f}s"
)

print("-" * 65)

print(
    f"Average total latency: "
    f"{average_latency:.4f}s"
)

print(
    f"P95 latency: "
    f"{p95_latency:.4f}s"
)

print(
    f"Minimum latency: "
    f"{minimum_latency:.4f}s"
)

print(
    f"Maximum latency: "
    f"{maximum_latency:.4f}s"
)

print("-" * 65)

if (
    average_inference_time
    >= average_person_time
    and average_inference_time
    >= average_garment_time
):
    bottleneck = "VTON inference"

elif (
    average_garment_time
    >= average_person_time
):
    bottleneck = "Garment preprocessing"

else:
    bottleneck = "Person preprocessing"


print(
    f"Primary bottleneck: {bottleneck}"
)

print(
    f"Results saved to: {OUTPUT_CSV}"
)

print("=" * 65)