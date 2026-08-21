import csv
import time
import cv2

from app.person_processor import prepare_tryon_input
from app.inference import run_tryon


PERSON_IMAGE_PATH = "test_data/persons/person.jpg"
GARMENT_IMAGE_PATH = "test_data/garments/tshirt.jpg"

TOTAL_REQUESTS = 10

results = []


for request_number in range(1, TOTAL_REQUESTS + 1):

    print(f"\nRunning request {request_number}/{TOTAL_REQUESTS}...")

    start_time = time.perf_counter()

    try:
        
        # Load images
        

        person_image = cv2.imread(
            PERSON_IMAGE_PATH
        )

        garment_image = cv2.imread(
            GARMENT_IMAGE_PATH
        )

        if person_image is None:
            raise ValueError(
                "Person image could not be loaded."
            )

        if garment_image is None:
            raise ValueError(
                "Garment image could not be loaded."
            )

        
        # Prepare inputs
        

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

        
        # Run try-on
        

        result_image, inference_metadata = run_tryon(
            person_input,
            garment_input,
            pose_data,
            person_mask,
        )

        processing_time = round(
            time.perf_counter() - start_time,
            4,
        )

        print(
            f"Request {request_number} successful "
            f"in {processing_time} seconds"
        )

        results.append({
            "request_number": request_number,
            "success": True,
            "processing_time": processing_time,
            "failure_reason": "",
        })

    except Exception as exc:

        processing_time = round(
            time.perf_counter() - start_time,
            4,
        )

        print(
            f"Request {request_number} failed: "
            f"{str(exc)}"
        )

        results.append({
            "request_number": request_number,
            "success": False,
            "processing_time": processing_time,
            "failure_reason": str(exc),
        })



# Save CSV


csv_path = "tryon_performance_results.csv"

with open(
    csv_path,
    mode="w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "request_number",
            "success",
            "processing_time",
            "failure_reason",
        ],
    )

    writer.writeheader()

    writer.writerows(results)



# Calculate statistics


successful_times = [
    result["processing_time"]
    for result in results
    if result["success"]
]

success_count = len(successful_times)

failure_count = TOTAL_REQUESTS - success_count

failure_rate = round(
    (failure_count / TOTAL_REQUESTS) * 100,
    2,
)


print("\n" + "=" * 50)
print("PERFORMANCE TEST RESULTS")
print("=" * 50)

print(f"Total requests: {TOTAL_REQUESTS}")
print(f"Successful requests: {success_count}")
print(f"Failed requests: {failure_count}")
print(f"Failure rate: {failure_rate}%")

if successful_times:

    average_time = round(
        sum(successful_times)
        / len(successful_times),
        4,
    )

    minimum_time = round(
        min(successful_times),
        4,
    )

    maximum_time = round(
        max(successful_times),
        4,
    )

    print(f"Average processing time: {average_time}s")
    print(f"Minimum processing time: {minimum_time}s")
    print(f"Maximum processing time: {maximum_time}s")

print("=" * 50)

print(
    f"\nResults saved to: {csv_path}"
)