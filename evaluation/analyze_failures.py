import csv
from pathlib import Path


RESULTS_PATH = Path("evaluation/evaluation_results.csv")
OUTPUT_PATH = Path("evaluation/failure_analysis.csv")


FAILURE_MAP = {
    "PERSON_NOT_DETECTED": {
        "failure_type": "Person not detected",
        "root_cause": "No reliable person detection in the input image.",
        "severity": "High",
        "possible_fix": "Use a clearer image containing exactly one visible person.",
    },
    "POSE_NOT_DETECTED": {
        "failure_type": "Poor pose",
        "root_cause": "Pose landmarks could not be detected reliably.",
        "severity": "High",
        "possible_fix": "Use a clear front-facing person image and improve pose validation.",
    },
    "LOW_POSE_QUALITY": {
        "failure_type": "Poor pose",
        "root_cause": "Landmark visibility/confidence was below threshold.",
        "severity": "Medium",
        "possible_fix": "Improve pose thresholding or reject unclear poses earlier.",
    },
    "LOW_MASK_QUALITY": {
        "failure_type": "Poor segmentation",
        "root_cause": "Person segmentation mask quality was below threshold.",
        "severity": "Medium",
        "possible_fix": "Improve segmentation cleanup and input-background handling.",
    },
    "LOW_GARMENT_QUALITY": {
        "failure_type": "Garment mask failure",
        "root_cause": "Garment foreground separation was insufficient.",
        "severity": "Medium",
        "possible_fix": "Improve garment cropping/background handling.",
    },
    "PARTIAL_BODY": {
        "failure_type": "Partial body / alignment risk",
        "root_cause": (
            "The current body-frame validation requires lower-body "
            "landmarks that are not sufficiently visible in the image."
        ),
        "severity": "Medium",
        "possible_fix": (
            "Use category-aware landmark validation and allow "
            "upper-body try-on when shoulders and hips are valid."
        ),
    },
}


with open(
    RESULTS_PATH,
    "r",
    encoding="utf-8",
    newline="",
) as file:
    rows = list(csv.DictReader(file))


failures = []


for row in rows:

    if row["success"].lower() == "true":
        continue

    raw_failure = row.get(
        "failure_type",
        "",
    )

    mapped = FAILURE_MAP.get(
        raw_failure,
        {
            "failure_type": raw_failure or "Unknown failure",
            "root_cause": row.get(
                "failure_reason",
                "Unknown root cause",
            ),
            "severity": "Medium",
            "possible_fix": "Inspect the failed input and pipeline stage.",
        },
    )

    failures.append(
        {
            "test_id": row["test_id"],
            "failure_type": mapped["failure_type"],
            "root_cause": mapped["root_cause"],
            "severity": mapped["severity"],
            "possible_fix": mapped["possible_fix"],
        }
    )


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline="",
) as file:

    fieldnames = [
        "test_id",
        "failure_type",
        "root_cause",
        "severity",
        "possible_fix",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(failures)


print("=" * 60)
print("FAILURE ANALYSIS COMPLETED")
print("=" * 60)
print(f"Failures recorded: {len(failures)}")
print(f"Saved to: {OUTPUT_PATH}")
print("=" * 60)