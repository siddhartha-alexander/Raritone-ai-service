import csv
from pathlib import Path


CSV_PATH = Path("vton_evaluation.csv")


if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Evaluation file not found: {CSV_PATH}"
    )



# Load evaluation results


with open(
    CSV_PATH,
    "r",
    encoding="utf-8",
    newline="",
) as file:

    reader = csv.DictReader(file)
    rows = list(reader)


if not rows:
    raise ValueError(
        "Evaluation CSV contains no results."
    )



# Score columns


score_columns = [
    "garment_alignment",
    "garment_preservation",
    "body_alignment",
    "boundary_quality",
    "face_preservation",
    "overall_realism",
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0



# Calculate average quality for every result


evaluated = []

for row in rows:

    scores = [
        safe_float(row.get(column))
        for column in score_columns
    ]

    average_score = (
        sum(scores) / len(scores)
    )

    evaluated.append(
        {
            "row": row,
            "average_score": average_score,
            "scores": dict(
                zip(
                    score_columns,
                    scores,
                )
            ),
        }
    )


# Lowest-quality results first
evaluated.sort(
    key=lambda item: item["average_score"]
)



# Failure source analysis


def determine_failure_source(scores):

    causes = []

    if scores["garment_alignment"] < 3.5:
        causes.append(
            "Pose/alignment issue"
        )

    if scores["body_alignment"] < 3.5:
        causes.append(
            "Pose/body alignment issue"
        )

    if scores["boundary_quality"] < 3.5:
        causes.append(
            "Segmentation/compositing boundary issue"
        )

    if scores["garment_preservation"] < 3.5:
        causes.append(
            "Garment preprocessing/distortion issue"
        )

    if scores["face_preservation"] < 4.0:
        causes.append(
            "Person/face preservation issue"
        )

    if scores["overall_realism"] < 3.5:
        causes.append(
            "Baseline model limitation"
        )

    if not causes:
        causes.append(
            "Minor visual quality limitation"
        )

    return "; ".join(causes)



# Display five weakest results


top_failures = evaluated[:5]


print("\n" + "=" * 70)
print("TOP 5 VTON FAILURE / WEAK QUALITY CASES")
print("=" * 70)


for index, item in enumerate(
    top_failures,
    start=1,
):

    row = item["row"]
    scores = item["scores"]

    pair_id = (
        row.get("image_pair")
        or row.get("pair_id")
        or row.get("pair")
        or f"result_{index}"
    )

    cause = determine_failure_source(
        scores
    )

    print(f"\nFailure Case {index}")
    print("-" * 50)

    print(
        f"Image pair: {pair_id}"
    )

    print(
        f"Average quality: "
        f"{item['average_score']:.2f}/5"
    )

    print(
        f"Garment alignment: "
        f"{scores['garment_alignment']:.2f}/5"
    )

    print(
        f"Garment preservation: "
        f"{scores['garment_preservation']:.2f}/5"
    )

    print(
        f"Body alignment: "
        f"{scores['body_alignment']:.2f}/5"
    )

    print(
        f"Boundary quality: "
        f"{scores['boundary_quality']:.2f}/5"
    )

    print(
        f"Face preservation: "
        f"{scores['face_preservation']:.2f}/5"
    )

    print(
        f"Overall realism: "
        f"{scores['overall_realism']:.2f}/5"
    )

    print(
        f"Likely cause: {cause}"
    )



# Overall summary


overall_average = sum(
    item["average_score"]
    for item in evaluated
) / len(evaluated)


weak_results = [
    item
    for item in evaluated
    if item["average_score"] < 3.5
]


failure_rate = (
    len(weak_results)
    / len(evaluated)
    * 100
)


print("\n" + "=" * 70)
print("QUALITY ANALYSIS SUMMARY")
print("=" * 70)

print(
    f"Evaluated results: {len(evaluated)}"
)

print(
    f"Average quality score: "
    f"{overall_average:.2f}/5"
)

print(
    f"Weak results (<3.5): "
    f"{len(weak_results)}"
)

print(
    f"Quality failure rate: "
    f"{failure_rate:.2f}%"
)

print("=" * 70)