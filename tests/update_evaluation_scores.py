import pandas as pd


CSV_PATH = "vton_evaluation.csv"


scores = {
    "pair_001": [3, 4, 3, 3, 5, 3],
    "pair_002": [4, 5, 4, 4, 5, 4],
    "pair_003": [3, 4, 3, 3, 5, 3],
    "pair_004": [5, 5, 4, 4, 5, 4],
    "pair_005": [3, 4, 3, 3, 5, 3],
    "pair_006": [4, 5, 4, 4, 5, 4],
}


columns = [
    "garment_alignment",
    "garment_preservation",
    "body_alignment",
    "boundary_quality",
    "face_preservation",
    "overall_realism",
]


df = pd.read_csv(CSV_PATH)


for pair_id, values in scores.items():

    for column, value in zip(
        columns,
        values,
    ):
        df.loc[
            df["pair_id"] == pair_id,
            column,
        ] = value


df.to_csv(
    CSV_PATH,
    index=False,
)


print("\nVTON QUALITY EVALUATION")
print("=" * 50)


for column in columns:

    average = df[column].mean()

    print(
        f"{column}: {average:.2f}/5"
    )


quality_average = (
    df[columns]
    .mean()
    .mean()
)


print("-" * 50)

print(
    f"Overall average quality: "
    f"{quality_average:.2f}/5"
)

print("=" * 50)

print(
    "\nScores saved to:",
    CSV_PATH,
)