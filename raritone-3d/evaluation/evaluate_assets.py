import csv
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pipeline import process_asset


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

INPUT_DIR = PROJECT_ROOT / "input"

OUTPUT_CSV = (
    PROJECT_ROOT
    / "evaluation"
    / "3d_asset_evaluation.csv"
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_assets():

    assets = sorted(
        list(INPUT_DIR.glob("*.glb"))
        + list(INPUT_DIR.glob("*.gltf"))
    )

    if not assets:
        print(
            "No GLB/GLTF assets found "
            "in the input directory."
        )
        return

    rows = []

    print(
        f"\nFound {len(assets)} assets.\n"
    )

    for index, asset_path in enumerate(
        assets,
        start=1,
    ):

        print(
            f"[{index}/{len(assets)}] "
            f"Processing {asset_path.name}..."
        )

        result = process_asset(
            str(asset_path)
        )

        # ---------------------------------------------
        # Failed processing
        # ---------------------------------------------

        if not result.get("success"):

            rows.append({
                "asset_id": "",
                "product_name": asset_path.stem,
                "original_file": asset_path.name,
                "optimized_file": "",
                "validation_passed": False,
                "original_polygons": "",
                "optimized_polygons": "",
                "polygon_reduction_percent": "",
                "original_size_mb": "",
                "optimized_size_mb": "",
                "size_reduction_percent": "",
                "original_load_time_seconds": "",
                "optimized_load_time_seconds": "",
                "load_time_improvement_percent": "",
                "material_count": "",
                "texture_count": "",
                "texture_resolution": "",
                "materials_preserved": "",
                "textures_preserved": "",
                "processing_time_seconds": "",
                "preview_generated": False,
                "license": "",
                "status": result.get(
                    "status",
                    "FAILED",
                ),
                "visual_quality_score_5": "",
                "optimization_score_5": "",
                "overall_quality_score_5": "",
                "review_notes": result.get(
                    "error",
                    "Processing failed.",
                ),
            })

            print("    FAILED")
            continue

        # ---------------------------------------------
        # Successful processing
        # ---------------------------------------------

        metadata = result["metadata"]

        optimization_result = (
            result["optimization"]
        )

        comparison = (
            optimization_result[
                "optimization"
            ]
        )

        before = (
            optimization_result["before"]
        )

        after = (
            optimization_result["after"]
        )

        preview = result.get(
            "preview",
            {}
        )

        texture_resolutions = (
            after.get(
                "texture_resolutions",
                [],
            )
        )

        rows.append({
            "asset_id": metadata["asset_id"],

            "product_name": (
                metadata["product_name"]
            ),

            "original_file": (
                asset_path.name
            ),

            "optimized_file": (
                result["optimized_asset"]
            ),

            "validation_passed": (
                metadata[
                    "validation_passed"
                ]
            ),

            "original_polygons": (
                before["polygon_count"]
            ),

            "optimized_polygons": (
                after["polygon_count"]
            ),

            "polygon_reduction_percent": (
                comparison[
                    "polygon_reduction_percent"
                ]
            ),

            "original_size_mb": (
                before["file_size_mb"]
            ),

            "optimized_size_mb": (
                after["file_size_mb"]
            ),

            "size_reduction_percent": (
                comparison[
                    "size_reduction_percent"
                ]
            ),

            # -----------------------------------------
            # Loading-time comparison
            # -----------------------------------------

            "original_load_time_seconds": (
                comparison[
                    "original_load_time_seconds"
                ]
            ),

            "optimized_load_time_seconds": (
                comparison[
                    "optimized_load_time_seconds"
                ]
            ),

            "load_time_improvement_percent": (
                comparison[
                    "load_time_improvement_percent"
                ]
            ),

            "material_count": (
                after["material_count"]
            ),

            "texture_count": (
                after["texture_count"]
            ),

            "texture_resolution": (
                "; ".join(
                    texture_resolutions
                )
            ),

            "materials_preserved": (
                comparison[
                    "materials_preserved"
                ]
            ),

            "textures_preserved": (
                comparison[
                    "textures_preserved"
                ]
            ),

            "processing_time_seconds": (
                optimization_result[
                    "processing_time"
                ]
            ),

            "preview_generated": (
                preview.get(
                    "success",
                    False,
                )
            ),

            "license": (
                metadata["license"]
            ),

            "status": (
                result["status"]
            ),

            # Manual review fields
            "visual_quality_score_5": "",
            "optimization_score_5": "",
            "overall_quality_score_5": "",

            "review_notes": "",
        })

        print(
            "    PASS | "
            f"{before['polygon_count']} -> "
            f"{after['polygon_count']} polygons | "
            f"{before['file_size_mb']} -> "
            f"{after['file_size_mb']} MB | "
            f"load "
            f"{comparison['original_load_time_seconds']}s -> "
            f"{comparison['optimized_load_time_seconds']}s"
        )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "asset_id",
        "product_name",
        "original_file",
        "optimized_file",
        "validation_passed",
        "original_polygons",
        "optimized_polygons",
        "polygon_reduction_percent",
        "original_size_mb",
        "optimized_size_mb",
        "size_reduction_percent",

        # Loading-time metrics
        "original_load_time_seconds",
        "optimized_load_time_seconds",
        "load_time_improvement_percent",

        "material_count",
        "texture_count",
        "texture_resolution",
        "materials_preserved",
        "textures_preserved",
        "processing_time_seconds",
        "preview_generated",
        "license",
        "status",
        "visual_quality_score_5",
        "optimization_score_5",
        "overall_quality_score_5",
        "review_notes",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\nEvaluation complete."
    )

    print(
        f"CSV saved to:\n{OUTPUT_CSV}"
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate_assets()