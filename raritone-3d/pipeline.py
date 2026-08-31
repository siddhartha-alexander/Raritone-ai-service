import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from validation.asset_validator import validate_asset
from processing.optimize_asset import optimize_asset
from generation.preview_generator import generate_preview
from metadata.asset_metadata import generate_metadata


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "processed"
REJECTED_DIR = PROJECT_ROOT / "rejected"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REJECTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_optimized_filename(input_path):
    """
    Convert:

    RAR-3D-001-avocado-original.glb

    into:

    RAR-3D-001-avocado-optimized.glb
    """

    input_path = Path(input_path)

    stem = input_path.stem

    if stem.endswith("-original"):
        stem = stem[:-len("-original")]

    return (
        f"{stem}-optimized"
        f"{input_path.suffix.lower()}"
    )


def get_preview_filename(input_path):
    """
    Convert:

    RAR-3D-001-avocado-original.glb

    into:

    RAR-3D-001-avocado-preview.png
    """

    input_path = Path(input_path)

    stem = input_path.stem

    if stem.endswith("-original"):
        stem = stem[:-len("-original")]

    if stem.endswith("-optimized"):
        stem = stem[:-len("-optimized")]

    return f"{stem}-preview.png"


# ---------------------------------------------------------
# Main 3D asset pipeline
# ---------------------------------------------------------

def process_asset(file_path):
    """
    Complete CPU-based 3D asset processing pipeline.

    Flow:

    Existing GLB/GLTF
        ↓
    Validate Original Asset
        ↓
    Optimize Asset
        ↓
    Validate Optimized Asset
        ↓
    Generate Preview
        ↓
    Generate Metadata
        ↓
    PENDING_REVIEW
    """

    input_path = Path(file_path)

    # -----------------------------------------------------
    # 1. Check input
    # -----------------------------------------------------

    if not input_path.exists():

        return {
            "success": False,
            "status": "REJECTED",
            "error": (
                "Input asset does not exist."
            ),
        }

    if input_path.suffix.lower() not in {
        ".glb",
        ".gltf",
    }:

        return {
            "success": False,
            "status": "REJECTED",
            "asset": input_path.name,
            "error": (
                "Only GLB and GLTF assets "
                "are supported."
            ),
        }

    # -----------------------------------------------------
    # 2. Validate original asset
    # -----------------------------------------------------

    try:

        original_validation = (
            validate_asset(
                str(input_path)
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": "REJECTED",
            "asset": input_path.name,
            "error": (
                "Original asset validation "
                f"failed: {str(exc)}"
            ),
        }

    # -----------------------------------------------------
    # 3. Reject invalid original asset
    # -----------------------------------------------------

    if not original_validation["valid"]:

        rejected_path = (
            REJECTED_DIR
            / input_path.name
        )

        try:

            if (
                input_path.resolve()
                != rejected_path.resolve()
            ):

                shutil.copy2(
                    input_path,
                    rejected_path,
                )

        except Exception as exc:

            return {
                "success": False,
                "status": "REJECTED",
                "asset": input_path.name,
                "validation": (
                    original_validation
                ),
                "error": (
                    "Asset failed validation "
                    "and could not be copied "
                    "to the rejected folder: "
                    f"{str(exc)}"
                ),
            }

        return {
            "success": False,
            "status": "REJECTED",
            "asset": input_path.name,
            "validation": (
                original_validation
            ),
            "rejected_path": str(
                rejected_path
            ),
        }

    # -----------------------------------------------------
    # 4. Create optimized output path
    # -----------------------------------------------------

    optimized_filename = (
        get_optimized_filename(
            input_path
        )
    )

    optimized_path = (
        PROCESSED_DIR
        / optimized_filename
    )

    # -----------------------------------------------------
    # 5. Optimize asset
    # -----------------------------------------------------

    try:

        optimization = (
            optimize_asset(
                str(input_path),
                str(optimized_path),
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": "PROCESSING_FAILED",
            "asset": input_path.name,
            "validation": (
                original_validation
            ),
            "error": (
                "Asset optimization failed: "
                f"{str(exc)}"
            ),
        }

    # -----------------------------------------------------
    # 6. Validate optimized asset
    # -----------------------------------------------------

    try:

        optimized_validation = (
            validate_asset(
                str(optimized_path)
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": "PROCESSING_FAILED",
            "asset": input_path.name,
            "optimized_asset": (
                optimized_filename
            ),
            "validation": (
                original_validation
            ),
            "error": (
                "Optimized asset validation "
                f"failed: {str(exc)}"
            ),
        }

    if not optimized_validation["valid"]:

        return {
            "success": False,
            "status": "PROCESSING_FAILED",
            "asset": input_path.name,
            "optimized_asset": (
                optimized_filename
            ),
            "validation": (
                original_validation
            ),
            "optimized_validation": (
                optimized_validation
            ),
            "error": (
                "Optimized asset failed "
                "validation."
            ),
        }

    # -----------------------------------------------------
    # 7. Generate preview
    # -----------------------------------------------------

    preview_filename = (
        get_preview_filename(
            input_path
        )
    )

    preview_path = (
        PROCESSED_DIR
        / preview_filename
    )

    try:

        preview = (
            generate_preview(
                str(optimized_path),
                str(preview_path),
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": "PROCESSING_FAILED",
            "asset": input_path.name,
            "optimized_asset": (
                optimized_filename
            ),
            "processed_path": str(
                optimized_path
            ),
            "validation": (
                original_validation
            ),
            "optimization": (
                optimization
            ),
            "optimized_validation": (
                optimized_validation
            ),
            "error": (
                "Preview generation failed: "
                f"{str(exc)}"
            ),
        }

    # -----------------------------------------------------
    # 8. Generate metadata
    # -----------------------------------------------------

    try:

        metadata = (
            generate_metadata(
                str(optimized_path)
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": "PROCESSING_FAILED",
            "asset": input_path.name,
            "optimized_asset": (
                optimized_filename
            ),
            "preview": (
                preview
            ),
            "error": (
                "Metadata generation failed: "
                f"{str(exc)}"
            ),
        }

    # -----------------------------------------------------
    # 9. Add review information
    # -----------------------------------------------------

    metadata["status"] = (
        "PENDING_REVIEW"
    )

    metadata["preview"] = str(
        preview_path
    )

    # -----------------------------------------------------
    # 10. Final result
    # -----------------------------------------------------

    return {
        "success": True,

        "status": (
            "PENDING_REVIEW"
        ),

        "asset": (
            input_path.name
        ),

        "optimized_asset": (
            optimized_filename
        ),

        "processed_path": str(
            optimized_path
        ),

        "validation": (
            original_validation
        ),

        "optimization": (
            optimization
        ),

        "optimized_validation": (
            optimized_validation
        ),

        "preview": (
            preview
        ),

        "metadata": (
            metadata
        ),
    }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python raritone-3d/pipeline.py "
            "<input.glb|input.gltf>"
        )

        return

    result = process_asset(
        sys.argv[1]
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()