import json
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from validation.asset_validator import validate_asset


# ---------------------------------------------------------
# Asset source and license registry
# ---------------------------------------------------------

ASSET_SOURCES = {
    "RAR-3D-001": {
        "name": "Avocado",
        "source": "Khronos glTF Sample Assets",
        "license": "CC0",
        "usage_permission": (
            "Permitted under source license"
        ),
    },

    "RAR-3D-002": {
        "name": "Water Bottle",
        "source": "Khronos glTF Sample Assets",
        "license": "CC0",
        "usage_permission": (
            "Permitted under source license"
        ),
    },

    "RAR-3D-003": {
        "name": "Boom Box",
        "source": "Khronos glTF Sample Assets",
        "license": "CC0",
        "usage_permission": (
            "Permitted under source license"
        ),
    },

    "RAR-3D-004": {
        "name": "Corset",
        "source": "Khronos glTF Sample Assets",
        "license": "CC0",
        "usage_permission": (
            "Permitted under source license"
        ),
    },

    "RAR-3D-005": {
        "name": "Barramundi Fish",
        "source": "Khronos glTF Sample Assets",
        "license": "CC0",
        "usage_permission": (
            "Permitted under source license"
        ),
    },

    "RAR-3D-006": {
        "name": "Damaged Helmet",
        "source": "Khronos glTF Sample Assets",
        "license": (
            "Mixed attribution/restriction - "
            "manual license review required"
        ),
        "usage_permission": (
            "Evaluation/testing only until "
            "license is manually verified"
        ),
    },
}


# ---------------------------------------------------------
# Asset ID extraction
# ---------------------------------------------------------

def extract_asset_id(filename):
    """
    Extract:

    RAR-3D-001

    from filenames such as:

    RAR-3D-001-avocado-original.glb
    RAR-3D-001-avocado-optimized.glb
    """

    parts = filename.split("-")

    if len(parts) >= 3:
        return "-".join(
            parts[:3]
        )

    return "UNKNOWN"


# ---------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------

def generate_metadata(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Asset not found: {path}"
        )

    validation = validate_asset(
        str(path)
    )

    asset_id = extract_asset_id(
        path.name
    )

    source_info = ASSET_SOURCES.get(
        asset_id,
        {
            "name": path.stem,
            "source": "Unknown",
            "license": "Unknown",
            "usage_permission": (
                "Requires manual verification"
            ),
        },
    )

    metadata = {

        # ---------------------------------------------
        # Asset identification
        # ---------------------------------------------

        "asset_id": asset_id,

        "product_id": (
            asset_id.replace(
                "RAR-3D",
                "PROD",
            )
        ),

        "product_name": (
            source_info["name"]
        ),

        # ---------------------------------------------
        # Asset properties
        # ---------------------------------------------

        "asset_type": "3d_asset",

        "format": (
            validation["format"]
        ),

        "mesh_count": (
            validation["mesh_count"]
        ),

        "vertex_count": (
            validation["vertex_count"]
        ),

        "polygon_count": (
            validation["polygon_count"]
        ),

        "material_count": (
            validation["material_count"]
        ),

        "texture_count": (
            validation["texture_count"]
        ),

        "texture_resolutions": (
            validation[
                "texture_resolutions"
            ]
        ),

        "dimensions": (
            validation["dimensions"]
        ),

        "orientation": (
            validation["orientation"]
        ),

        "file_size_mb": (
            validation["file_size_mb"]
        ),

        # ---------------------------------------------
        # Review state
        # ---------------------------------------------

        "status": (
            "PENDING_REVIEW"
            if validation["valid"]
            else "REJECTED"
        ),

        # ---------------------------------------------
        # Source / licensing
        # ---------------------------------------------

        "source": (
            source_info["source"]
        ),

        "license": (
            source_info["license"]
        ),

        "usage_permission": (
            source_info[
                "usage_permission"
            ]
        ),

        # ---------------------------------------------
        # Generation information
        # ---------------------------------------------

        "model": "N/A",
        "model_version": "N/A",

        # ---------------------------------------------
        # Validation information
        # ---------------------------------------------

        "validation_passed": (
            validation["valid"]
        ),

        "warnings": (
            validation["warnings"]
        ),

        "errors": (
            validation["errors"]
        ),

        # ---------------------------------------------
        # Processing timestamp
        # ---------------------------------------------

        "processed_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
    }

    return metadata


# ---------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------

def save_metadata(metadata):

    output_path = (
        PROJECT_ROOT
        / "metadata"
        / "assets.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():

        try:

            with open(
                output_path,
                "r",
                encoding="utf-8",
            ) as file:

                existing = json.load(
                    file
                )

            if not isinstance(
                existing,
                list,
            ):
                existing = []

        except Exception:
            existing = []

    else:
        existing = []

    # Replace previous metadata for
    # the same asset ID.
    existing = [
        item
        for item in existing
        if item.get("asset_id")
        != metadata["asset_id"]
    ]

    existing.append(
        metadata
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            existing,
            file,
            indent=4,
        )

    return str(output_path)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python "
            "raritone-3d/metadata/"
            "asset_metadata.py "
            "<asset.glb|asset.gltf>"
        )

        return

    try:

        metadata = generate_metadata(
            sys.argv[1]
        )

        metadata_file = (
            save_metadata(
                metadata
            )
        )

        result = {
            "success": True,
            "metadata_file": (
                metadata_file
            ),
            "metadata": metadata,
        }

        print(
            json.dumps(
                result,
                indent=4,
            )
        )

    except Exception as exc:

        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(exc),
                },
                indent=4,
            )
        )

        sys.exit(1)


if __name__ == "__main__":
    main()