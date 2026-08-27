import json
import os
import sys
from pathlib import Path

import trimesh


MAX_POLYGON_COUNT = 200_000
MAX_FILE_SIZE_MB = 50.0


def validate_asset(asset_path):
    """
    Validate a GLB/GLTF candidate before human review.

    Checks:
    - File exists
    - Supported format
    - File size
    - Loadable GLB/GLTF
    - Mesh exists
    - Mesh is non-empty
    - Polygon count
    - Material presence
    - Texture/visual information
    """

    path = Path(asset_path)

    result = {
        "valid": False,
        "file": str(path),
        "format": None,
        "mesh": False,
        "texture": False,
        "materials": False,
        "polygon_count": 0,
        "vertex_count": 0,
        "mesh_count": 0,
        "file_size_mb": 0.0,
        "warnings": [],
        "errors": [],
    }

    # -----------------------------------------------------
    # File existence
    # -----------------------------------------------------

    if not path.exists():
        result["errors"].append(
            "3D asset file does not exist."
        )
        return result

    if not path.is_file():
        result["errors"].append(
            "Asset path is not a file."
        )
        return result

    # -----------------------------------------------------
    # Format
    # -----------------------------------------------------

    extension = path.suffix.lower()

    if extension not in {".glb", ".gltf"}:
        result["errors"].append(
            "Unsupported format. Expected GLB or GLTF."
        )
        return result

    result["format"] = extension.lstrip(".")

    # -----------------------------------------------------
    # File size
    # -----------------------------------------------------

    file_size_mb = (
        path.stat().st_size
        / (1024 * 1024)
    )

    result["file_size_mb"] = round(
        file_size_mb,
        4,
    )

    if file_size_mb <= 0:
        result["errors"].append(
            "Asset file is empty."
        )
        return result

    if file_size_mb > MAX_FILE_SIZE_MB:
        result["warnings"].append(
            (
                f"Large asset: {file_size_mb:.2f} MB. "
                "Optimization is recommended."
            )
        )

    # -----------------------------------------------------
    # Load asset
    # -----------------------------------------------------

    try:
        scene = trimesh.load(
            str(path),
            force="scene",
        )

    except Exception as exc:
        result["errors"].append(
            f"Unable to load 3D asset: {exc}"
        )
        return result

    # -----------------------------------------------------
    # Collect meshes
    # -----------------------------------------------------

    meshes = []

    for geometry in scene.geometry.values():

        if isinstance(
            geometry,
            trimesh.Trimesh,
        ):
            meshes.append(
                geometry
            )

    result["mesh_count"] = len(
        meshes
    )

    if not meshes:
        result["errors"].append(
            "No valid mesh found."
        )
        return result

    result["mesh"] = True

    # -----------------------------------------------------
    # Geometry checks
    # -----------------------------------------------------

    vertex_count = 0
    polygon_count = 0

    materials_found = False
    texture_found = False

    for mesh in meshes:

        vertex_count += len(
            mesh.vertices
        )

        polygon_count += len(
            mesh.faces
        )

        # Visual information
        visual = mesh.visual

        if visual is not None:

            material = getattr(
                visual,
                "material",
                None,
            )

            if material is not None:
                materials_found = True

                image = getattr(
                    material,
                    "image",
                    None,
                )

                if image is not None:
                    texture_found = True

            # Vertex/face colors still represent
            # valid visual surface information.
            kind = getattr(
                visual,
                "kind",
                None,
            )

            if kind in {
                "texture",
                "vertex",
                "face",
            }:
                texture_found = True

    result["vertex_count"] = (
        vertex_count
    )

    result["polygon_count"] = (
        polygon_count
    )

    result["materials"] = (
        materials_found
    )

    result["texture"] = (
        texture_found
    )

    # -----------------------------------------------------
    # Mesh validation
    # -----------------------------------------------------

    if vertex_count == 0:
        result["errors"].append(
            "Mesh contains no vertices."
        )

    if polygon_count == 0:
        result["errors"].append(
            "Mesh contains no polygon faces."
        )

    if (
        polygon_count
        > MAX_POLYGON_COUNT
    ):
        result["warnings"].append(
            (
                f"High polygon count: "
                f"{polygon_count}. "
                "Optimization recommended."
            )
        )

    # -----------------------------------------------------
    # Material/texture validation
    # -----------------------------------------------------

    if not materials_found:
        result["warnings"].append(
            "No material detected."
        )

    if not texture_found:
        result["warnings"].append(
            "No texture information detected."
        )

    # -----------------------------------------------------
    # Final validity
    # -----------------------------------------------------

    result["valid"] = (
        len(result["errors"]) == 0
    )

    return result


def main():
    if len(sys.argv) < 2:

        print(
            "Usage: python validation/asset_validator.py "
            "<asset.glb>"
        )

        return

    asset_path = sys.argv[1]

    result = validate_asset(
        asset_path
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()