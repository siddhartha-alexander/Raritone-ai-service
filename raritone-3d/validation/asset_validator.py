import os
import sys
import json
from pathlib import Path

import trimesh


def validate_asset(file_path: str):
    result = {
        "valid": False,
        "file": file_path,
        "format": None,

        "mesh": False,
        "mesh_count": 0,
        "vertex_count": 0,
        "polygon_count": 0,

        "materials": False,
        "material_count": 0,

        "textures": False,
        "texture_count": 0,
        "texture_resolutions": [],

        "bounding_box": None,
        "dimensions": None,
        "scene_scale": None,
        "orientation": "Y-up (glTF convention)",

        "file_size_mb": 0,

        "warnings": [],
        "errors": []
    }

    path = Path(file_path)

    # --------------------------------------------------
    # FILE VALIDATION
    # --------------------------------------------------
    if not path.exists():
        result["errors"].append("File does not exist.")
        return result

    if not path.is_file():
        result["errors"].append("Path is not a file.")
        return result

    extension = path.suffix.lower()

    if extension not in [".glb", ".gltf"]:
        result["errors"].append(
            "Unsupported format. Expected GLB or GLTF."
        )
        return result

    result["format"] = extension.replace(".", "")
    result["file_size_mb"] = round(
        path.stat().st_size / (1024 * 1024),
        4
    )

    # --------------------------------------------------
    # LOAD / CORRUPTION CHECK
    # --------------------------------------------------
    try:
        loaded = trimesh.load(
            str(path),
            force="scene",
            process=False
        )
    except Exception as e:
        result["errors"].append(
            f"Unable to load asset. File may be corrupted: {str(e)}"
        )
        return result

    if loaded is None:
        result["errors"].append(
            "Unable to load asset."
        )
        return result

    # Convert single mesh into scene if necessary
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    else:
        scene = loaded

    geometries = list(scene.geometry.values())

    if len(geometries) == 0:
        result["errors"].append(
            "No mesh geometry found."
        )
        return result

    # --------------------------------------------------
    # MESH INFORMATION
    # --------------------------------------------------
    mesh_count = 0
    vertex_count = 0
    polygon_count = 0

    material_ids = set()
    texture_ids = set()
    texture_resolutions = set()

    for mesh in geometries:

        if not isinstance(mesh, trimesh.Trimesh):
            continue

        mesh_count += 1

        vertices = len(mesh.vertices)
        faces = len(mesh.faces)

        vertex_count += vertices
        polygon_count += faces

        if vertices == 0:
            result["warnings"].append(
                "A mesh contains no vertices."
            )

        if faces == 0:
            result["warnings"].append(
                "A mesh contains no faces."
            )

        # --------------------------------------------------
        # MATERIAL INFORMATION
        # --------------------------------------------------
        try:
            visual = mesh.visual
            material = getattr(visual, "material", None)

            if material is not None:
                material_ids.add(id(material))

                # ------------------------------------------
                # TEXTURE INFORMATION
                # ------------------------------------------
                image = getattr(material, "image", None)

                if image is not None:
                    texture_ids.add(id(image))

                    try:
                        width, height = image.size

                        texture_resolutions.add(
                            f"{width}x{height}"
                        )

                    except Exception:
                        pass

                # Some PBR materials may expose texture
                # information through other attributes
                for attr in [
                    "baseColorTexture",
                    "normalTexture",
                    "emissiveTexture",
                    "metallicRoughnessTexture",
                    "occlusionTexture"
                ]:
                    texture = getattr(material, attr, None)

                    if texture is not None:
                        texture_ids.add(id(texture))

                        try:
                            if hasattr(texture, "size"):
                                width, height = texture.size
                                texture_resolutions.add(
                                    f"{width}x{height}"
                                )
                        except Exception:
                            pass

        except Exception as e:
            result["warnings"].append(
                f"Could not fully inspect material: {str(e)}"
            )

    result["mesh_count"] = mesh_count
    result["vertex_count"] = int(vertex_count)
    result["polygon_count"] = int(polygon_count)

    result["mesh"] = (
        mesh_count > 0
        and vertex_count > 0
        and polygon_count > 0
    )

    # --------------------------------------------------
    # MATERIAL RESULTS
    # --------------------------------------------------
    result["material_count"] = len(material_ids)
    result["materials"] = len(material_ids) > 0

    if not result["materials"]:
        result["warnings"].append(
            "No material detected."
        )

    # --------------------------------------------------
    # TEXTURE RESULTS
    # --------------------------------------------------
    result["texture_count"] = len(texture_ids)
    result["textures"] = len(texture_ids) > 0

    result["texture_resolutions"] = sorted(
        list(texture_resolutions)
    )

    if not result["textures"]:
        result["warnings"].append(
            "No linked texture detected."
        )

    # --------------------------------------------------
    # BOUNDING BOX / DIMENSIONS / SCALE
    # --------------------------------------------------
    try:
        bounds = scene.bounds

        if bounds is not None:

            minimum = bounds[0]
            maximum = bounds[1]

            dimensions = maximum - minimum

            result["bounding_box"] = {
                "min": [
                    round(float(x), 6)
                    for x in minimum
                ],
                "max": [
                    round(float(x), 6)
                    for x in maximum
                ]
            }

            result["dimensions"] = {
                "x": round(float(dimensions[0]), 6),
                "y": round(float(dimensions[1]), 6),
                "z": round(float(dimensions[2]), 6)
            }

    except Exception as e:
        result["warnings"].append(
            f"Could not calculate bounding box: {str(e)}"
        )

    try:
        result["scene_scale"] = round(
            float(scene.scale),
            6
        )
    except Exception:
        result["warnings"].append(
            "Could not determine scene scale."
        )

    # --------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------
    if not result["mesh"]:
        result["errors"].append(
            "Asset does not contain valid mesh geometry."
        )

    if result["polygon_count"] <= 0:
        result["errors"].append(
            "Asset contains no polygons."
        )

    result["valid"] = len(result["errors"]) == 0

    return result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python asset_validator.py <asset.glb|asset.gltf>"
        )
        sys.exit(1)

    asset_path = sys.argv[1]

    validation_result = validate_asset(asset_path)

    print(
        json.dumps(
            validation_result,
            indent=4
        )
    )