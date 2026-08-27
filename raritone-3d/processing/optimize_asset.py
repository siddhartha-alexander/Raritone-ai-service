import json
import sys
import time
from pathlib import Path

import trimesh


TARGET_RATIO = 0.50


def load_scene(path):
    scene = trimesh.load(
        str(path),
        force="scene",
    )

    if not scene.geometry:
        raise ValueError(
            "No mesh geometry found."
        )

    return scene


def get_stats(scene, path):
    vertices = 0
    polygons = 0

    for geometry in scene.geometry.values():

        if isinstance(
            geometry,
            trimesh.Trimesh,
        ):
            vertices += len(
                geometry.vertices
            )

            polygons += len(
                geometry.faces
            )

    file_size_mb = (
        Path(path).stat().st_size
        / (1024 * 1024)
    )

    return {
        "vertex_count": vertices,
        "polygon_count": polygons,
        "file_size_mb": round(
            file_size_mb,
            4,
        ),
    }


def optimize_asset(
    input_path,
    output_path,
    target_ratio=TARGET_RATIO,
):
    start_time = time.perf_counter()

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input asset not found: {input_path}"
        )

    scene = load_scene(
        input_path
    )

    before = get_stats(
        scene,
        input_path,
    )

    optimized_scene = trimesh.Scene()

    for name, geometry in (
        scene.geometry.items()
    ):

        if not isinstance(
            geometry,
            trimesh.Trimesh,
        ):
            continue

        current_faces = len(
            geometry.faces
        )

        target_faces = max(
            int(
                current_faces
                * target_ratio
            ),
            100,
        )

        optimized_mesh = (
            geometry.simplify_quadric_decimation(
                face_count=target_faces
            )
        )

        optimized_scene.add_geometry(
            optimized_mesh,
            node_name=name,
            geom_name=name,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    glb_data = (
        optimized_scene.export(
            file_type="glb"
        )
    )

    with open(
        output_path,
        "wb",
    ) as file:
        file.write(
            glb_data
        )

    optimized_loaded = (
        load_scene(
            output_path
        )
    )

    after = get_stats(
        optimized_loaded,
        output_path,
    )

    processing_time = round(
        time.perf_counter()
        - start_time,
        4,
    )

    result = {
        "input_asset": str(
            input_path
        ),
        "output_asset": str(
            output_path
        ),
        "target_ratio": (
            target_ratio
        ),
        "before": before,
        "after": after,
        "processing_time": (
            processing_time
        ),
    }

    return result


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python processing/optimize_asset.py "
            "<input.glb> <output.glb>"
        )

        return

    result = optimize_asset(
        sys.argv[1],
        sys.argv[2],
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()