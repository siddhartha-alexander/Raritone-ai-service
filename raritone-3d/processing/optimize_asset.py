import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import trimesh


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.asset_validator import validate_asset


# ---------------------------------------------------------
# glTF Transform CLI
# ---------------------------------------------------------

def get_gltf_transform_command():
    """
    Locate the globally installed glTF Transform CLI.
    """

    command = shutil.which("gltf-transform")

    if command is None:
        raise RuntimeError(
            "glTF Transform CLI was not found. "
            "Install it using: "
            "npm install -g @gltf-transform/cli"
        )

    return command


# ---------------------------------------------------------
# Asset load-time benchmark
# ---------------------------------------------------------

def measure_load_time(asset_path, runs=3):
    """
    Measure average CPU load time for a GLB/GLTF asset.

    Multiple runs are used because a single timing
    measurement can be noisy.
    """

    asset_path = Path(asset_path)

    timings = []

    for _ in range(runs):

        start = time.perf_counter()

        scene = trimesh.load(
            str(asset_path),
            force="scene",
            process=False,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if not scene.geometry:
            raise ValueError(
                f"No geometry found while "
                f"benchmarking: {asset_path}"
            )

        timings.append(
            elapsed
        )

    average = (
        sum(timings)
        / len(timings)
    )

    return round(
        average,
        6,
    )


# ---------------------------------------------------------
# Optimization
# ---------------------------------------------------------

def optimize_asset(
    input_path,
    output_path,
):
    """
    Optimize an existing GLB/GLTF asset using
    glTF Transform.

    The optimizer preserves textures and materials
    and avoids mesh compression that caused
    compatibility issues with Trimesh.
    """

    start_time = time.perf_counter()

    input_path = Path(input_path)
    output_path = Path(output_path)

    # -----------------------------------------------------
    # 1. Input checks
    # -----------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input asset not found: {input_path}"
        )

    if input_path.suffix.lower() not in {
        ".glb",
        ".gltf",
    }:
        raise ValueError(
            "Only GLB and GLTF assets are supported."
        )

    # -----------------------------------------------------
    # 2. Validate original asset
    # -----------------------------------------------------

    before = validate_asset(
        str(input_path)
    )

    if not before["valid"]:
        raise ValueError(
            "Input asset failed validation and "
            "cannot be optimized."
        )

    # -----------------------------------------------------
    # 3. Measure original loading time
    # -----------------------------------------------------

    original_load_time = (
        measure_load_time(
            input_path
        )
    )

    # -----------------------------------------------------
    # 4. Prepare output
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gltf_transform = (
        get_gltf_transform_command()
    )

    command = [
        gltf_transform,
        "optimize",
        str(input_path),
        str(output_path),

        # Preserve original texture encoding.
        "--texture-compress",
        "false",

        # Avoid meshopt compression because it caused
        # compatibility issues with Trimesh validation.
        "--compress",
        "false",
    ]

    # -----------------------------------------------------
    # 5. Windows CLI handling
    # -----------------------------------------------------

    if (
        sys.platform == "win32"
        and Path(
            gltf_transform
        ).suffix.lower()
        in {".cmd", ".bat"}
    ):
        command = [
            "cmd",
            "/c",
            *command,
        ]

    # -----------------------------------------------------
    # 6. Run optimization
    # -----------------------------------------------------

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if process.returncode != 0:

        error_message = (
            process.stderr.strip()
            or process.stdout.strip()
            or "Unknown glTF Transform error."
        )

        raise RuntimeError(
            "3D asset optimization failed: "
            f"{error_message}"
        )

    if not output_path.exists():

        raise RuntimeError(
            "Optimization completed but no "
            "output asset was created."
        )

    # -----------------------------------------------------
    # 7. Validate optimized asset
    # -----------------------------------------------------

    after = validate_asset(
        str(output_path)
    )

    if not after["valid"]:

        raise RuntimeError(
            "Optimized asset failed validation."
        )

    # -----------------------------------------------------
    # 8. Preservation checks
    # -----------------------------------------------------

    if (
        before["materials"]
        and not after["materials"]
    ):
        raise RuntimeError(
            "Optimization removed the asset materials."
        )

    if (
        before["textures"]
        and not after["textures"]
    ):
        raise RuntimeError(
            "Optimization removed the asset textures."
        )

    # -----------------------------------------------------
    # 9. Measure optimized loading time
    # -----------------------------------------------------

    optimized_load_time = (
        measure_load_time(
            output_path
        )
    )

    # -----------------------------------------------------
    # 10. Calculate improvements
    # -----------------------------------------------------

    before_size = (
        before["file_size_mb"]
    )

    after_size = (
        after["file_size_mb"]
    )

    before_polygons = (
        before["polygon_count"]
    )

    after_polygons = (
        after["polygon_count"]
    )

    if before_size > 0:

        size_reduction_percent = round(
            (
                (
                    before_size
                    - after_size
                )
                / before_size
            )
            * 100,
            2,
        )

    else:

        size_reduction_percent = 0.0

    if before_polygons > 0:

        polygon_reduction_percent = round(
            (
                (
                    before_polygons
                    - after_polygons
                )
                / before_polygons
            )
            * 100,
            2,
        )

    else:

        polygon_reduction_percent = 0.0

    # -----------------------------------------------------
    # Load-time improvement
    # -----------------------------------------------------

    if original_load_time > 0:

        load_time_improvement_percent = round(
            (
                (
                    original_load_time
                    - optimized_load_time
                )
                / original_load_time
            )
            * 100,
            2,
        )

    else:

        load_time_improvement_percent = 0.0

    # -----------------------------------------------------
    # Total pipeline optimization time
    # -----------------------------------------------------

    processing_time = round(
        time.perf_counter()
        - start_time,
        4,
    )

    # -----------------------------------------------------
    # 11. Result
    # -----------------------------------------------------

    result = {

        "success": True,

        "input_asset": str(
            input_path
        ),

        "output_asset": str(
            output_path
        ),

        "before": before,

        "after": after,

        "optimization": {

            "size_reduction_percent": (
                size_reduction_percent
            ),

            "polygon_reduction_percent": (
                polygon_reduction_percent
            ),

            "materials_preserved": (
                not before["materials"]
                or after["materials"]
            ),

            "textures_preserved": (
                not before["textures"]
                or after["textures"]
            ),

            "original_load_time_seconds": (
                original_load_time
            ),

            "optimized_load_time_seconds": (
                optimized_load_time
            ),

            "load_time_improvement_percent": (
                load_time_improvement_percent
            ),
        },

        "processing_time": (
            processing_time
        ),
    }

    return result


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python "
            "raritone-3d/processing/"
            "optimize_asset.py "
            "<input.glb|input.gltf> "
            "<output.glb|output.gltf>"
        )

        return

    try:

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