import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import trimesh


# ---------------------------------------------------------
# Preview generation
# ---------------------------------------------------------

def generate_preview(
    asset_path,
    output_path,
):
    """
    Generate a lightweight CPU-based PNG preview
    of a GLB/GLTF asset.

    This is intended as an asset-review thumbnail,
    not a photorealistic renderer.
    """

    asset_path = Path(asset_path)
    output_path = Path(output_path)

    if not asset_path.exists():
        raise FileNotFoundError(
            f"3D asset not found: {asset_path}"
        )

    if asset_path.suffix.lower() not in {
        ".glb",
        ".gltf",
    }:
        raise ValueError(
            "Only GLB and GLTF assets are supported."
        )

    # -----------------------------------------------------
    # 1. Load scene
    # -----------------------------------------------------

    scene = trimesh.load(
        str(asset_path),
        force="scene",
        process=False,
    )

    if not scene.geometry:
        raise ValueError(
            "No mesh geometry found in asset."
        )

    # -----------------------------------------------------
    # 2. Collect vertices
    # -----------------------------------------------------

    all_vertices = []

    for geometry in scene.geometry.values():

        if not isinstance(
            geometry,
            trimesh.Trimesh,
        ):
            continue

        if len(geometry.vertices) == 0:
            continue

        all_vertices.append(
            geometry.vertices
        )

    if not all_vertices:
        raise ValueError(
            "No vertices available for preview."
        )

    import numpy as np

    vertices = np.vstack(
        all_vertices
    )

    # -----------------------------------------------------
    # 3. Downsample large assets for preview
    # -----------------------------------------------------

    max_preview_points = 15000

    if len(vertices) > max_preview_points:

        step = max(
            len(vertices)
            // max_preview_points,
            1,
        )

        vertices = vertices[::step]

    # -----------------------------------------------------
    # 4. Generate preview
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(6, 6)
    )

    axis = fig.add_subplot(
        111,
        projection="3d",
    )

    axis.scatter(
        vertices[:, 0],
        vertices[:, 2],
        vertices[:, 1],
        s=1,
    )

    axis.set_box_aspect(
        (
            max(
                vertices[:, 0].max()
                - vertices[:, 0].min(),
                0.001,
            ),
            max(
                vertices[:, 2].max()
                - vertices[:, 2].min(),
                0.001,
            ),
            max(
                vertices[:, 1].max()
                - vertices[:, 1].min(),
                0.001,
            ),
        )
    )

    axis.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # 5. Result
    # -----------------------------------------------------

    return {
        "success": True,
        "asset": str(asset_path),
        "preview": str(output_path),
        "preview_type": "cpu_vertex_preview",
        "points_rendered": len(vertices),
    }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python "
            "raritone-3d/generation/"
            "preview_generator.py "
            "<asset.glb|asset.gltf> "
            "<preview.png>"
        )

        return

    try:

        result = generate_preview(
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