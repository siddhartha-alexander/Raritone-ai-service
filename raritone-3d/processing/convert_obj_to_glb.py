import sys
from pathlib import Path

import trimesh


def convert_obj_to_glb(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input OBJ not found: {input_path}"
        )

    if input_path.suffix.lower() != ".obj":
        raise ValueError(
            "Input file must be an OBJ file."
        )

    scene = trimesh.load(
        str(input_path),
        force="scene",
    )

    if not scene.geometry:
        raise ValueError(
            "OBJ contains no mesh geometry."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    glb_data = scene.export(
        file_type="glb"
    )

    with open(
        output_path,
        "wb",
    ) as file:
        file.write(glb_data)

    print("OBJ → GLB conversion successful")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python processing/convert_obj_to_glb.py "
            "<input.obj> <output.glb>"
        )
        sys.exit(1)

    convert_obj_to_glb(
        sys.argv[1],
        sys.argv[2],
    )