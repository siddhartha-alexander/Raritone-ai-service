import json
from pathlib import Path

from asset_validator import validate_asset


INPUT_DIR = Path("raritone-3d/input")
OUTPUT_FILE = Path("raritone-3d/validation/validation_report.json")


def main():
    results = []

    asset_files = sorted(
        list(INPUT_DIR.glob("*.glb")) +
        list(INPUT_DIR.glob("*.gltf"))
    )

    if not asset_files:
        print("No GLB/GLTF assets found in input folder.")
        return

    print(f"Found {len(asset_files)} assets.\n")

    for asset in asset_files:
        print(f"Validating: {asset.name}")

        result = validate_asset(str(asset))
        results.append(result)

        status = "PASS" if result["valid"] else "FAIL"

        print(
            f"{status} | "
            f"Polygons: {result['polygon_count']} | "
            f"Meshes: {result['mesh_count']} | "
            f"Size: {result['file_size_mb']} MB"
        )
        print()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    passed = sum(1 for r in results if r["valid"])
    failed = len(results) - passed

    print("=" * 50)
    print(f"Total assets : {len(results)}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")
    print(f"Report saved : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()