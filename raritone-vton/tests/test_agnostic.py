import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_DIR / "app"

sys.path.insert(0, str(APP_DIR))

from agnostic_generator import AgnosticGenerator


person_path = input(
    "Enter person image path: "
).strip().strip('"')

output_dir = (
    PROJECT_DIR
    / "outputs"
    / "agnostic_test"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

agnostic_output = (
    output_dir
    / "agnostic_person_v2.jpg"
)


print("\nLoading agnostic generator...")

generator = AgnosticGenerator()


print("\nGenerating agnostic person...")

result = generator.generate(
    person_image_path=person_path,
    output_path=agnostic_output
)

print(result)

if result.get("success"):

    print(
        "\nAgnostic image generated successfully."
    )

    print(
        "Output:",
        agnostic_output
    )