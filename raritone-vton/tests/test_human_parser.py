import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_DIR / "app"

sys.path.insert(0, str(APP_DIR))

from human_parser import HumanParser


person_path = input("Enter person image path: ").strip().strip('"')

output_dir = PROJECT_DIR / "outputs" / "human_parser_test"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "human_parsing.png"


print("\nLoading parser...")

parser = HumanParser()

print("\nParsing image...")

result = parser.parse(
    image_path=person_path,
    output_path=output_path
)

print(result)

if result.get("success"):
    print("\nHuman parsing completed.")
    print("Output:", output_path)