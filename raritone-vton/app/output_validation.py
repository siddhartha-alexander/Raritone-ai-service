from pathlib import Path
from PIL import Image


MIN_OUTPUT_WIDTH = 256
MIN_OUTPUT_HEIGHT = 256


def validate_output(image_path: str) -> dict:
    path = Path(image_path)

    if not path.exists():
        return {
            "valid": False,
            "error": "Try-on output was not generated."
        }

    try:
        with Image.open(path) as image:
            width, height = image.size

            if width < MIN_OUTPUT_WIDTH or height < MIN_OUTPUT_HEIGHT:
                return {
                    "valid": False,
                    "error": "Generated output resolution is too small."
                }

            image.verify()

        return {
            "valid": True,
            "width": width,
            "height": height,
            "output": str(path)
        }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Generated output is invalid: {str(e)}"
        }