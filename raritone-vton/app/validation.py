from pathlib import Path
from PIL import Image

ALLOWED_FORMATS = {"JPEG", "PNG"}
MIN_WIDTH = 256
MIN_HEIGHT = 256
MAX_FILE_SIZE_MB = 10


def validate_image(image_path: str) -> dict:
    path = Path(image_path)

    if not path.exists():
        return {
            "valid": False,
            "error": "Image file does not exist."
        }

    file_size_mb = path.stat().st_size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        return {
            "valid": False,
            "error": f"Image exceeds {MAX_FILE_SIZE_MB} MB limit."
        }

    try:
        with Image.open(path) as img:
            image_format = img.format
            width, height = img.size

            if image_format not in ALLOWED_FORMATS:
                return {
                    "valid": False,
                    "error": "Only JPG/JPEG and PNG images are supported."
                }

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return {
                    "valid": False,
                    "error": f"Minimum supported resolution is {MIN_WIDTH}x{MIN_HEIGHT}."
                }

            return {
                "valid": True,
                "format": image_format,
                "width": width,
                "height": height,
                "file_size_mb": round(file_size_mb, 3)
            }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Invalid or corrupted image: {str(e)}"
        }


if __name__ == "__main__":
    test_path = input("Enter image path: ")
    print(validate_image(test_path))