import cv2
import numpy as np
from pathlib import Path
from rembg import remove

from validation import validate_image


class GarmentProcessor:
    def __init__(self, target_size=(768, 1024)):
        self.target_width = target_size[0]
        self.target_height = target_size[1]

    def process(self, image_path: str, output_dir: str) -> dict:
        # 1. Validate garment image
        validation_result = validate_image(image_path)

        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"]
            }

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            return {
                "success": False,
                "error": "Unable to read garment image."
            }

        # Convert input to BGRA
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # 2. Background removal
        rgba_input = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        removed = remove(rgba_input)

        if removed is None:
            return {
                "success": False,
                "error": "Garment background removal failed."
            }

        removed = np.array(removed)

        # rembg output should be RGBA
        if removed.shape[2] != 4:
            return {
                "success": False,
                "error": "Background removal did not return an RGBA image."
            }

        # 3. Garment mask from alpha channel
        alpha = removed[:, :, 3]

        _, garment_mask = cv2.threshold(
            alpha,
            10,
            255,
            cv2.THRESH_BINARY
        )

        # 4. Find garment bounding box
        points = cv2.findNonZero(garment_mask)

        if points is None:
            return {
                "success": False,
                "error": "No garment region detected."
            }

        x, y, w, h = cv2.boundingRect(points)

        cropped_rgba = removed[y:y+h, x:x+w]
        cropped_mask = garment_mask[y:y+h, x:x+w]

        # 5. Preserve aspect ratio while resizing
        scale = min(
            self.target_width / w,
            self.target_height / h
        )

        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))

        resized_rgba = cv2.resize(
            cropped_rgba,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        resized_mask = cv2.resize(
            cropped_mask,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST
        )

        # 6. Place garment on fixed canvas
        processed_canvas = np.zeros(
            (self.target_height, self.target_width, 4),
            dtype=np.uint8
        )

        mask_canvas = np.zeros(
            (self.target_height, self.target_width),
            dtype=np.uint8
        )

        x_offset = (self.target_width - new_width) // 2
        y_offset = (self.target_height - new_height) // 2

        processed_canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized_rgba

        mask_canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized_mask

        # 7. Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed_path = output_path / "garment_processed.png"
        mask_path = output_path / "garment_mask.png"

        processed_bgra = cv2.cvtColor(
            processed_canvas,
            cv2.COLOR_RGBA2BGRA
        )

        cv2.imwrite(
            str(processed_path),
            processed_bgra
        )

        cv2.imwrite(
            str(mask_path),
            mask_canvas
        )

        garment_pixels = int(np.count_nonzero(mask_canvas))

        coverage = garment_pixels / (
            self.target_width * self.target_height
        )

        return {
            "success": True,
            "original_width": validation_result["width"],
            "original_height": validation_result["height"],
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "processed_width": self.target_width,
            "processed_height": self.target_height,
            "garment_coverage": round(coverage, 4),
            "processed_output": str(processed_path),
            "mask_output": str(mask_path)
        }


if __name__ == "__main__":
    processor = GarmentProcessor()

    image_path = input("Enter garment image path: ").strip().strip('"')

    result = processor.process(
        image_path=image_path,
        output_dir="raritone-vton/dataset/garments/processed"
    )

    print(result)