import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PersonSegmenter:
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False
        )

        self.segmenter = vision.ImageSegmenter.create_from_options(
            options
        )

    def segment(self, image_path: str, output_dir: str) -> dict:
        image = cv2.imread(image_path)

        if image is None:
            return {
                "success": False,
                "error": "Unable to read image."
            }

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.segmenter.segment(mp_image)

        category_mask = result.category_mask

        mask = category_mask.numpy_view()

        # Convert any foreground/person class to white
        binary_mask = np.where(
            mask > 0,
            255,
            0
        ).astype(np.uint8)

        output_path = Path(output_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_path = output_path / "person_mask.png"

        cv2.imwrite(
            str(mask_path),
            binary_mask
        )

        return {
            "success": True,
            "mask_output": str(mask_path),
            "mask_width": binary_mask.shape[1],
            "mask_height": binary_mask.shape[0]
        }


if __name__ == "__main__":
    segmenter = PersonSegmenter(
        model_path="raritone-vton/models/selfie_multiclass_256x256.tflite"
    )

    image_path = input(
        "Enter person image path: "
    )

    result = segmenter.segment(
        image_path=image_path,
        output_dir="raritone-vton/dataset/masks"
    )

    print(result)