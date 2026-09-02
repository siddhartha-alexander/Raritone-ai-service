from pathlib import Path

import cv2
import numpy as np

from human_parser import HumanParser


class AgnosticGenerator:
    """
    Builds a VTON-style agnostic image using semantic human parsing.

    Main idea:
    - preserve face/hair/lower body
    - remove upper clothing region
    - neutralize arms around clothing area
    - preserve the general body shape
    """

    # SegFormer clothing parser labels
    UPPER_CLOTHES = 4
    DRESS = 7
    LEFT_ARM = 14
    RIGHT_ARM = 15
    SCARF = 17

    def __init__(self):
        self.parser = HumanParser()

    def generate(
        self,
        person_image_path,
        output_path,
    ):
        person_image_path = Path(person_image_path)
        output_path = Path(output_path)

        image = cv2.imread(str(person_image_path))

        if image is None:
            return {
                "success": False,
                "error": "Unable to read person image."
            }

        height, width = image.shape[:2]

        # --------------------------------------------------
        # 1. Human parsing
        # --------------------------------------------------

        parse_result = self.parser.parse(
            image_path=person_image_path
        )

        if not parse_result.get("success"):
            return {
                "success": False,
                "error": "Human parsing failed."
            }

        segmentation = parse_result["segmentation"]

        if segmentation.shape[:2] != (height, width):
            segmentation = cv2.resize(
                segmentation,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        # --------------------------------------------------
        # 2. Build clothing-removal mask
        # --------------------------------------------------

        clothing_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        clothing_labels = [
            self.UPPER_CLOTHES,
            self.DRESS,
            self.SCARF,
        ]

        for label in clothing_labels:
            clothing_mask[
                segmentation == label
            ] = 255

        # --------------------------------------------------
        # 3. Arm mask
        # --------------------------------------------------

        arm_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        arm_mask[
            segmentation == self.LEFT_ARM
        ] = 255

        arm_mask[
            segmentation == self.RIGHT_ARM
        ] = 255

        # --------------------------------------------------
        # 4. Expand clothing region slightly
        # --------------------------------------------------

        kernel_size = max(
            int(min(width, height) * 0.015),
            5
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        clothing_mask = cv2.dilate(
            clothing_mask,
            kernel,
            iterations=1
        )

        # --------------------------------------------------
        # 5. Only remove arms near clothing region
        #
        # This avoids deleting the entire forearm/hand area.
        # --------------------------------------------------

        expanded_clothing = cv2.dilate(
            clothing_mask,
            kernel,
            iterations=3
        )

        arm_near_clothing = cv2.bitwise_and(
            arm_mask,
            expanded_clothing
        )

        # --------------------------------------------------
        # 6. Final agnostic mask
        # --------------------------------------------------

        agnostic_mask = cv2.bitwise_or(
            clothing_mask,
            arm_near_clothing
        )

        # Smooth mask edges
        agnostic_mask = cv2.morphologyEx(
            agnostic_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        # --------------------------------------------------
        # 7. Create neutral body region
        # --------------------------------------------------

        agnostic = image.copy()

        neutral_color = np.array(
            [128, 128, 128],
            dtype=np.uint8
        )

        agnostic[
            agnostic_mask > 0
        ] = neutral_color

        # --------------------------------------------------
        # 8. Smooth only the masked region
        # --------------------------------------------------

        blurred = cv2.GaussianBlur(
            agnostic,
            (5, 5),
            0
        )

        mask_3 = cv2.merge(
            [
                agnostic_mask,
                agnostic_mask,
                agnostic_mask,
            ]
        )

        agnostic = np.where(
            mask_3 > 0,
            blurred,
            agnostic
        ).astype(np.uint8)

        # --------------------------------------------------
        # 9. Save output
        # --------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        saved = cv2.imwrite(
            str(output_path),
            agnostic
        )

        if not saved:
            return {
                "success": False,
                "error": "Failed to save agnostic image."
            }

        return {
            "success": True,
            "output": str(output_path),
            "width": width,
            "height": height,
        }