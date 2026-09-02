from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


class HumanParser:
    MODEL_NAME = "mattmdjaga/segformer_b2_clothes"

    LABELS = {
        0: "Background",
        1: "Hat",
        2: "Hair",
        3: "Sunglasses",
        4: "Upper-clothes",
        5: "Skirt",
        6: "Pants",
        7: "Dress",
        8: "Belt",
        9: "Left-shoe",
        10: "Right-shoe",
        11: "Face",
        12: "Left-leg",
        13: "Right-leg",
        14: "Left-arm",
        15: "Right-arm",
        16: "Bag",
        17: "Scarf",
    }

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading human parser on {self.device}...")

        self.processor = SegformerImageProcessor.from_pretrained(
            self.MODEL_NAME
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(self.MODEL_NAME)
            .to(self.device)
        )

        self.model.eval()

        print("Human parser loaded.")

    def parse(self, image_path, output_path=None):
        image_path = Path(image_path)

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

            segmentation = (
                logits.argmax(dim=1)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

        result = {
            "success": True,
            "segmentation": segmentation,
            "width": width,
            "height": height,
        }

        if output_path is not None:

            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            visualization = self._visualize(
                segmentation
            )

            cv2.imwrite(
                str(output_path),
                visualization
            )

            result["output"] = str(output_path)

        return result

    def _visualize(self, segmentation):
        """
        Create color-coded visualization for testing.
        """

        colors = np.array(
            [
                [0, 0, 0],          # Background
                [128, 0, 0],        # Hat
                [255, 0, 0],        # Hair
                [0, 85, 0],         # Sunglasses
                [170, 0, 51],       # Upper clothes
                [255, 85, 0],       # Skirt
                [0, 0, 85],         # Pants
                [0, 119, 221],      # Dress
                [85, 85, 0],        # Belt
                [0, 85, 85],        # Left shoe
                [85, 51, 0],        # Right shoe
                [52, 86, 128],      # Face
                [0, 128, 0],        # Left leg
                [0, 0, 255],        # Right leg
                [51, 170, 221],     # Left arm
                [0, 255, 255],      # Right arm
                [85, 255, 170],     # Bag
                [170, 255, 85],     # Scarf
            ],
            dtype=np.uint8
        )

        visualization = colors[segmentation]

        # Convert RGB -> BGR for OpenCV
        visualization = cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )

        return visualization