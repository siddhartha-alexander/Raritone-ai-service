import cv2
import numpy as np
from pathlib import Path
from output_validation import validate_output
from person_processor import PersonProcessor
from garment_processor import GarmentProcessor
from alignment import align_garment


class TryOnPipeline:
    def __init__(
        self,
        pose_model_path: str
    ):
        self.person_processor = PersonProcessor(
            pose_model_path
        )

        self.garment_processor = GarmentProcessor()

    def run(
        self,
        person_image: str,
        garment_image: str,
        output_dir: str
    ) -> dict:

        output_path = Path(output_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # 1. Person processing
        person_result = self.person_processor.process(
            image_path=person_image,
            output_dir=str(
                output_path / "person"
            )
        )

        if not person_result["success"]:
            return person_result

        # 2. Garment processing
        garment_result = self.garment_processor.process(
            image_path=garment_image,
            output_dir=str(
                output_path / "garment"
            )
        )

        if not garment_result["success"]:
            return garment_result

        # 3. Garment alignment
        alignment_result = align_garment(
            person_image_path=person_image,
            garment_rgba_path=garment_result[
                "processed_output"
            ],
            landmarks=person_result[
                "landmarks"
            ]
        )

        if not alignment_result["success"]:
            return alignment_result

        person = cv2.imread(person_image)

        aligned = alignment_result[
            "aligned_garment"
        ]

        garment_rgb = aligned[:, :, :3]
        alpha = aligned[:, :, 3].astype(
            np.float32
        ) / 255.0

        alpha = np.expand_dims(
            alpha,
            axis=2
        )

        # 4. Basic compositing try-on
        result = (
            garment_rgb * alpha
            +
            person * (1 - alpha)
        ).astype(np.uint8)

        result_path = (
            output_path
            /
            "tryon_result.jpg"
        )

        cv2.imwrite(
            str(result_path),
            result
        )
        output_validation = validate_output(
            str(result_path)
        )

        if not output_validation["valid"]:
            return {
                "success": False,
                "error": output_validation["error"]
            }
        

        return {
            "success": True,
            "result": str(result_path),
            "person_pose": person_result[
                "pose_output"
            ],
            "garment_processed": garment_result[
                "processed_output"
            ],
            "garment_mask": garment_result[
                "mask_output"
            ],
            "placement": alignment_result[
                "placement"
            ],
            "output_validation": output_validation
        }
if __name__ == "__main__":
    pipeline = TryOnPipeline(
        pose_model_path=(
            "raritone-vton/models/"
            "pose_landmarker_lite.task"
        )
    )

    person = input(
        "Enter person image path: "
    ).strip().strip('"')

    garment = input(
        "Enter garment image path: "
    ).strip().strip('"')

    result = pipeline.run(
        person_image=person,
        garment_image=garment,
        output_dir=(
            "raritone-vton/outputs/demo"
        )
    )

    print(result)