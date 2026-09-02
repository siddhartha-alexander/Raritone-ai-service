from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove

from agnostic_generator import AgnosticGenerator
from inference import VTONInference


class RawTryOnPipeline:
    """
    Public workflow:

        raw person image
              +
        raw garment image
              ↓
        agnostic generation
        garment segmentation
              ↓
        agnostic + garment + garment mask
              ↓
        trained RaritoneVTONNet
              ↓
        try-on result
    """

    def __init__(self):
        print("\nInitializing raw try-on pipeline...")

        self.agnostic_generator = AgnosticGenerator()
        self.inference = VTONInference()

        print("Raw try-on pipeline ready.\n")

    def _prepare_garment(
        self,
        garment_image_path,
        garment_output_path,
        mask_output_path,
    ):
        garment_image_path = Path(garment_image_path)
        garment_output_path = Path(garment_output_path)
        mask_output_path = Path(mask_output_path)

        if not garment_image_path.exists():
            raise FileNotFoundError(
                f"Garment image not found: {garment_image_path}"
            )

        image = Image.open(
            garment_image_path
        ).convert("RGBA")

        # Remove garment background
        segmented = remove(image)

        if not isinstance(segmented, Image.Image):
            segmented = Image.open(segmented).convert("RGBA")

        segmented = segmented.convert("RGBA")

        rgba = np.array(segmented)

        # Alpha channel becomes garment mask
        alpha = rgba[:, :, 3]

        mask = np.where(
            alpha > 20,
            255,
            0
        ).astype(np.uint8)

        # RGB garment
        rgb = rgba[:, :, :3]

        garment_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        Image.fromarray(
            rgb
        ).save(
            garment_output_path
        )

        Image.fromarray(
            mask,
            mode="L"
        ).save(
            mask_output_path
        )

        return {
            "garment": str(garment_output_path),
            "mask": str(mask_output_path),
        }

    def run(
        self,
        person_image_path,
        garment_image_path,
        output_path,
    ):
        person_image_path = Path(person_image_path)
        garment_image_path = Path(garment_image_path)
        output_path = Path(output_path)

        if not person_image_path.exists():
            raise FileNotFoundError(
                f"Person image not found: {person_image_path}"
            )

        if not garment_image_path.exists():
            raise FileNotFoundError(
                f"Garment image not found: {garment_image_path}"
            )

        # Folder for temporary preprocessing files
        work_dir = (
            output_path.parent
            / f"{output_path.stem}_processing"
        )

        work_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        agnostic_path = (
            work_dir
            / "agnostic_person.jpg"
        )

        processed_garment_path = (
            work_dir
            / "garment.png"
        )

        garment_mask_path = (
            work_dir
            / "garment_mask.png"
        )

        # --------------------------------------------------
        # STEP 1: Generate agnostic person
        # --------------------------------------------------

        print("1. Generating agnostic person...")

        agnostic_result = (
            self.agnostic_generator.generate(
                person_image_path=str(person_image_path),
                output_path=str(agnostic_path),
            )
        )

        if not agnostic_result.get("success"):
            raise RuntimeError(
                "Agnostic generation failed: "
                + str(agnostic_result)
            )

        print(
            "   Agnostic:",
            agnostic_path
        )

        # --------------------------------------------------
        # STEP 2: Prepare garment + garment mask
        # --------------------------------------------------

        print("2. Preparing garment...")

        garment_result = (
            self._prepare_garment(
                garment_image_path=garment_image_path,
                garment_output_path=processed_garment_path,
                mask_output_path=garment_mask_path,
            )
        )

        print(
            "   Garment:",
            garment_result["garment"]
        )

        print(
            "   Mask:",
            garment_result["mask"]
        )

        # --------------------------------------------------
        # STEP 3: Run trained model
        # --------------------------------------------------

        print("3. Running trained VTON model...")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.inference.predict(
            agnostic_path=str(agnostic_path),
            garment_path=str(processed_garment_path),
            garment_mask_path=str(garment_mask_path),
            output_path=str(output_path),
        )

        print("\nTry-on completed.")
        print("Result:", output_path)

        return {
            "success": True,
            "result": str(output_path),
            "agnostic": str(agnostic_path),
            "garment": str(processed_garment_path),
            "garment_mask": str(garment_mask_path),
        }


if __name__ == "__main__":

    person_path = input(
        "Enter person image path: "
    ).strip().strip('"')

    garment_path = input(
        "Enter garment image path: "
    ).strip().strip('"')

    project_dir = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    output_path = (
        project_dir
        / "outputs"
        / "raw_tryon_test"
        / "tryon_result.jpg"
    )

    pipeline = RawTryOnPipeline()

    result = pipeline.run(
        person_image_path=person_path,
        garment_image_path=garment_path,
        output_path=output_path,
    )

    print("\nPipeline result:")
    print(result)