import os
import time
import cv2

from app.garment_processor import process_garment
from app.tryon_pipeline import (
    calculate_garment_region,
    align_garment,
    composite_garment,
)


MODEL_VERSION = "tryon-v1-baseline"


def run_tryon(
    person_input,
    garment_input,
    pose_data,
    mask,
):
    

    total_start = time.perf_counter()
    inference_start = time.perf_counter()

    try:
        
        processed_garment, garment_mask = process_garment(
            garment_input
        )


        image_height, image_width = person_input.shape[:2]

       

        region = calculate_garment_region(
            pose_data,
            image_width,
            image_height,
        )

       
        aligned_garment, aligned_mask = align_garment(
            processed_garment,
            garment_mask,
            region,
        )

       

        result_image = composite_garment(
            person_input,
            aligned_garment,
            aligned_mask,
            region,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Try-on inference failed: {str(exc)}"
        )

    inference_time = round(
        time.perf_counter() - inference_start,
        4,
    )

   
    os.makedirs(
        "outputs",
        exist_ok=True,
    )

  
    result_path = "outputs/tryon_result.png"

    success = cv2.imwrite(
        result_path,
        result_image,
    )

    if not success:
        raise RuntimeError(
            "Failed to save try-on result."
        )

    total_time = round(
        time.perf_counter() - total_start,
        4,
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "inference_time": inference_time,
        "total_time": total_time,
        "result_path": result_path,
        "garment_region": region,
    }

    return result_image, metadata