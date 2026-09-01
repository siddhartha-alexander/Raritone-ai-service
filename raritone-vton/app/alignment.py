import cv2
import numpy as np


def align_garment(
    person_image_path: str,
    garment_rgba_path: str,
    landmarks: dict
) -> dict:
    person = cv2.imread(person_image_path)
    garment = cv2.imread(garment_rgba_path, cv2.IMREAD_UNCHANGED)

    if person is None:
        return {
            "success": False,
            "error": "Unable to read person image."
        }

    if garment is None or garment.shape[2] != 4:
        return {
            "success": False,
            "error": "Garment image must be RGBA."
        }

    h, w, _ = person.shape

    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    left_hip = landmarks["left_hip"]
    right_hip = landmarks["right_hip"]

    shoulder_left = np.array([
        left_shoulder["pixel_x"],
        left_shoulder["pixel_y"]
    ])

    shoulder_right = np.array([
        right_shoulder["pixel_x"],
        right_shoulder["pixel_y"]
    ])

    hip_left = np.array([
        left_hip["pixel_x"],
        left_hip["pixel_y"]
    ])

    hip_right = np.array([
        right_hip["pixel_x"],
        right_hip["pixel_y"]
    ])

    shoulder_width = np.linalg.norm(
        shoulder_left - shoulder_right
    )

    torso_height = (
        ((hip_left[1] + hip_right[1]) / 2)
        -
        ((shoulder_left[1] + shoulder_right[1]) / 2)
    )

    target_width = int(shoulder_width * 1.8)
    target_height = int(torso_height * 1.7)

    target_width = max(target_width, 1)
    target_height = max(target_height, 1)

    garment_rgb = garment[:, :, :3]
    garment_alpha = garment[:, :, 3]

    resized_rgb = cv2.resize(
        garment_rgb,
        (target_width, target_height)
    )

    resized_alpha = cv2.resize(
        garment_alpha,
        (target_width, target_height)
    )

    center_x = int(
        (
            shoulder_left[0]
            +
            shoulder_right[0]
        ) / 2
    )

    top_y = int(
        (
            shoulder_left[1]
            +
            shoulder_right[1]
        ) / 2
    )

    x1 = center_x - target_width // 2
    y1 = top_y - int(target_height * 0.15)

    x2 = x1 + target_width
    y2 = y1 + target_height

    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(w, x2)
    y2_clip = min(h, y2)

    crop_x1 = x1_clip - x1
    crop_y1 = y1_clip - y1
    crop_x2 = crop_x1 + (x2_clip - x1_clip)
    crop_y2 = crop_y1 + (y2_clip - y1_clip)

    aligned_rgba = np.zeros(
        (h, w, 4),
        dtype=np.uint8
    )

    aligned_rgba[
        y1_clip:y2_clip,
        x1_clip:x2_clip,
        :3
    ] = resized_rgb[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    aligned_rgba[
        y1_clip:y2_clip,
        x1_clip:x2_clip,
        3
    ] = resized_alpha[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    return {
        "success": True,
        "aligned_garment": aligned_rgba,
        "placement": {
            "x1": int(x1_clip),
            "y1": int(y1_clip),
            "x2": int(x2_clip),
            "y2": int(y2_clip)
        }
    }