import cv2
import numpy as np


def calculate_garment_region(landmarks, image_width, image_height):
    """
    Calculate the target upper-body region using
    shoulder and hip landmarks.
    """

    required_landmarks = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]

    for landmark in required_landmarks:
        if landmark not in landmarks:
            raise ValueError(
                f"Required landmark missing: {landmark}"
            )

    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    left_hip = landmarks["left_hip"]
    right_hip = landmarks["right_hip"]

    # Convert normalized landmark coordinates to pixels
    ls_x = int(left_shoulder["x"] * image_width)
    ls_y = int(left_shoulder["y"] * image_height)

    rs_x = int(right_shoulder["x"] * image_width)
    rs_y = int(right_shoulder["y"] * image_height)

    lh_x = int(left_hip["x"] * image_width)
    lh_y = int(left_hip["y"] * image_height)

    rh_x = int(right_hip["x"] * image_width)
    rh_y = int(right_hip["y"] * image_height)

    # Shoulder width
    shoulder_width = int(
        np.sqrt(
            (rs_x - ls_x) ** 2 +
            (rs_y - ls_y) ** 2
        )
    )

    # Upper body height
    shoulder_center_y = int((ls_y + rs_y) / 2)
    hip_center_y = int((lh_y + rh_y) / 2)

    body_height = abs(
        hip_center_y - shoulder_center_y
    )

    # Center of garment
    center_x = int(
        (ls_x + rs_x + lh_x + rh_x) / 4
    )

    center_y = int(
        (shoulder_center_y + hip_center_y) / 2
    )

    # Slightly enlarge garment
    target_width = int(shoulder_width * 1.35)
    target_height = int(body_height * 1.15)

    # Approximate shoulder rotation
    angle = np.degrees(
        np.arctan2(
            rs_y - ls_y,
            rs_x - ls_x,
        )
    )

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": target_width,
        "height": target_height,
        "angle": float(angle),
    }


def align_garment(
    garment,
    mask,
    region,
):
    """
    Resize and rotate garment according to
    the person's upper-body pose.
    """

    target_width = region["width"]
    target_height = region["height"]
    angle = region["angle"]

    # Resize garment and mask
    garment_resized = cv2.resize(
        garment,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    mask_resized = cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    center = (
        target_width // 2,
        target_height // 2,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    garment_rotated = cv2.warpAffine(
        garment_resized,
        rotation_matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask_rotated = cv2.warpAffine(
        mask_resized,
        rotation_matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return garment_rotated, mask_rotated


def composite_garment(
    person_image,
    garment,
    mask,
    region,
):
    """
    Place the aligned garment onto the person image.
    This is a baseline 2D compositing prototype,
    not a photorealistic AI try-on model.
    """

    result = person_image.copy()

    garment_height, garment_width = garment.shape[:2]

    center_x = region["center_x"]
    center_y = region["center_y"]

    # Calculate placement
    x1 = center_x - garment_width // 2
    y1 = center_y - garment_height // 2

    x2 = x1 + garment_width
    y2 = y1 + garment_height

    # Clip placement to image boundaries
    image_height, image_width = result.shape[:2]

    x1_clip = max(0, x1)
    y1_clip = max(0, y1)

    x2_clip = min(image_width, x2)
    y2_clip = min(image_height, y2)

    if x1_clip >= x2_clip or y1_clip >= y2_clip:
        raise ValueError(
            "Garment region is outside the image."
        )

    # Corresponding garment crop
    garment_x1 = x1_clip - x1
    garment_y1 = y1_clip - y1

    garment_x2 = garment_x1 + (
        x2_clip - x1_clip
    )

    garment_y2 = garment_y1 + (
        y2_clip - y1_clip
    )

    garment_crop = garment[
        garment_y1:garment_y2,
        garment_x1:garment_x2,
    ]

    mask_crop = mask[
        garment_y1:garment_y2,
        garment_x1:garment_x2,
    ]

    roi = result[
        y1_clip:y2_clip,
        x1_clip:x2_clip,
    ]

    # Convert mask to alpha
    alpha = (
        mask_crop.astype(np.float32) / 255.0
    )

    alpha = np.expand_dims(
        alpha,
        axis=2,
    )

    # Blend garment with person image
    blended = (
        garment_crop.astype(np.float32) * alpha
        + roi.astype(np.float32) * (1 - alpha)
    )

    result[
        y1_clip:y2_clip,
        x1_clip:x2_clip,
    ] = blended.astype(np.uint8)

    return result