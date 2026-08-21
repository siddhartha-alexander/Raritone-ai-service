import cv2
import numpy as np


def calculate_garment_region(landmarks, image_width, image_height):
    """
    Calculate a pose-aware upper-body garment region.

    Uses:
    - left shoulder
    - right shoulder
    - left hip
    - right hip

    The garment is positioned from slightly above the shoulder line
    down toward the hip region.
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

   
    # Convert normalized coordinates to pixels
   

    ls_x = int(left_shoulder["x"] * image_width)
    ls_y = int(left_shoulder["y"] * image_height)

    rs_x = int(right_shoulder["x"] * image_width)
    rs_y = int(right_shoulder["y"] * image_height)

    lh_x = int(left_hip["x"] * image_width)
    lh_y = int(left_hip["y"] * image_height)

    rh_x = int(right_hip["x"] * image_width)
    rh_y = int(right_hip["y"] * image_height)

   
    # Shoulder center
   

    shoulder_center_x = int((ls_x + rs_x) / 2)
    shoulder_center_y = int((ls_y + rs_y) / 2)

   
    # Hip center
   

    hip_center_x = int((lh_x + rh_x) / 2)
    hip_center_y = int((lh_y + rh_y) / 2)

   
    # Shoulder width
   

    shoulder_width = np.sqrt(
        (rs_x - ls_x) ** 2
        + (rs_y - ls_y) ** 2
    )

   
    # Torso height
   

    torso_height = np.sqrt(
        (hip_center_x - shoulder_center_x) ** 2
        + (hip_center_y - shoulder_center_y) ** 2
    )

   
    # Garment dimensions
   

    # Slightly wider than shoulders
    target_width = int(shoulder_width * 1.5)

    # Shirt should extend from shoulders toward hips
    target_height = int(torso_height * 1.5)

    # Prevent extremely small dimensions
    target_width = max(target_width, 25)
    target_height = max(target_height, 25)

   
    # Garment center
   

    center_x = shoulder_center_x

    # Move garment center slightly downward from shoulders
    center_y = int(
        shoulder_center_y
        + target_height * 0.30
    )

   
    # Shoulder angle
   

    angle = np.degrees(
        np.arctan2(
            rs_y - ls_y,
            rs_x - ls_x,
        )
    )

    # Normalize angle to [-90, 90]
    if angle > 90:
        angle -= 180

    if angle < -90:
        angle += 180

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": target_width,
        "height": target_height,
        "angle": float(angle),

        "shoulder_center_y": shoulder_center_y,
        "hip_center_y": hip_center_y,
    }


def align_garment(
    garment,
    mask,
    region,
):
    """
    Resize the garment while preserving its aspect ratio.

    The garment is fitted inside the pose-based target region
    instead of being stretched to fill it.
    """

    target_width = region["width"]
    target_height = region["height"]
    angle = region["angle"]

   
    # Validate mask
   

    if mask is None:
        raise ValueError("Garment mask is missing.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2GRAY,
        )

    # Binary mask
    _, mask = cv2.threshold(
        mask,
        30,
        255,
        cv2.THRESH_BINARY,
    )

   
    # Original garment dimensions
   

    garment_height, garment_width = garment.shape[:2]

    if garment_width == 0 or garment_height == 0:
        raise ValueError(
            "Invalid garment dimensions."
        )

   
    # Preserve aspect ratio
   

    scale_x = target_width / garment_width
    scale_y = target_height / garment_height

    scale = min(scale_x, scale_y)

    new_width = max(
        1,
        int(garment_width * scale),
    )

    new_height = max(
        1,
        int(garment_height * scale),
    )

    garment_resized = cv2.resize(
        garment,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    mask_resized = cv2.resize(
        mask,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

   
    # Create transparent placement canvas
   

    canvas_width = target_width
    canvas_height = target_height

    garment_canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        dtype=np.uint8,
    )

    mask_canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
        ),
        dtype=np.uint8,
    )

    # Center resized garment inside target region
    x_offset = max(
        0,
        (canvas_width - new_width) // 2,
    )

    y_offset = max(
        0,
        (canvas_height - new_height) // 2,
    )

    x_end = min(
        canvas_width,
        x_offset + new_width,
    )

    y_end = min(
        canvas_height,
        y_offset + new_height,
    )

    garment_canvas[
        y_offset:y_end,
        x_offset:x_end,
    ] = garment_resized[
        0:y_end - y_offset,
        0:x_end - x_offset,
    ]

    mask_canvas[
        y_offset:y_end,
        x_offset:x_end,
    ] = mask_resized[
        0:y_end - y_offset,
        0:x_end - x_offset,
    ]

   
    # Rotate only for meaningful shoulder tilt
   

    center = (
        canvas_width // 2,
        canvas_height // 2,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    garment_rotated = cv2.warpAffine(
        garment_canvas,
        rotation_matrix,
        (
            canvas_width,
            canvas_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask_rotated = cv2.warpAffine(
        mask_canvas,
        rotation_matrix,
        (
            canvas_width,
            canvas_height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Final cleanup
    _, mask_rotated = cv2.threshold(
        mask_rotated,
        30,
        255,
        cv2.THRESH_BINARY,
    )

    return garment_rotated, mask_rotated


def composite_garment(
    person_image,
    garment,
    mask,
    region,
):
    """
    Composite the aligned garment onto the person image.
    """

    result = person_image.copy()

    garment_height, garment_width = garment.shape[:2]

    center_x = region["center_x"]
    center_y = region["center_y"]

   
    # Calculate garment placement
   

    x1 = center_x - garment_width // 2
    y1 = center_y - garment_height // 2

    x2 = x1 + garment_width
    y2 = y1 + garment_height

    image_height, image_width = result.shape[:2]

   
    # Clip to image boundaries
   

    x1_clip = max(0, x1)
    y1_clip = max(0, y1)

    x2_clip = min(image_width, x2)
    y2_clip = min(image_height, y2)

    if x1_clip >= x2_clip or y1_clip >= y2_clip:
        raise ValueError(
            "Garment region is outside the image."
        )

   
    # Calculate corresponding garment crop
   

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

   
    # Clean mask
   

    if len(mask_crop.shape) == 3:
        mask_crop = cv2.cvtColor(
            mask_crop,
            cv2.COLOR_BGR2GRAY,
        )

    # Slight blur gives smoother garment boundaries
    mask_crop = cv2.GaussianBlur(
        mask_crop,
        (3, 3),
        0,
    )

    # Convert to alpha
    alpha = (
        mask_crop.astype(np.float32)
        / 255.0
    )

    alpha = np.expand_dims(
        alpha,
        axis=2,
    )

   
    # Blend garment with person
   

    blended = (
        garment_crop.astype(np.float32)
        * alpha
        + roi.astype(np.float32)
        * (1.0 - alpha)
    )

    result[
        y1_clip:y2_clip,
        x1_clip:x2_clip,
    ] = blended.astype(np.uint8)

    return result