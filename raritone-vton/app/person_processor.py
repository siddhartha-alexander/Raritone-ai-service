import cv2
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from validation import validate_image


class PersonProcessor:
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, image_path: str, output_dir: str) -> dict:
        validation_result = validate_image(image_path)

        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"]
            }

        image = cv2.imread(image_path)

        if image is None:
            return {
                "success": False,
                "error": "Unable to read image."
            }

        height, width, _ = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.detector.detect(mp_image)

        if not result.pose_landmarks:
            return {
                "success": False,
                "person_detected": False,
                "error": "No person detected."
            }

        if len(result.pose_landmarks) > 1:
            return {
                "success": False,
                "person_detected": True,
                "person_count": len(result.pose_landmarks),
                "error": "Multiple people detected. Please upload an image containing only one person."
            }

        landmarks_raw = result.pose_landmarks[0]

        important_indices = {
            "nose": 0,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28
        }

        landmarks = {}

        for name, index in important_indices.items():
            lm = landmarks_raw[index]

            landmarks[name] = {
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
                "visibility": round(getattr(lm, "visibility", 0.0), 4),
                "pixel_x": int(lm.x * width),
                "pixel_y": int(lm.y * height)
            }

        debug_image = image.copy()

        connections = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28)
        ]

        for start_idx, end_idx in connections:
            start = landmarks_raw[start_idx]
            end = landmarks_raw[end_idx]

            p1 = (
                int(start.x * width),
                int(start.y * height)
            )

            p2 = (
                int(end.x * width),
                int(end.y * height)
            )

            cv2.line(debug_image, p1, p2, (0, 255, 0), 2)

        for lm in landmarks_raw:
            x = int(lm.x * width)
            y = int(lm.y * height)

            if 0 <= x < width and 0 <= y < height:
                cv2.circle(debug_image, (x, y), 3, (0, 0, 255), -1)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pose_output = output_path / "person_pose.jpg"

        cv2.imwrite(str(pose_output), debug_image)

        return {
            "success": True,
            "person_detected": True,
            "person_count": 1,
            "image_width": width,
            "image_height": height,
            "landmarks": landmarks,
            "pose_output": str(pose_output)
        }


if __name__ == "__main__":
    processor = PersonProcessor(
        model_path="raritone-vton/models/pose_landmarker_lite.task"
    )

    image_path = input("Enter person image path: ")

    result = processor.process(
        image_path=image_path,
        output_dir="raritone-vton/dataset/poses"
    )

    print(result)