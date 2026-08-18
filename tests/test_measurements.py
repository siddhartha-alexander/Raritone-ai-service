import cv2

from app.measurements import calculate_measurements
from app.pose_detector import PoseDetector


image = cv2.imread("sample_images/person.jpg")

if image is None:
    raise FileNotFoundError(
        "Could not load sample_images/person.jpg"
    )

detector = PoseDetector()

landmarks = detector.detect(image)

if landmarks is None:
    raise RuntimeError("No person detected.")

measurements = calculate_measurements(landmarks)

print("Body proportions:")

for name, value in measurements.items():
    print(f"{name}: {value:.4f}")