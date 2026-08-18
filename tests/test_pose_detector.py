import cv2

from app.pose_detector import PoseDetector


image = cv2.imread("sample_images/person.jpg")

if image is None:
    raise FileNotFoundError(
        "Could not load sample_images/person.jpg"
    )

detector = PoseDetector()

landmarks = detector.detect(image)

if landmarks is None:
    print("No person detected.")
else:
    print("Person detected!")
    print(f"Landmarks returned: {len(landmarks)}")

    for name, landmark in landmarks.items():
        print(name, landmark)