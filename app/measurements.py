import math


def distance(point1, point2):
    """Calculate 2D Euclidean distance between two landmarks."""
    return math.sqrt(
        (point2["x"] - point1["x"]) ** 2
        + (point2["y"] - point1["y"]) ** 2
    )


def calculate_measurements(landmarks):
    """Calculate relative body proportions from pose landmarks."""

    shoulder_width = distance(
        landmarks["left_shoulder"],
        landmarks["right_shoulder"],
    )

    hip_width = distance(
        landmarks["left_hip"],
        landmarks["right_hip"],
    )

    left_arm = (
        distance(
            landmarks["left_shoulder"],
            landmarks["left_elbow"],
        )
        + distance(
            landmarks["left_elbow"],
            landmarks["left_wrist"],
        )
    )

    right_arm = (
        distance(
            landmarks["right_shoulder"],
            landmarks["right_elbow"],
        )
        + distance(
            landmarks["right_elbow"],
            landmarks["right_wrist"],
        )
    )

    left_leg = (
        distance(
            landmarks["left_hip"],
            landmarks["left_knee"],
        )
        + distance(
            landmarks["left_knee"],
            landmarks["left_ankle"],
        )
    )

    right_leg = (
        distance(
            landmarks["right_hip"],
            landmarks["right_knee"],
        )
        + distance(
            landmarks["right_knee"],
            landmarks["right_ankle"],
        )
    )

    shoulder_midpoint = {
        "x": (
            landmarks["left_shoulder"]["x"]
            + landmarks["right_shoulder"]["x"]
        ) / 2,
        "y": (
            landmarks["left_shoulder"]["y"]
            + landmarks["right_shoulder"]["y"]
        ) / 2,
    }

    hip_midpoint = {
        "x": (
            landmarks["left_hip"]["x"]
            + landmarks["right_hip"]["x"]
        ) / 2,
        "y": (
            landmarks["left_hip"]["y"]
            + landmarks["right_hip"]["y"]
        ) / 2,
    }

    torso = distance(
        shoulder_midpoint,
        hip_midpoint,
    )

    shoulder_to_hip_ratio = (
        shoulder_width / hip_width
        if hip_width != 0
        else 0
    )

    return {
        "shoulder_ratio": shoulder_width,
        "hip_ratio": hip_width,
        "left_arm_ratio": left_arm,
        "right_arm_ratio": right_arm,
        "left_leg_ratio": left_leg,
        "right_leg_ratio": right_leg,
        "torso_ratio": torso,
        "shoulder_to_hip_ratio": shoulder_to_hip_ratio,
    }