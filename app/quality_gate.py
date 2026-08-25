MIN_POSE_QUALITY = 0.60
MIN_MASK_QUALITY = 0.10
MIN_GARMENT_QUALITY = 0.10


class QualityGateError(Exception):
    """
    Exception raised when an input fails the VTON quality gate.
    """

    def __init__(
        self,
        error_code,
        message,
        quality=None,
    ):
        self.error_code = error_code
        self.message = message
        self.quality = quality or {}

        super().__init__(message)


def validate_person_quality(person_result):
    """
    Validate person preprocessing output.
    """

    if not person_result:
        raise QualityGateError(
            "INVALID_PERSON",
            "Person preprocessing returned no result.",
        )

    if not person_result.get("valid", False):
        raise QualityGateError(
            person_result.get(
                "error_code",
                "PERSON_QUALITY_FAILED",
            ),
            person_result.get(
                "message",
                "Person image failed quality validation.",
            ),
            {
                "pose": person_result.get(
                    "pose_quality",
                    0.0,
                ),
                "mask": person_result.get(
                    "mask_quality",
                    0.0,
                ),
            },
        )

    pose_quality = float(
        person_result.get(
            "pose_quality",
            0.0,
        )
    )

    mask_quality = float(
        person_result.get(
            "mask_quality",
            0.0,
        )
    )

    person_count = int(
        person_result.get(
            "person_count",
            0,
        )
    )

    if person_count != 1:
        raise QualityGateError(
            "INVALID_PERSON_COUNT",
            "Please upload an image containing exactly one person.",
            {
                "pose": pose_quality,
                "mask": mask_quality,
            },
        )

    if pose_quality < MIN_POSE_QUALITY:
        raise QualityGateError(
            "LOW_POSE_QUALITY",
            (
                "Please upload a clear full-body image "
                "with the person facing the camera."
            ),
            {
                "pose": pose_quality,
                "mask": mask_quality,
            },
        )

    if mask_quality < MIN_MASK_QUALITY:
        raise QualityGateError(
            "LOW_MASK_QUALITY",
            (
                "The person could not be separated clearly "
                "from the background."
            ),
            {
                "pose": pose_quality,
                "mask": mask_quality,
            },
        )

    return True


def validate_garment_quality(garment_result):
    """
    Validate garment preprocessing output.
    """

    if not garment_result:
        raise QualityGateError(
            "INVALID_GARMENT",
            "Garment preprocessing returned no result.",
        )

    if not garment_result.get(
        "valid",
        False,
    ):
        raise QualityGateError(
            garment_result.get(
                "error_code",
                "GARMENT_QUALITY_FAILED",
            ),
            garment_result.get(
                "message",
                "Garment image failed quality validation.",
            ),
            {
                "garment": garment_result.get(
                    "garment_quality",
                    0.0,
                )
            },
        )

    garment_quality = float(
        garment_result.get(
            "garment_quality",
            0.0,
        )
    )

    if garment_quality < MIN_GARMENT_QUALITY:
        raise QualityGateError(
            "LOW_GARMENT_QUALITY",
            (
                "Please upload a clear garment image "
                "with a simple background."
            ),
            {
                "garment": garment_quality,
            },
        )

    if not garment_result.get(
        "mask_available",
        False,
    ):
        raise QualityGateError(
            "GARMENT_MASK_MISSING",
            "A valid garment mask could not be generated.",
            {
                "garment": garment_quality,
            },
        )

    return True


def run_quality_gate(
    person_result,
    garment_result,
):
    """
    Final quality gate before VTON inference.

    Bad input -> reject
    Good input -> allow inference
    """

    validate_person_quality(
        person_result
    )

    validate_garment_quality(
        garment_result
    )

    return {
        "passed": True,
        "quality": {
            "pose": round(
                float(
                    person_result[
                        "pose_quality"
                    ]
                ),
                4,
            ),
            "mask": round(
                float(
                    person_result[
                        "mask_quality"
                    ]
                ),
                4,
            ),
            "garment": round(
                float(
                    garment_result[
                        "garment_quality"
                    ]
                ),
                4,
            ),
        },
    }