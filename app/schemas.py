from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: str


class Landmark(BaseModel):
    x: float
    y: float
    visibility: float


class Measurements(BaseModel):
    shoulder_ratio: float
    hip_ratio: float
    left_arm_ratio: float
    right_arm_ratio: float
    left_leg_ratio: float
    right_leg_ratio: float
    torso_ratio: float
    shoulder_to_hip_ratio: float


class AIResponse(BaseModel):
    success: bool
    person_detected: bool
    landmarks: dict[str, Landmark]
    measurements: Optional[Measurements] = None
    model_version: str
    processing_time: float