"""
Local face screen run before the vision API: keeps only single, large, forward-facing faces.

Uses the MediaPipe Tasks FaceLandmarker (the face-mesh model, 478 landmarks). The legacy
mediapipe.solutions.face_mesh API was removed in mediapipe 0.10.30 and isn't installable on
Python 3.13+. Head pose comes from the landmarker's facial transformation matrix.

Rejection reasons, checked in this order:
  overexposed (mean grayscale brightness above a threshold; checked before the model runs),
  no_face, multiple_faces, face_too_small, yaw_out_of_range, pitch_out_of_range
"""

import math
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path.home() / ".cache" / "fairface_harness" / "face_landmarker.task"

REASONS = ["overexposed", "no_face", "multiple_faces", "face_too_small", "yaw_out_of_range", "pitch_out_of_range"]


def is_overexposed(image, threshold: float = 200) -> tuple[bool, float]:
    """(too bright?, mean brightness 0-255) for a PIL image, from its grayscale mean."""
    mean_brightness = float(np.array(image.convert("L")).mean())
    return mean_brightness > threshold, mean_brightness


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the FaceLandmarker model (~4 MB) once."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(MODEL_URL, follow_redirects=True, timeout=120)
        response.raise_for_status()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(response.content)
        tmp.replace(path)
    return path


def head_pose(matrix) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 4x4 (or 3x3) rotation matrix. 0, 0, 0 is facing the camera."""
    r = np.asarray(matrix, dtype=float)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


@dataclass
class FaceCheck:
    ok: bool
    reason: str = ""  # "" when ok, otherwise one of REASONS
    n_faces: int = 0
    face_area: float | None = None  # landmark bounding box as a fraction of the image area
    yaw: float | None = None
    pitch: float | None = None
    brightness: float | None = None  # mean grayscale value, 0-255


def evaluate(face_landmarks, matrices, max_yaw: float, max_pitch: float, min_face_area: float) -> FaceCheck:
    """The checks themselves, on a FaceLandmarker result's fields (separate from the model, so it can be tested)."""
    n = len(face_landmarks)
    if n == 0:
        return FaceCheck(False, "no_face")
    if n > 1:
        return FaceCheck(False, "multiple_faces", n_faces=n)

    xs = np.clip([p.x for p in face_landmarks[0]], 0, 1)
    ys = np.clip([p.y for p in face_landmarks[0]], 0, 1)
    area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    yaw, pitch, _ = head_pose(matrices[0]) if matrices else (None, None, None)
    check = FaceCheck(True, n_faces=1, face_area=area, yaw=yaw, pitch=pitch)

    if area < min_face_area:
        check.ok, check.reason = False, "face_too_small"
    elif yaw is None or abs(yaw) > max_yaw:
        check.ok, check.reason = False, "yaw_out_of_range"
    elif abs(pitch) > max_pitch:
        check.ok, check.reason = False, "pitch_out_of_range"
    return check


class FacePrefilter:
    def __init__(self, max_yaw: float = 15, max_pitch: float = 10, min_face_area: float = 0.15,
                 max_brightness: float = 200):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.max_yaw, self.max_pitch, self.min_face_area = max_yaw, max_pitch, min_face_area
        self.max_brightness = max_brightness
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,  # 2, not 1, so a second face is detected and the image can be rejected
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def check(self, image) -> FaceCheck:
        """image: a PIL image."""
        too_bright, brightness = is_overexposed(image, self.max_brightness)
        if too_bright:
            return FaceCheck(False, "overexposed", brightness=brightness)
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        check = evaluate(result.face_landmarks, result.facial_transformation_matrixes,
                         self.max_yaw, self.max_pitch, self.min_face_area)
        check.brightness = brightness
        return check

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
