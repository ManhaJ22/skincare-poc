import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image  # noqa: E402

from face_prefilter import evaluate, head_pose, is_overexposed  # noqa: E402


def rotation(yaw=0.0, pitch=0.0):
    """4x4 matrix: pitch about x, then yaw about y (degrees)."""
    p, y = math.radians(pitch), math.radians(yaw)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    return m


def face(x0=0.1, y0=0.1, x1=0.9, y1=0.9):
    """A fake landmark list whose bounding box is (x0, y0)-(x1, y1)."""
    return [SimpleNamespace(x=x0, y=y0), SimpleNamespace(x=x1, y=y1), SimpleNamespace(x=0.5, y=0.5)]


@pytest.mark.parametrize("yaw, pitch", [(0, 0), (20, 0), (-30, 0), (0, 12), (0, -8), (10, -5)])
def test_head_pose_recovers_angles(yaw, pitch):
    got_yaw, got_pitch, _ = head_pose(rotation(yaw, pitch))
    assert got_yaw == pytest.approx(yaw, abs=1e-6)
    assert got_pitch == pytest.approx(pitch, abs=1e-6)


LIMITS = {"max_yaw": 15, "max_pitch": 10, "min_face_area": 0.15}


@pytest.mark.parametrize(
    "landmarks, matrices, reason",
    [
        ([], [], "no_face"),
        ([face(), face()], [rotation(), rotation()], "multiple_faces"),
        ([face(0.4, 0.4, 0.6, 0.6)], [rotation()], "face_too_small"),        # 4% of the image
        ([face(0.4, 0.4, 0.6, 0.6)], [rotation(yaw=40)], "face_too_small"),  # size is checked before pose
        ([face()], [rotation(yaw=16)], "yaw_out_of_range"),
        ([face()], [rotation(yaw=-20, pitch=20)], "yaw_out_of_range"),        # yaw before pitch
        ([face()], [rotation(pitch=-11)], "pitch_out_of_range"),
        ([face()], [rotation(yaw=14, pitch=9)], ""),
        ([face(-0.2, -0.2, 0.5, 0.5)], [rotation()], ""),                     # off-image landmarks clipped: 25%
    ],
)
def test_evaluate_reasons_in_order(landmarks, matrices, reason):
    check = evaluate(landmarks, matrices, **LIMITS)
    assert check.reason == reason
    assert check.ok == (reason == "")


def test_evaluate_reports_metrics():
    check = evaluate([face()], [rotation(yaw=5, pitch=-3)], **LIMITS)
    assert check.n_faces == 1
    assert check.face_area == pytest.approx(0.64)
    assert check.yaw == pytest.approx(5) and check.pitch == pytest.approx(-3)


@pytest.mark.parametrize("value, threshold, expected", [
    (250, 200, True),
    (128, 200, False),
    (200, 200, False),   # strictly greater than the threshold
    (180, 170, True),    # custom threshold
])
def test_is_overexposed(value, threshold, expected):
    image = Image.new("RGB", (32, 32), (value, value, value))
    too_bright, brightness = is_overexposed(image, threshold)
    assert too_bright is expected
    assert brightness == pytest.approx(value)
