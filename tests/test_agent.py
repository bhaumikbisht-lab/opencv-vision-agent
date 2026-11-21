"""Tests for the Vision Agent. Run: python -m pytest tests/ -q
(also runnable without pytest: python tests/test_agent.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision_agent import tools
from vision_agent.agent import VisionAgent


def test_demo_scene_has_three_blobs():
    found = tools.detect_color_blobs(tools.make_demo_image())
    labels = {d["label"] for d in found}
    assert labels == {"red", "green", "blue"}, labels


def test_quality_gate_flags_dark_blur():
    import cv2
    img = tools.make_demo_image()
    bad = cv2.GaussianBlur(cv2.convertScaleAbs(img, alpha=0.15, beta=0), (21, 21), 0)
    q = tools.check_quality(bad)
    assert q["too_dark"] and q["blurry"], q


def test_agent_handles_bad_input():
    agent = VisionAgent(tasks=("color",))
    result = agent.process_image(None)
    assert result["ok"] is False
    assert agent.state.failures, "bad input must be logged as a failure mode"


def test_agent_demo_run_detects():
    agent = VisionAgent(tasks=("color", "shapes"))
    result = agent.process_image(tools.make_demo_image())
    assert result["ok"] and result["detections"] >= 3, result


def _moved_scene():
    import cv2
    a = tools.make_demo_image()
    b = tools.make_demo_image()
    cv2.circle(b, (140, 140), 45, (200, 200, 200), -1)  # erase red blob
    cv2.circle(b, (200, 140), 45, (0, 0, 255), -1)      # redraw it moved
    return a, b


def test_motion_detects_movement():
    a, b = _moved_scene()
    boxes = tools.detect_motion(a, b)
    assert boxes, "moved blob must produce at least one motion box"


def test_motion_quiet_on_same_frame():
    img = tools.make_demo_image()
    assert tools.detect_motion(img, img.copy()) == []


def test_agent_motion_across_frames():
    a, b = _moved_scene()
    agent = VisionAgent(tasks=("motion",))
    first = agent.process_image(a)
    second = agent.process_image(b)
    assert first["detections"] == 0, first
    assert second["detections"] >= 1, second


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_")}.items()):
        fn()
        print(f"PASS {name}")
