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


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_")}.items()):
        fn()
        print(f"PASS {name}")
