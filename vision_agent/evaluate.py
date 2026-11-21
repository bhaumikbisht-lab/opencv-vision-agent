"""Evaluation harness for the Vision Agent.

The demo scene has KNOWN ground truth (we drew it), so success is
scored per component, not as one vague pass/fail:

  - colour blobs: 3 expected (red, green, blue) at known centres (±30 px)
  - shapes:       at least 3 contours expected in the demo scene
  - quality gate: a deliberately darkened + blurred copy must be
                  flagged by check_quality (too_dark / blurry)

Run:  python -m vision_agent.evaluate
"""

from __future__ import annotations

import cv2
import numpy as np

from . import tools
from .agent import VisionAgent

EXPECTED_BLOBS = {"red": (140, 140), "green": (320, 240), "blue": (500, 340)}
TOLERANCE_PX = 30


def score_blobs(found: list) -> dict:
    hits, detail = 0, {}
    for label, (ex, ey) in EXPECTED_BLOBS.items():
        match = next((d for d in found if d["label"] == label), None)
        ok = bool(match) and abs(match["x"] - ex) <= TOLERANCE_PX \
            and abs(match["y"] - ey) <= TOLERANCE_PX
        detail[label] = "hit" if ok else "miss"
        hits += int(ok)
    return {"score": hits, "max": len(EXPECTED_BLOBS), "detail": detail}


def main() -> None:
    agent = VisionAgent(session_name="evaluation")
    img = tools.make_demo_image()
    result = agent.process_image(img)

    blobs = [d for d in agent.state.detections if d["kind"] == "color_blob"] \
        if agent.state.detections and "kind" in agent.state.detections[0] else \
        tools.detect_color_blobs(img)
    # state stores kind at top level per detection dict via add_detections
    blob_score = score_blobs(tools.detect_color_blobs(img))
    shapes = tools.detect_shapes(img)

    # Motion: erase the red blob at (140,140), redraw it at (200,140)
    moved = tools.make_demo_image()
    cv2.circle(moved, (140, 140), 45, (200, 200, 200), -1)
    cv2.circle(moved, (200, 140), 45, (0, 0, 255), -1)
    motion_boxes = tools.detect_motion(img, moved)
    motion_ok = any(b["x"] < 245 and b["x"] + b["w"] > 95 and
                    b["y"] < 185 and b["y"] + b["h"] > 95 for b in motion_boxes)

    bad = cv2.convertScaleAbs(img, alpha=0.15, beta=0)          # darken
    bad = cv2.GaussianBlur(bad, (21, 21), 0)                    # blur
    q = tools.check_quality(bad)
    quality_ok = q["too_dark"] and q["blurry"]

    print("=== Vision Agent — evaluation ===")
    print(f"Colour blobs : {blob_score['score']}/{blob_score['max']}  {blob_score['detail']}")
    print(f"Shapes found : {len(shapes)} (expected >= 3)")
    print(f"Motion       : {'PASS' if motion_ok else 'FAIL'}  ({len(motion_boxes)} box(es) on a moved blob)")
    print(f"Quality gate : {'PASS' if quality_ok else 'FAIL'}  "
          f"(dark+blurred copy flagged: dark={q['too_dark']}, blurry={q['blurry']})")
    print(f"Frames: {agent.state.frames_processed} | "
          f"Failures handled: {len(agent.state.failures)}")
    overall = blob_score["score"] == blob_score["max"] and len(shapes) >= 3 and quality_ok and motion_ok
    print(f"OVERALL: {'PASS' if overall else 'CHECK COMPONENTS ABOVE'}")


if __name__ == "__main__":
    main()
