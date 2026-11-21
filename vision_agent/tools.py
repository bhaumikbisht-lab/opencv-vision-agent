"""Tools the Vision Agent can use.

Each tool does ONE job and returns plain data (dicts/lists), so the
agent's decisions are easy to test and explain. No tool decides
anything on its own — that is the agent's job.
"""

from __future__ import annotations

import os
import cv2
import numpy as np


# HSV ranges for the colour-blob tool (OpenCV Hue is 0-179)
COLOR_RANGES = {
    "red":   [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green": [((40, 70, 70), (80, 255, 255))],
    "blue":  [((90, 70, 70), (130, 255, 255))],
}


def load_image(path: str):
    """Load an image from disk. Returns None (not a crash) on a bad path."""
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def make_demo_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic test scene with known ground truth.

    Scene: one red, one green and one blue filled circle, plus one
    rectangle outline, on a mid-grey background. Because we drew it,
    we know exactly what a correct agent must find — this is what the
    evaluation harness scores against.
    """
    img = np.full((height, width, 3), 200, dtype=np.uint8)  # grey bg (BGR)
    cv2.circle(img, (140, 140), 45, (0, 0, 255), -1)        # red    (BGR)
    cv2.circle(img, (320, 240), 45, (0, 255, 0), -1)        # green  (BGR)
    cv2.circle(img, (500, 340), 45, (255, 0, 0), -1)        # blue   (BGR)
    cv2.rectangle(img, (60, 330), (220, 430), (0, 0, 0), 3)
    cv2.putText(img, "vision-agent demo", (180, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img


def check_quality(img: np.ndarray) -> dict:
    """Quality gate: brightness + blur, checked BEFORE any detection.

    Blur is measured with the variance of the Laplacian — a sharp
    image has strong edges (high variance), a blurry one does not.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": round(brightness, 1),
        "blur_score": round(blur_score, 1),
        "too_dark": brightness < 40,
        "too_bright": brightness > 220,
        "blurry": blur_score < 100.0,
    }


def enhance(img: np.ndarray, brighter: bool = False, sharpen: bool = False) -> np.ndarray:
    """Simple corrective actions the agent can choose when quality fails."""
    out = img.copy()
    if brighter:
        out = cv2.convertScaleAbs(out, alpha=1.4, beta=40)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out = cv2.filter2D(out, -1, kernel)
    return out


def detect_color_blobs(img: np.ndarray, min_area: int = 500) -> list:
    """Find solid colour blobs (red/green/blue). Returns centre + radius."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    found = []
    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            found.append({"label": name, "x": int(x), "y": int(y),
                          "radius": int(r), "area": int(area)})
    return sorted(found, key=lambda d: d["label"])


def detect_shapes(img: np.ndarray) -> list:
    """Find contours and label them by approximate polygon sides."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        sides = len(approx)
        label = {3: "triangle", 4: "quadrilateral"}.get(sides, "circle/other")
        shapes.append({"label": label, "sides": sides, "x": x, "y": y,
                       "w": w, "h": h, "area": int(area)})
    return shapes


def detect_faces(img: np.ndarray) -> list:
    """Haar-cascade face detection (for real photos / webcam frames)."""
    cascade_path = os.path.join(cv2.data.haarcascades,
                                "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        return []
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(40, 40))
    return [{"label": "face", "x": int(x), "y": int(y),
             "w": int(w), "h": int(h)} for (x, y, w, h) in faces]


def annotate(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw detection boxes/circles + labels on a copy of the image."""
    out = img.copy()
    for d in detections:
        if "radius" in d:  # colour blob
            cv2.circle(out, (d["x"], d["y"]), d["radius"], (0, 0, 0), 2)
            cv2.circle(out, (d["x"], d["y"]), 4, (0, 0, 0), -1)
            text_at = (d["x"] - d["radius"], d["y"] - d["radius"] - 8)
        else:              # face / shape box
            cv2.rectangle(out, (d["x"], d["y"]),
                          (d["x"] + d.get("w", 30), d["y"] + d.get("h", 30)),
                          (0, 0, 0), 2)
            text_at = (d["x"], d["y"] - 8)
        cv2.putText(out, d["label"], text_at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
    return out


def detect_motion(prev_img, curr_img, min_area: int = 800) -> list:
    """Detect moving regions between two frames (frame differencing).

    Compares the previous and current frame: absdiff -> threshold ->
    dilate -> contours. Returns bounding boxes of regions that changed
    enough to count as motion. Identical frames return an empty list.
    Note: frame differencing marks BOTH the old and the new position
    of a moved object — that is expected, the union is the motion area.
    """
    if prev_img is None or curr_img is None:
        return []
    if prev_img.shape != curr_img.shape:
        return []
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    curr_gray = cv2.GaussianBlur(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append({"label": "motion", "x": int(x), "y": int(y),
                      "w": int(w), "h": int(h)})
    return boxes


def save_image(path: str, img: np.ndarray) -> bool:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return bool(cv2.imwrite(path, img))
