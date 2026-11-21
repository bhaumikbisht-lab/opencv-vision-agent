"""The Vision Agent itself.

A deliberate, inspectable loop — no black-box framework:

    OBSERVE  -> read a frame / image and check its quality
    DECIDE   -> if quality failed, choose a corrective action first;
                otherwise choose which detectors to run (task-driven)
    ACT      -> run the chosen tools
    VERIFY   -> detections are only written to memory after they are
                confirmed (non-empty, inside the frame); a zero result
                is logged as a failure mode, not silently ignored

Tasks: "all" (default), "color", "shapes", "faces", "motion"
"""

from __future__ import annotations

import cv2
import numpy as np

from . import tools
from .state import AgentState


class VisionAgent:
    def __init__(self, session_name: str = "vision-session",
                 tasks: tuple = ("color", "shapes", "faces", "motion")):
        self.state = AgentState(session_name=session_name)
        self.tasks = tasks
        self._prev_frame = None  # last raw frame, for motion detection

    # ---- one full loop on one image -------------------------------------
    def process_image(self, img) -> dict:
        if img is None:
            self.state.log_failure("bad_input", "image was None (bad path / no frame)")
            return {"ok": False, "reason": "bad_input"}

        self.state.frames_processed += 1

        # OBSERVE
        quality = tools.check_quality(img)
        self.state.quality_log.append(quality)
        self.state.log_action("observe", f"brightness={quality['brightness']} "
                                         f"blur={quality['blur_score']}")

        # DECIDE + ACT (correct first, if the quality gate failed)
        if quality["too_dark"] or quality["blurry"]:
            img = tools.enhance(img, brighter=quality["too_dark"],
                                sharpen=quality["blurry"])
            self.state.log_action("enhance",
                                  f"dark={quality['too_dark']} blurry={quality['blurry']}")
            if quality["too_dark"]:
                self.state.log_failure("too_dark", "corrected with brightness boost")
            if quality["blurry"]:
                self.state.log_failure("blurry", "corrected with sharpening")

        # MOTION (needs the previous frame — kept in agent memory)
        motion_items = []
        if "motion" in self.tasks:
            if self._prev_frame is not None:
                motion_items = tools.detect_motion(self._prev_frame, img)
            else:
                self.state.log_action("motion", "no previous frame yet — skipped")
        self._prev_frame = img.copy()

        # ACT (detectors chosen by task)
        found_by_kind = {}
        if "motion" in self.tasks:
            found_by_kind["motion"] = motion_items
        if "color" in self.tasks:
            found_by_kind["color_blob"] = tools.detect_color_blobs(img)
        if "shapes" in self.tasks:
            found_by_kind["shape"] = tools.detect_shapes(img)
        if "faces" in self.tasks:
            found_by_kind["face"] = tools.detect_faces(img)

        # VERIFY + remember (only confirmed, in-frame detections)
        h, w = img.shape[:2]
        total = 0
        for kind, items in found_by_kind.items():
            confirmed = [d for d in items if 0 <= d["x"] < w and 0 <= d["y"] < h]
            if items and not confirmed:
                self.state.log_failure("out_of_frame", f"{kind} detections rejected")
            self.state.add_detections(kind, confirmed)
            total += len(confirmed)
            self.state.log_action("detect", f"{kind}: {len(confirmed)} confirmed")
        if total == 0:
            self.state.log_failure("no_detection", "no tool confirmed any detection")

        annotated = tools.annotate(img, self.state.detections[-total:] if total else [])
        return {"ok": True, "detections": total, "by_kind":
                {k: len(v) for k, v in found_by_kind.items()},
                "quality": quality, "annotated": annotated}

    # ---- run modes --------------------------------------------------------
    def run_demo(self, out_path: str = "outputs/demo_annotated.png") -> dict:
        """Run on the synthetic scene (no camera or file needed)."""
        img = tools.make_demo_image()
        result = self.process_image(img)
        if result.get("annotated") is not None:
            tools.save_image(out_path, result.pop("annotated"))
        return result

    def run_file(self, path: str, out_path: str = "outputs/file_annotated.png") -> dict:
        img = tools.load_image(path)
        result = self.process_image(img)
        if result.get("ok") and result.get("annotated") is not None:
            tools.save_image(out_path, result.pop("annotated"))
        return result

    def run_webcam(self, max_frames: int = 100, camera_index: int = 0,
                   show: bool = True) -> dict:
        """Process webcam frames until max_frames or 'q' is pressed."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            self.state.log_failure("no_camera", f"camera {camera_index} did not open")
            return {"ok": False, "reason": "no_camera"}
        done = 0
        while done < max_frames:
            ok, frame = cap.read()
            if not ok:
                self.state.log_failure("frame_drop", "camera returned no frame")
                break
            result = self.process_image(frame)
            done += 1
            if show and result.get("annotated") is not None:
                cv2.imshow("Vision Agent — press q to quit", result["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if show:
            cv2.destroyAllWindows()
        return {"ok": True, "frames": done}
