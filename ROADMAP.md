# Roadmap

Planned work, roughly in order. Items move to CHANGELOG.md when done.

## Next
- [ ] Background-subtraction motion detection (MOG2) as an alternative
      to frame differencing, with a comparison in the evaluation harness
- [ ] ArUco marker detection tool (robot localisation)
- [ ] Object tracking tool (CSRT/KCF) — follow one object across frames

## Later
- [ ] HSV colour-range auto-calibration helper
- [ ] Optional YOLO/ONNX detector tool behind a feature flag
- [ ] Save short annotated clips from webcam sessions

## Principles
- Tools do one job and return plain data; the agent decides.
- Every new tool ships with a test and a ground-truth evaluation check.
