# Changelog

All notable changes to this project are documented here.

## [1.1.0] — 2026-11-21
### Added
- Motion detection tool (`detect_motion`) using frame differencing:
  absdiff -> threshold -> dilate -> contours, with a minimum-area
  noise filter.
- The agent now keeps the previous frame in memory, so motion is
  detected frame-to-frame in webcam/live mode.
- 3 new tests (7 total) and a Motion component in the evaluation
  harness — moved-blob ground truth, PASS.
- GitHub Actions CI: tests + evaluation run on every push.

## [1.0.0] — 2026-11-21
### Added
- Initial release: Observe -> Decide -> Act -> Verify agent loop.
- Tools: colour-blob tracker, shape/contour detector, Haar face
  detector, image quality gate (brightness + blur), annotation.
- Explicit session state with JSON save/load.
- Evaluation harness scored against known ground truth.
- Demo mode (synthetic scene — no camera or file needed).
