# Contributing

Contributions are welcome — new vision tools, better detectors,
tests, and docs.

1. Fork the repo and create a feature branch.
2. Keep the design rule: **tools do one job and return plain data;
   the agent decides.** No decision logic inside tools.
3. Add or update a test for any new tool (`tests/test_agent.py`).
4. Make sure the evaluation harness still passes:
   `python -m vision_agent.evaluate`
5. Open a pull request describing what changed and how you tested it.

Good first issues: a new colour range, a background-subtraction
(MOG2) motion tool, or a YOLO/ONNX detector added as an optional tool.
