# OpenCV Vision Agent

A small, inspectable **computer-vision agent** built with Python + OpenCV.
It watches an image or webcam feed, decides what to do, runs vision tools,
verifies the result, and remembers everything in explicit state.
No black-box framework — the whole loop fits in one readable file.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x%20%2F%205.x-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

## How it works
```
OBSERVE  read frame + quality gate (brightness, blur)
DECIDE   quality failed? correct first. Otherwise pick detectors by task
ACT      colour-blob tracker · shape/contour detector · Haar face detector
VERIFY   only in-frame, confirmed detections are written to memory;
         zero/failed results are logged as failure modes, not ignored
```
State (`vision_agent/state.py`) holds frames processed, all confirmed
detections, actions taken, failures handled and per-frame quality —
and saves to JSON, so any session can be replayed and audited.

## Quick start
```bash
pip install -r requirements.txt
python run_agent.py --source demo            # no camera/file needed
python run_agent.py --source image --path photo.jpg
python run_agent.py --source webcam --frames 200   # press q to quit
python -m vision_agent.evaluate              # evaluation harness
python tests/test_agent.py                   # tests
python examples/demo.py                      # minimal example
```

## Evaluation
The demo scene is synthetic with **known ground truth** (3 colour blobs
at known centres, ≥3 shapes, and a dark+blurred copy the quality gate
must flag), so the harness scores each component separately:

```
Colour blobs : 3/3   {'red': 'hit', 'green': 'hit', 'blue': 'hit'}
Shapes found : 4 (expected >= 3)
Quality gate : PASS
OVERALL: PASS
```

## Failure modes handled
| Failure | Mitigation |
|---|---|
| Bad path / no frame | `load_image` returns None; agent logs `bad_input`, does not crash |
| Too dark / blurry frame | Quality gate flags it; agent brightens / sharpens before detecting |
| No camera | Logged as `no_camera` with a clean exit |
| Nothing detected | Logged as `no_detection` failure mode in state |

## Project structure
```
run_agent.py            CLI entry point
config.example.json     copy to config.json and make it yours
vision_agent/
  agent.py              the Observe→Decide→Act→Verify loop
  tools.py              colour, shape, face, quality & annotation tools
  state.py              session memory (JSON save/load)
  evaluate.py           evaluation harness (known ground truth)
tests/test_agent.py     4 tests
examples/demo.py        minimal usage example
```

## Extending it
Add a tool in `tools.py` (one job, returns plain data), wire it into
the DECIDE step in `agent.py`, and add a ground-truth check in
`evaluate.py`. See `CONTRIBUTING.md`.

## License
MIT — see `LICENSE`. Originally built by Bhaumik Bisht for robotics
perception practice; free for anyone to learn from and build on.
