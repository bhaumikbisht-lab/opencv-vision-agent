"""Run the OpenCV Vision Agent.

Examples:
  python run_agent.py --source demo
  python run_agent.py --source image --path photo.jpg
  python run_agent.py --source webcam --frames 200
  python run_agent.py --source demo --tasks color shapes
"""

import argparse
import json

from vision_agent import VisionAgent


def main():
    p = argparse.ArgumentParser(description="OpenCV Vision Agent")
    p.add_argument("--source", choices=["demo", "image", "webcam"], default="demo")
    p.add_argument("--path", default="", help="image path (for --source image)")
    p.add_argument("--frames", type=int, default=100, help="webcam frame limit")
    p.add_argument("--tasks", nargs="*", default=["color", "shapes", "faces", "motion"],
                   choices=["color", "shapes", "faces", "motion"])
    p.add_argument("--out", default="outputs/annotated.png")
    p.add_argument("--state", default="outputs/session_state.json")
    args = p.parse_args()

    agent = VisionAgent(session_name=f"vision-{args.source}",
                        tasks=tuple(args.tasks))
    if args.source == "demo":
        result = agent.run_demo(out_path=args.out)
    elif args.source == "image":
        result = agent.run_file(args.path, out_path=args.out)
    else:
        result = agent.run_webcam(max_frames=args.frames)

    result.pop("annotated", None)
    print(json.dumps(result, indent=2))
    print(json.dumps(agent.state.summary(), indent=2))
    agent.state.save(args.state)
    print(f"Annotated image -> {args.out} | Session state -> {args.state}")


if __name__ == "__main__":
    main()
