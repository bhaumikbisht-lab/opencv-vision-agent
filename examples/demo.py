"""Smallest possible example: run the agent on the demo scene.

    python examples/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision_agent import VisionAgent

agent = VisionAgent(session_name="example-demo", tasks=("color", "shapes"))
result = agent.run_demo(out_path="outputs/example_annotated.png")
print(result)
print(agent.state.summary())
