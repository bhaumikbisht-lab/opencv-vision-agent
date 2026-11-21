"""OpenCV Vision Agent — a small, inspectable vision agent.

Loop: Observe -> Decide -> Act -> Verify, with explicit state/memory.
"""

from .agent import VisionAgent
from .state import AgentState

__all__ = ["VisionAgent", "AgentState"]
__version__ = "1.1.0"
