"""Memory / state for the Vision Agent.

The agent keeps one explicit state object for the whole session:
what it has seen, what it decided, what it found, and what failed.
State is only updated from *confirmed* tool results, never guesses,
and it can be saved to JSON so a session can be replayed / audited.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict


@dataclass
class AgentState:
    session_name: str = "vision-session"
    frames_processed: int = 0
    detections: list = field(default_factory=list)   # every confirmed detection
    actions: list = field(default_factory=list)      # every action the agent took
    failures: list = field(default_factory=list)     # failure modes seen + handled
    quality_log: list = field(default_factory=list)  # per-frame quality checks
    started_at: float = field(default_factory=time.time)

    def log_action(self, action: str, detail: str = "") -> None:
        self.actions.append({"t": round(time.time() - self.started_at, 3),
                             "action": action, "detail": detail})

    def log_failure(self, kind: str, detail: str = "") -> None:
        self.failures.append({"t": round(time.time() - self.started_at, 3),
                              "kind": kind, "detail": detail})

    def add_detections(self, kind: str, items: list) -> None:
        for it in items:
            self.detections.append({"kind": kind, **it})

    def summary(self) -> dict:
        return {
            "session": self.session_name,
            "frames_processed": self.frames_processed,
            "total_detections": len(self.detections),
            "actions_taken": len(self.actions),
            "failures_handled": len(self.failures),
            "elapsed_sec": round(time.time() - self.started_at, 2),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AgentState":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
