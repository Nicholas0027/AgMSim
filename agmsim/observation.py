"""Observations: what an agent is shown when it is asked to decide.

An :class:`Observation` is the *only* channel from the world to a decision
model. It is a plain, JSON-serialisable structure, so the very same object is
handed to a rule, an optimiser, a learned policy, an LLM prompt or a remote
agent. It states

* ``trigger`` -- why the agent is being asked now;
* ``state``   -- the agent's own state (persona, objective, position, ...);
* ``context`` -- trigger-specific information within the agent's observation scope;
* ``actions`` -- the actions the agent is permitted to take right now.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from .actions import ActionSpec, Decision, Param

OBJECTIVES = {
    "profit": "Maximise your own net earnings (income minus costs).",
    "service": "Maximise service quality for the people you serve: keep their waiting short and serve as many as possible.",
    "balanced": "Balance your own net earnings with good service to the people you serve.",
}


def minutes(seconds: Optional[float]) -> Optional[float]:
    return None if seconds is None else round(seconds / 60.0, 1)


def km(metres: float) -> float:
    return round(metres / 1000.0, 2)


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@dataclass
class Observation:
    agent_id: str
    kind: str                     # traveler | vehicle | bus | evcs | operator | <plugin>
    tick: int
    time_s: float
    trigger: str
    state: dict
    context: dict
    actions: tuple
    default: Decision
    aux: dict = field(default_factory=dict, repr=False)   # engine-private; never shown to decision models

    def has(self, action: str) -> bool:
        return any(spec.name == action for spec in self.actions)

    def action(self, name: str) -> Optional[ActionSpec]:
        return next((spec for spec in self.actions if spec.name == name), None)

    def choices(self, action: str, param: str) -> list:
        """Allowed values of ``param`` for ``action`` (empty if not permitted)."""
        spec = self.action(action)
        p = spec.param(param) if spec else None
        return list(p.choices) if p is not None and p.choices is not None else []

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        """Rebuild an observation from its dict form (as shown to a model, or stored in a trace).

        The default action is not part of that form; it is taken to be the first permitted action
        with the first allowed value of each required argument, which is how the built-in kinds
        order their actions.
        """
        actions = []
        for item in data["permitted_actions"]:
            params = tuple(Param(name, spec.get("type", "string"),
                                 tuple(spec["choices"]) if "choices" in spec else None,
                                 spec.get("required", True), spec.get("description", ""))
                           for name, spec in (item.get("args") or {}).items())
            actions.append(ActionSpec(item["action"], item.get("description", ""), params))
        first = actions[0]
        default = Decision(first.name, {p.name: p.choices[0] for p in first.params if p.choices and p.required})
        return cls(data["agent_id"], data["kind"], data["tick"], float(data.get("time_s", 0.0)), data["trigger"],
                   data["state"], data["context"], tuple(actions), default)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "kind": self.kind,
            "time": clock(self.time_s),
            "tick": self.tick,
            "trigger": self.trigger,
            "state": self.state,
            "context": self.context,
            "permitted_actions": [spec.to_dict() for spec in self.actions],
        }

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
