"""Replay of recorded decisions.

Running a simulation with every agent driven by :class:`ReplayPolicy`
re-creates a recorded run exactly, without calling any decision model. This is
how a run that involved an LLM can be re-examined later: the LLM's outputs are
part of the record, so the trajectory no longer depends on the model being
available or deterministic.
"""
from __future__ import annotations

from collections import deque

from ..actions import Decision
from .base import Policy


class ReplayDivergence(RuntimeError):
    """The replayed run reached a decision point that differs from the recording."""


class ReplayPolicy(Policy):
    name = "replay"

    def __init__(self, records: list, strict: bool = True, **params):
        super().__init__(**params)
        self._records = deque(records)
        self.strict = strict

    def decide(self, obs):
        if not self._records:
            raise ReplayDivergence(
                f"{obs.agent_id}: decision point at tick {obs.tick} ({obs.trigger}) has no recorded counterpart"
            )
        rec = self._records.popleft()
        if rec["tick"] != obs.tick or rec["trigger"] != obs.trigger:
            raise ReplayDivergence(
                f"{obs.agent_id}: expected {rec['trigger']} at tick {rec['tick']}, got {obs.trigger} at tick {obs.tick}"
            )
        if self.strict and rec.get("obs_digest") and rec["obs_digest"] != obs.digest():
            raise ReplayDivergence(
                f"{obs.agent_id}: observation at tick {obs.tick} ({obs.trigger}) differs from the recording"
            )
        d = rec["decision"]
        return Decision(d["action"], dict(d.get("args") or {}), d.get("rationale", ""), {"replayed": True})

    def remaining(self) -> int:
        return len(self._records)
