"""Replay and verification of recorded runs.

``replay(run_dir)`` re-executes a run with every decision taken from the
trace instead of from a decision model, and checks after every recorded tick
that the state fingerprint matches. A run that involved an LLM can therefore
be reproduced exactly without the model, and any change in the simulator's
dynamics is detected at the first tick where it matters.
"""
from __future__ import annotations

from .config import Config
from .engine import Simulation
from .policies.replay import ReplayDivergence
from .recorder import Recorder, read_trace


def replay(run_dir) -> dict:
    records = read_trace(run_dir)
    header = records[0]
    if header.get("type") != "header":
        raise ValueError("trace does not start with a header record")
    data = dict(header["config"])
    data["plugins"] = header.get("plugins", [])
    cfg = Config.from_dict(data)
    cfg.record.dir = None
    by_agent: dict[str, list] = {}
    for rec in records:
        if rec["type"] == "decision" and rec["source"] != "forced":
            by_agent.setdefault(rec["agent"], []).append(rec)
    recorded = [(rec["tick"], rec["digest"]) for rec in records if rec["type"] == "state"]
    summary = next((rec for rec in records if rec["type"] == "summary"), {})

    sim = Simulation(cfg, replay=by_agent, recorder=Recorder(None, keep=False))
    report = {"run": str(run_dir), "ticks": sim.n_ticks, "decisions_replayed": sum(len(v) for v in by_agent.values()),
              "identical": False, "first_divergence_tick": None, "error": None}
    try:
        sim.run()
    except ReplayDivergence as exc:
        report["error"] = str(exc)
        report["first_divergence_tick"] = sim.world.tick
        return report
    replayed = sim._state_chain
    for (tick, digest), again in zip(recorded, replayed):
        if digest != again:
            report["first_divergence_tick"] = tick
            return report
    report["identical"] = len(recorded) == len(replayed) and sim.trace_digest() == summary.get("trace_digest")
    report["unused_decisions"] = sum(p.remaining() for p in sim.policies.values() if hasattr(p, "remaining"))
    return report
