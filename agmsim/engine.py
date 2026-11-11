"""The simulation engine.

One tick advances the state by ``S_{t+dt} = F(S_t, A_t, xi_t)`` in a fixed,
documented order:

1. exogenous events (interventions, new trip needs);
2. every agent that has a decision point observes the **same** state ``S_t``;
3. decisions are produced -- by the agent's decision model where its agency
   gives it a say, by the default action where it does not;
4. actions are validated and applied kind by kind (operator, charging
   stations, travellers, buses, vehicles), in agent-id order within a kind;
5. interactions are settled;
6. physics: congestion, movement, boarding, charging, timers;
7. the tick is recorded (decisions, events, state fingerprint).

Decision timing is event-driven and independent of the time step: an agent is
consulted only when one of its triggers fires.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from . import kinds as _kinds  # noqa: F401  (registers built-in kinds)
from . import policies as _policies  # noqa: F401  (registers built-in policies)
from .actions import Decision, validate_decision
from .agency import parse_agency
from .config import Config, LLMCfg
from .kinds.base import KINDS
from .llm import build_llm_client
from .network import build_network
from .observation import Observation
from .plans import parse_clock
from .policies.base import PolicyContext, create_policy, instantiate
from .policies.replay import ReplayPolicy
from .recorder import Recorder
from .registry import load_plugins
from .rng import RngStreams
from .world import World


UPDATE_ORDER = [
    "exogenous events (interventions, new trip needs)",
    "all due agents observe the same state S_t",
    "decisions: the agent's decision model where it has authority, the default action where it has not",
    "actions applied kind by kind (operator, charging stations, travellers, buses, vehicles), by agent id within a kind",
    "interactions settled",
    "physics: congestion, movement, boarding, charging, timers",
]


@dataclass
class AgentSpec:
    kind: str
    use_llm: bool
    policy: str                       # name of the decision model actually used
    objective: str = ""
    non_llm_policy: str = ""          # also the fallback of the LLM path
    group: int = 0                    # 0: the kind's base settings; k: the k-th entry of ``mix``


@dataclass
class DecisionRecord:
    tick: int
    agent: str
    kind: str
    trigger: str
    source: str                       # "<policy name>", "forced" (no authority), or "replay"
    obs: Optional[Observation]
    decision: Decision
    default: Optional[Decision] = None
    valid: bool = True
    error: Optional[str] = None
    result: str = ""
    note: str = ""

    def to_dict(self, observations: str = "full") -> dict:
        out: dict[str, Any] = {"type": "decision", "tick": self.tick, "agent": self.agent, "kind": self.kind,
                               "trigger": self.trigger, "source": self.source,
                               "decision": self.decision.to_dict()}
        if self.obs is not None:
            out["obs_digest"] = self.obs.digest()
            out["permitted"] = [spec.name for spec in self.obs.actions]
            if observations == "full" and self.source != "forced":
                out["obs"] = self.obs.to_dict()
        if self.default is not None:
            out["default"] = self.default.to_dict()
            out["exercised"] = not self.decision.same_choice(self.default)
        if not self.valid:
            out["valid"], out["error"] = False, self.error
        if self.result:
            out["result"] = self.result
        if self.note:
            out["note"] = self.note
        if self.decision.info:
            out["info"] = self.decision.info
        return out


@dataclass
class Result:
    metrics: dict
    run_dir: Optional[str]
    state_digest: str
    trace_digest: str
    records: list = field(default_factory=list, repr=False)


class Simulation:
    """A configured run.

    ``policies`` maps an agent kind to a decision model that overrides the
    config for every agent of that kind: a registered name, a ``Policy``
    subclass, or a plain function ``obs -> decision``.
    """

    def __init__(self, config, overrides=None, policies: Optional[dict] = None,
                 replay: Optional[dict] = None, recorder: Optional[Recorder] = None):
        self.cfg = Config.load(config, overrides) if (overrides or not isinstance(config, Config)) else config
        self.plugins = load_plugins(self.cfg.plugins, self.cfg.base_dir)
        cfg = self.cfg
        self.dt = cfg.simulation.time_step_seconds
        self.n_ticks = int(math.ceil(cfg.simulation.duration_seconds / self.dt - 1e-9))
        self.streams = RngStreams(cfg.simulation.seed)
        self.world = World(build_network(cfg.network, cfg.base_dir), self.dt)
        self.world.clock0 = parse_clock(cfg.simulation.clock_start)
        self.recorder = recorder if recorder is not None else Recorder(cfg)
        self._policy_overrides = policies or {}
        self._replay = replay
        self._llm_clients: dict[str, Any] = {}
        self._pool: Optional[ThreadPoolExecutor] = None
        self.wall = {"deciding_s": 0.0}
        self._tick_records: list[DecisionRecord] = []
        self._tick_events: list[dict] = []
        self._state_chain: list[str] = []
        self.event_counts: dict[str, int] = {}

        self.kinds: dict[str, Any] = {name: KINDS.get(name)(self) for name in cfg.active_kinds()}
        self.agency: dict[str, Any] = {}
        self.specs: dict[str, AgentSpec] = {}
        self.policies: dict[str, Any] = {}
        self.stats: dict[str, dict] = {}
        for name, kind in self.kinds.items():
            kcfg = getattr(cfg.agents, name)
            profile = parse_agency(kind.agency, kcfg.agency)
            self.stats[name] = {"decision_points": 0, "decided": 0, "forced": 0, "invalid": 0,
                                "exercised": 0, "fallback": 0}
            ids = kind.setup()
            groups = self._assign_groups(name, ids, kcfg.mix)
            for agent_id in ids:
                g = groups.get(agent_id, 0)
                part = kcfg.mix[g - 1] if g else {}
                use_llm = bool(part.get("use_llm", kcfg.use_llm))
                non_llm = part.get("non_llm_policy", kcfg.non_llm_policy)
                self.agency[agent_id] = parse_agency(kind.agency, part["agency"]) if "agency" in part else profile
                self.specs[agent_id] = AgentSpec(name, use_llm, kcfg.llm_policy if use_llm else non_llm,
                                                 getattr(kcfg, "objective", ""), non_llm, g)
        for agent_id, spec in self.specs.items():
            self.policies[agent_id] = self._make_policy(agent_id, spec)
        # simultaneous model calls within one decision phase: what the configured backends allow
        self._concurrency = max([c.cfg.max_concurrency for c in self._llm_clients.values()], default=1)
        self._recording = self.recorder.active
        self._interventions = self._prepare_interventions()
        self._frame_every = max(1, int(round(cfg.record.frames_every_seconds / self.dt)))
        self._hash_every = max(1, int(cfg.record.state_hash_every_ticks))
        self._finished = False
        self.recorder.write(self._header())
        if self._recording:
            self._record_frame()

    # ------------------------------------------------------------------ setup helpers

    def _assign_groups(self, kind_name: str, ids: list, mix: list) -> dict:
        """Split a kind's agents into sub-populations: ``mix: [{share: 0.3, agency: L1}, ...]``.

        Membership is drawn from its own random stream, so it depends only on the seed and the shares.
        A larger share always contains the agents of a smaller one (nested penetration levels).
        """
        if not mix:
            return {}
        allowed = {"share", "agency", "use_llm", "non_llm_policy"}
        order = list(ids)
        self.streams.get("mix", kind_name).shuffle(order)
        groups, start = {}, 0
        for g, part in enumerate(mix, start=1):
            unknown = set(part) - allowed
            if unknown or "share" not in part:
                raise ValueError(f"agents.{kind_name}.mix entries need 'share' and may set {sorted(allowed)}")
            end = min(len(order), start + int(round(float(part["share"]) * len(order))))
            for agent_id in order[start:end]:
                groups[agent_id] = g
            start = end
        return groups

    def _llm_for(self, kind_name: str):
        kcfg = getattr(self.cfg.agents, kind_name)
        merged = {**self.cfg.to_dict()["llm"], **(kcfg.llm or {})}
        key = json.dumps(merged, sort_keys=True, default=str)
        if key not in self._llm_clients:
            self._llm_clients[key] = build_llm_client(LLMCfg(**merged))
        return self._llm_clients[key]

    def _make_policy(self, agent_id: str, spec: AgentSpec):
        kind = self.kinds[spec.kind]
        kcfg = getattr(self.cfg.agents, spec.kind)
        rng = self.streams.get("agent", agent_id)

        def context(with_llm: bool) -> PolicyContext:
            return PolicyContext(
                agent_id=agent_id, role=spec.kind, rng=rng, objective=spec.objective,
                llm=self._llm_for(spec.kind) if with_llm else None,
                fallback_factory=fallback, config=self.cfg,
                extras={"kind": kind})

        def fallback():
            policy = create_policy(spec.kind, spec.non_llm_policy, kcfg.policy_params)
            policy.bind(context(False))
            return policy

        if self._replay is not None:
            policy = ReplayPolicy(self._replay.get(agent_id, []))
        elif spec.kind in self._policy_overrides:
            override = self._policy_overrides[spec.kind]
            policy = create_policy(spec.kind, override, {}) if isinstance(override, str) else instantiate(override)
            spec.policy = policy.name
        elif spec.use_llm:
            policy = create_policy(spec.kind, kcfg.llm_policy, kcfg.llm_policy_params)
        else:
            policy = create_policy(spec.kind, spec.non_llm_policy, kcfg.policy_params)
        policy.bind(context(policy.uses_llm))
        return policy

    def decision_model(self, agent_id: str):
        """A fresh decision model, configured like this agent's own and with no memory of earlier decisions."""
        return self._make_policy(agent_id, self.specs[agent_id])

    @property
    def llm_concurrency(self) -> int:
        """Simultaneous model calls the configured LLM backends allow (1 if no agent uses one)."""
        return self._concurrency

    def close(self) -> None:
        """Release worker threads and connections without computing results (for uses other than a run)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for client in self._llm_clients.values():
            client.close()

    def _prepare_interventions(self) -> dict[int, list]:
        """Index scheduled interventions by tick."""
        schedule: dict[int, list] = {}
        net = self.world.net

        def at(seconds: float) -> int:
            return int(seconds // self.dt)

        for item in self.cfg.interventions:
            kind = item.get("type")
            if kind == "link_condition":
                links = []
                for u, v in item["links"]:
                    for a, b in ((u, v), (v, u)) if item.get("both_directions", True) else ((u, v),):
                        link = net.link_between(int(a), int(b))
                        if link is None:
                            raise ValueError(f"intervention refers to a missing link {a}->{b}")
                        links.append(link.id)
                schedule.setdefault(at(item["at_seconds"]), []).append(("links", links, float(item["factor"])))
                if item.get("until_seconds") is not None:
                    schedule.setdefault(at(item["until_seconds"]), []).append(("links", links, 1.0))
            elif kind == "set_agency":
                schedule.setdefault(at(item["at_seconds"]), []).append(
                    ("agency", item["kind"], item["agency"], item.get("agents")))
            else:
                raise ValueError(f"unknown intervention type '{kind}' (use link_condition or set_agency)")
        return schedule

    def _header(self) -> dict:
        from .recorder import code_digest, environment
        return {
            "type": "header",
            "config": self.cfg.to_dict(),
            "config_digest": self.cfg.digest(),
            "code_digest": code_digest(),
            "environment": environment(),
            "plugins": self.plugins,
            "network": self.world.net.to_dict(),
            "lines": {k: v["stops"] for k, v in self.world.lines.items()},
            "stations": {s.id: s.node for s in self.world.stations.values()},
            "signals": {s.id: {"node": s.node, "names": s.names,
                               "from": [[self.world.net.links[lid].u for lid in ids] for ids in s.phases]}
                        for s in self.world.signals.values()},
            "kinds": {name: {"side": k.side, "title": k.title, "count": sum(1 for s in self.specs.values() if s.kind == name),
                             "agency": next((self.agency[a].to_dict() for a, s in self.specs.items() if s.kind == name), None),
                             "policy": next((s.policy for s in self.specs.values() if s.kind == name), None),
                             "use_llm": bool(getattr(self.cfg.agents, name).use_llm)}
                      for name, k in self.kinds.items()},
            "update_order": UPDATE_ORDER,
        }

    # ------------------------------------------------------------------ services for kinds

    def emit(self, event: str, **data) -> None:
        """Record something that happened in the world."""
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        self._tick_events.append({"type": "event", "tick": self.world.tick, "event": event, **data})

    def forced(self, agent_id: str, kind: str, trigger: str, decision: Decision, note: str = "") -> None:
        """Record a default action taken because the agent had no authority (already applied by the caller)."""
        self.stats[kind]["decision_points"] += 1
        self.stats[kind]["forced"] += 1
        self._tick_records.append(DecisionRecord(self.world.tick, agent_id, kind, trigger, "forced", None, decision, note=note))

    def feedback(self, agent_id: str, event: dict) -> None:
        """Deliver an outcome to the agent's decision model (for memory and learning)."""
        policy = self.policies.get(agent_id)
        if policy is not None:
            policy.feedback(event)

    def conditions_changed(self, notify_operator: bool = True) -> None:
        """Link conditions, or what is publicly known about them, changed."""
        for name in ("vehicle", "ondemand"):
            if name in self.kinds:
                self.kinds[name].conditions_changed()
        if notify_operator and self.world.operator is not None:
            self.world.operator.incident_pending = True

    # ------------------------------------------------------------------ the tick

    def step(self) -> bool:
        """Advance one tick; returns False once the run is over."""
        w = self.world
        if w.tick >= self.n_ticks:
            return False
        tick = w.tick
        for item in self._interventions.get(tick, ()):
            self._apply_intervention(item)
        for kind in self.kinds.values():
            kind.begin_tick()

        # 2. observe: every due agent sees the same state
        records: list[DecisionRecord] = []
        for name, kind in self.kinds.items():
            for agent_id, trigger in kind.due():
                obs = kind.observe(agent_id, trigger)
                if obs is None:
                    kind.nothing_to_decide(agent_id, trigger)
                    continue
                source = self.policies[agent_id].name if kind.authority(agent_id, trigger) else "forced"
                records.append(DecisionRecord(tick, agent_id, name, trigger, source, obs,
                                              Decision(obs.default.action, dict(obs.default.args)), obs.default))

        # 3. decide
        asked = [rec for rec in records if rec.source != "forced"]
        raw: dict[int, Any] = {}
        waiting = {}
        started = time.perf_counter()
        parallel = self._concurrency > 1 and sum(self.policies[rec.agent].concurrent for rec in asked) > 1
        for i, rec in enumerate(asked):
            policy = self.policies[rec.agent]
            if parallel and policy.concurrent:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=self._concurrency)
                waiting[i] = self._pool.submit(policy.decide, rec.obs)
            else:
                raw[i] = policy.decide(rec.obs)
        for i, future in waiting.items():
            raw[i] = future.result()
        self.wall["deciding_s"] += time.perf_counter() - started
        for i, rec in enumerate(asked):
            decision, error = validate_decision(raw[i], rec.obs.actions)
            if decision is None:
                rec.valid, rec.error = False, error
            else:
                rec.decision = decision

        # 4. apply, in kind order then agent order
        for rec in records:
            stats = self.stats[rec.kind]
            stats["decision_points"] += 1
            if rec.source == "forced":
                stats["forced"] += 1
            else:
                stats["decided"] += 1
                stats["invalid"] += int(not rec.valid)
                stats["exercised"] += int(not rec.decision.same_choice(rec.default))
                stats["fallback"] += int("fallback" in rec.decision.info)
            self.kinds[rec.kind].apply(rec)

        # 5-6. settle interactions, then physics
        for kind in self.kinds.values():
            kind.resolve()
        for kind in self.kinds.values():
            kind.physics()
        w.tick += 1

        # 7. record
        recording = self._recording        # batch runs keep no trace: skip building one
        if recording:
            mode = self.cfg.record.observations
            for rec in records + self._tick_records:
                self.recorder.write(rec.to_dict(mode))
            for event in self._tick_events:
                self.recorder.write(event)
        self._tick_records, self._tick_events = [], []
        if w.tick % self._hash_every == 0 or w.tick == self.n_ticks:
            digest = w.digest()
            self._state_chain.append(digest)
            if recording:
                self.recorder.write({"type": "state", "tick": w.tick, "digest": digest})
        if recording and (w.tick % self._frame_every == 0 or w.tick == self.n_ticks):
            self._record_frame()
        return w.tick < self.n_ticks

    def _apply_intervention(self, item) -> None:
        if item[0] == "links":
            changed = [self.world.net.set_factor(lid, item[2]) for lid in item[1]]
            if any(changed):
                self.emit("link_condition", links=item[1], factor=item[2])
                self.conditions_changed()
        else:
            _, kind, spec, agents = item
            targets = agents or [a for a, s in self.specs.items() if s.kind == kind]
            for agent_id in targets:
                self.agency[agent_id] = parse_agency(self.kinds[kind].agency, spec, base=self.agency[agent_id])
            self.emit("agency_changed", kind=kind, agency=self.agency[targets[0]].to_dict() if targets else None)

    def _record_frame(self) -> None:
        w = self.world
        frame = {"type": "frame", "tick": w.tick, "t": round(w.time_s, 1)}
        for name, kind in self.kinds.items():
            frame[name] = kind.frame()
        slow = {l.id: round(l.factor, 3) for l in w.net.links if abs(l.factor - 1.0) > 1e-9}
        if slow:
            frame["slow_links"] = slow
        busy = {l.id: round(l.flow, 2) for l in w.net.links if l.flow < 0.95}     # congested: share of free-flow speed
        if busy:
            frame["busy_links"] = busy
        frame["stats"] = self._frame_stats()
        self.recorder.write(frame)

    def _frame_stats(self) -> dict:
        """Aggregates for charts: how many travellers are in which stage, and how fast traffic moves."""
        w = self.world
        people = {"waiting": 0, "activity": 0, "moving": 0, "done": 0, "gave_up": 0}
        stage = {"planned": "waiting", "at_activity": "activity", "arrived": "done", "abandoned": "gave_up"}
        for t in w.travelers.values():
            people[stage.get(t.state.value, "moving")] += 1
        ratios = [w.net.links[m.link].cur_speed / w.net.links[m.link].speed
                  for movers in (w.vehicles.values(), w.cabs.values()) for m in movers if m.link is not None]
        return {"people": people, "on_road": len(ratios),
                "speed": round(sum(ratios) / len(ratios), 3) if ratios else None}    # mean share of free-flow speed

    # ------------------------------------------------------------------ running

    def run(self, progress=None) -> Result:
        """Run to the end and return metrics; ``progress(tick, n_ticks)`` is called once per tick."""
        while self.step():
            if progress is not None:
                progress(self.world.tick, self.n_ticks)
        return self.finish()

    def llm_usage(self) -> list[dict]:
        return [client.summary() for client in self._llm_clients.values()]

    def trace_digest(self) -> str:
        import hashlib
        return hashlib.sha256("".join(self._state_chain).encode()).hexdigest()[:20]

    def finish(self) -> Result:
        from .metrics import compute_metrics
        if self._finished:
            return self._result
        self._finished = True
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        for client in self._llm_clients.values():
            client.close()
        for agent_id, policy in self.policies.items():
            policy.end_episode({})
        metrics = compute_metrics(self)
        state, trace = self.world.digest(), self.trace_digest()
        self.recorder.write({"type": "summary", "metrics": metrics, "state_digest": state, "trace_digest": trace})
        run_dir = self.recorder.close(metrics, {"state_digest": state, "trace_digest": trace,
                                                "config_digest": self.cfg.digest()})
        self._result = Result(metrics, run_dir, state, trace, self.recorder.records)
        return self._result
