"""Traffic lights as agents.

A junction with traffic lights holds vehicles at the end of their link until
their approach has green. Approaches are grouped into two phases by the
direction of the road (east-west and north-south); one phase has green at a
time, a green lets vehicles through at the saturation flow, and a switch costs
a few seconds in which nobody moves. Vehicles that have to stop wait in
arrival order and stay on their link, so a queue also slows that link down.

What a signal may decide:

* L0 non-agentic: a fixed-time plan -- every phase gets the same green, in turn;
* L1 context-aware: once the minimum green has passed it may keep the green
  or give it to the other phase, knowing the queues and the vehicles
  approaching on each of its own approaches;
* L2 cooperative: it also sees the roads to the next junctions and what the
  orchestrator asks for -- a direction to favour everywhere, or green for one
  phase at this junction -- and by default it does what was asked.

Bus priority set by the orchestrator lets buses through on red.
"""
from __future__ import annotations

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..entities import Signal
from ..network import StopLine
from .base import Kind, register_kind

OBJECTIVE = "Keep vehicles moving through your junction: keep queues and waits short on every approach."
PHASES = ("east-west", "north-south")


@register_kind("signal")
class SignalKind(Kind):
    name = "signal"
    side = "machine"
    title = "Traffic light"
    role_prompt = (
        "You control the traffic lights of one road junction. The approaches form two phases, east-west and "
        "north-south; exactly one phase has green at a time. A green lets waiting vehicles through one after "
        "another; switching the green to the other phase costs a few seconds in which nobody moves. An "
        "orchestrator who sees the whole network may ask you to favour a direction or to give one phase green."
    )
    triggers = {"phase": "the current green has lasted its minimum; keep it or give green to the other phase"}
    actions = {
        "keep_phase": "Leave the green where it is for now.",
        "switch_phase": "Give green to the chosen phase, after the short all-red.",
    }
    agency = AgencySpec(
        "signal",
        {
            "actuate": "keep or end a green according to the queues at its own junction (context awareness)",
            "coordinate": "see the roads to the next junctions and act on what the orchestrator asks for (cooperation)",
        },
        {"L0": (), "L1": ("actuate",), "L2": ("actuate", "coordinate"),
         "L3": ("actuate", "coordinate"), "L4": ("actuate", "coordinate")},
    )

    # ------------------------------------------------------------------ setup

    def setup(self):
        w, cfg, net = self.world, self.cfg, self.world.net
        every = cfg.junctions == "all"
        nodes = sorted(net.nodes) if every else [int(n) for n in cfg.junctions]
        width = max(2, len(str(max(net.nodes))))
        incoming: dict[int, list] = {}
        for link in net.links:
            incoming.setdefault(link.v, []).append(link)
        for n in nodes:
            groups: tuple[list, list] = ([], [])
            for link in incoming.get(n, []):
                (ux, uy), (vx, vy) = net.xy(link.u), net.xy(link.v)
                groups[0 if abs(vx - ux) >= abs(vy - uy) else 1].append(link.id)
            crossing = groups[0] and groups[1] and len(groups[0]) + len(groups[1]) >= 3
            if not crossing:
                if every:
                    continue                    # a bend or a dead end: nothing to control
                raise ValueError(f"agents.signal.junctions: node {n} is not a junction of crossing roads")
            sig = Signal(f"T{n:0{width}d}", n, [sorted(g) for g in groups], list(PHASES), phase=n % 2,
                         wait_by_phase=[0, 0])
            for i, ids in enumerate(sig.phases):
                for lid in ids:
                    net.stop_lines[lid] = StopLine(green=(i == sig.phase))
            w.signals[sig.id] = sig
        self._counted, self._on = -1, {}
        return list(w.signals)

    # ------------------------------------------------------------------ what a signal knows

    def _on_links(self) -> dict:
        """Number of vehicles and buses on every link, counted once per tick."""
        w = self.world
        if self._counted != w.tick:
            on: dict = {}
            for movers in (w.vehicles.values(), w.buses.values(), w.cabs.values()):
                for m in movers:
                    if m.link is not None:
                        on[m.link] = on.get(m.link, 0) + 1
            self._counted, self._on = w.tick, on
        return self._on

    def demand(self, sig) -> list:
        """Per phase: vehicles held at the stop lines, and vehicles still approaching."""
        net, on = self.world.net, self._on_links()
        out = []
        for ids in sig.phases:
            waiting = sum(len(net.stop_lines[lid].queue) for lid in ids)
            out.append((waiting, sum(on.get(lid, 0) for lid in ids) - waiting))
        return out

    def _request(self, sig):
        """Index of the phase the orchestrator currently asks for, if this signal acts on requests."""
        if sig.request is None or not self.allows(sig.id, "coordinate"):
            return None
        if self.world.tick >= sig.request["until"]:
            sig.request = None
            return None
        return sig.request["phase"]

    def _planned_green(self, sig) -> float:
        """Green of the current phase in the fixed-time plan, stretched or cut for a favoured direction."""
        cfg, op = self.cfg, self.world.operator
        favoured = op.favoured if op is not None and self.allows(sig.id, "coordinate") else None
        if favoured not in sig.names:
            return cfg.green_seconds
        return cfg.green_seconds * (cfg.favour_factor if sig.names[sig.phase] == favoured else 1.0 / cfg.favour_factor)

    # ------------------------------------------------------------------ decision points

    def due(self):
        w, cfg = self.world, self.cfg
        points = []
        for sig in w.signals.values():
            if w.tick < sig.red_until or w.tick < sig.next_decision:
                continue
            elapsed = (w.tick - sig.since) * w.dt
            if elapsed < cfg.min_green_seconds:
                continue
            if not self.allows(sig.id, "actuate"):
                if elapsed >= self._planned_green(sig):          # the fixed-time plan switches now
                    points.append((sig.id, "phase"))
                continue
            wanted = any(held + coming > 0 for i, (held, coming) in enumerate(self.demand(sig)) if i != sig.phase)
            if wanted or elapsed >= cfg.max_green_seconds or self._request(sig) not in (None, sig.phase):
                points.append((sig.id, "phase"))
            else:                                                 # nobody else wants the green: nothing to decide
                sig.next_decision = w.tick + w.ticks(cfg.decision_seconds)
        return points

    def authority(self, agent_id, trigger):
        return self.allows(agent_id, "actuate")

    def observe(self, agent_id, trigger):
        w, cfg, net = self.world, self.cfg, self.world.net
        sig = w.signals[agent_id]
        elapsed = (w.tick - sig.since) * w.dt
        demand, on = self.demand(sig), self._on_links()
        state = {"junction_node": sig.node, "objective": OBJECTIVE, "green_now": sig.names[sig.phase],
                 "green_for_s": round(elapsed), "min_green_s": cfg.min_green_seconds,
                 "max_green_s": cfg.max_green_seconds, "switching_costs_s": cfg.lost_seconds}
        context = {"phases": [
            {"phase": sig.names[i], "has_green": i == sig.phase, "waiting": demand[i][0], "approaching": demand[i][1],
             "approaches": [{"from_node": net.links[lid].u, "waiting": len(net.stop_lines[lid].queue),
                             "approaching": on.get(lid, 0) - len(net.stop_lines[lid].queue)} for lid in ids]}
            for i, ids in enumerate(sig.phases)]}
        requested = self._request(sig)
        if self.allows(sig.id, "coordinate"):
            context["next_junctions"] = [
                {"to_node": link.v, "vehicles_on_the_way": on.get(link.id, 0),
                 "speed_pct_of_free_flow": round(100 * link.flow * link.factor)}
                for link in net.links if link.u == sig.node]
            context["orchestrator"] = {"favours": w.operator.favoured if w.operator else None,
                                       "asks_for": sig.names[requested] if requested is not None else None}
        others = tuple(name for i, name in enumerate(sig.names) if i != sig.phase)
        must_switch = elapsed >= cfg.max_green_seconds
        actions = ([] if must_switch else [self.spec("keep_phase")]) + [
            self.spec("switch_phase", Param("phase", "string", others))]
        # The default is the fixed-time plan; a signal that cooperates gives the orchestrator what it asked for.
        if requested is not None and requested != sig.phase:
            default = Decision("switch_phase", {"phase": sig.names[requested]})
        elif must_switch or (requested is None and elapsed >= self._planned_green(sig)):
            default = Decision("switch_phase", {"phase": sig.names[(sig.phase + 1) % len(sig.names)]})
        else:
            default = Decision("keep_phase")
        return self.obs(sig.id, trigger, state, context, actions, default)

    # ------------------------------------------------------------------ applying decisions

    def apply(self, rec):
        w, cfg = self.world, self.cfg
        sig = w.signals[rec.agent]
        sig.next_decision = w.tick + w.ticks(cfg.decision_seconds)
        if rec.decision.action != "switch_phase":
            # look again no later than when the plan (or the maximum green) would end this green
            for seconds in (self._planned_green(sig), cfg.max_green_seconds):
                end = sig.since + w.ticks(seconds)
                if w.tick < end < sig.next_decision:
                    sig.next_decision = end
            return
        k = sig.names.index(rec.decision.args["phase"])
        if self._request(sig) == k:
            sig.requests_followed, sig.request = sig.requests_followed + 1, None
        lost = w.ticks(cfg.lost_seconds) if cfg.lost_seconds > 0 else 0
        sig.phase, sig.switches = k, sig.switches + 1
        sig.red_until = sig.since = w.tick + lost
        rec.result = f"green to {sig.names[k]}"

    # ------------------------------------------------------------------ physics

    def physics(self):
        """Set every stop line for this tick and let waiting vehicles through, first come first."""
        w, cfg, net = self.world, self.cfg, self.world.net
        scale = self.sim.cfg.network.capacity_scale
        for sig in w.signals.values():
            open_now, queued = w.tick >= sig.red_until, 0
            for i, ids in enumerate(sig.phases):
                for lid in ids:
                    line = net.stop_lines[lid]
                    line.green = open_now and i == sig.phase
                    if line.green:
                        through = cfg.saturation_flow * net.links[lid].lanes * scale * w.dt
                        line.credit = min(line.credit + through, max(1.0, through))
                        while line.queue and line.credit >= 1.0:
                            mover = line.queue.pop(0)
                            mover.queued, mover.go = False, True
                            line.credit -= 1.0
                            sig.held += 1
                    queued += len(line.queue)
                    sig.wait_by_phase[i] += len(line.queue)
            sig.wait_ticks += queued
            sig.queue_max = max(sig.queue_max, queued)

    def frame(self):
        net, tick = self.world.net, self.world.tick
        return [[s.id, s.node, s.phase if tick >= s.red_until else -1,
                 sum(len(net.stop_lines[lid].queue) for ids in s.phases for lid in ids)]
                for s in self.world.signals.values()]
