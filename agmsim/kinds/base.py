"""Agent kinds: the unit of extension for new agentic entities.

A *kind* describes one type of agent in the mobility system -- travellers,
ride vehicles, buses, charging stations, the network operator, or anything a
plugin adds. A kind declares

* its **agency vocabulary** (permissions and how they accumulate over levels);
* its **decision points** (``due``), the **observation** shown at each, and the
  **default action** that applies when the agent has no authority there;
* how a validated decision changes the world (``apply``), how interactions are
  settled (``resolve``) and how the entity evolves physically (``physics``).

The engine treats every kind identically: all due agents observe the same
state, decisions are produced by each agent's decision model (rule, algorithm,
LLM, remote, ...), and actions are applied kind by kind in a fixed order.
"""
from __future__ import annotations

from typing import Optional

from ..actions import ActionSpec, Decision, Param
from ..agency import AgencySpec
from ..observation import Observation
from ..registry import Registry

KINDS = Registry("agent kind")


def register_kind(name: str):
    return KINDS.register(name)


class Kind:
    name = "kind"
    side = "machine"                 # "human" or "machine": which LLM switch of the 2x2 design it belongs to
    title = ""
    role_prompt = ""                 # how an LLM decision model is told who it is
    triggers: dict = {}              # trigger -> explanation (used in prompts and docs)
    actions: dict = {}               # action -> explanation
    agency: Optional[AgencySpec] = None

    def __init__(self, sim):
        self.sim = sim
        self.world = sim.world
        self.cfg = getattr(sim.cfg.agents, self.name, None)

    # ------------------------------------------------------------------ life-cycle hooks

    def setup(self) -> list[str]:
        """Create this kind's entities in the world and return the ids of its agents."""
        return []

    def begin_tick(self) -> None:
        """Exogenous updates at the start of a tick (new needs, timers)."""

    def due(self) -> list[tuple[str, str]]:
        """Decision points of this tick as ``(agent_id, trigger)``, in agent-id order."""
        return []

    def authority(self, agent_id: str, trigger: str) -> bool:
        """Does the agent's agency give it a say at this decision point?"""
        return True

    def observe(self, agent_id: str, trigger: str) -> Optional[Observation]:
        """Build the observation, or None if there is nothing to decide."""
        raise NotImplementedError

    def nothing_to_decide(self, agent_id: str, trigger: str) -> None:
        """Called when ``observe`` returned None."""

    def apply(self, rec) -> None:
        """Apply a validated decision (``rec.decision``) to the world."""

    def resolve(self) -> None:
        """Settle interactions after all actions of the tick were applied."""

    def physics(self) -> None:
        """Advance this kind's entities by one time step."""

    def frame(self) -> list:
        """Compact snapshot for visualisation."""
        return []

    # ------------------------------------------------------------------ helpers

    def allows(self, agent_id: str, permission: str) -> bool:
        return self.sim.agency[agent_id].allows(permission)

    def spec(self, action: str, *params: Param) -> ActionSpec:
        return ActionSpec(action, self.actions[action], tuple(params))

    def obs(self, agent_id, trigger, state, context, actions, default: Decision, aux=None) -> Observation:
        w = self.world
        return Observation(agent_id, self.name, w.tick, w.time_s + w.clock0, trigger, state, context,
                           tuple(actions), default, aux or {})


# --------------------------------------------------------------------------- movement


def _may_cross(line, m, gated: bool) -> bool:
    """May the mover leave this link now? Movers that have to stop wait in arrival order for the light."""
    if not gated:                       # the trip ends here, or the mover has priority: no green needed
        if m.queued:
            line.queue.remove(m)
        m.queued = m.go = False
        return True
    if m.go:                            # released by the traffic light in this tick
        m.go = False
        return True
    if m.queued:
        return False
    if line.green and not line.queue and line.credit >= 1.0:
        line.credit -= 1.0
        line.passed += 1
        return True
    line.queue.append(m)
    m.queued = True
    return False


def advance(net, m, dt: float, factor: float = 1.0, own_lane: bool = False, preempt: bool = False) -> tuple[float, bool]:
    """Move ``m`` along ``m.path`` for ``dt`` seconds.

    ``m`` is any entity with ``node``, ``link``, ``offset`` and ``path``. With
    ``own_lane`` the mover is unaffected by congestion (but still by incidents).
    At a signalised junction the mover stops at the end of its link until the
    light lets it through, unless its trip ends there or ``preempt`` is set.
    Returns the metres travelled and whether the end of the path was reached.
    """
    budget = dt
    moved = 0.0
    while m.path and budget > 1e-9:
        link = net.links[m.path[0]]
        if m.link is None:
            m.link, m.offset = link.id, 0.0
        speed = (link.cur_speed / link.flow if own_lane else link.cur_speed) * factor
        remaining = link.length - m.offset
        need = remaining / speed
        if need <= budget + 1e-9:
            line = net.stop_lines.get(link.id)
            if line is not None and not _may_cross(line, m, len(m.path) > 1 and not preempt):
                moved += remaining
                m.offset = link.length
                break
            moved += remaining
            budget -= need
            m.node, m.link, m.offset = link.v, None, 0.0
            m.path.pop(0)
        else:
            step = speed * budget
            m.offset += step
            moved += step
            budget = 0.0
    return moved, (not m.path and m.link is None)


def path_to(world, m, node: int, dynamic: bool = False) -> list[int]:
    """Link sequence from the mover's position to ``node`` (finishing the current link first)."""
    if m.link is not None:
        link = world.net.links[m.link]
        return [m.link] + list(world.net.route(link.v, node, dynamic).links)
    return list(world.net.route(m.node, node, dynamic).links)
