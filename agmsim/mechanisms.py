"""The agency mechanisms: what an agent knows when it decides.

An agent's *level* says what it may do. Three further settings, written per
kind next to the level, say what it has to go on:

``observation``
    what it perceives of the current state. Here: how much of the road
    conditions it knows as they are now -- ``traffic: none | local | global``,
    with ``radius_m`` for ``local``.
``memory``
    what it carries from the past: its own latest decisions and what came of
    them (``decisions``; with ``reasons`` also why it took each, in its
    decision model's own words), and for how long a road condition it saw or
    was told about stays with it (``traffic_seconds``).
``interaction``
    what it exchanges with other agents. Driving agents within ``range_m`` of
    each other pass on what they know about slow roads (``share_traffic``).

All three are kept here, by the framework, and not inside a decision model: a
rule, an algorithm, a language model and a remote service are handed the same
observation and the same memory, so they can be compared like for like.

How knowledge of traffic is put together for one driving agent:

1. roads it *sees* (within ``radius_m``, if ``traffic: local``): their travel
   time as it is now;
2. roads it has a *report* about -- something it saw earlier or was told, not
   older than ``traffic_seconds``: the travel time in the report;
3. every other road: the travel time without any traffic.

A report is made about a road that is slow (below ``SLOW`` of its normal
speed), and about a road reported slow that is found to be clear again.
Agents take note of the roads around them and exchange reports every
``PERIOD_SECONDS``. Without memory a report lasts until the next exchange
only: an agent then knows what the agents in range see now, and nothing of
what they heard. With memory, reports are kept and passed on, so news
travels from car to car -- and may be out of date when it arrives.

An agent that learns of a slow road on the route it is following reconsiders
that route at once: its decision points follow from what it comes to know.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from .network import current_weight, free_flow_weight
from .observation import clock

SLOW = 0.8              # a road is worth a report below this share of its normal speed
PERIOD_SECONDS = 10.0   # how often agents take note of the roads around them and exchange reports
MAX_REPORTS_SHOWN = 6
MAX_REASON_CHARS = 280  # of the reason an agent gave for a decision, as kept in its memory


class Memories:
    """Each agent's latest decisions and outcomes, oldest first."""

    def __init__(self, sim):
        self.world, self._kinds = sim.world, sim.kinds
        self._size = {name: max(0, int(getattr(sim.cfg.agents, name).memory.decisions)) for name in sim.kinds}
        self._reasons = {name: bool(getattr(sim.cfg.agents, name).memory.reasons) for name in sim.kinds}
        self._kind = {agent_id: spec.kind for agent_id, spec in sim.specs.items()}
        self._items: dict[str, deque] = {}

    def of(self, agent_id: str) -> tuple:
        items = self._items.get(agent_id)
        return tuple(items) if items else ()

    def _add(self, agent_id: str, item: dict) -> None:
        size = self._size.get(self._kind.get(agent_id), 0)
        if size:
            items = self._items.get(agent_id)
            if items is None:
                items = self._items[agent_id] = deque(maxlen=size)
            items.append({"at": clock(self.world.time_s + self.world.clock0), **item})

    def decided(self, rec) -> None:
        """The agent took a decision of its own (``rec`` has been applied)."""
        if not self._size.get(rec.kind):
            return
        item = self._kinds[rec.kind].remember(rec)
        if rec.decision.rationale and self._reasons.get(rec.kind):
            item["why"] = rec.decision.rationale[:MAX_REASON_CHARS]
        self._add(rec.agent, item)

    def outcome(self, agent_id: str, event: dict) -> None:
        """Something that concerns the agent came about (trip completed, offer declined, ...)."""
        self._add(agent_id, {"outcome": event.get("event"), **{k: v for k, v in event.items() if k != "event"}})


class TrafficKnowledge:
    """What every driving agent knows about the roads, given its kind's observation, memory and interaction."""

    def __init__(self, sim):
        self.sim, self.world = sim, sim.world
        self.kinds: dict[str, SimpleNamespace] = {}
        for name, kind in sim.kinds.items():
            if not kind.drives:
                continue
            cfg = kind.cfg
            keep = cfg.memory.traffic_seconds
            self.kinds[name] = SimpleNamespace(
                name=name, kind=kind, traffic=cfg.observation.traffic, radius=float(cfg.observation.radius_m),
                keep=self.world.ticks(keep) if keep > 0 else 0, share=bool(cfg.interaction.share_traffic),
                range=float(cfg.interaction.range_m))
        # Kinds whose agents keep reports: everything but "sees it all" and "sees nothing", each on its own.
        self._noting = [k for k in self.kinds.values() if k.traffic == "local" or k.share or k.keep]
        self._period = self.world.ticks(PERIOD_SECONDS)
        self.reports: dict[str, dict] = {}      # agent -> {link id: (travel time s, tick observed, heard from another)}
        self._slow: dict[str, frozenset] = {}   # agent -> roads it took to be slow at the last look
        self._views: dict = {}
        self._heard: set = set()                # (agent, road) of every report taken over from another agent

    # ------------------------------------------------------------------ what an agent sees

    def view(self, radius: float, m) -> tuple:
        """Roads whose middle lies within ``radius`` of the road the mover is on (or the junction it is at)."""
        key = (radius, m.link if m.link is not None else -1 - m.node)
        links = self._views.get(key)
        if links is None:
            net = self.world.net
            x, y = net.xy(m.node) if m.link is None else net.position(m.link, net.links[m.link].length / 2.0)
            r2, found = radius * radius, []
            for link in net.links:
                mx, my = net.position(link.id, link.length / 2.0)
                if (mx - x) ** 2 + (my - y) ** 2 <= r2:
                    found.append(link.id)
            links = self._views[key] = tuple(found)
        return links

    def _sees(self, k, m) -> tuple:
        return self.view(k.radius, m) if k.traffic != "none" else ()

    # ------------------------------------------------------------------ every few seconds

    def update(self) -> None:
        """Agents take note of the roads around them, forget what is too old, and exchange reports.

        An agent that learns of a slow road on the route it is following reconsiders that route.
        """
        w = self.world
        if not self._noting or w.tick % self._period:
            return
        tick, links = w.tick, w.net.links
        present = []
        for k in self._noting:
            oldest = tick - k.keep              # without memory: nothing from before this look survives
            for m in k.kind.on_the_road():
                store = self.reports.get(m.id)
                if store:
                    for lid in [lid for lid, report in store.items() if report[1] < oldest]:
                        del store[lid]
                for lid in self._sees(k, m):
                    link = links[lid]
                    if link.cur_time * SLOW > link.free_time or (store and lid in store):
                        if store is None:
                            store = self.reports[m.id] = {}
                        store[lid] = (link.cur_time, tick, False)
                present.append((m, k, store))
        self._exchange([item for item in present if item[1].share])
        for m, k, _ in present:
            store = self.reports.get(m.id)
            slow = frozenset(lid for lid, report in store.items()
                             if report[0] * SLOW > links[lid].free_time) if store else frozenset()
            fresh = slow - self._slow.get(m.id, frozenset())
            self._slow[m.id] = slow
            if fresh and m.path and not fresh.isdisjoint(m.path):
                m.route_pending = True

    def _exchange(self, sharing: list) -> None:
        """Every sharing agent hears what the agents in range knew before this exchange (one hop per exchange)."""
        if len(sharing) < 2:
            return
        w, links, tick = self.world, self.world.net.links, self.world.tick
        cell = max(k.range for _, k, _ in sharing)
        grid: dict = {}
        placed = []
        for m, k, store in sharing:
            x, y = w.xy(m)
            placed.append((m, k, store, x, y))
            if store:
                grid.setdefault((int(x // cell), int(y // cell)), []).append((x, y, k.range, store, m.id))
        if not grid:
            return
        news = []
        for m, k, store, x, y in placed:
            cx, cy, oldest, got = int(x // cell), int(y // cell), tick - k.keep, None
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for ox, oy, reach, other, other_id in grid.get((gx, gy), ()):
                        if other_id == m.id or (ox - x) ** 2 + (oy - y) ** 2 > min(reach, k.range) ** 2:
                            continue
                        for lid, report in other.items():
                            if report[1] < oldest:
                                continue                # too old for this listener to take in
                            mine = (got.get(lid) if got else None) or (store.get(lid) if store else None)
                            if mine is None:
                                if report[0] * SLOW <= links[lid].free_time:
                                    continue            # "that road is clear": no news to someone who thought so anyway
                            elif report[1] <= mine[1]:
                                continue
                            if got is None:
                                got = {}
                            got[lid] = (report[0], report[1], True)
            if got:
                news.append((m.id, got))
        for agent_id, got in news:
            self.reports.setdefault(agent_id, {}).update(got)
            self._heard.update((agent_id, lid) for lid in got)

    # ------------------------------------------------------------------ what an agent knows

    def can_know(self, name: str) -> bool:
        """Can an agent of this kind know anything about current traffic by itself or from others?"""
        k = self.kinds[name]
        return k.traffic != "none" or k.share

    def knowledge(self, name: str, m) -> tuple:
        """``(weight, seen, reports)``: travel time of each road as the agent knows it, the roads it sees right
        now (None: all of them) and the reports it holds about roads it does not see."""
        w, k = self.world, self.kinds[name]
        if w.advisory or k.traffic == "global":
            return current_weight, None, {}
        seen = frozenset(self._sees(k, m))
        reports = {lid: report for lid, report in (self.reports.get(m.id) or {}).items() if lid not in seen}
        if not seen and not reports:
            return free_flow_weight, seen, reports

        def weight(link):
            if link.id in seen:
                return link.cur_time
            report = reports.get(link.id)
            return link.free_time if report is None else report[0]

        return weight, seen, reports

    def weight(self, name: str, m):
        return self.knowledge(name, m)[0]

    def explain(self, name: str, m, routes, seen, reports: dict) -> dict | None:
        """What the agent's estimates rest on: how far it sees, and what it holds about the roads of
        ``routes`` that it does not see. None for an agent that knows every road as it is."""
        if seen is None:
            return None
        w, net, k = self.world, self.world.net, self.kinds[name]
        out = ({"you_see_roads_within_km": round(k.radius / 1000.0, 2)} if k.traffic == "local"
               else {"you_see": "no traffic yourself"})
        out["roads_you_do_not_see"] = "reckoned as clear, unless you hold a report about them"
        on_routes = {lid for route in routes for lid in route.links}
        rows = []
        for lid in sorted(on_routes & set(reports), key=lambda lid: (-reports[lid][1], lid))[:MAX_REPORTS_SHOWN]:
            seconds, tick, heard = reports[lid]
            link = net.links[lid]
            rows.append({"road": [link.u, link.v], "speed_pct_of_normal": round(100 * link.free_time / seconds),
                         "age_min": round((w.tick - tick) * w.dt / 60.0, 1),
                         "source": "told by another vehicle" if heard else "seen earlier"})
        if rows:
            out["reports_about_roads_you_do_not_see"] = rows
        return out

    # ------------------------------------------------------------------ results

    def metrics(self) -> dict:
        """How far news travelled: reports are counted once per agent and road."""
        members = sum(1 for spec in self.sim.specs.values() if spec.kind in self.kinds and self.kinds[spec.kind].share)
        listeners = len({agent_id for agent_id, _ in self._heard})
        return {"reports_heard": len(self._heard), "agents_that_heard": listeners,
                "share_of_agents_that_heard": round(listeners / members, 3) if members else 0.0}
