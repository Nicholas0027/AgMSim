"""The world: container of the full system state ``S_t`` plus read-only queries."""
from __future__ import annotations

import hashlib
import math
from typing import Optional

from .entities import Bus, Cab, OperatorState, RideRequest, Signal, Station, Traveler, TState, Vehicle, VState
from .network import RoadNetwork

_IDLE = frozenset((TState.PLANNED, TState.ARRIVED, TState.ABANDONED))


class World:
    def __init__(self, net: RoadNetwork, dt: float):
        self.net = net
        self.dt = float(dt)
        self.tick = 0
        self.clock0 = 0.0                                # time of day at tick 0, seconds (display only)
        self.travelers: dict[str, Traveler] = {}
        self.vehicles: dict[str, Vehicle] = {}
        self.buses: dict[str, Bus] = {}
        self.lines: dict[str, dict] = {}                 # line id -> {"stops": [...], "legs_s": [...], "headway_s": float}
        self.stations: dict[str, Station] = {}
        self.signals: dict[str, Signal] = {}
        self.cabs: dict[str, Cab] = {}                   # on-demand cars
        self.requests: dict[str, RideRequest] = {}       # ride requests, open and past
        self.operator: Optional[OperatorState] = None
        self.cordon: frozenset = frozenset()             # link ids inside the operator's charging cordon
        self.stop_departures: dict[tuple[str, int], int] = {}   # (line, stop index) -> tick of the last departure
        self.car_factor = 1.0                            # speed multipliers set by the operator
        self.bus_factor = 1.0
        self._digest_cache: dict[str, tuple] = {}        # fingerprint parts of idle agents (see digest)

    # ------------------------------------------------------------------ time

    @property
    def time_s(self) -> float:
        return self.tick * self.dt

    def ticks(self, seconds: float) -> int:
        """Convert a duration to a whole number of ticks (at least one)."""
        return max(1, int(math.ceil(seconds / self.dt - 1e-9)))

    # ------------------------------------------------------------------ public knowledge

    @property
    def advisory(self) -> bool:
        """Are real-time link conditions public knowledge (published by the operator)?"""
        return bool(self.operator and self.operator.advisory)

    @property
    def toll(self) -> float:
        return self.operator.toll if self.operator else 0.0

    def route_toll(self, links) -> float:
        """Charge for driving ``links`` (paid once per trip when the cordon is entered)."""
        if self.toll > 0 and any(lid in self.cordon for lid in links):
            return self.toll
        return 0.0

    # ------------------------------------------------------------------ movers (vehicles and buses)

    def anchor(self, m) -> tuple[int, float, float, float]:
        """Next node the mover will be at, with the free-flow time, current time and distance to reach it."""
        if m.link is None:
            return m.node, 0.0, 0.0, 0.0
        link = self.net.links[m.link]
        remaining = max(0.0, link.length - m.offset)
        return link.v, remaining / link.speed, remaining / link.cur_speed, remaining

    def eta(self, m, node: int, dynamic: bool = False) -> float:
        """Seconds for the mover to reach ``node`` along the fastest route."""
        anchor, lead_free, lead_cur, _ = self.anchor(m)
        return (lead_cur if dynamic else lead_free) + self.net.time(anchor, node, dynamic)

    def distance(self, m, node: int) -> float:
        """Metres for the mover to reach ``node`` along the free-flow fastest route."""
        anchor, _, _, lead_m = self.anchor(m)
        return lead_m + self.net.length(anchor, node)

    def xy(self, m) -> tuple[float, float]:
        if m.link is None:
            return self.net.xy(m.node)
        return self.net.position(m.link, m.offset)

    # ------------------------------------------------------------------ state fingerprint

    def digest(self) -> str:
        """Fingerprint of the dynamic state, used to verify that two runs evolve identically.

        Most agents are idle at any moment (trip not yet started or already over, vehicle parked),
        so their part of the fingerprint is reused until something about them changes.
        """
        cache = self._digest_cache
        vehicles = []
        for v in self.vehicles.values():
            if v.state is VState.PARKED:     # a parked vehicle changes only by moving, charging or a new plan
                key = (v.m_loaded, v.m_empty, v.soc, v.tolls, v.suggested, v.transfer is None, v.target, len(v.path))
                hit = cache.get(v.id)
                if hit is not None and hit[0] == key:
                    vehicles.append(hit[1])
                    continue
            else:
                key = None
            item = (v.id, v.node, v.link, round(v.offset, 4), v.state.value, v.passenger, v.station, v.target,
                    tuple(v.path), round(v.soc, 6), round(v.m_loaded + v.m_empty, 3), round(v.tolls, 4),
                    v.transfer is not None, v.suggested)
            if key is not None:
                cache[v.id] = (key, repr(item))
                item = cache[v.id][1]
            vehicles.append(item)
        travelers = []
        for t in self.travelers.values():
            idle = t.state in _IDLE
            if idle:                          # nothing about the traveller changes while in these states
                hit = cache.get(t.id)
                if hit is not None and hit[0] is t.state:
                    travelers.append(hit[1])
                    continue
            item = (t.id, t.state.value, t.mode, t.cancels, t.t_depart, t.t_arrival, t.phase_end, t.waited_ticks,
                    t.suggestion is not None, round(t.paid, 4))
            if t.plan is not None:
                item += (t.plan.at, round(t.plan.leave_s, 3), tuple(a.skipped for a in t.plan.activities))
            if idle:
                cache[t.id] = (t.state, repr(item))
                item = cache[t.id][1]
            travelers.append(item)
        buses = [
            (b.id, b.node, b.link, round(b.offset, 4), b.state.value, b.next_stop, tuple(b.onboard),
             b.dwell_until, round(b.held_s, 3), b.skip_next, tuple(b.path))
            for b in self.buses.values()
        ]
        stations = [
            (s.id, round(s.multiplier, 4), tuple(s.queue), tuple(s.plugged), round(s.kwh, 4), round(s.revenue, 4))
            for s in self.stations.values()
        ]
        op = self.operator
        operator = (op.advisory, op.bus_priority, round(op.toll, 4)) if op else None
        links = tuple((round(link.factor, 6), round(link.flow, 6)) for link in self.net.links)
        signals = [(s.id, s.phase, s.since, s.red_until, s.request and (s.request["phase"], s.request["until"]),
                    tuple((lid, round(self.net.stop_lines[lid].credit, 6), tuple(m.id for m in self.net.stop_lines[lid].queue))
                          for ids in s.phases for lid in ids)) for s in self.signals.values()]
        parts = (self.tick, vehicles, travelers, buses, stations, operator, links)
        if signals:
            parts += ((signals, op and op.favoured),)
        if self.cabs:
            parts += ([(c.id, c.node, c.link, round(c.offset, 4), c.state.value, c.request, c.proposal, tuple(c.path),
                        c.queued, c.go) for c in self.cabs.values()],
                      [(r.id, r.state, r.car, r.offered, tuple(sorted(r.declined_by))) for r in self.requests.values()])
        blob = repr(parts)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]
