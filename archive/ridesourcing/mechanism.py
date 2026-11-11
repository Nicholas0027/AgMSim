"""Interaction mechanism: the rules by which human and machine decisions meet.

The mechanism is part of the transition function, not of any agent. It
defines, explicitly and in one place:

* **quoting** -- what a traveller is told about the ride service;
* **pricing** -- how a fare is computed;
* **central matching** -- which vehicle the platform proposes for an open
  request (the *matcher* is itself a plugin);
* **conflict resolution** -- who wins when several vehicles claim the same
  request.

Whether a proposal binds a vehicle, and whether an offer binds a traveller,
is *not* decided here: that depends on each agent's agency profile and is
enforced by the engine.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .entities import Request, Traveler, Vehicle
from .registry import Registry

MECHANISMS = Registry("mechanism")
MATCHERS = Registry("matcher")

INFEASIBLE = 1e9


def register_mechanism(name: str):
    return MECHANISMS.register(name)


def register_matcher(name: str):
    """Register ``(requests, vehicles, eta, max_eta) -> [(request, vehicle), ...]``.

    ``eta(vehicle, request)`` returns the pickup time in seconds, or ``inf``
    if the pair is not allowed.
    """
    return MATCHERS.register(name)


@dataclass
class Claim:
    """A vehicle's bid to serve an open request."""

    vehicle: str
    request: str
    multiplier: float = 1.0
    message: str = ""
    eta_s: float = 0.0
    price: float = 0.0


# --------------------------------------------------------------------------- matchers


@register_matcher("nearest")
def match_nearest(requests, vehicles, eta: Callable, max_eta: float):
    """Oldest request first; each takes the vehicle with the shortest pickup time."""
    free = list(vehicles)
    pairs = []
    for request in requests:
        best, best_eta = None, math.inf
        for vehicle in free:
            e = eta(vehicle, request)
            if e <= max_eta and e < best_eta - 1e-9:
                best, best_eta = vehicle, e
        if best is not None:
            pairs.append((request, best))
            free.remove(best)
    return pairs


def hungarian(cost: list[list[float]]) -> list[int]:
    """Minimum-cost assignment for an n x m matrix with n <= m; returns the column of each row."""
    n, m = len(cost), len(cost[0])
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


@register_matcher("hungarian")
def match_hungarian(requests, vehicles, eta: Callable, max_eta: float):
    """Batch assignment that serves as many requests as possible at minimum total pickup time."""
    if not requests or not vehicles:
        return []
    cost = []
    for request in requests:
        row = []
        for vehicle in vehicles:
            e = eta(vehicle, request)
            row.append(e if e <= max_eta else INFEASIBLE)
        cost.append(row)
    pairs = []
    if len(requests) <= len(vehicles):
        for i, j in enumerate(hungarian(cost)):
            if j >= 0 and cost[i][j] < INFEASIBLE:
                pairs.append((requests[i], vehicles[j]))
    else:
        transposed = [[cost[i][j] for i in range(len(requests))] for j in range(len(vehicles))]
        for j, i in enumerate(hungarian(transposed)):
            if i >= 0 and cost[i][j] < INFEASIBLE:
                pairs.append((requests[i], vehicles[j]))
        pairs.sort(key=lambda pair: (pair[0].t_created, pair[0].id))
    return pairs


# --------------------------------------------------------------------------- default mechanism


@register_mechanism("dispatch")
class DispatchMechanism:
    """Request -> (proposal) -> (offer) -> assignment, with optional vehicle-initiated claims."""

    name = "dispatch"

    def __init__(self, cfg, world):
        self.cfg = cfg
        self.m = cfg.mechanism
        self.world = world
        self.matcher = MATCHERS.get(self.m.matcher)

    # ------------------------------------------------------------------ platform knowledge

    @property
    def dynamic(self) -> bool:
        """Does the platform use real-time link conditions (configured, or published by the operator)?"""
        return self.m.routing == "dynamic" or self.world.advisory

    def eta(self, vehicle: Vehicle, node: int) -> float:
        return self.world.eta(vehicle, node, self.dynamic)

    # ------------------------------------------------------------------ pricing & quoting

    def fare(self, trip_m: float, trip_s: float) -> float:
        p = self.m.pricing
        base = p.base + p.per_km * trip_m / 1000.0 + p.per_min * trip_s / 60.0
        return round(max(p.min_fare, base), 2)

    def quote(self, traveler: Traveler) -> dict:
        """What the platform tells a traveller who asks about a ride."""
        candidates = self.world.available_vehicles()
        etas = [self.eta(v, traveler.origin) for v in candidates]
        best = min(etas) if etas else math.inf
        supply = best <= self.m.max_pickup_seconds
        return {
            "est_wait_s": round(best if supply else self.m.pricing.no_supply_wait_seconds, 1),
            "est_travel_s": round(self.world.net.time(traveler.origin, traveler.destination, self.dynamic), 1),
            "price": self.fare(traveler.trip_m, traveler.trip_s),
            "supply": supply,
            "available_vehicles": len(candidates),
        }

    # ------------------------------------------------------------------ matching & conflicts

    def match(self, requests: list[Request], vehicles: list[Vehicle]) -> list[tuple[Request, Vehicle]]:
        """Platform proposals for open requests among eligible vehicles."""
        if not requests or not vehicles:
            return []

        def eta(vehicle: Vehicle, request: Request) -> float:
            if vehicle.id in request.excluded:
                return math.inf
            return self.eta(vehicle, request.origin)

        return self.matcher(requests, vehicles, eta, self.m.max_pickup_seconds)

    def resolve_claims(self, claims: list[Claim], rng: random.Random) -> tuple[list[Claim], list[Claim]]:
        """Pick one winner per request among competing claims."""
        by_request: dict[str, list[Claim]] = {}
        for claim in claims:
            by_request.setdefault(claim.request, []).append(claim)
        winners, losers = [], []
        for rid in sorted(by_request):
            group = sorted(by_request[rid], key=lambda c: c.vehicle)
            if self.m.claim_rule == "eta":
                group.sort(key=lambda c: (c.eta_s, c.price, c.vehicle))
            elif self.m.claim_rule == "price":
                group.sort(key=lambda c: (c.price, c.eta_s, c.vehicle))
            else:
                rng.shuffle(group)
            winners.append(group[0])
            losers.extend(group[1:])
        return winners, losers

    # ------------------------------------------------------------------ documentation

    def describe(self) -> dict:
        """Machine-readable statement of the rules, stored with every run."""
        m = self.m
        return {
            "type": self.name,
            "central_matching": m.central_matching,
            "matcher": m.matcher,
            "match_interval_seconds": m.match_interval_seconds,
            "max_pickup_seconds": m.max_pickup_seconds,
            "dispatch_self_selecting": m.dispatch_self_selecting,
            "claim_rule": m.claim_rule,
            "request_timeout_seconds": m.request_timeout_seconds,
            "routing": m.routing,
            "update_order": [
                "exogenous events (interventions, new trip needs)",
                "all due agents observe the same state S_t",
                "decisions are produced (decision model per agent; default action where the agent has no authority)",
                "human actions are applied in id order",
                "machine actions are applied in id order",
                "acceptances commit; competing claims are resolved; the platform proposes vehicles for open requests",
                "vehicles move, pickups and drop-offs complete, timers expire",
            ],
        }
