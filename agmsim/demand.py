"""Exogenous inputs of a run: who wants to travel where and when.

Trips and personas are drawn up-front from the ``demand`` random stream. They
therefore depend only on the seed and the scenario, never on what the agents
later decide, so every experimental arm that shares a seed faces exactly the
same travellers.
"""
from __future__ import annotations

import math
import random

from .entities import Persona, Traveler
from .network import RoadNetwork
from .plans import build_plan
from .registry import Registry
from .rng import lognormal, weighted_choice

DEMANDS = Registry("demand generator")

AGE_GROUPS = (("18-24", 0.15), ("25-34", 0.25), ("35-49", 0.25), ("50-64", 0.20), ("65+", 0.15))
PURPOSES = (("commute", 0.40), ("errand", 0.20), ("leisure", 0.20), ("school", 0.12), ("medical", 0.08))


def register_demand(name: str):
    """Register a generator ``(cfg, net, rng, n) -> [(time_s, origin, destination), ...]``."""
    return DEMANDS.register(name)


# --------------------------------------------------------------------------- helpers


def _sample_time(rng: random.Random, start: float, end: float, params: dict) -> float:
    """Uniform by default; a truncated normal peak if ``peak_seconds`` is given."""
    if "peak_seconds" in params:
        peak = float(params["peak_seconds"])
        width = float(params.get("peak_width_seconds", (end - start) / 6.0))
        for _ in range(50):
            t = rng.gauss(peak, width)
            if start <= t < end:
                return t
    return start + rng.random() * (end - start)


def _other_node(rng: random.Random, net: RoadNetwork, nodes: list[int], origin: int, min_m: float, weights=None) -> int:
    """Draw a destination at least ``min_m`` away from ``origin`` (best effort)."""
    best, best_len = None, -1.0
    for _ in range(40):
        cand = weighted_choice(rng, nodes, weights) if weights else nodes[rng.randrange(len(nodes))]
        if cand == origin:
            continue
        length = net.length(origin, cand)
        if length >= min_m:
            return cand
        if length > best_len:
            best, best_len = cand, length
    if best is None:
        best = next(n for n in nodes if n != origin)
    return best


def _centre_node(net: RoadNetwork) -> int:
    xs = [x for x, _ in net.nodes.values()]
    ys = [y for _, y in net.nodes.values()]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    return min(net.nodes, key=lambda n: (math.hypot(net.nodes[n][0] - cx, net.nodes[n][1] - cy), n))


# --------------------------------------------------------------------------- generators


@register_demand("uniform")
def demand_uniform(cfg, net, rng, n):
    """Origins and destinations uniform over nodes."""
    start, end = cfg.demand_window()
    nodes = sorted(net.nodes)
    trips = []
    for _ in range(n):
        t = _sample_time(rng, start, end, cfg.demand.params)
        origin = nodes[rng.randrange(len(nodes))]
        trips.append((t, origin, _other_node(rng, net, nodes, origin, cfg.demand.min_trip_m)))
    return trips


@register_demand("cbd")
def demand_cbd(cfg, net, rng, n):
    """Monocentric demand: most trips head to (or leave) a centre.

    ``params``: ``center`` (node, default the geometric centre), ``scale_m``
    (decay length of attraction), ``inbound_share`` (default 0.7).
    """
    params = cfg.demand.params
    start, end = cfg.demand_window()
    nodes = sorted(net.nodes)
    centre = int(params.get("center", _centre_node(net)))
    span = max(net.euclid(centre, node) for node in nodes) or 1.0
    scale = float(params.get("scale_m", 0.35 * span))
    inbound = float(params.get("inbound_share", 0.7))
    attraction = [math.exp(-net.euclid(centre, node) / scale) for node in nodes]
    trips = []
    for _ in range(n):
        t = _sample_time(rng, start, end, params)
        if rng.random() < inbound:
            origin = nodes[rng.randrange(len(nodes))]
            destination = _other_node(rng, net, nodes, origin, cfg.demand.min_trip_m, attraction)
        else:
            origin = weighted_choice(rng, nodes, attraction)
            destination = _other_node(rng, net, nodes, origin, cfg.demand.min_trip_m)
        trips.append((t, origin, destination))
    return trips


@register_demand("hotspots")
def demand_hotspots(cfg, net, rng, n):
    """Weighted origins and destinations mixed with a uniform background.

    ``params``: ``origins`` and ``destinations`` (``{node: weight}``),
    ``background_share`` (default 0.3).
    """
    params = cfg.demand.params
    start, end = cfg.demand_window()
    nodes = sorted(net.nodes)
    background = float(params.get("background_share", 0.3))

    def weights(table):
        table = {int(k): float(v) for k, v in (table or {}).items()}
        return [table.get(node, 0.0) for node in nodes] if table else None

    w_origin, w_dest = weights(params.get("origins")), weights(params.get("destinations"))
    trips = []
    for _ in range(n):
        t = _sample_time(rng, start, end, params)
        use_background = rng.random() < background
        if w_origin and not use_background:
            origin = weighted_choice(rng, nodes, w_origin)
        else:
            origin = nodes[rng.randrange(len(nodes))]
        dest_weights = None if (use_background or not w_dest) else w_dest
        trips.append((t, origin, _other_node(rng, net, nodes, origin, cfg.demand.min_trip_m, dest_weights)))
    return trips


@register_demand("event")
def demand_event(cfg, net, rng, n):
    """Uniform background plus a burst of departures from one node (e.g. a venue emptying).

    ``params``: ``node``, ``share`` (default 0.5), ``event_start_seconds``,
    ``event_end_seconds``.
    """
    params = cfg.demand.params
    start, end = cfg.demand_window()
    nodes = sorted(net.nodes)
    venue = int(params.get("node", _centre_node(net)))
    share = float(params.get("share", 0.5))
    ev_start = float(params.get("event_start_seconds", start + 0.4 * (end - start)))
    ev_end = float(params.get("event_end_seconds", start + 0.6 * (end - start)))
    trips = []
    for _ in range(n):
        if rng.random() < share:
            t = ev_start + rng.random() * (ev_end - ev_start)
            origin = venue
        else:
            t = _sample_time(rng, start, end, params)
            origin = nodes[rng.randrange(len(nodes))]
        trips.append((t, origin, _other_node(rng, net, nodes, origin, cfg.demand.min_trip_m)))
    return trips


@register_demand("corridor")
def demand_corridor(cfg, net, rng, n):
    """Through traffic: most trips cross the network along one axis, from one side to the other.

    ``params``: ``axis`` (``east_west``, the default, or ``north_south``), ``through_share`` (default
    0.75; the rest are uniform trips), ``one_way`` (default false: both directions).
    """
    params = cfg.demand.params
    start, end = cfg.demand_window()
    nodes = sorted(net.nodes)
    along, across = (0, 1) if params.get("axis", "east_west") == "east_west" else (1, 0)
    lines: dict = {}
    for node in nodes:
        lines.setdefault(round(net.nodes[node][across], 1), []).append(node)
    lines = [sorted(members, key=lambda m: (net.nodes[m][along], m)) for _, members in sorted(lines.items()) if len(members) > 1]
    share, one_way = float(params.get("through_share", 0.75)), bool(params.get("one_way", False))
    trips = []
    for _ in range(n):
        t = _sample_time(rng, start, end, params)
        if lines and rng.random() < share:
            line = lines[rng.randrange(len(lines))]
            origin, destination = (line[0], line[-1]) if one_way or rng.random() < 0.5 else (line[-1], line[0])
        else:
            origin = nodes[rng.randrange(len(nodes))]
            destination = _other_node(rng, net, nodes, origin, cfg.demand.min_trip_m)
        trips.append((t, origin, destination))
    return trips


# --------------------------------------------------------------------------- population & fleet


def _persona(rng: random.Random, p) -> Persona:
    age = weighted_choice(rng, [a for a, _ in AGE_GROUPS], [w for _, w in AGE_GROUPS])
    purpose = weighted_choice(rng, [a for a, _ in PURPOSES], [w for _, w in PURPOSES])
    vot = min(150.0, max(4.0, lognormal(rng, p.vot_median, p.vot_sigma)))
    patience = min(1800.0, max(120.0, lognormal(rng, p.patience_median_s, p.patience_sigma)))
    limited = rng.random() < p.mobility_limited_share
    return Persona(age, purpose, vot, patience, p.walk_speed_mps, p.max_walk_m, limited)


def generate_travelers(cfg, net: RoadNetwork, rng: random.Random, dt: float, clock0: float = 0.0) -> dict[str, Traveler]:
    """Create the traveller population, ordered by need time: one trip need each, or a day plan each.

    With ``demand.plan`` the demand pattern supplies where the first and the last activity are and
    when the first one ends; the template supplies the rest (see :func:`agmsim.plans.build_plan`).
    """
    n = int(cfg.agents.traveler.count)
    trips = DEMANDS.get(cfg.demand.pattern)(cfg, net, rng, n)
    trips = sorted(trips, key=lambda trip: (trip[0], trip[1], trip[2]))
    width = max(3, len(str(n)))
    travelers: dict[str, Traveler] = {}
    for i, (t, origin, destination) in enumerate(trips):
        persona = _persona(rng, cfg.agents.traveler.personas)
        route = net.route(origin, destination)
        tid = f"H{i + 1:0{width}d}"
        travelers[tid] = Traveler(
            id=tid,
            persona=persona,
            origin=origin,
            destination=destination,
            t_need=int(t // dt),
            trip_m=route.length,
            trip_s=route.time,
        )
        if cfg.demand.plan:
            travelers[tid].plan = build_plan(cfg.demand.plan, net, rng, t, origin, destination, clock0,
                                             cfg.simulation.duration_seconds, cfg.demand.plan_time_factor)
    return travelers
