"""Destinations as points of interest (POIs).

A POI is a named place of some type -- work, shop, school, leisure, medical, or whatever a scenario
calls it -- at a node of the network, optionally with opening hours and a weight (how strongly it
attracts trips). POIs are part of the physical infrastructure, so they are the same for every seed.

They matter in three places: a trip's destination can be a POI that fits the trip's purpose
(``demand.to_pois``); an activity of a day plan can take place at a POI of a given type
(``at: {poi: shop}``), whose opening hours it then has; and an external event can close a POI early
(``poi_closure``).

``pois`` in a scenario lists them. An entry either names one place
(``{id: market, type: shop, node: 14, name: Central market, closes: "18:30"}``) or asks for several of a
type (``{type: shop, count: 4, closes: "18:30"}``), which are put down at nodes that are still free.
"""
from __future__ import annotations

import math
import random

from .entities import Poi
from .plans import parse_clock
from .rng import weighted_choice

#: The type of place a trip of each purpose goes to.
PURPOSE_PLACE = {"commute": "work", "errand": "shop", "leisure": "leisure", "school": "school", "medical": "medical"}


def build_pois(specs: list, net, clock0: float = 0.0) -> dict:
    pois: dict = {}
    nodes = sorted(net.nodes)
    for spec in specs:
        kind = str(spec["type"])
        hours = {"open_s": parse_clock(spec["opens"]) - clock0 if "opens" in spec else None,
                 "close_s": parse_clock(spec["closes"]) - clock0 if "closes" in spec else None}
        weight = float(spec.get("weight", 1.0))
        if "count" in spec:
            rng = random.Random(f"poi|{kind}")              # the same places for every seed
            for k in range(int(spec["count"])):
                free = [n for n in nodes if n not in {p.node for p in pois.values()}] or nodes
                pid = f"{kind}{k + 1}"
                pois[pid] = Poi(pid, kind, free[rng.randrange(len(free))], f"{kind} {k + 1}", weight, **hours)
        else:
            node = int(spec["node"])
            if node not in net.nodes:
                raise ValueError(f"pois: there is no node {node}")
            pid = str(spec.get("id") or f"{kind}{len(pois) + 1}")
            pois[pid] = Poi(pid, kind, node, str(spec.get("name") or pid), weight, **hours)
    return pois


def choose_for_purpose(pois: dict, purpose: str, origin: int, net, rng, scale_m: float):
    """A place for a trip of this purpose: nearer and weightier places are likelier. None if no place fits."""
    kind = PURPOSE_PLACE.get(purpose, purpose)
    fits = [p for p in pois.values() if p.type == kind and p.node != origin]
    if not fits:
        return None
    return weighted_choice(rng, fits, [p.weight * math.exp(-net.length(origin, p.node) / scale_m) for p in fits])


def on_the_way(pois: dict, kind: str, previous: int, destination: int, net):
    """The place of this type that adds the least detour between two nodes. None if there is none."""
    fits = [p for p in pois.values() if p.type == kind]
    if not fits:
        return None
    return min(fits, key=lambda p: (net.length(previous, p.node) + net.length(p.node, destination), p.id))
