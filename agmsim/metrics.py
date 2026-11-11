"""Outcome metrics, computed from the final world state.

Outcomes for travellers, use of private agentic vehicles, bus and
charging-station operation, the operator's interventions, and -- specific to agentic systems -- how often
agents were asked, how often they used their authority, how often their
output was invalid, and what the LLM calls cost.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from .entities import TState
from .plans import rates, score
from .registry import Registry

METRICS = Registry("metric")


def register_metric(name: str):
    """Register a custom metric ``(sim) -> value``; it appears under ``custom`` in the results."""
    return METRICS.register(name)


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 3) if values else 0.0


def _pct(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return round(values[min(len(values) - 1, int(math.ceil(q * len(values))) - 1)], 3)


def gini(values) -> float:
    """Gini coefficient of non-negative values (0 = equal)."""
    values = sorted(max(0.0, v) for v in values)
    n, total = len(values), sum(values)
    if n == 0 or total <= 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(values))
    return round((2.0 * cum) / (n * total) - (n + 1.0) / n, 4)


def _plan_trips(planners) -> list:
    """Every trip of the day plans, made or under way, in the shape the trip metrics expect."""
    trips = []
    for t in planners:
        for leg in t.plan.legs:
            trips.append(SimpleNamespace(persona=t.persona, car=t.car, state=TState.ARRIVED, mode=leg["mode"],
                                         t_need=leg["depart"], t_arrival=leg["arrive"], paid=leg["paid"],
                                         waited_ticks=leg["waited"], cancels=leg["cancels"]))
        leg = t.plan.leg
        if leg is not None:
            trips.append(SimpleNamespace(persona=t.persona, car=t.car, state=t.state, mode=t.mode, t_need=leg["depart"],
                                         t_arrival=None, paid=t.paid - leg["paid"],
                                         waited_ticks=t.waited_ticks - leg["waited"], cancels=t.cancels - leg["cancels"]))
    return trips


def _plan_metrics(sim, planners) -> dict:
    """How the day plans went: MATSim's score of each executed plan, in USD, and what was changed."""
    horizon = sim.n_ticks * sim.dt
    scores = [score(t.plan, rates(t.persona.value_of_time, sim.cfg.scoring), horizon, t.paid) for t in planners]
    late = [s["late_s"] for s in scores]
    kind, reviews = sim.kinds["traveler"], sum(t.plan.reviews for t in planners)
    (worth_n, worth_edit), (idle_n, idle_edit) = kind.review_edit[True], kind.review_edit[False]
    return {
        "people": len(planners),
        "completed": sum(1 for t in planners if t.state == TState.ARRIVED),
        "score_usd_mean": _mean(s["total"] for s in scores),
        "score_usd_p10": _pct([s["total"] for s in scores], 0.1),
        "activities_usd_mean": _mean(s["activities"] for s in scores),
        "late_usd_mean": _mean(s["late"] for s in scores),
        "travel_usd_mean": _mean(s["travel"] for s in scores),
        "money_usd_mean": _mean(s["money"] for s in scores),
        "people_late": sum(1 for x in late if x > 0),
        "late_min_mean": round(sum(late) / 60.0 / len(planners), 3),
        "activities_missed": sum(s["missed"] for s in scores),          # kept in the plan but not performed at all
        "activities_skipped": sum(a.skipped for t in planners for a in t.plan.activities),
        "reviews": reviews,
        "edits": sum(t.plan.edits for t in planners),
        # per review: worth of the best listed option minus worth of the option taken, in the person's own terms
        "review_regret_usd_mean": round(kind.review_regret / reviews, 4) if reviews else None,
        "review_best_rate": round(kind.review_best / reviews, 4) if reviews else None,
        # changed the plan when a change was worth making (projected gain of at least REVIEW_GAIN_USD), and when not;
        # their difference is 0 for a model that ignores what the options lead to and 1 for one that follows it
        "changed_when_worth_it": round(worth_edit / worth_n, 4) if worth_n else None,
        "changed_when_not": round(idle_edit / idle_n, 4) if idle_n else None,
        "review_sensitivity": round(worth_edit / worth_n - idle_edit / idle_n, 4) if worth_n and idle_n else None,
    }


def compute_metrics(sim) -> dict:
    w, cfg, dt = sim.world, sim.cfg, sim.dt
    out: dict = {}

    # ---------------------------------------------------------------- travellers
    planners = [t for t in w.travelers.values() if t.plan is not None]
    travelers = [t for t in w.travelers.values() if t.plan is None and t.t_need < sim.n_ticks] + _plan_trips(planners)
    if travelers:
        arrived = [t for t in travelers if t.state == TState.ARRIVED]
        modes = {}
        for t in arrived:
            modes[t.mode] = modes.get(t.mode, 0) + 1
        limited = [t for t in travelers if t.persona.mobility_limited]
        times = [(t.t_arrival - t.t_need) * dt for t in arrived]
        costs = [t.paid + t.persona.value_of_time * (t.t_arrival - t.t_need) * dt / 3600.0 for t in arrived]
        waits = [t.waited_ticks * dt for t in arrived if t.waited_ticks]
        ranked = sorted(t.persona.value_of_time for t in arrived) or [0.0]
        vot_low, vot_high = ranked[len(ranked) // 3], ranked[min(len(ranked) - 1, 2 * len(ranked) // 3)]
        out["travelers"] = {
            "trips": len(travelers),
            "completed": len(arrived),
            "completion_rate": round(len(arrived) / len(travelers), 4),
            "abandoned": sum(1 for t in travelers if t.state == TState.ABANDONED),
            "unfinished": sum(1 for t in travelers if t.state not in (TState.ARRIVED, TState.ABANDONED)),
            "mode_share": {m: round(n / len(arrived), 4) for m, n in sorted(modes.items())} if arrived else {},
            "trip_time_s_mean": _mean(times),
            "trip_time_s_p90": _pct(times, 0.9),
            "trip_time_gini": gini(times),
            "bus_wait_s_mean": _mean(waits),
            "paid_usd_mean": _mean(t.paid for t in arrived),
            "generalised_cost_usd_mean": _mean(costs),
            "generalised_cost_usd_mean_low_vot": _mean(c for c, t in zip(costs, arrived) if t.persona.value_of_time <= vot_low),
            "generalised_cost_usd_mean_high_vot": _mean(c for c, t in zip(costs, arrived) if t.persona.value_of_time >= vot_high),
            "trip_time_s_mean_car_owners": _mean(x for x, t in zip(times, arrived) if t.car),
            "gave_up_waiting": sum(t.cancels for t in travelers),
            "completion_rate_mobility_limited": (
                round(sum(1 for t in limited if t.state == TState.ARRIVED) / len(limited), 4) if limited else None),
        }

    if planners:
        out["plans"] = _plan_metrics(sim, planners)

    # ---------------------------------------------------------------- outcome variables of the whole system
    # The four headline measures: what travellers wait, and how much, how long and how fast vehicles move.
    done = [t for t in travelers if t.state == TState.ARRIVED]
    metres = (sum(v.m_loaded + v.m_empty for v in w.vehicles.values()) + sum(b.m_total for b in w.buses.values())
              + sum(c.m_loaded + c.m_empty for c in w.cabs.values()))
    seconds = (sum(v.s_moving for v in w.vehicles.values()) + sum(b.s_moving for b in w.buses.values())
               + sum(c.s_moving for c in w.cabs.values()))
    out["outcomes"] = {
        # time spent waiting for transport (at a stop, or for a car that was called), per completed trip;
        # trips that needed no wait count as 0
        "wait_time_s_mean": _mean(t.waited_ticks * dt for t in done),
        "vmt_miles": round(metres / 1609.344, 2),                  # vehicle miles travelled: cars, on-demand cars, buses
        "vkt_km": round(metres / 1000.0, 2),
        "vht_hours": round(seconds / 3600.0, 3),                   # vehicle hours travelled: time on the road
        "traffic_speed_mph": round(metres / 1609.344 / (seconds / 3600.0), 2) if seconds else None,   # VMT / VHT
        "traffic_speed_kmh": round(metres / 1000.0 / (seconds / 3600.0), 2) if seconds else None,
    }

    # ---------------------------------------------------------------- private agentic vehicles
    if w.vehicles:
        vehicles = list(w.vehicles.values())
        loaded, empty = sum(v.m_loaded for v in vehicles), sum(v.m_empty for v in vehicles)
        total_m = loaded + empty
        drive, free = sum(v.drive_s_owner for v in vehicles), sum(v.free_s_owner for v in vehicles)
        stops = sum(v.charge_stops for v in vehicles)
        by_level: dict = {}
        for v in vehicles:
            d, f = by_level.setdefault(sim.agency[v.id].level, [0.0, 0.0])
            by_level[sim.agency[v.id].level] = [d + v.drive_s_owner, f + v.free_s_owner]
        vk = sim.kinds["vehicle"]
        (pay_n, pay_paid), (free_n, free_paid) = vk.owner_pay[True], vk.owner_pay[False]
        out["vehicles"] = {
            "vehicles": len(vehicles),
            "trips": sum(v.trips for v in vehicles),
            "vkt_km": round(total_m / 1000.0, 2),
            "empty_km_share": round(empty / total_m, 4) if total_m else 0.0,
            "travel_time_ratio": round(drive / free, 4) if free else 0.0,   # door-to-door car time / free-flow time
            "travel_time_ratio_by_level": {k: round(d / f, 4) for k, (d, f) in sorted(by_level.items()) if f},
            "owner_choices": vk.owner_choices,                          # route choices made knowing the owner's preferences
            "owner_alignment_rate": round(vk.owner_aligned / vk.owner_choices, 4) if vk.owner_choices else None,
            "owner_regret_usd_mean": round(vk.owner_regret / vk.owner_choices, 4) if vk.owner_choices else None,
            # paid a toll when the owner was better off paying, minus paid when better off not paying:
            # near 0 for a vehicle that ignores its owner's preferences, 1 for one that follows them
            "owner_sensitivity": (round(pay_paid / pay_n - free_paid / free_n, 4) if pay_n and free_n else None),
            "reroutes": sum(v.reroutes for v in vehicles),
            "suggestions": sum(v.suggestions for v in vehicles),
            "suggestions_accepted": sum(v.suggestions_accepted for v in vehicles),
            "charge_stops": sum(v.charge_stops for v in vehicles),
            "charge_queue_s_mean": round(sum(v.s_queue for v in vehicles) / stops, 2) if stops else 0.0,
            "energy_kwh": round(sum(v.kwh_charged for v in vehicles), 2),
            "energy_cost_usd": round(sum(v.energy_cost for v in vehicles), 2),
            "tolls_usd": round(sum(v.tolls for v in vehicles), 2),
            "battery_depleted_ticks": sum(v.depleted for v in vehicles),
            "co2_kg": round(total_m / 1000.0 * cfg.metrics.vehicle_co2_g_per_km / 1000.0, 2),
        }

    # ---------------------------------------------------------------- buses, stations, operator
    if w.buses:
        buses = list(w.buses.values())
        bus_km = sum(b.m_total for b in buses) / 1000.0
        out["bus"] = {
            "buses": len(buses),
            "boardings": sum(b.boardings for b in buses),
            "km": round(bus_km, 2),
            "holds": sum(b.holds for b in buses),
            "stops_skipped": sum(b.skips for b in buses),
            "left_behind": sum(b.left_behind for b in buses),
            "co2_kg": round(bus_km * cfg.metrics.bus_co2_g_per_km / 1000.0, 2),
        }
    if w.stations:
        stations = list(w.stations.values())
        capacity = sum(s.chargers for s in stations) * max(1, w.tick)
        out["evcs"] = {
            "stations": len(stations),
            "sessions": sum(s.sessions for s in stations),
            "energy_kwh": round(sum(s.kwh for s in stations), 2),
            "revenue_usd": round(sum(s.revenue for s in stations), 2),
            "utilisation": round(sum(s.busy_charger_ticks for s in stations) / capacity, 4),
            "sessions_gini": gini(s.sessions for s in stations),        # how unevenly demand is spread over stations
            "price_usd_per_kwh_mean_paid": round(sum(s.revenue for s in stations) / max(1e-9, sum(s.kwh for s in stations)), 4)
            if any(s.kwh for s in stations) else 0.0,
        }
    if w.pois:
        by_type: dict = {}
        for poi in w.pois.values():
            by_type[poi.type] = by_type.get(poi.type, 0) + poi.visits
        out["pois"] = {
            "places": len(w.pois),
            "visits": sum(p.visits for p in w.pois.values()),
            "visits_by_type": dict(sorted(by_type.items())),
            "found_closed": sum(p.found_closed for p in w.pois.values()),     # arrived after the place had closed
            "visits_gini": gini(p.visits for p in w.pois.values()),          # how unevenly visits are spread over places
        }
    if w.cabs:
        cabs, kind = list(w.cabs.values()), sim.kinds["ondemand"]
        loaded, empty = sum(c.m_loaded for c in cabs), sum(c.m_empty for c in cabs)
        revenue = sum(c.revenue for c in cabs)
        cost = (loaded + empty) / 1000.0 * cfg.agents.ondemand.operating_cost_per_km
        out["ondemand"] = {
            "cars": len(cabs),
            "requests": kind.requests,
            "rides": sum(c.trips for c in cabs),
            "served_share": round(sum(c.trips for c in cabs) / kind.requests, 4) if kind.requests else 0.0,
            "cancelled": kind.cancelled,                                   # the traveller gave up waiting
            "pickup_wait_s_mean": round(kind.pickup_wait_ticks * dt / kind.pickups, 2) if kind.pickups else 0.0,
            "declined": sum(c.declined for c in cabs),
            "repositions": sum(c.repositions for c in cabs),
            "reroutes": sum(c.reroutes for c in cabs),
            "vkt_km": round((loaded + empty) / 1000.0, 2),
            "empty_km_share": round(empty / (loaded + empty), 4) if loaded + empty else 0.0,
            "revenue_usd": round(revenue, 2),
            "profit_usd": round(revenue - cost, 2),
            "rides_per_car_gini": gini(c.trips for c in cabs),             # how unevenly the work is spread over the cars
        }
    if w.signals:
        signals = list(w.signals.values())
        lines = [w.net.stop_lines[lid] for s in signals for ids in s.phases for lid in ids]
        held, through = sum(s.held for s in signals), sum(line.passed for line in lines)
        out["signals"] = {
            "junctions": len(signals),
            "switches": sum(s.switches for s in signals),
            "crossings": held + through,                                   # vehicles that passed a traffic light
            "stopped_share": round(held / (held + through), 4) if held + through else 0.0,
            "wait_s_mean": round(sum(s.wait_ticks for s in signals) * dt / (held + through), 2) if held + through else 0.0,
            "queue_max": max(s.queue_max for s in signals),               # most vehicles held at one junction at once
            "requests_followed": sum(s.requests_followed for s in signals),
        }
    if w.operator is not None:
        out["operator"] = {
            "advisories_published": w.operator.advisories,
            "bus_priority_time_share": round(w.operator.priority_ticks / max(1, w.tick), 4),
            "toll_usd_final": w.operator.toll,
            "toll_changes": w.operator.toll_changes,
            "signal_requests": w.operator.signal_requests,
            "toll_revenue_usd": round(w.operator.toll_revenue, 2),
        }

    if any(k.share for k in sim.traffic.kinds.values()):     # how far news about slow roads travelled between agents
        out["knowledge"] = sim.traffic.metrics()

    # ---------------------------------------------------------------- decisions and LLM use
    decisions = {}
    for name, s in sim.stats.items():
        decisions[name] = {
            **s,
            "agency_level": next((sim.agency[a].level for a, sp in sim.specs.items() if sp.kind == name), None),
            "policy": next((sp.policy for sp in sim.specs.values() if sp.kind == name), None),
            "exercised_rate": round(s["exercised"] / s["decided"], 4) if s["decided"] else 0.0,
            "invalid_rate": round(s["invalid"] / s["decided"], 4) if s["decided"] else 0.0,
        }
    out["decisions"] = decisions
    usage = sim.llm_usage()
    out["llm"] = {
        "calls": sum(u["calls"] for u in usage),
        "replayed": sum(u["replayed"] for u in usage),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "cached_input_tokens": sum(u.get("cached_input_tokens", 0) for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "cost_usd": round(sum(u["cost_usd"] for u in usage), 6),
        "errors": sum(u["errors"] for u in usage),
        "fallbacks": sum(s["fallback"] for s in sim.stats.values()),
        "backends": [{k: u[k] for k in ("backend", "model", "calls", "latency_ms_mean", "latency_ms_p50",
                                         "latency_ms_p95")} for u in usage],
        "deciding_wall_s": round(sim.wall["deciding_s"], 2),     # wall-clock time spent waiting for decision models
    }
    custom = {name: METRICS.get(name)(sim) for name in METRICS.names()}
    if custom:
        out["custom"] = custom
    return out


def flatten(metrics: dict, prefix: str = "") -> dict:
    """Flatten nested metrics into ``a.b.c -> number`` (for tables and CSV)."""
    flat = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
    return flat
