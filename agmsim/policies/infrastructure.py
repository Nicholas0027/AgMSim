"""Non-LLM decision models for buses, charging stations and the network operator.

Each kind gets a ``rule`` policy (the non-agentic reference behaviour, which
never uses the authority it is given) and an ``algorithm`` policy (a simple
transparent heuristic that does).
"""
from __future__ import annotations

from ..actions import Decision
from .base import Policy, register_policy


@register_policy("rule", role="bus")
@register_policy("rule", role="evcs")
@register_policy("rule", role="operator")
@register_policy("rule", role="signal")
class DefaultPolicy(Policy):
    """Always takes the default action: timetable bus, fixed tariff, hands-off operator, fixed-time signal
    (which, where it cooperates, means doing what the orchestrator asks for)."""

    def decide(self, obs):
        return Decision(obs.default.action, dict(obs.default.args), "rule: default behaviour")


@register_policy("algorithm", role="bus")
class HeadwayBus(Policy):
    """Holds for passengers about to arrive and for bunching; skips empty stops."""

    def __init__(self, max_onboard_for_hold: int = 12, bunching_ratio: float = 0.6, **params):
        super().__init__(**params)
        self.max_onboard_for_hold = int(max_onboard_for_hold)
        self.bunching_ratio = float(bunching_ratio)

    def decide(self, obs):
        ctx = obs.context
        holds = sorted(obs.choices("hold", "seconds"))
        if holds:
            arriving = ctx["passengers_walking_to_this_stop_arrive_in_s"]
            if arriving and obs.state["passengers_on_board"] <= self.max_onboard_for_hold:
                fit = [h for h in holds if h >= arriving[0]]
                if fit:
                    return Decision("hold", {"seconds": fit[0]}, "algorithm: a passenger is about to arrive")
            since = ctx["seconds_since_previous_bus_left_this_stop"]
            if since is not None and since < self.bunching_ratio * ctx["target_headway_s"]:
                return Decision("hold", {"seconds": holds[0]}, "algorithm: too close to the bus ahead")
        if obs.has("depart_skip_next") and ctx["passengers_waiting_at_next_stop"] == 0:
            return Decision("depart_skip_next", {}, "algorithm: nobody needs the next stop")
        return Decision("depart", {}, "algorithm: depart")


@register_policy("algorithm", role="evcs")
class LoadPricingStation(Policy):
    """Raises the price when busy, lowers it when empty; serves the emptiest battery first."""

    def decide(self, obs):
        if obs.trigger == "queue":
            row = min(obs.context["queue"], key=lambda r: (r["battery_pct"], r["vehicle_id"]))
            return Decision("serve", {"vehicle_id": row["vehicle_id"]}, "algorithm: lowest battery first")
        ctx = obs.context
        choices = sorted(obs.choices("set_price", "price_multiplier"))
        if ctx["vehicles_queued"] >= 1:
            want = choices[-1]
        elif ctx["utilisation_since_last_review_pct"] >= 70:
            want = min(choices, key=lambda c: abs(c - 1.25))
        elif ctx["utilisation_since_last_review_pct"] <= 20 and ctx["chargers_in_use"] == 0:
            want = choices[0]
        else:
            want = 1.0
        return Decision("set_price", {"price_multiplier": want}, f"algorithm: load-based multiplier {want}")


@register_policy("algorithm", role="signal")
class PressureSignal(Policy):
    """Gives green to the phase with more vehicles; does what the orchestrator asks for.

    A phase's pressure is its vehicles held at the stop line plus a share of those approaching,
    raised for a direction the orchestrator favours. The green moves when the other phase's
    pressure exceeds the current one's by ``switch_margin`` (a switch costs time), or when the
    current phase has run empty.
    """

    def __init__(self, approaching_weight: float = 0.5, switch_margin: float = 1.0, favour_bonus: float = 1.3, **params):
        super().__init__(**params)
        self.approaching_weight = float(approaching_weight)
        self.switch_margin = float(switch_margin)
        self.favour_bonus = float(favour_bonus)

    def decide(self, obs):
        if not obs.has("keep_phase"):
            return Decision(obs.default.action, dict(obs.default.args), "algorithm: maximum green reached")
        told = obs.context.get("orchestrator") or {}
        phases = obs.context["phases"]
        current = next(p for p in phases if p["has_green"])
        if told.get("asks_for"):
            if told["asks_for"] == current["phase"]:
                return Decision("keep_phase", {}, "algorithm: the orchestrator asks for this phase")
            return Decision("switch_phase", {"phase": told["asks_for"]}, "algorithm: the orchestrator asks for it")

        def pressure(p):
            value = p["waiting"] + self.approaching_weight * p["approaching"]
            return value * (self.favour_bonus if p["phase"] == told.get("favours") else 1.0)

        best = max((p for p in phases if not p["has_green"]), key=lambda p: (pressure(p), p["phase"]))
        empty = current["waiting"] + current["approaching"] == 0
        if pressure(best) > pressure(current) + self.switch_margin or (empty and pressure(best) > 0):
            return Decision("switch_phase", {"phase": best["phase"]},
                            f"algorithm: {best['phase']} has more vehicles ({pressure(best):.1f} against {pressure(current):.1f})")
        return Decision("keep_phase", {}, "algorithm: the current phase still has the most vehicles")


@register_policy("algorithm", role="operator")
class ResponsiveOperator(Policy):
    """Publishes advisories when the network is disturbed, prices the cordon by its congestion,
    and gives buses priority when they carry more people than cars do."""

    def __init__(self, congested_pct: float = 70.0, free_pct: float = 90.0, imbalance: float = 2.0,
                 min_waiting: float = 60.0, hotspot: int = 3, **params):
        super().__init__(**params)
        self.congested_pct = float(congested_pct)
        self.free_pct = float(free_pct)
        self.imbalance = float(imbalance)      # one direction has this many times the other's queue: favour it
        self.min_waiting = float(min_waiting)  # ... and at least this many vehicle seconds since the last review
        self.hotspot = int(hotspot)            # a queue at red of this length: ask that junction for green

    def decide(self, obs):
        ctx, state = obs.context, obs.state
        disturbed = bool(ctx["incidents"]) or ctx["network_speed_pct_of_free_flow"] < self.congested_pct
        if disturbed and obs.has("publish_advisory"):
            return Decision("publish_advisory", {}, "algorithm: network disturbed")
        if not disturbed and obs.has("withdraw_advisory") and ctx["network_speed_pct_of_free_flow"] >= self.free_pct:
            return Decision("withdraw_advisory", {}, "algorithm: network back to normal")
        lights = ctx.get("traffic_lights")
        if lights and obs.has("favour_direction"):       # where the queues are: favour that direction, then single junctions
            waiting, now = lights["vehicle_seconds_waited_since_last_review"], lights["favoured_direction"]
            most, least = sorted(waiting, key=lambda d: (-waiting[d], d))[:2]
            if waiting[most] >= self.min_waiting and waiting[most] >= self.imbalance * max(1, waiting[least]) and now != most:
                return Decision("favour_direction", {"direction": most},
                                f"algorithm: {waiting[most]} vehicle seconds waited {most}, {waiting[least]} {least}")
            other = least if now == most else most
            if now is not None and waiting[other] >= self.imbalance * max(1, waiting[now]):      # it has tipped the other way
                return Decision("favour_direction", {"direction": "none"}, f"algorithm: {other} now waits far more")
            worst = lights["longest_queues_at_red"][:1]
            if worst and worst[0]["waiting"] >= self.hotspot and obs.has("guide_signal"):
                return Decision("guide_signal", {"node": worst[0]["junction"], "phase": worst[0]["phase"]},
                                f"algorithm: {worst[0]['waiting']} vehicles held at junction {worst[0]['junction']}")
        if obs.has("set_toll"):
            choices = sorted(obs.choices("set_toll", "usd"))
            now = state["cordon_toll_usd"]
            idx = min(range(len(choices)), key=lambda i: abs(choices[i] - now))
            speed = ctx["cordon"]["speed_pct_of_free_flow"]
            if speed < self.congested_pct and idx + 1 < len(choices):
                return Decision("set_toll", {"usd": choices[idx + 1]}, f"algorithm: cordon at {speed}% of free flow")
            if speed >= self.free_pct and idx > 0:
                return Decision("set_toll", {"usd": choices[idx - 1]}, f"algorithm: cordon flowing at {speed}%")
        bus_people = ctx["travellers_waiting_for_a_bus"] + ctx["travellers_on_buses"]
        if obs.has("enable_bus_priority") and bus_people > ctx["travellers_in_cars"]:
            return Decision("enable_bus_priority", {}, "algorithm: more people depend on buses")
        if obs.has("disable_bus_priority") and bus_people <= ctx["travellers_in_cars"] // 2:
            return Decision("disable_bus_priority", {}, "algorithm: few bus passengers")
        return Decision("no_change", {}, "algorithm: nothing to change")
