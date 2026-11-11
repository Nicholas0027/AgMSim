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
class DefaultPolicy(Policy):
    """Always takes the default action: timetable bus, fixed tariff, hands-off operator."""

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


@register_policy("algorithm", role="operator")
class ResponsiveOperator(Policy):
    """Publishes advisories when the network is disturbed, prices the cordon by its congestion,
    and gives buses priority when they carry more people than cars do."""

    def __init__(self, congested_pct: float = 70.0, free_pct: float = 90.0, **params):
        super().__init__(**params)
        self.congested_pct = float(congested_pct)
        self.free_pct = float(free_pct)

    def decide(self, obs):
        ctx, state = obs.context, obs.state
        disturbed = bool(ctx["incidents"]) or ctx["network_speed_pct_of_free_flow"] < self.congested_pct
        if disturbed and obs.has("publish_advisory"):
            return Decision("publish_advisory", {}, "algorithm: network disturbed")
        if not disturbed and obs.has("withdraw_advisory") and ctx["network_speed_pct_of_free_flow"] >= self.free_pct:
            return Decision("withdraw_advisory", {}, "algorithm: network back to normal")
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
