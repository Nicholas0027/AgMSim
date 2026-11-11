"""Non-LLM decision models for on-demand cars."""
from __future__ import annotations

from ..actions import Decision
from ..observation import OBJECTIVES
from .base import Policy, register_policy


@register_policy("rule", role="ondemand")
class RuleCab(Policy):
    """Takes every request, keeps its route, waits where it is: never uses the authority it is given."""

    def decide(self, obs):
        return Decision(obs.default.action, dict(obs.default.args), "rule: default behaviour")


@register_policy("algorithm", role="ondemand")
class GreedyCab(Policy):
    """A transparent heuristic for each of the car's decisions.

    * A proposed request is taken unless it pays too little for the time it takes: net earnings
      (fare minus driving cost, the drive to the traveller included) per minute below
      ``min_usd_per_min``. A car told to pursue profit applies this always, a balanced one only
      while other cars are free to take the request, and one told to serve never turns anything down.
    * The fastest listed route is taken.
    * A free car moves to the place with the most recent requests per free car already there, if
      that beats where it is by ``reposition_margin`` and is at most ``max_reposition_min`` away.
    """

    def __init__(self, min_usd_per_min: float = 0.6, max_reposition_min: float = 8.0, reposition_margin: float = 1.0,
                 **params):
        super().__init__(**params)
        self.min_usd_per_min = float(min_usd_per_min)
        self.max_reposition_min = float(max_reposition_min)
        self.reposition_margin = float(reposition_margin)

    def decide(self, obs):
        ctx = obs.context
        if obs.trigger == "proposal":
            objective = obs.state["objective"]
            if obs.has("decline_request") and objective != OBJECTIVES["service"]:
                request = ctx["request"]
                rate = (request["fare_usd"] - ctx["your_driving_cost_usd"]) / max(1.0, request["pickup_min"] + request["trip_min"])
                if rate < self.min_usd_per_min and (objective == OBJECTIVES["profit"] or ctx["other_free_cars"] > 0):
                    return Decision("decline_request", {}, f"algorithm: only {rate:.2f} USD per minute")
            return Decision("accept_request", {}, "algorithm: take the request")
        if obs.trigger == "route":
            best = min(ctx["routes"], key=lambda r: (r["eta_min"], r["route"]))
            return Decision("choose_route", {"route": best["route"]}, f"algorithm: fastest route ({best['eta_min']} min)")
        here = ctx["here"]["recent_requests"] / (1 + ctx["here"]["other_free_cars_here"])
        reachable = [p for p in ctx["places"] if p["drive_min"] <= self.max_reposition_min]
        if reachable:
            best = max(reachable, key=lambda p: (p["recent_requests"] / (1 + p["free_cars_there"]), -p["drive_min"], -p["node"]))
            pull = best["recent_requests"] / (1 + best["free_cars_there"])
            if pull >= here + self.reposition_margin:
                return Decision("reposition", {"node": best["node"]},
                                f"algorithm: {best['recent_requests']} recent requests at node {best['node']}, "
                                f"{best['free_cars_there']} free cars there")
        return Decision("stay", {}, "algorithm: nowhere better to wait")
