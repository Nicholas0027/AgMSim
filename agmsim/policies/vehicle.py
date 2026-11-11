"""Non-LLM decision models for private agentic vehicles."""
from __future__ import annotations

from ..actions import Decision
from .base import Policy, register_policy


@register_policy("rule", role="vehicle")
class RuleVehicle(Policy):
    """Fixed rules: take the fastest known route, never propose anything, charge only when forced to."""

    def decide(self, obs):
        if obs.trigger == "route":
            best = min(obs.context["routes"], key=lambda r: (r["eta_min"], r["route"]))
            return Decision("choose_route", {"route": best["route"]}, "rule: fastest known route")
        return Decision(obs.default.action, dict(obs.default.args), "rule: default behaviour")


@register_policy("algorithm", role="vehicle")
class OwnerCostVehicle(Policy):
    """Minimises the owner's generalised cost where the owner's preferences are known.

    * Route: fastest route; if the owner's value of time is known (``state.owner``), the route with the lowest
      ``toll + driving cost + value_of_time * time`` instead.
    * Transfer: proposes parking and continuing by bus when that saves at least ``transfer_gain_min`` minutes
      over the quickest listed driving route. By default on time alone; with ``weigh_fare: true`` only when
      the time saved is also worth more than the fare at the owner's value of time.
    * Charging: the station with the lowest detour, queue and price burden.

    A myopic heuristic, not an optimal controller.
    """

    def __init__(self, transfer_gain_min: float = 3.0, top_up_below_pct: float = 55.0, explain: bool = True,
                 weigh_fare: bool = False, **params):
        super().__init__(**params)
        self.transfer_gain_min = float(transfer_gain_min)
        self.weigh_fare = bool(weigh_fare)
        self.top_up_below_pct = float(top_up_below_pct)
        self.explain = bool(explain)                # attach an explanation when proposing a transfer

    @staticmethod
    def _vot(obs):
        owner = obs.state.get("owner")
        return owner["value_of_time_usd_per_h"] / 60.0 if owner else None

    def _station(self, obs, why: str) -> Decision:
        vot = self._vot(obs) or 0.3

        def burden(row):
            queue_min = 8.0 * max(0, row["vehicles_queued"] + row["chargers_in_use"] - row["chargers"] + 1)
            detour_min = 2.0 * (row["km"] + row.get("then_to_destination_km", 0.0))
            return vot * (detour_min + queue_min) + 5.0 * row["price_usd_per_kwh"]

        best = min(obs.context["stations"], key=lambda r: (burden(r), r["station"]))
        return Decision("charge", {"station": best["station"]}, why)

    def decide(self, obs):
        if obs.trigger == "route":
            routes = obs.context["routes"]
            vot = self._vot(obs)
            options = obs.context.get("bus_transfer_options")
            if options and obs.has("suggest_transfer"):
                best = min(options, key=lambda o: (o["total_min"], o["stop"]))
                gain = min(r["eta_min"] for r in routes) - best["total_min"]
                worth = not self.weigh_fare or vot is None or vot * gain > best["bus_fare_usd"]
                if gain >= self.transfer_gain_min and worth:
                    message = (f"The road ahead is slow. Taking bus line {best['line']} from stop {best['stop']} "
                               f"should get you there about {gain:.0f} min sooner.")
                    args = {"stop": best["stop"], "message": message} if self.explain else {"stop": best["stop"]}
                    return Decision("suggest_transfer", args,
                                    f"algorithm: bus transfer saves {gain:.1f} min")
            if vot is None:
                best = min(routes, key=lambda r: (r["eta_min"], r["route"]))
                return Decision("choose_route", {"route": best["route"]}, "algorithm: fastest known route")
            per_km = obs.state["driving_cost_usd_per_km"]
            best = min(routes, key=lambda r: (r["toll_usd"] + per_km * r["km"] + vot * r["eta_min"], r["route"]))
            return Decision("choose_route", {"route": best["route"]}, "algorithm: lowest cost for the owner")
        if obs.trigger == "charge":
            return self._station(obs, "algorithm: battery will not reach the destination")
        if obs.trigger == "parked":
            if obs.state["battery"]["charge_pct"] < self.top_up_below_pct:
                return self._station(obs, "algorithm: idle time used for charging")
            return Decision("stay_parked", {}, "algorithm: enough charge")
        return obs.default
