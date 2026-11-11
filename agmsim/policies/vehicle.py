"""Non-LLM decision models for private agentic vehicles."""
from __future__ import annotations

from ..actions import Decision
from ..observation import clock_seconds
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
    * With ``settle_min`` it uses its memory: for that many minutes after it changed route it changes
      again only for a saving of at least ``resettle_saving_min`` minutes. A vehicle that remembers
      nothing (``memory.decisions: 0``) has nothing to go on, and the rule does nothing.

    A myopic heuristic, not an optimal controller.
    """

    def __init__(self, transfer_gain_min: float = 3.0, top_up_below_pct: float = 55.0, explain: bool = True,
                 weigh_fare: bool = False, settle_min: float = 0.0, resettle_saving_min: float = 5.0, **params):
        super().__init__(**params)
        self.transfer_gain_min = float(transfer_gain_min)
        self.weigh_fare = bool(weigh_fare)
        self.top_up_below_pct = float(top_up_below_pct)
        self.explain = bool(explain)                # attach an explanation when proposing a transfer
        self.settle_min = float(settle_min)
        self.resettle_saving_min = float(resettle_saving_min)

    def _settled(self, obs, routes, best):
        """Minutes since the vehicle last changed route, if that is too recent for this change to be worth it."""
        if not self.settle_min or best["route"] == 0:
            return None
        last = next((item for item in reversed(obs.memory) if item.get("did") == "changed route"), None)
        if last is None:
            return None
        since = (obs.time_s - clock_seconds(last["at"])) / 60.0
        if since < self.settle_min and routes[0]["eta_min"] - best["eta_min"] < self.resettle_saving_min:
            return since
        return None

    def _route(self, obs, routes, best, why: str) -> Decision:
        since = self._settled(obs, routes, best)
        if since is not None:
            return Decision("choose_route", {"route": 0},
                            f"algorithm: changed route {since:.0f} min ago; staying unless a change saves "
                            f"{self.resettle_saving_min:g} min or more")
        return Decision("choose_route", {"route": best["route"]}, why)

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
                return self._route(obs, routes, best, "algorithm: fastest known route")
            per_km = obs.state["driving_cost_usd_per_km"]
            best = min(routes, key=lambda r: (r["toll_usd"] + per_km * r["km"] + vot * r["eta_min"], r["route"]))
            return self._route(obs, routes, best, "algorithm: lowest cost for the owner")
        if obs.trigger == "charge":
            return self._station(obs, "algorithm: battery will not reach the destination")
        if obs.trigger == "parked":
            if obs.state["battery"]["charge_pct"] < self.top_up_below_pct:
                return self._station(obs, "algorithm: idle time used for charging")
            return Decision("stay_parked", {}, "algorithm: enough charge")
        return obs.default
