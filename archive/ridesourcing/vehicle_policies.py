"""Non-LLM decision models for vehicles."""
from __future__ import annotations

from ..actions import Decision
from .base import Policy, register_policy


def _fastest_route(obs) -> Decision:
    routes = obs.context["routes"]
    best = min(routes, key=lambda r: (r["eta_min"], r["route"]))
    return Decision("choose_route", {"route": best["route"]}, "fastest known route")


@register_policy("rule", role="vehicle")
class RuleVehicle(Policy):
    """Compliant vehicle with fixed rules.

    Accepts every proposal, never repositions, takes the fastest known route,
    and -- if it is allowed to pick its own work -- claims the closest request.
    """

    def decide(self, obs):
        if obs.trigger == "proposal":
            return Decision("accept_request", {}, "rule: accept every proposal")
        if obs.trigger == "idle":
            rows = obs.context.get("open_requests") or []
            if rows and obs.has("claim_request"):
                row = min(rows, key=lambda r: (r["pickup_eta_min"], r["request_id"]))
                return Decision("claim_request", {"request_id": row["request_id"]}, "rule: closest open request")
            return Decision("stay", {}, "rule: wait for work")
        if obs.trigger == "route":
            return _fastest_route(obs)
        return obs.default          # e.g. low battery: nearest station


@register_policy("algorithm", role="vehicle")
class GreedyVehicle(Policy):
    """Objective-driven heuristic.

    Scores each option by expected net benefit under the vehicle's objective
    (``profit``, ``service`` or ``balanced``; anything else is treated as
    balanced):

    * proposals are declined when they are unprofitable or too far away
      (profit), never (service), or only when both hold (balanced);
    * open requests are claimed by net earnings per minute (profit), by rider
      waiting time (service), or a mix;
    * an idle vehicle moves towards the candidate node with the largest excess
      of recent demand over nearby supply, if that beats staying put.

    This is a myopic baseline, not an optimal controller.
    """

    def __init__(self, max_pickup_min: float = 10.0, min_net_usd: float = 0.0,
                 max_reposition_min: float = 6.0, scarcity_multiplier: float = 1.25, **params):
        super().__init__(**params)
        self.max_pickup_min = float(max_pickup_min)
        self.min_net_usd = float(min_net_usd)
        self.max_reposition_min = float(max_reposition_min)
        self.scarcity_multiplier = float(scarcity_multiplier)
        self.w_profit, self.w_service = 0.5, 0.5

    def bind(self, ctx):
        super().bind(ctx)
        self.w_profit, self.w_service = {"profit": (1.0, 0.0), "service": (0.0, 1.0)}.get(ctx.objective, (0.5, 0.5))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _net(obs, row) -> float:
        return row["fare_usd"] - obs.state["operating_cost_usd_per_km"] * (row["pickup_km"] + row["trip_km"])

    def _offer_args(self, obs, action: str) -> dict:
        choices = obs.choices(action, "price_multiplier")
        if not choices or self.w_profit < 1.0:
            return {}
        market = obs.context.get("market", {})
        scarce = market.get("open_requests_visible", 0) > market.get("other_available_vehicles_visible", 0) + 1
        want = self.scarcity_multiplier if scarce else 1.0
        return {"price_multiplier": min(choices, key=lambda c: (abs(c - want), c))}

    # -- decisions ----------------------------------------------------------

    def decide(self, obs):
        if obs.trigger == "proposal":
            row = obs.context["proposal"]
            net = self._net(obs, row)
            if obs.has("decline_request") and self.w_service < 1.0:
                too_far = row["pickup_eta_min"] > self.max_pickup_min
                unprofitable = net < self.min_net_usd
                if self.w_profit >= 1.0 and (too_far or unprofitable):
                    return Decision("decline_request", {}, f"algorithm: net {net:.2f} USD, pickup {row['pickup_eta_min']} min")
                if self.w_profit < 1.0 and too_far and unprofitable:
                    return Decision("decline_request", {}, "algorithm: far away and unprofitable")
            return Decision("accept_request", self._offer_args(obs, "accept_request"),
                            f"algorithm: net {net:.2f} USD")

        if obs.trigger == "idle":
            rows = obs.context.get("open_requests") or []
            if rows and obs.has("claim_request"):
                def score(row):
                    minutes = row["pickup_eta_min"] + row["trip_min"] + 1.0
                    return (self.w_profit * 10.0 * self._net(obs, row) / minutes
                            + self.w_service * (row["rider_waited_min"] - row["pickup_eta_min"]))

                best = min(rows, key=lambda r: (-score(r), r["request_id"]))
                if not (self.w_profit >= 1.0 and self._net(obs, best) < self.min_net_usd):
                    args = {"request_id": best["request_id"], **self._offer_args(obs, "claim_request")}
                    return Decision("claim_request", args, "algorithm: best-scoring open request")
            if obs.has("reposition"):
                def pull(row):
                    return row["recent_requests_nearby"] - row.get("other_available_vehicles_nearby", 0)

                here_pull = pull(obs.context["here"])
                reachable = [r for r in obs.context["reposition_candidates"] if r["eta_min"] <= self.max_reposition_min]
                if reachable:
                    best = min(reachable, key=lambda r: (-(pull(r) - 0.3 * r["eta_min"]), r["node"]))
                    if pull(best) >= 1 and pull(best) > here_pull:
                        return Decision("reposition", {"node": best["node"]},
                                        f"algorithm: demand-supply gap {pull(best)} vs {here_pull} here")
            if obs.has("charge") and not rows and obs.state["battery"]["charge_pct"] < 45:
                return self._charge(obs, "algorithm: quiet moment, topping up")
            return Decision("stay", {}, "algorithm: nothing better to do")

        if obs.trigger == "route":
            return _fastest_route(obs)
        if obs.trigger == "charge":
            return self._charge(obs, "algorithm: battery low")
        return obs.default

    def _charge(self, obs, why: str) -> Decision:
        """Pick the station with the lowest time-plus-price burden."""
        def burden(row):
            wait = 8.0 * max(0, row["vehicles_queued"] + row["chargers_in_use"] - row["chargers"] + 1)
            return row["eta_min"] + wait + 20.0 * self.w_profit * row["price_usd_per_kwh"]

        best = min(obs.context["stations"], key=lambda r: (burden(r), r["station"]))
        return Decision("charge", {"station": best["station"]}, why)
