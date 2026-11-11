"""Non-LLM decision models for travellers."""
from __future__ import annotations

import math

from ..actions import Decision
from ..plans import REVIEW_GAIN_USD, option_value
from .base import Policy, register_policy


@register_policy("rule", role="traveler")
class RuleTraveler(Policy):
    """Fixed behavioural rules.

    Takes the habitual mode, follows the vehicle's suggestions, and leaves a
    bus stop once the wait exceeds the traveller's patience.
    """

    def decide(self, obs):
        if obs.trigger == "need":
            return Decision("choose_service", dict(obs.default.args), "rule: habitual mode")
        if obs.trigger == "replan":
            best = min(obs.context["options"], key=lambda o: (o["wait_min"] + o["travel_min"], o["option"]))
            return Decision("choose_service", {"option": best["option"]}, "rule: fastest remaining option")
        if obs.trigger == "suggestion":
            return Decision("accept_suggestion", {}, "rule: follow the vehicle")
        if obs.trigger == "wait_check":
            patience = obs.state["persona"]["patience_min"]
            if obs.context["waited_min"] > patience and obs.context["next_bus_eta_min"] > 2.0:
                return Decision("give_up_waiting", {}, "rule: waited longer than patience")
            return Decision("keep_waiting", {}, "rule: within patience")
        return obs.default


@register_policy("utility", role="traveler")
class UtilityTraveler(Policy):
    """Random-utility (logit) choice using the traveller's own value of time.

    Generalised cost of an option, in USD::

        cost = price + VOT * (wait_weight * wait + travel + (walk_weight - 1) * walking)

    Options are chosen with probability proportional to ``exp(-cost / scale)``
    (set ``deterministic: true`` to always take the cheapest).

    The parameters are behavioural assumptions, not estimates from data;
    calibrate them before drawing empirical conclusions.
    """

    def __init__(self, scale: float = 1.5, wait_weight: float = 1.5, walk_weight: float = 2.0,
                 price_tolerance: float = 0.2, deterministic: bool = False, plan_gain_usd: float = REVIEW_GAIN_USD,
                 **params):
        super().__init__(**params)
        self.scale = float(scale)
        self.wait_weight = float(wait_weight)
        self.walk_weight = float(walk_weight)
        self.price_tolerance = float(price_tolerance)
        self.deterministic = bool(deterministic)
        self.plan_gain_usd = float(plan_gain_usd)       # a change of plan must be worth at least this much

    def _vot(self, obs) -> float:
        return obs.state["persona"]["value_of_time_usd_per_h"] / 60.0

    def _cost(self, obs, option: dict) -> float:
        walking = option["travel_min"] if option["option"] == "walk" else option.get("walk_min", 0.0)
        return option["price_usd"] + self._vot(obs) * (
            self.wait_weight * option["wait_min"] + option["travel_min"] + (self.walk_weight - 1.0) * walking)

    def decide(self, obs):
        if obs.trigger in ("need", "replan"):
            options = obs.context["options"]
            costs = [self._cost(obs, o) for o in options]
            if self.deterministic or len(options) == 1:
                idx = min(range(len(options)), key=lambda i: (costs[i], options[i]["option"]))
            else:
                low = min(costs)
                weights = [math.exp(-(c - low) / self.scale) for c in costs]
                x = self.rng.random() * sum(weights)
                idx, acc = len(options) - 1, 0.0
                for i, w in enumerate(weights):
                    acc += w
                    if x < acc:
                        idx = i
                        break
            return Decision("choose_service", {"option": options[idx]["option"]},
                            f"utility: generalised cost {costs[idx]:.2f} USD")

        if obs.trigger == "plan_review":          # the edit whose projected outcome is worth most to this person
            options = obs.context["options"]
            values = [option_value(obs.state["preferences"], obs.context, o) for o in options]
            best = max(range(len(options)), key=lambda i: (values[i], -i))
            if values[best] - values[0] < self.plan_gain_usd:
                return Decision("keep_plan", {}, "utility: no change of plan is worth the trouble")
            choice = options[best]["choice"]
            return Decision(choice["action"], {k: v for k, v in choice.items() if k != "action"},
                            f"utility: worth {values[best] - values[0]:+.2f} USD against keeping the plan")

        persona = obs.state["persona"]
        if obs.trigger == "suggestion":
            proposal = obs.context["proposal"]
            vot = self._vot(obs)
            bus = proposal["bus_fare_usd"] + vot * (
                proposal["drive_to_stop_min"] + self.wait_weight * proposal["bus_wait_min"]
                + proposal["bus_ride_min"] + self.walk_weight * proposal["walk_min"])
            car = vot * obs.context["if_you_keep_driving_min"]
            if bus < car:
                return Decision("accept_suggestion", {}, f"utility: transfer costs {bus:.2f} vs {car:.2f} USD")
            return Decision("decline_suggestion", {}, f"utility: staying in the car costs {car:.2f} vs {bus:.2f} USD")

        if obs.trigger == "wait_check":
            ctx = obs.context
            waited, remaining = ctx["waited_min"], ctx["next_bus_eta_min"]
            alternatives = ctx.get("alternatives") or []
            if alternatives and waited + remaining > 1.5 * persona["patience_min"]:
                return Decision("give_up_waiting", {}, "utility: expected wait exceeds tolerance")
            return Decision("keep_waiting", {}, "utility: waiting still worthwhile")
        return obs.default
