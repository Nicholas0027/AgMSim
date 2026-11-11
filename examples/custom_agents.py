"""Plugging your own agentic components into AgMSim.

Run it with::

    agmsim run examples/custom_scenario.yaml

The scenario file lists this module under ``plugins:``, which is all it takes
for the names registered below to be usable in the config.
"""
from agmsim import Decision, Policy, register_metric, register_policy


# 1. The shortest possible agent: a function from observation to decision.
@register_policy("toll_averse", role="vehicle")
def toll_averse(obs):
    """A vehicle that always takes the cheapest route, however slow."""
    if obs.trigger == "route":
        best = min(obs.context["routes"], key=lambda r: (r["toll_usd"], r["eta_min"]))
        return Decision("choose_route", {"route": best["route"]}, "cheapest route first")
    return obs.default          # everything else: the mechanism's default action


# 2. An agent with memory that learns from feedback during the run.
@register_policy("cautious_bus", role="bus")
class CautiousBus(Policy):
    """Holds for approaching passengers, but stops doing so once it has held too often."""

    def __init__(self, max_holds: int = 10, **params):
        super().__init__(**params)
        self.max_holds = max_holds
        self.holds = 0

    def decide(self, obs):
        arriving = obs.context["passengers_walking_to_this_stop_arrive_in_s"]
        choices = obs.choices("hold", "seconds")          # empty if the bus has no authority to hold
        if arriving and choices and self.holds < self.max_holds:
            self.holds += 1
            return Decision("hold", {"seconds": min(choices)}, f"someone arrives in {arriving[0]} s")
        return Decision("depart")

    def feedback(self, event):
        """Outcomes that concern this agent arrive here (used by LLM agents as memory, too)."""


# 3. A custom metric, reported under ``custom`` in the results.
@register_metric("car_trips_rerouted_share")
def rerouted_share(sim):
    vehicles = sim.world.vehicles.values()
    trips = sum(v.trips for v in vehicles)
    return round(sum(1 for v in vehicles if v.reroutes) / trips, 4) if trips else 0.0
