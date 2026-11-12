"""AgMSim -- Agentic Mobility Simulator.

A testbed for short-term mobility dynamics in which travellers, vehicles,
buses, charging stations and the network operator are all decision-making
agents. For every kind of agent the decision model (rule, algorithm, LLM,
remote, ...) and the agency (what the agent is allowed to decide) are
configured independently.

Quick start::

    from agmsim import Simulation
    result = Simulation("montreal").run()
    print(result.metrics["travelers"])

Plug in your own agent::

    from agmsim import register_policy, Decision

    @register_policy("toll_averse", role="vehicle")
    def toll_averse(obs):
        if obs.trigger == "route":
            best = min(obs.context["routes"], key=lambda r: (r["toll_usd"], r["eta_min"]))
            return Decision("choose_route", {"route": best["route"]}, "cheapest route first")
        return obs.default
"""
__version__ = "0.1.0"

from .actions import ActionSpec, Decision, Param  # noqa: E402
from .agency import AgencyProfile, AgencySpec, parse_agency  # noqa: E402
from .config import Config, ConfigError, list_scenarios  # noqa: E402
from .demand import register_demand  # noqa: E402
from .engine import Result, Simulation  # noqa: E402
from .kinds import Kind, register_kind  # noqa: E402
from .llm import LLMClient, LLMRequest, LLMResponse, register_llm  # noqa: E402
from .metrics import register_metric  # noqa: E402
from .network import RoadNetwork, register_network  # noqa: E402
from .observation import Observation  # noqa: E402
from .policies import Policy, PolicyContext, register_policy  # noqa: E402

__all__ = [
    "ActionSpec", "AgencyProfile", "AgencySpec", "Config", "ConfigError", "Decision", "Kind", "LLMClient",
    "LLMRequest", "LLMResponse", "Observation", "Param", "Policy", "PolicyContext", "Result", "RoadNetwork",
    "Simulation", "list_scenarios", "parse_agency", "register_demand", "register_kind", "register_llm",
    "register_metric", "register_network", "register_policy",
]
