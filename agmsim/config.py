"""Typed, validated configuration.

A run is fully described by one :class:`Config`. For every kind of agent
(traveller, vehicle, bus, charging station, network operator) three things are
configured *independently*, because they answer different questions:

* ``use_llm`` / ``non_llm_policy`` -- **how** the agent decides (decision model);
* ``agency`` -- **what** the agent is permitted to decide (decision authority);
* timing and observation options -- **when** it is asked and what it sees.

Unknown keys are rejected so that a typo can never silently change an
experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_type_hints

import yaml

EVENT_TYPES = ("link_condition", "set_agency", "weather", "demand_surge", "poi_closure")
MACHINE_KINDS = ("vehicle", "bus", "evcs", "operator", "signal", "ondemand")
KIND_ORDER = ("operator", "signal", "evcs", "traveler", "bus", "vehicle", "ondemand")   # order in which actions are applied


class ConfigError(ValueError):
    """Raised for invalid configuration."""


# --------------------------------------------------------------------------- world


@dataclass
class SimulationCfg:
    time_step_seconds: float = 1.0      # physical update resolution (decision timing is configured separately)
    duration_seconds: float = 1800.0
    seed: int = 42
    clock_start: Any = 0                # time of day at which the run starts: "16:30" or seconds (display and plans)


@dataclass
class NetworkCfg:
    type: str = "grid"                  # registered network builder: grid | bridge | file | <plugin>
    rows: int = 3
    cols: int = 3
    block_m: float = 400.0
    speed_mps: float = 10.0
    congestion: bool = True             # link speed falls with the number of vehicles on the link
    jam_spacing_m: float = 7.0          # road length one stopped vehicle occupies
    capacity_scale: float = 1.0         # <1 when each simulated car stands for several real cars
    params: dict = field(default_factory=dict)


@dataclass
class DemandCfg:
    pattern: str = "uniform"            # registered demand generator: uniform | cbd | hotspots | event | <plugin>
    start_seconds: float = 0.0
    end_seconds: Optional[float] = None  # default: 60% of the simulated duration
    min_trip_m: float = 600.0
    params: dict = field(default_factory=dict)
    plan: list = field(default_factory=list)   # day-plan template (see agmsim.plans.build_plan); empty: one trip each
    plan_time_factor: float = 1.2       # planned arrivals assume this multiple of the free-flow driving time
    to_pois: bool = False               # a trip goes to a point of interest that fits its purpose (see ``pois``)
    poi_scale_m: Optional[float] = None  # how fast a place's pull falls with distance (default: 40% of the network's span)


# --------------------------------------------------------------------------- agents


DRIVING_KINDS = ("vehicle", "ondemand")     # kinds that find their own way through traffic


@dataclass
class ObservationCfg:
    """What an agent perceives of the current state."""

    traffic: str = "global"             # road conditions it knows as they are now: none | local | global
    radius_m: float = 800.0             # 'local': roads within this distance of where it is
    route_names: bool = False           # each route it is offered has the name of its way, the same at every decision
    route_roads: bool = False           # each route it is offered also lists the junctions it passes


@dataclass
class MemoryCfg:
    """What an agent carries from the past."""

    decisions: int = 4                  # its own latest decisions and outcomes, shown at every decision (0: none)
    reasons: bool = False               # ... each decision with the reason its decision model gave at the time
    traffic_seconds: float = 0.0        # how long a road condition it saw or was told about stays with it


@dataclass
class InteractionCfg:
    """What an agent exchanges with other agents, beyond what its level lets it do."""

    share_traffic: bool = False         # pass on what it knows about slow roads to agents within range_m that do the same
    range_m: float = 500.0


@dataclass
class AgentCfg:
    """Settings shared by every kind of agent."""

    use_llm: bool = False
    llm_policy: str = "llm"             # registered policy used when use_llm is true
    non_llm_policy: str = "rule"        # registered policy used when use_llm is false
    policy_params: dict = field(default_factory=dict)       # constructor arguments of the non-LLM policy
    llm_policy_params: dict = field(default_factory=dict)   # constructor arguments of the LLM policy
    agency: Any = "L0"                  # level name, list of permissions, or mapping of can_* flags
    llm: Optional[dict] = None          # per-kind override of the top-level ``llm`` block
    mix: list = field(default_factory=list)  # sub-populations: [{share: 0.3, agency: L1, use_llm: ..., non_llm_policy: ...}]
    # The agency mechanisms. They are kept by the framework, so every decision model is handed the same.
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    interaction: InteractionCfg = field(default_factory=InteractionCfg)


@dataclass
class PersonaCfg:
    vot_median: float = 20.0            # value of time, USD per hour (log-normal median)
    vot_sigma: float = 0.5
    patience_median_s: float = 480.0    # tolerated wait (log-normal median)
    patience_sigma: float = 0.4
    mobility_limited_share: float = 0.1
    walk_speed_mps: float = 1.35
    max_walk_m: float = 2500.0


@dataclass
class TravelerCfg(AgentCfg):
    count: int = 30
    car_ownership: float = 0.7          # share of travellers who own an agentic vehicle
    default_mode: str = "car"           # what a traveller without choice authority does: car | bus | walk
    wait_check_seconds: float = 60.0    # how often a traveller waiting at a stop may reconsider
    walk: bool = True                   # walking is available as an option
    plan_check_seconds: float = 120.0   # how often a person at an activity looks at how the rest of the day is going
    plan_review_threshold_seconds: float = 300.0   # an arrival drifting this far from what they reckon with prompts a review
    plan_review_horizon_seconds: float = 3600.0    # ... but not earlier than this before the planned departure
    personas: PersonaCfg = field(default_factory=PersonaCfg)


@dataclass
class BatteryCfg:
    """Used only when charging stations exist; otherwise vehicles have unlimited range."""

    capacity_kwh: float = 40.0
    consumption_kwh_per_km: float = 0.18
    initial_soc_min: float = 0.3
    initial_soc_max: float = 0.9
    target_soc: float = 0.8


@dataclass
class VehicleCfg(AgentCfg):
    """Private agentic vehicles (AgVs). There is one per car-owning traveller."""

    operating_cost_per_km: float = 0.20  # wear and energy, charged to the owner
    parking_fee: float = 0.0            # paid at the destination
    reroute_check_seconds: float = 120.0  # how often a driving vehicle may reconsider its route
    min_saving_seconds: float = 30.0    # a faster alternative must save at least this much to be worth asking about
    route_alternatives: int = 3
    battery: BatteryCfg = field(default_factory=BatteryCfg)


@dataclass
class BusCfg(AgentCfg):
    lines: list = field(default_factory=list)   # [{id, stops: [node, ...], buses, capacity}]
    dwell_seconds: float = 15.0
    dwell_per_passenger_seconds: float = 2.0
    hold_choices: list = field(default_factory=lambda: [30, 60])   # seconds a bus may hold at a stop
    max_hold_seconds: float = 120.0     # cumulative holding allowed at one stop
    fare: float = 1.5
    max_access_m: float = 500.0         # longest walk to or from a stop
    speed_factor: float = 0.85          # bus speed relative to link speed


@dataclass
class EvcsCfg(AgentCfg):
    stations: list = field(default_factory=list)   # [{id, node, chargers, power_kw}]
    price_per_kwh: float = 0.40
    price_multipliers: list = field(default_factory=lambda: [0.75, 1.0, 1.25, 1.5])
    review_seconds: float = 300.0       # how often a station may revise its price


@dataclass
class OperatorCfg(AgentCfg):
    enabled: bool = False
    review_seconds: float = 300.0
    bus_priority_gain: float = 1.2      # bus speed multiplier while transit priority is on
    car_priority_penalty: float = 0.92  # car speed multiplier while transit priority is on
    cordon_center: Optional[int] = None  # node at the centre of the charging cordon (default: geometric centre)
    cordon_radius_m: float = 0.0        # 0: no cordon
    toll_choices: list = field(default_factory=lambda: [0.0, 1.0, 2.0, 4.0])
    initial_toll: float = 0.0           # settings in force from the start (they stay fixed at level L0)
    initial_bus_priority: bool = False
    initial_advisory: bool = False


@dataclass
class SignalCfg(AgentCfg):
    """Traffic lights. A junction listed here holds vehicles at its stop lines unless their approach has green."""

    junctions: Any = field(default_factory=list)   # nodes with traffic lights: a list, or "all" for every junction with crossing roads
    green_seconds: float = 30.0         # green per phase of the fixed-time plan (what a signal without authority follows)
    min_green_seconds: float = 8.0
    max_green_seconds: float = 90.0
    lost_seconds: float = 4.0           # after a switch nobody moves for this long
    decision_seconds: float = 10.0      # how often a signal with authority may reconsider after the minimum green
    saturation_flow: float = 0.5        # vehicles per second and lane that a green lets through (scaled by network.capacity_scale)
    request_seconds: float = 60.0       # how long a request of the orchestrator stays in force
    favour_factor: float = 1.5          # green of the favoured direction in the fixed-time plan (the other gets 1/factor)


@dataclass
class OnDemandCfg(AgentCfg):
    """On-demand cars: a fleet that carries one traveller at a time on request."""

    fleet: int = 0                      # number of cars; 0: no on-demand service
    depots: list = field(default_factory=list)   # nodes at which the cars start (default: anywhere)
    objective: str = "balanced"         # what the cars are told to pursue: profit | service | balanced
    base_fare: float = 2.0              # USD per ride ...
    fare_per_km: float = 1.0            # ... plus per km of the trip ...
    fare_per_minute: float = 0.3        # ... plus per minute of its free-flow driving time
    operating_cost_per_km: float = 0.25  # what driving costs the car, with or without a passenger
    reroute_check_seconds: float = 120.0
    min_saving_seconds: float = 30.0
    route_alternatives: int = 3
    idle_check_seconds: float = 120.0   # how often a free car may reconsider where to wait
    demand_window_seconds: float = 300.0  # "recent requests" shown to a free car cover this long


@dataclass
class AgentsCfg:
    traveler: TravelerCfg = field(default_factory=TravelerCfg)
    vehicle: VehicleCfg = field(default_factory=VehicleCfg)
    bus: BusCfg = field(default_factory=BusCfg)
    evcs: EvcsCfg = field(default_factory=EvcsCfg)
    operator: OperatorCfg = field(default_factory=OperatorCfg)
    signal: SignalCfg = field(default_factory=SignalCfg)
    ondemand: OnDemandCfg = field(default_factory=OnDemandCfg)


# --------------------------------------------------------------------------- LLM, recording, metrics


@dataclass
class LLMCfg:
    preset: Optional[str] = None        # a named bundle from LLM_PRESETS (deepseek, qwen-local, ...); see resolved()
    backend: Optional[str] = None       # openai_compatible | mock | <plugin>; required when any use_llm is true
    model: str = ""
    base_url: Optional[str] = None      # e.g. http://localhost:8080/v1 for a local server
    api_key_env: Optional[str] = None   # name of the environment variable holding the key (never the key itself)
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 400
    max_concurrency: int = 4            # simultaneous calls within one decision phase
    retries_on_invalid: int = 1         # re-ask with the validation error before falling back
    max_calls: Optional[int] = None     # hard cap; further decisions use the non-LLM fallback policy
    budget_usd: Optional[float] = None
    cassette: Optional[str] = None      # JSONL file of recorded responses (deterministic re-execution)
    cassette_mode: str = "auto"         # auto: replay if recorded, else call and record | replay | record
    json_mode: bool = False             # ask the server for a JSON object (not all local servers support it)
    timeout_seconds: float = 120.0
    pricing: dict = field(default_factory=dict)  # USD per million tokens: {input, cached_input, output}
    params: dict = field(default_factory=dict)   # backend-specific extras

    def resolved(self) -> "LLMCfg":
        """The settings actually used: the preset fills every field left at its default value.

        The stored config keeps only the preset's name and what was set explicitly, so switching
        ``llm.preset`` by an override never leaves values of the previous preset behind.
        """
        if not self.preset:
            return self
        if self.preset not in LLM_PRESETS:
            raise ConfigError(f"llm.preset must be one of {sorted(LLM_PRESETS)}, got {self.preset!r}")
        blank, values = LLMCfg(), asdict(self)
        for key, value in LLM_PRESETS[self.preset].items():
            if key in ("params", "pricing"):
                values[key] = {**copy.deepcopy(value), **values[key]}
            elif values[key] == getattr(blank, key):
                values[key] = copy.deepcopy(value)
        return LLMCfg(**values)


# Named LLM settings. Hosted presets read their key from the environment (or a local ``.env`` file).
# Prices are estimates for the cost column only: list prices at peak hours converted to USD
# (DeepSeek publishes them in CNY; 7.2 CNY/USD assumed). Check the provider's page before relying on them.
LLM_PRESETS: dict[str, dict] = {
    "qwen-local": {"backend": "openai_compatible", "base_url": "http://127.0.0.1:8089/v1",
                   "model": "Qwen/Qwen2.5-7B-Instruct", "max_concurrency": 8},   # mlx_lm.server batches requests
    "deepseek": {"backend": "openai_compatible", "base_url": "https://api.deepseek.com", "model": "deepseek-flash",
                 "api_key_env": "DEEPSEEK_API_KEY", "max_concurrency": 16,
                 "params": {"extra_body": {"thinking": {"type": "disabled"}}},
                 "pricing": {"input": 0.28, "cached_input": 0.0056, "output": 1.11}},
    # the same model with its reasoning phase switched on (lowest effort); reasoning counts as output tokens
    "deepseek-thinking": {"backend": "openai_compatible", "base_url": "https://api.deepseek.com",
                          "model": "deepseek-flash", "api_key_env": "DEEPSEEK_API_KEY", "max_concurrency": 16,
                          "max_tokens": 4000,
                          "params": {"extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}},
                          "pricing": {"input": 0.28, "cached_input": 0.0056, "output": 1.11}},
    "deepseek-pro": {"backend": "openai_compatible", "base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro",
                     "api_key_env": "DEEPSEEK_API_KEY", "max_concurrency": 16,
                     "params": {"extra_body": {"thinking": {"type": "disabled"}}},
                     "pricing": {"input": 1.25, "cached_input": 0.042, "output": 3.75}},
    "mock": {"backend": "mock"},
}


@dataclass
class ScoringCfg:
    """How an executed day plan is scored: MATSim's Charypar-Nagel parameters and defaults (utils per hour)."""

    performing: float = 6.0             # time at an activity of its usual length
    traveling: float = -6.0             # time spent travelling, any mode
    late_arrival: float = -18.0         # starting an activity after its latest start
    money: float = 1.0                  # utils per USD; used when scores are not scaled per person
    scale_by_value_of_time: bool = True  # express scores in USD so that (performing - traveling) is the person's value of time


@dataclass
class RecordCfg:
    dir: Optional[str] = "runs"         # None: keep the trace in memory only
    run_name: Optional[str] = None
    observations: str = "full"          # store full observations with each decision, or only their digest
    frames_every_seconds: float = 2.0
    state_hash_every_ticks: int = 1


@dataclass
class MetricsCfg:
    vehicle_co2_g_per_km: float = 120.0  # distance-based accounting factors; replace with your fleet's values
    bus_co2_g_per_km: float = 800.0


@dataclass
class Config:
    name: str = "agmsim-run"
    description: str = ""
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    network: NetworkCfg = field(default_factory=NetworkCfg)
    demand: DemandCfg = field(default_factory=DemandCfg)
    agents: AgentsCfg = field(default_factory=AgentsCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    record: RecordCfg = field(default_factory=RecordCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    interventions: list = field(default_factory=list)   # external events; ``events`` is accepted as another name
    pois: list = field(default_factory=list)            # points of interest (see agmsim.pois)
    plugins: list = field(default_factory=list)
    base_dir: Optional[str] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_dict(cls, data: dict | None, base_dir: str | Path | None = None) -> "Config":
        cfg = _build(cls, _expand_aliases(data or {}), "config")
        cfg.base_dir = str(base_dir) if base_dir is not None else None
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, source: "str | Path | dict | Config", overrides: "list[str] | dict | None" = None) -> "Config":
        """Load from a YAML file, a packaged scenario name, a dict or a Config.

        ``overrides`` are dotted assignments such as ``agents.vehicle.agency=L3``.
        """
        base_dir = None
        if isinstance(source, Config):
            data, base_dir = source.to_dict(), source.base_dir
        elif isinstance(source, dict):
            data = copy.deepcopy(source)
        else:
            path = resolve_scenario(source)
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            base_dir = path.parent
        if overrides:
            data = apply_overrides(data, overrides)
        return cls.from_dict(data, base_dir=base_dir)

    # ------------------------------------------------------------------ helpers

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("base_dir", None)
        return data

    def digest(self) -> str:
        """Fingerprint of everything that defines the experiment (not where it is stored)."""
        data = self.to_dict()
        data["record"].pop("dir", None)
        data["record"].pop("run_name", None)
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def demand_window(self) -> tuple[float, float]:
        """Interval in which trip needs arise (default: the first 60% of the run)."""
        end = self.demand.end_seconds
        if end is None:
            end = round(0.6 * self.simulation.duration_seconds, 3)
        return float(self.demand.start_seconds), float(end)

    def active_kinds(self) -> list[str]:
        """Agent kinds present in this run, in action-application order."""
        a = self.agents
        present = {
            "operator": a.operator.enabled,
            "signal": bool(a.signal.junctions),
            "evcs": bool(a.evcs.stations),
            "traveler": a.traveler.count > 0,
            "bus": bool(a.bus.lines),
            "vehicle": a.traveler.count > 0 and a.traveler.car_ownership > 0,
            "ondemand": a.traveler.count > 0 and a.ondemand.fleet > 0,
        }
        return [k for k in KIND_ORDER if present[k]]

    def validate(self) -> None:
        sim = self.simulation
        if sim.time_step_seconds <= 0 or sim.duration_seconds <= 0:
            raise ConfigError("simulation.time_step_seconds and duration_seconds must be positive")
        start, end = self.demand_window()
        if end <= start:
            raise ConfigError("demand.end_seconds must be greater than demand.start_seconds")
        _choice("record.observations", self.record.observations, ("full", "digest"))
        _choice("llm.cassette_mode", self.llm.cassette_mode, ("auto", "replay", "record"))
        self.llm.resolved()     # rejects an unknown llm.preset
        _choice("agents.traveler.default_mode", self.agents.traveler.default_mode, ("car", "bus", "walk", "ride"))
        _choice("agents.ondemand.objective", self.agents.ondemand.objective, ("profit", "service", "balanced"))
        for kind in KIND_ORDER:
            k, path = getattr(self.agents, kind), f"agents.{kind}"
            _choice(f"{path}.observation.traffic", k.observation.traffic, ("none", "local", "global"))
            if k.observation.radius_m <= 0 or k.interaction.range_m <= 0:
                raise ConfigError(f"{path}.observation.radius_m and {path}.interaction.range_m must be positive")
            if int(k.memory.decisions) < 0 or k.memory.traffic_seconds < 0:
                raise ConfigError(f"{path}.memory.decisions and {path}.memory.traffic_seconds must not be negative")
            if kind not in DRIVING_KINDS and (k.observation.traffic != "global" or k.observation.route_roads
                                              or k.observation.route_names
                                              or k.memory.traffic_seconds > 0 or k.interaction.share_traffic):
                raise ConfigError(
                    f"{path}: observation.traffic, observation.route_names, observation.route_roads, memory.traffic_seconds and "
                    f"interaction.share_traffic are settings of the kinds that find their own way through traffic "
                    f"({', '.join(DRIVING_KINDS)})")
        for item in self.interventions:
            if not isinstance(item, dict) or item.get("type") not in EVENT_TYPES:
                raise ConfigError(f"each external event needs a 'type' out of {list(EVENT_TYPES)}")
        for item in self.pois:
            if not isinstance(item, dict) or "type" not in item or ("node" not in item and "count" not in item):
                raise ConfigError("each entry of 'pois' needs a 'type' and either a 'node' or a 'count'")
        if self.demand.plan and (len(self.demand.plan) < 2 or any("type" not in item for item in self.demand.plan)):
            raise ConfigError("demand.plan needs at least two activities, each with a 'type'")
        if self.scoring.performing <= self.scoring.traveling or self.scoring.money <= 0:
            raise ConfigError("scoring.performing must exceed scoring.traveling, and scoring.money must be positive")
        if not 0.0 <= self.agents.traveler.car_ownership <= 1.0:
            raise ConfigError("agents.traveler.car_ownership must be between 0 and 1")
        if 1.0 not in [float(x) for x in self.agents.evcs.price_multipliers]:
            raise ConfigError("agents.evcs.price_multipliers must contain 1.0")
        if 0.0 not in [float(x) for x in self.agents.operator.toll_choices]:
            raise ConfigError("agents.operator.toll_choices must contain 0")


# --------------------------------------------------------------------------- loading helpers


def _choice(path: str, value: Any, allowed: tuple) -> None:
    if value not in allowed:
        raise ConfigError(f"{path} must be one of {list(allowed)}, got {value!r}")


def _expand_aliases(data: dict) -> dict:
    """Accept the two-class shorthand ``human_agents`` / ``machine_agents``.

    ``human_agents`` is the traveller block. In ``machine_agents`` the
    decision-model switches (``use_llm``, ``llm``, ``llm_policy``) apply to every
    machine kind that does not set them itself; all other keys configure the
    vehicles.
    """
    data = dict(data)
    agents = {k: dict(v or {}) for k, v in (data.get("agents") or {}).items()}
    if "human_agents" in data:
        agents["traveler"] = {**(data.pop("human_agents") or {}), **agents.get("traveler", {})}
    if "machine_agents" in data:
        machine = dict(data.pop("machine_agents") or {})
        shared = {k: machine.pop(k) for k in ("use_llm", "llm", "llm_policy") if k in machine}
        for kind in MACHINE_KINDS:
            agents[kind] = {**shared, **agents.get(kind, {})}
        agents["vehicle"] = {**machine, **agents["vehicle"]}
    vehicle = agents.get("vehicle")
    if vehicle:                             # earlier names of the vehicle's knowledge of traffic
        seen = dict(vehicle.get("observation") or {})
        for name, key in (("link_conditions", "traffic"), ("local_radius_m", "radius_m")):
            if name in vehicle:
                seen.setdefault(key, vehicle.pop(name))
        if seen:
            vehicle["observation"] = seen
    if agents:
        data["agents"] = agents
    if "events" in data:                    # the scenario's external events, under their other name (it wins)
        data["interventions"] = data.pop("events") or []
    return data


def _build(cls, data: Any, path: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' must be a mapping, got {type(data).__name__}")
    data = dict(data)
    known = {f.name for f in fields(cls)} - {"base_dir"}
    if "agency" in known:
        # ``can_replan: true`` written next to ``use_llm`` is an agency switch.
        flags = {k: data.pop(k) for k in list(data) if k.startswith("can_")}
        if flags:
            base = data.get("agency")
            if base is None:
                data["agency"] = dict(flags)
            elif isinstance(base, str):
                data["agency"] = {"level": base, **flags}
            elif isinstance(base, list):
                data["agency"] = {"permissions": list(base), **flags}
            elif isinstance(base, dict):
                data["agency"] = {**base, **flags}
            else:
                raise ConfigError(f"'{path}.agency' has an unsupported type")
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"'{path}': unknown key(s) {unknown}. Valid keys: {sorted(known)}")
    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        target = hints.get(name)
        if is_dataclass(target):
            kwargs[name] = _build(target, value, f"{path}.{name}")
        else:
            kwargs[name] = copy.deepcopy(value)
    return cls(**kwargs)


def apply_overrides(data: dict, overrides: "list[str] | dict") -> dict:
    """Apply dotted-path assignments (``a.b.c=value``) to a config dict."""
    data = _expand_aliases(copy.deepcopy(data))
    if isinstance(overrides, dict):
        raw_items = list(overrides.items())
    else:
        raw_items = []
        for item in overrides:
            key, sep, raw = item.partition("=")
            if not sep:
                raise ConfigError(f"override '{item}' must look like key.path=value")
            raw_items.append((key.strip(), yaml.safe_load(raw)))
    items = []
    for key, value in raw_items:      # translate the two-class shorthand into per-kind keys
        head, _, rest = key.partition(".")
        if head == "human_agents" and rest:
            items.append((f"agents.traveler.{rest}", value))
        elif head == "machine_agents" and rest:
            if rest.split(".")[0] in ("use_llm", "llm", "llm_policy"):
                items.extend((f"agents.{kind}.{rest}", value) for kind in MACHINE_KINDS)
            else:
                items.append((f"agents.vehicle.{rest}", value))
        else:
            items.append((key, value))
    for key, value in items:
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return data


SCENARIO_DIR = Path(__file__).parent / "scenarios"


TESTBED = "montreal"        # the scenario experiments are run on; listed first, so it is what the page opens with


def list_scenarios() -> list[str]:
    return sorted((p.stem for p in SCENARIO_DIR.glob("*.yaml")), key=lambda name: (name != TESTBED, name))


def resolve_scenario(source: "str | Path") -> Path:
    """Map a file path or a packaged scenario name to a path."""
    path = Path(source)
    if path.exists():
        return path
    packaged = SCENARIO_DIR / f"{source}.yaml"
    if packaged.exists():
        return packaged
    raise ConfigError(
        f"config '{source}' is neither an existing file nor a packaged scenario "
        f"({', '.join(list_scenarios()) or 'none'})"
    )
