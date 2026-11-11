"""State variables of the simulated world.

Everything that evolves during a run lives in these plain dataclasses, so the
system state ``S_t`` is explicit and hashable: travellers (human layer), their
agentic vehicles, buses, charging stations and the network operator (machine
and infrastructure layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TState(str, Enum):
    """Traveller life-cycle."""

    PLANNED = "planned"            # the trip need has not arisen yet
    DECIDING = "deciding"          # need arisen (or re-planning); choosing how to travel
    AT_ACTIVITY = "at_activity"    # at an activity of their day plan
    IN_CAR = "in_car"              # travelling in their own vehicle
    TO_STOP = "to_stop"            # walking to a bus stop
    AT_STOP = "at_stop"            # waiting for a bus
    ON_BUS = "on_bus"
    EGRESS = "egress"              # walking from the bus stop to the destination
    WALKING = "walking"            # walking the whole trip
    WAITING_RIDE = "waiting_ride"  # has called an on-demand car and waits for it
    IN_RIDE = "in_ride"            # travelling in an on-demand car
    ARRIVED = "arrived"
    ABANDONED = "abandoned"        # the trip was not made


class VState(str, Enum):
    """Life-cycle of a private agentic vehicle."""

    PARKED = "parked"
    DRIVING = "driving"            # heading to the trip destination, or to a transfer stop
    TO_CHARGER = "to_charger"
    QUEUED = "queued"              # waiting for a charger
    CHARGING = "charging"


class CState(str, Enum):
    """Life-cycle of an on-demand car."""

    IDLE = "idle"
    TO_PICKUP = "to_pickup"
    OCCUPIED = "occupied"          # carrying a traveller
    REPOSITIONING = "repositioning"  # driving empty to wait somewhere else


class BState(str, Enum):
    MOVING = "moving"
    DWELLING = "dwelling"          # at a stop (boarding, alighting, or holding)


@dataclass
class Persona:
    """Attributes of a traveller. The same numbers drive behavioural models and LLM personas."""

    age_group: str
    purpose: str
    value_of_time: float       # USD per hour
    patience_s: float          # tolerated wait, seconds
    walk_speed: float          # m/s
    max_walk_m: float
    mobility_limited: bool

    def to_dict(self) -> dict:
        return {
            "age_group": self.age_group,
            "trip_purpose": self.purpose,
            "value_of_time_usd_per_h": round(self.value_of_time, 1),
            "patience_min": round(self.patience_s / 60.0, 1),
            "mobility_limited": self.mobility_limited,
        }


@dataclass
class Poi:
    """A point of interest: a named place of some type at a node, possibly with opening hours."""

    id: str
    type: str
    node: int
    name: str = ""
    weight: float = 1.0                       # how strongly it attracts trips
    open_s: Optional[float] = None            # seconds into the run; None: always open
    close_s: Optional[float] = None
    visits: int = 0
    found_closed: int = 0                     # visitors who arrived after it had closed


@dataclass
class Traveler:
    id: str
    persona: Persona
    origin: int
    destination: int
    t_need: int                               # tick at which the trip need arises
    trip_m: float = 0.0                       # network distance of the trip
    trip_s: float = 0.0                       # free-flow driving time of the trip
    car: Optional[str] = None                 # id of the traveller's own vehicle, if they have one
    state: TState = TState.PLANNED
    mode: Optional[str] = None                # how the trip was (or is being) made: car | bus | walk | car+bus
    cancels: int = 0
    replan_pending: bool = False
    suggestion: Optional[dict] = None         # plan change proposed by the traveller's vehicle, awaiting an answer
    next_check: Optional[int] = None          # tick of the next waiting re-evaluation
    wait_start: Optional[int] = None          # tick at which the current wait began
    waited_ticks: int = 0                     # total ticks spent waiting at bus stops
    t_depart: Optional[int] = None            # tick of the first travel choice
    t_arrival: Optional[int] = None
    phase_start: Optional[int] = None         # timed phases (walking, access, egress)
    phase_end: Optional[int] = None
    bus: Optional[dict] = None                # bus plan: line, board, alight, access_s, egress_s
    paid: float = 0.0
    plan: Optional[object] = None             # day plan (agmsim.plans.Plan); None: a single trip
    ride: Optional[str] = None                # the ride request this traveller has open, if any
    poi: Optional[str] = None                 # the point of interest a single trip goes to, if any


@dataclass
class Vehicle:
    id: str
    owner: str
    node: int                                 # last node reached (current node when not on a link)
    state: VState = VState.PARKED
    link: Optional[int] = None                # link currently being traversed
    offset: float = 0.0                       # metres travelled on that link
    path: list = field(default_factory=list)  # remaining link ids; path[0] is the current link when on one
    target: Optional[int] = None              # node currently driven to
    destination: Optional[int] = None         # trip destination while the owner is on board
    passenger: Optional[str] = None
    station: Optional[str] = None             # charging station being used or approached
    transfer: Optional[dict] = None           # accepted park-and-ride plan
    soc: float = 1.0                          # battery state of charge (0..1)
    charge_target: float = 0.8
    needs_charge: bool = False                # cannot reach the destination without charging
    park_pending: bool = False                # just parked: may decide what to do next
    route_pending: bool = False               # conditions changed: reconsider the route now
    next_route_check: int = 0
    hold_until: int = 0                       # does not move before this tick (deciding how to set off)
    suggested: bool = False                   # already proposed a plan change on this trip
    queued: bool = False                      # held at a traffic light, waiting its turn
    go: bool = False                          # released by the traffic light: may cross now
    toll_paid: bool = False
    trip_m: float = 0.0                       # metres driven on the current trip
    trip_start: Optional[int] = None
    # accounting ------------------------------------------------------------
    m_loaded: float = 0.0                     # metres driven with the owner on board
    m_empty: float = 0.0                      # metres driven empty (self-initiated charging trips)
    s_drive: float = 0.0
    s_moving: float = 0.0                     # seconds on the road, whatever the purpose (for vehicle hours travelled)
    s_charge: float = 0.0                     # driving to, queueing at and using chargers
    s_queue: float = 0.0                      # of which: waiting for a free charger
    drive_s_owner: float = 0.0                # in-vehicle time experienced by owners (incl. charging stops)
    free_s_owner: float = 0.0                 # free-flow time of the same trips
    trips: int = 0
    reroutes: int = 0
    suggestions: int = 0
    suggestions_accepted: int = 0
    charge_stops: int = 0
    kwh_charged: float = 0.0
    energy_cost: float = 0.0
    tolls: float = 0.0
    depleted: int = 0                         # ticks driven with an empty battery


@dataclass
class Cab:
    """An on-demand car: carries one traveller at a time on request."""

    id: str
    node: int
    state: CState = CState.IDLE
    link: Optional[int] = None
    offset: float = 0.0
    path: list = field(default_factory=list)
    target: Optional[int] = None
    request: Optional[str] = None             # the request it is serving
    proposal: Optional[str] = None            # a request the service proposes to it, not yet answered
    queued: bool = False                      # held at a traffic light
    go: bool = False
    route_pending: bool = False
    next_route_check: int = 0
    next_idle_check: int = 0
    # accounting ------------------------------------------------------------
    m_loaded: float = 0.0                     # metres with a traveller on board
    m_empty: float = 0.0                      # metres empty: to pick-ups and while repositioning
    s_moving: float = 0.0
    trips: int = 0
    declined: int = 0
    repositions: int = 0
    reroutes: int = 0
    revenue: float = 0.0


@dataclass
class RideRequest:
    id: str
    traveler: str
    origin: int
    destination: int
    opened: int                               # tick at which the traveller called
    fare: float
    trip_m: float
    state: str = "open"                       # open | proposed | assigned | on_board | done | cancelled
    offered: Optional[str] = None             # the car it is proposed to
    car: Optional[str] = None                 # the car that took it
    declined_by: set = field(default_factory=set)
    picked: Optional[int] = None


@dataclass
class Bus:
    id: str
    line: str
    node: int
    next_stop: int = 0                        # index (in the line's stop list) of the stop being approached or served
    state: BState = BState.DWELLING
    link: Optional[int] = None
    offset: float = 0.0
    path: list = field(default_factory=list)
    onboard: list = field(default_factory=list)   # traveller ids
    capacity: int = 30
    dwell_until: int = 0
    held_s: float = 0.0                       # holding already applied at the current stop
    asked: bool = False                       # the at-stop decision is due
    skip_next: bool = False
    queued: bool = False                      # held at a traffic light
    go: bool = False                          # released by the traffic light
    m_total: float = 0.0
    s_moving: float = 0.0                     # seconds on the road between stops
    boardings: int = 0
    holds: int = 0
    skips: int = 0
    left_behind: int = 0                      # travellers who could not board because the bus was full


@dataclass
class Station:
    id: str
    node: int
    chargers: int
    power_kw: float
    base_price: float                         # USD per kWh
    multiplier: float = 1.0
    queue: list = field(default_factory=list)     # vehicle ids waiting, arrival order
    plugged: list = field(default_factory=list)   # vehicle ids charging
    choose_pending: bool = False              # a charger is free and the station may pick who is next
    next_review: int = 0
    kwh: float = 0.0
    revenue: float = 0.0
    sessions: int = 0
    busy_charger_ticks: int = 0

    @property
    def price(self) -> float:
        return round(self.base_price * self.multiplier, 4)


@dataclass
class Signal:
    """The traffic lights of one junction. One phase (a group of approaches) has green at a time."""

    id: str
    node: int
    phases: list                              # per phase: ids of the incoming links that have green together
    names: list                               # per phase: "east-west", "north-south"
    phase: int = 0                            # the phase that has (or is about to get) green
    since: int = 0                            # tick at which that phase's green began
    red_until: int = 0                        # everybody is held until this tick (lost time after a switch)
    next_decision: int = 0
    request: Optional[dict] = None            # what the orchestrator asked for: {"phase": index, "until": tick}
    switches: int = 0
    held: int = 0                             # vehicles that had to stop here
    wait_ticks: int = 0                       # ticks vehicles spent held here
    wait_by_phase: list = field(default_factory=list)   # the same, per phase
    queue_max: int = 0
    requests_followed: int = 0


@dataclass
class OperatorState:
    id: str = "OP"
    advisory: bool = False                    # real-time link conditions are published to everyone
    bus_priority: bool = False                # transit priority is switched on
    toll: float = 0.0                         # charge for entering the cordon, USD
    next_review: int = 0
    incident_pending: bool = False            # link conditions changed since the last decision
    advisories: int = 0
    priority_ticks: int = 0
    toll_revenue: float = 0.0
    toll_changes: int = 0
    favoured: Optional[str] = None            # direction the traffic lights are asked to favour
    signal_requests: int = 0                  # requests and directions given to traffic lights
