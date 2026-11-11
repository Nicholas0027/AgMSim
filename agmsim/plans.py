"""Day plans: MATSim's action mechanism, executed within one simulated period.

In MATSim an agent acts through a *plan* -- a chain of activities joined by trips. The mobility
simulation executes the plan and asks the agent only at fixed points (an activity ends, a node is
reached, a bus arrives); the executed plan is *scored*; and between iterations of the same day
plans are mutated at random and selected by score, until nobody gains by changing.

AgMSim keeps the first three parts:

* a :class:`Plan` is a list of :class:`Activity`; the trips between them use the decision points
  that already exist (how to travel at departure, the route while driving, waiting at a stop);
* an activity ends at its planned end time, or at once if the person arrives after it (MATSim's
  ``tryEndTimeThenDuration``);
* the executed plan is scored with the Charypar-Nagel function (:func:`score`), per person and in
  USD, so that an hour of travel instead of an hour at an activity costs exactly the person's
  value of time.

It replaces the fourth. Rather than replaying the day, a person may *edit the rest of the plan
while the day unfolds* -- the edits of MATSim's within-day ``EditPlans``: end the current activity
at another time, drop an activity. Which edit to make is a decision like any other, shown as an
observation (:func:`project` gives what each edit leads to) and taken by a rule, an algorithm or a
language model.

Differences from MATSim, all deliberate: a run covers a period of the day rather than 24 hours, so
instead of merging the first and last activity each activity may carry time spent outside the
period (``outside_s``); an activity's ``priority`` sets its zero-utility duration directly; and an
activity that is skipped is worth nothing, while one that is kept but not performed counts as zero
duration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional


#: A change of plan "is worth making" when its projected gain over keeping the plan is at least this, USD.
REVIEW_GAIN_USD = 0.5


def parse_clock(value) -> float:
    """Seconds since midnight from ``"17:30"``, ``"17:30:00"`` or a number of seconds.

    Quote clock times in YAML: an unquoted ``17:30`` is read as the number 1050.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        parts = [float(part) for part in str(value).strip().split(":")]
    except ValueError:
        parts = []
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"cannot read {value!r} as a time of day (use \"HH:MM\" or seconds)")
    return parts[0] * 3600.0 + parts[1] * 60.0 + (parts[2] if len(parts) == 3 else 0.0)


def hhmm(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


@dataclass
class Activity:
    """One activity of a plan. Times are seconds into the run."""

    type: str
    node: int
    end_s: Optional[float] = None            # planned end; None: by duration, or open-ended (the last activity)
    duration_s: Optional[float] = None       # leaves this long after arriving, when no end time is set
    open_s: Optional[float] = None           # opening hours: time outside them does not count
    close_s: Optional[float] = None
    latest_start_s: Optional[float] = None   # starting later is penalised
    typical_s: float = 0.0                   # usual length; the activity is worth most per minute below it
    outside_s: float = 0.0                   # time spent at it before or after the simulated period
    priority: float = 1.0                    # what doing it at all is worth: priority x rate x usual length
    optional: bool = False                   # may be dropped from the plan
    poi: Optional[str] = None                # the point of interest it takes place at, if any
    # what happened ----------------------------------------------------------
    arrived_s: Optional[float] = None
    left_s: Optional[float] = None
    skipped: bool = False

    def leaves(self, arrive_s: float, horizon_s: float) -> float:
        """When a person who arrives at ``arrive_s`` leaves, if the plan is followed."""
        if self.end_s is not None:
            return max(arrive_s, self.end_s)
        if self.duration_s is not None:
            return arrive_s + self.duration_s
        return max(arrive_s, horizon_s)

    def stay(self, arrive_s: float, leave_s: float) -> tuple[float, float]:
        """Seconds actually spent on the activity (within opening hours) and seconds late."""
        start = arrive_s if self.open_s is None else max(arrive_s, self.open_s)
        end = leave_s if self.close_s is None else min(leave_s, self.close_s)
        late = 0.0 if self.latest_start_s is None else max(0.0, start - self.latest_start_s)
        return max(0.0, end - start), late


@dataclass
class Plan:
    activities: list
    at: int = 0                                  # the activity the person is at, or has last left
    leave_s: float = 0.0                         # when the current activity ends
    legs: list = field(default_factory=list)     # trips made: {to, mode, depart, arrive, paid, waited, cancels}
    leg: Optional[dict] = None                   # the trip under way
    expected: Optional[dict] = None              # activity index -> arrival time the person reckons with
    pending: Optional[dict] = None               # a review prepared for this tick
    notice: str = ""                             # news that calls for a look at the plan (a place closing early)
    follow_up: bool = False                      # the plan was just changed: one more look at it
    next_check: int = 0                          # tick of the next look at how the day is going
    reviews: int = 0
    edits: int = 0

    def ahead(self, skip=frozenset()) -> list:
        """Indices of the activities still to come."""
        return [k for k in range(self.at + 1, len(self.activities))
                if not self.activities[k].skipped and k not in skip]


# --------------------------------------------------------------------------- what a plan is worth


def rates(value_of_time: float, scoring) -> dict:
    """What an hour is worth to one person, USD: at an activity, travelling, being late.

    MATSim's parameters are in utils per hour. With ``scale_by_value_of_time`` they are converted
    person by person so that ``(performing - traveling)``, the value of travel time savings, equals
    the person's own value of time.
    """
    if scoring.scale_by_value_of_time:
        scale = value_of_time / (scoring.performing - scoring.traveling)
    else:
        scale = 1.0 / scoring.money
    return {"activity": scoring.performing * scale, "travel": scoring.traveling * scale,
            "late": scoring.late_arrival * scale}


def preferences(value_of_time: float, scoring) -> dict:
    """The same rates, as shown to a decision model."""
    r = rates(value_of_time, scoring)
    return {"time_at_activities_usd_per_h": round(r["activity"], 2), "travel_time_usd_per_h": round(r["travel"], 2),
            "arriving_late_usd_per_h": round(r["late"], 2)}


def performing(seconds: float, typical_s: float, rate_per_h: float, priority: float = 1.0) -> float:
    """Charypar-Nagel utility of performing an activity for ``seconds``.

    ``rate * typical * ln(duration / t0)`` with ``t0 = typical * exp(-priority)``. At the usual
    length the activity is worth ``priority * rate * typical`` and one more hour of it is worth
    ``rate``, whatever the priority: as in MATSim, the priority sets what doing the activity at
    all is worth (through the zero-utility duration ``t0``), not how long one wants to stay.
    Priority 1 is MATSim's ``relative`` setting (``t0 = typical / e``). Below ``t0`` the curve
    continues as a straight line, as in MATSim, down to ``-rate * typical`` at zero duration.
    """
    if typical_s <= 0:
        return 0.0
    zero = typical_s * math.exp(-priority)
    worth = rate_per_h * typical_s / 3600.0
    if seconds >= zero:
        return worth * math.log(seconds / zero)
    return -worth * (zero - seconds) / zero


def score(plan: Plan, r: dict, now_s: float, paid: float) -> dict:
    """Score of the plan as executed up to ``now_s``, USD, with its parts.

    Activities still ahead that were not dropped count as kept but not performed.
    """
    out = {"activities": 0.0, "late": 0.0, "travel": 0.0, "money": -paid, "late_s": 0.0, "missed": 0}
    for a in plan.activities:
        if a.skipped:
            continue
        there = 0.0
        if a.arrived_s is not None:
            there, late = a.stay(a.arrived_s, now_s if a.left_s is None else a.left_s)
            out["late"] += r["late"] * late / 3600.0
            out["late_s"] += late
        if there <= 0.0:
            out["missed"] += 1
        out["activities"] += performing(there + a.outside_s, a.typical_s, r["activity"], a.priority)
    travel = sum(leg["arrive_s"] - leg["depart_s"] for leg in plan.legs)
    if plan.leg is not None:
        travel += now_s - plan.leg["depart_s"]
    out["travel"] = r["travel"] * travel / 3600.0
    out["total"] = out["activities"] + out["late"] + out["travel"] + out["money"]
    return out


def project(plan: Plan, leave_s: float, estimate: Callable, horizon_s: float, skip=frozenset()):
    """The rest of the day if the person leaves the current activity at ``leave_s``.

    ``estimate(a, b)`` gives the travel time and cost between two nodes as they stand now.
    Returns ``[(activity index, arrival, seconds spent there, seconds late), ...]``, the total
    travel time and the total travel cost.
    """
    acts = plan.activities
    t, node, rows, travel, cost = leave_s, acts[plan.at].node, [], 0.0, 0.0
    for k in plan.ahead(skip):
        a = acts[k]
        leg_s, leg_usd = estimate(node, a.node)
        arrive = t + leg_s
        t = a.leaves(arrive, horizon_s)
        there, late = a.stay(arrive, t)
        rows.append((k, arrive, there, late))
        travel, cost, node = travel + leg_s, cost + leg_usd, a.node
    return rows, travel, cost


def option_value(prefs: dict, context: dict, option: dict) -> float:
    """Worth, USD, of one option of a plan review, from what the observation itself says.

    ``prefs`` is ``state.preferences``; ``context`` and ``option`` are the review's context and one
    entry of its ``options``. Only the part of the day that the option can still change is counted,
    so values are comparable between the options of one review and not beyond.
    """
    rate = prefs["time_at_activities_usd_per_h"]

    def worth(row: dict, minutes: float) -> float:
        return performing(60.0 * (minutes + row.get("outside_min", 0.0)), 60.0 * row["typical_min"], rate,
                          row.get("priority", 1.0))

    static = {row["activity"]: row for row in context["plan"]}
    value = worth(context["now_at"], option["minutes_here"])
    for row in option["then"]:
        value += worth(static[row["activity"]], row["minutes_there"])
        value += prefs["arriving_late_usd_per_h"] * row.get("late_min", 0.0) / 60.0
    return value + prefs["travel_time_usd_per_h"] * option["travel_min"] / 60.0 - option["travel_cost_usd"]


# --------------------------------------------------------------------------- building plans


def build_plan(template: list, net, rng, first_end_s: float, origin: int, destination: int, clock0: float,
               horizon_s: float, time_factor: float, pois: "dict | None" = None) -> Plan:
    """One person's plan from the scenario's template (``demand.plan``).

    Each entry describes an activity: ``type``; ``at`` (``origin`` for the first, ``destination``
    for the last, otherwise ``on_the_way`` -- a node that adds little detour --, ``random``, a
    node, a list of nodes to draw from, or ``{poi: <type>}`` -- the point of interest of that type
    that adds the least detour, whose opening hours the activity then has); ``share`` (how many
    people have it; middle activities
    only); ``end`` (a time of day, with ``end_spread_min``) or ``duration_min``; ``opens``,
    ``closes``; ``latest_start`` (a time of day) or ``latest_start_slack_min`` (that long after
    the planned arrival); ``typical_min`` (default: the planned length); ``priority``;
    ``optional``. The first activity ends at the time drawn by the demand pattern unless it has an
    ``end``. Planned arrivals assume ``time_factor`` times the free-flow driving time.
    """
    nodes = sorted(net.nodes)
    last = len(template) - 1
    chosen = [item for i, item in enumerate(template)
              if i in (0, last) or rng.random() < float(item.get("share", 1.0))]

    def at_s(value) -> float:
        return parse_clock(value) - clock0

    def place(spec, previous: int):
        if isinstance(spec, dict):              # at a point of interest of that type
            from .pois import on_the_way
            found = on_the_way(pois or {}, str(spec.get("poi")), previous, destination, net)
            if found is None:
                raise ValueError(f"demand.plan: there is no point of interest of type '{spec.get('poi')}'")
            return found
        if spec == "origin":
            return origin
        if spec == "destination":
            return destination
        if spec == "random":
            return nodes[rng.randrange(len(nodes))]
        if spec == "on_the_way":
            candidates = [nodes[rng.randrange(len(nodes))] for _ in range(4)]
            return min(candidates, key=lambda n: (net.length(previous, n) + net.length(n, destination), n))
        if isinstance(spec, (list, tuple)):
            return int(spec[rng.randrange(len(spec))])
        return int(spec)

    acts, t, node = [], 0.0, origin
    for n, item in enumerate(chosen):
        first, final = n == 0, n == len(chosen) - 1
        here = place(item.get("at", "origin" if first else "destination" if final else "on_the_way"), node)
        spot, here = (here, here.node) if not isinstance(here, int) else (None, here)
        arrive = 0.0 if first else t + time_factor * net.time(node, here)
        a = Activity(str(item["type"]), here, priority=float(item.get("priority", 1.0)),
                     optional=bool(item.get("optional", False)) and not (first or final),
                     poi=spot.id if spot is not None else None)
        if not final:
            if "end" in item:
                spread = 60.0 * float(item.get("end_spread_min", 0.0))
                a.end_s = at_s(item["end"]) + spread * (2.0 * rng.random() - 1.0)
            elif first:
                a.end_s = first_end_s
            else:
                a.duration_s = 60.0 * float(item.get("duration_min", 30.0))
        a.open_s = at_s(item["opens"]) if "opens" in item else spot.open_s if spot is not None else None
        a.close_s = at_s(item["closes"]) if "closes" in item else spot.close_s if spot is not None else None
        if "latest_start" in item:
            a.latest_start_s = at_s(item["latest_start"])
        elif "latest_start_slack_min" in item:
            a.latest_start_s = arrive + 60.0 * float(item["latest_start_slack_min"])
        t = a.leaves(arrive, horizon_s)
        planned = max(a.stay(arrive, t)[0], 60.0)
        a.typical_s = 60.0 * float(item["typical_min"]) if "typical_min" in item else planned
        if first or final:
            a.outside_s = max(0.0, a.typical_s - planned)
        acts.append(a)
        node = here
    acts[0].arrived_s = 0.0
    return Plan(acts, leave_s=acts[0].leaves(0.0, horizon_s))
