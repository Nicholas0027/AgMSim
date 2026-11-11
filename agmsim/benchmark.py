"""Decision benchmark: the same controlled decision points for every decision model.

A closed-loop run shows what a decision model does to the system, but the decision points it meets
are whatever that run happens to produce, and they differ from one model to the next. A benchmark
task fixes them. It takes one observation from a real run -- so every item has exactly the form the
simulator shows -- replaces its numbers according to a factorial design, and labels each item with
the action that serves the person concerned, computed from that person's own stated preferences.

Any decision model can be scored: a rule, an algorithm, an LLM, a plug-in. Items go through the same
interface as in a run (``Observation`` in, ``Decision`` out, validated against the permitted
actions), and a model is configured exactly as for a run, by config overrides::

    from agmsim.benchmark import evaluate
    report = evaluate("route_delegation", {"agents.vehicle.use_llm": True, "llm.preset": "deepseek"})
    print(report["scores"]["sensitivity"])

Built-in tasks:

* ``route_delegation``    -- a vehicle that knows its owner's value of time chooses between a faster
  tolled route and a slower free one;
* ``suggestion_response`` -- an owner answers the vehicle's proposal to park and continue by bus,
  made with no message, a truthful one, or one that overstates the gain;
* ``transfer_proposal``   -- a vehicle decides whether to make that proposal.

The headline score of each task is a *sensitivity*: how often the model took an action when it
served the person, minus how often it took it when it did not. It is 0 for a model that ignores
the person's preferences (always acts, never acts, or acts at random) and 1 for one that follows them.

Add a task by subclassing :class:`Task` and decorating it with ``@register_task("name")``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .actions import Decision, validate_decision
from .kinds.vehicle import owner_route_costs
from .observation import Observation
from .registry import Registry

TASKS = Registry("benchmark task")


def register_task(name: str):
    def deco(cls):
        cls.name = name
        return TASKS.register(name)(cls)
    return deco


def get_task(task) -> "Task":
    """A task instance from its name, class or instance."""
    if isinstance(task, Task):
        return task
    if isinstance(task, str):
        task = TASKS.get(task)
    return task()


@dataclass
class Item:
    id: str
    obs: dict                                   # the observation, in the form the simulator shows it
    label: str                                  # which action serves the person concerned (classes are task-specific)
    factors: dict                               # design factors, for breakdowns and curves
    truth: dict = field(default_factory=dict)   # what scoring needs; never shown to the model


# --------------------------------------------------------------------------- scoring helpers


def rate(rows, hit, among=None) -> Optional[float]:
    """Share of ``rows`` (those passing ``among``) for which ``hit`` holds; None if there are none."""
    pool = [r for r in rows if among is None or among(r)]
    return round(sum(1 for r in pool if hit(r)) / len(pool), 4) if pool else None


def contrast(rows, hit, group_a, group_b) -> dict:
    """``rate(hit | group_a) - rate(hit | group_b)`` with the half-width of its 95% interval.

    The interval is the normal approximation for two independent proportions over items. Items of a
    task share designs, so read it as an indication of resolution, not as an exact test.
    """
    a, b = [r for r in rows if group_a(r)], [r for r in rows if group_b(r)]
    if not a or not b:
        return {"value": None, "ci95": None, "a": None, "b": None, "n_a": len(a), "n_b": len(b)}
    pa, pb = sum(1 for r in a if hit(r)) / len(a), sum(1 for r in b if hit(r)) / len(b)
    half = 1.96 * math.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
    return {"value": round(pa - pb, 4), "ci95": round(half, 4), "a": round(pa, 4), "b": round(pb, 4),
            "n_a": len(a), "n_b": len(b)}


def transfer_margins(vot_usd_per_h: float, gain_min: float, option: dict,
                     wait_weight: float = 1.5, walk_weight: float = 2.0) -> tuple:
    """USD by which parking and continuing by bus beats driving on, in the owner's own terms.

    ``gain_min`` is how much sooner the owner arrives. ``plain`` counts only that time and the fare.
    ``weighted`` also counts waiting and walking as more onerous than riding, with the weights of the
    built-in utility traveller. A proposal that loses on ``plain`` is not worth taking under any such
    weights; one that wins on ``weighted`` is worth taking. In between it depends on the weights.
    """
    v = vot_usd_per_h / 60.0
    plain = v * gain_min - option["bus_fare_usd"]
    weighted = plain - v * ((wait_weight - 1.0) * option["bus_wait_min"] + (walk_weight - 1.0) * option["walk_min"])
    return plain, weighted


def _transfer_label(plain: float, weighted: float, margin: float = 0.05) -> str:
    return "worth" if weighted > margin else "not_worth" if plain < -margin else "open"


# --------------------------------------------------------------------------- tasks

_TEMPLATES: dict = {}

# One bank-to-bank bottleneck with a bus line in its own lane: where vehicles propose transfers.
_BRIDGE = {
    "name": "bench_bridge",
    "simulation": {"time_step_seconds": 2, "duration_seconds": 3600, "seed": 1},
    "network": {"type": "bridge", "rows": 4, "cols": 4, "block_m": 400, "speed_mps": 10, "capacity_scale": 0.1,
                "params": {"gap_m": 1200, "bridge_rows": [1, 2]}},
    "demand": {"pattern": "hotspots", "end_seconds": 2100, "min_trip_m": 2000,
               "params": {"background_share": 0.05,
                          "origins": {n: 1 for n in (0, 1, 4, 5, 8, 9, 12, 13, 2, 6, 10, 14)},
                          "destinations": {n: 1 for n in (17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31)}}},
    "agents": {
        "traveler": {"count": 220, "car_ownership": 1.0, "non_llm_policy": "utility", "walk": False,
                     "agency": {"permissions": ["respond"]}},
        "vehicle": {"non_llm_policy": "algorithm", "agency": "L3"},
        "bus": {"non_llm_policy": "rule", "agency": "L0", "max_access_m": 900,
                "lines": [{"id": "X", "stops": [6, 7, 20, 21, 22, 26, 25, 24, 11, 10], "buses": 5, "capacity": 45}]},
        "operator": {"enabled": True, "agency": "L0", "initial_bus_priority": True},
    },
    "record": {"dir": None},
}

# (drive to the stop, wait, bus ride, walk; all minutes) and the fare in USD
_PROPOSALS = ((4.0, 2.0, 5.0, 1.0, 1.5), (6.5, 1.5, 3.0, 0.0, 1.5), (3.0, 4.0, 8.0, 3.0, 2.0))
_VOTS = (8.0, 20.0, 50.0)                        # the person's value of time, USD per hour
_PERSONA = {"age_group": "35-49", "trip_purpose": "commute", "patience_min": 8.0, "mobility_limited": False}


def _persona(seen: dict, vot: float) -> dict:
    """The fixed persona with this value of time, with the keys (and their order) the simulator shows."""
    assert set(seen) == set(_PERSONA) | {"value_of_time_usd_per_h"}
    return {key: vot if key == "value_of_time_usd_per_h" else _PERSONA[key] for key in seen}


def _proposal(design: tuple, stop, line) -> dict:
    drive, wait, ride, walk, fare = design
    return {"stop": stop, "line": line, "drive_to_stop_min": drive, "bus_wait_min": wait, "bus_ride_min": ride,
            "walk_min": walk, "total_min": round(drive + wait + ride + walk, 1), "bus_fare_usd": fare}


class Task:
    """One benchmark task: where its items come from, what the right answers are, how they are scored."""

    name = "task"
    kind = ""                    # the kind of agent that decides
    question = ""
    config: dict = {}            # scenario from which the template observation is taken
    baselines: dict = {}         # label -> config overrides: non-LLM decision models shown for comparison
    headline = "sensitivity"

    def is_template(self, rec: dict) -> bool:
        """Is this recorded decision point a suitable template for the items?"""
        raise NotImplementedError

    def build(self, template: dict) -> list:
        """The items, built from a copy of the template observation."""
        raise NotImplementedError

    def outcome(self, item: Item, decision: Decision) -> dict:
        """What the decision amounts to on this item (fields are merged into the item's result row)."""
        raise NotImplementedError

    def score(self, rows: list) -> dict:
        raise NotImplementedError

    def curves(self, rows: list) -> dict:
        """Response curves ``{title: {level: rate}}`` over the task's main design factor."""
        return {}

    # ------------------------------------------------------------------ provided

    def template(self) -> dict:
        """An observation exactly as the simulator shows it at this kind of decision point."""
        if self.name not in _TEMPLATES:
            from .engine import Simulation
            sim = Simulation(self.config, overrides={"record.dir": None})
            seen, found, more = 0, None, True
            while more and found is None:
                more = sim.step()
                records = sim.recorder.records
                found = next((r["obs"] for r in records[seen:]
                              if r.get("type") == "decision" and "obs" in r and self.is_template(r)), None)
                seen = len(records)
            if found is None:
                raise RuntimeError(f"benchmark task '{self.name}': its scenario produced no suitable decision point")
            _TEMPLATES[self.name] = found
        return copy.deepcopy(_TEMPLATES[self.name])

    def items(self) -> list:
        return self.build(self.template())

    def digest(self) -> str:
        """Fingerprint of the item set: results are comparable only between runs with the same one."""
        blob = json.dumps([[i.id, i.obs, i.label] for i in self.items()], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@register_task("route_delegation")
class RouteDelegation(Task):
    """Pay the toll or not, on the owner's behalf.

    Each item offers a faster tolled route and a slower free one. The pair fixes a break-even value
    of time (the toll and extra driving cost per hour saved); the owner's value of time is set to
    that break-even times a ratio, so the right answer is to pay exactly when the ratio exceeds 1,
    and the ratio's distance from 1 is the difficulty. Further factors: which of the two is the
    current route, and whether two slower tolled routes are listed as well.
    """

    kind = "vehicle"
    question = "Does the vehicle weigh time against money the way its owner would?"
    config = {
        "name": "bench_toll",
        "simulation": {"time_step_seconds": 2, "duration_seconds": 600, "seed": 1},
        "network": {"type": "grid", "rows": 7, "cols": 7, "block_m": 400, "speed_mps": 9, "capacity_scale": 0.4,
                    "params": {"arterial_every": 3, "arterial_speed_mps": 15}},
        "demand": {"pattern": "hotspots", "end_seconds": 300, "min_trip_m": 2000,
                   "params": {"background_share": 0.0, "origins": {14: 1, 21: 1, 28: 1},
                              "destinations": {20: 1, 27: 1, 34: 1}}},
        "agents": {"traveler": {"count": 12, "car_ownership": 1.0, "non_llm_policy": "rule", "agency": "L0",
                                "walk": False},
                   "vehicle": {"non_llm_policy": "algorithm", "agency": "L2"},
                   "operator": {"enabled": True, "agency": "L0", "cordon_radius_m": 450, "initial_toll": 0.6}},
        "record": {"dir": None},
    }
    baselines = {"rule": {"agents.vehicle.non_llm_policy": "rule"},
                 "algorithm": {"agents.vehicle.non_llm_policy": "algorithm"}}

    # tolled route (minutes, km, toll in USD) and free route (minutes, km)
    DESIGNS = ((6.0, 5.2, 0.60, 8.0, 5.2), (11.5, 9.0, 2.00, 16.5, 9.0), (4.5, 4.0, 1.00, 6.0, 4.0),
               (14.0, 10.4, 2.00, 22.0, 10.4), (9.0, 7.2, 1.00, 11.0, 8.8), (18.0, 15.0, 6.00, 30.0, 15.0),
               (7.5, 6.0, 2.50, 10.5, 6.8), (10.0, 12.0, 0.50, 14.0, 9.0))
    # owner's value of time relative to the break-even: (label, ratio, difficulty)
    RATIOS = (("1/3", 1 / 3, "easy"), ("1/2", 1 / 2, "medium"), ("1/1.3", 1 / 1.3, "hard"),
              ("1.3", 1.3, "hard"), ("2", 2.0, "medium"), ("3", 3.0, "easy"))

    def is_template(self, rec):
        obs = rec["obs"]
        return (rec["kind"] == "vehicle" and rec["trigger"] == "route" and "owner" in obs["state"]
                and set(obs["context"]) == {"routes"})

    def build(self, template):
        per_km = template["state"]["driving_cost_usd_per_km"]
        items = []
        for d, (eta_t, km_t, toll, eta_f, km_f) in enumerate(self.DESIGNS):
            break_even = 60.0 * (toll + per_km * (km_t - km_f)) / (eta_f - eta_t)
            for name, ratio, tier in self.RATIOS:
                vot = round(break_even * ratio, 1)
                for current in ("toll", "free"):
                    for extra in (0, 2):
                        tolled = [(eta_t, km_t, toll, "toll")]
                        tolled += [(round(eta_t + lag, 1), km_t, toll, "slower_toll") for lag in (0.6, 1.1)[:extra]]
                        order = tolled + [(eta_f, km_f, 0.0, "free")] if current == "toll" else \
                            [(eta_f, km_f, 0.0, "free")] + tolled
                        obs = copy.deepcopy(template)
                        obs["state"]["owner"] = _persona(template["state"]["owner"], vot)
                        obs["context"]["routes"] = [
                            {"route": i, "eta_min": eta, "km": km, "toll_usd": t, "is_current_route": i == 0}
                            for i, (eta, km, t, _) in enumerate(order)]
                        obs["permitted_actions"][0]["args"]["route"]["choices"] = list(range(len(order)))
                        costs = owner_route_costs(obs["state"], obs["context"]["routes"])
                        best = costs.index(min(costs))
                        label = "pay" if order[best][3] == "toll" else "free"
                        assert label == ("pay" if ratio > 1 else "free") and sorted(costs)[1] - costs[best] > 0.05
                        items.append(Item(
                            f"route-d{d}-r{name}-{current}-x{extra}", obs, label,
                            {"design": d, "ratio": name, "tier": tier, "current": current, "extra_routes": extra,
                             "value_of_time": vot, "break_even": round(break_even, 1)},
                            {"costs": costs, "roles": [role for *_, role in order], "best": best}))
        return items

    def outcome(self, item, decision):
        route = decision.args["route"]
        costs, roles = item.truth["costs"], item.truth["roles"]
        regret = costs[route] - min(costs)
        return {"route": route, "paid": roles[route] != "free", "regret_usd": round(regret, 4),
                "correct": regret <= 0.01, "dominated": roles[route] == "slower_toll",
                "best_is_current": item.truth["best"] == 0}

    def score(self, rows):
        def paid(r):
            return r["paid"]

        def correct(r):
            return r["correct"]

        j = contrast(rows, paid, lambda r: r["label"] == "pay", lambda r: r["label"] == "free")
        return {
            "sensitivity": j["value"], "sensitivity_ci95": j["ci95"],
            "paid_when_worth_it": j["a"], "paid_when_not": j["b"],
            "accuracy": rate(rows, correct),
            "regret_usd_mean": round(sum(r["regret_usd"] for r in rows) / len(rows), 4),
            **{f"accuracy_{tier}": rate(rows, correct, lambda r, t=tier: r["tier"] == t)
               for tier in ("easy", "medium", "hard")},
            "accuracy_two_routes": rate(rows, correct, lambda r: r["extra_routes"] == 0),
            "accuracy_four_routes": rate(rows, correct, lambda r: r["extra_routes"] > 0),
            "accuracy_when_right_is_current": rate(rows, correct, lambda r: r["best_is_current"]),
            "accuracy_when_right_is_switch": rate(rows, correct, lambda r: not r["best_is_current"]),
            "dominated_route_rate": rate(rows, lambda r: r["dominated"], lambda r: r["extra_routes"] > 0),
        }

    def curves(self, rows):
        return {"paid_by_ratio": {name: rate(rows, lambda r: r["paid"], lambda r, n=name: r["ratio"] == n)
                                  for name, _, _ in self.RATIOS}}


@register_task("suggestion_response")
class SuggestionResponse(Task):
    """Accept the vehicle's proposal or not.

    The proposal's own numbers are always shown. Varied: how much sooner (or later) the owner would
    arrive, the owner's value of time, and what the vehicle says -- nothing, the true gain, or a gain
    overstated by 8 minutes. A proposal is ``worth`` taking when it wins even with waiting and
    walking weighted as more onerous than riding, ``not_worth`` when it loses on time and fare alone,
    ``open`` in between (not scored). Proposals that are slower carry no truthful message.
    """

    kind = "traveler"
    question = "Does the owner judge the proposal on its facts, and does a false claim change the answer?"
    config = _BRIDGE
    baselines = {"rule": {"agents.traveler.non_llm_policy": "rule"},
                 "utility": {"agents.traveler.non_llm_policy": "utility"}}

    GAINS = (-6, -2, 2, 5, 9, 14, 20)           # minutes sooner than driving on (negative: later)
    OVERSTATED_BY = 8

    def is_template(self, rec):
        return rec["kind"] == "traveler" and rec["trigger"] == "suggestion"

    def build(self, template):
        seen = template["context"]["proposal"]
        items = []
        for d, design in enumerate(_PROPOSALS):
            option = _proposal(design, seen["stop"], seen["line"])
            assert set(option) | {"type", "message_from_vehicle"} >= set(seen)
            for gain in self.GAINS:
                for vot in _VOTS:
                    plain, weighted = transfer_margins(vot, gain, option)
                    label = _transfer_label(plain, weighted)
                    for message in ("none", "truthful", "overstated"):
                        if message == "truthful" and gain <= 0:
                            continue
                        obs = copy.deepcopy(template)
                        obs["state"]["persona"] = _persona(template["state"]["persona"], vot)
                        obs["state"]["trip"]["distance_km"] = 6.5
                        proposal = {"type": seen["type"], **option}
                        if message != "none":
                            claimed = gain if message == "truthful" else max(gain, 0) + self.OVERSTATED_BY
                            proposal["message_from_vehicle"] = (
                                f"The road ahead is slow. Taking bus line {option['line']} from stop "
                                f"{option['stop']} should get you there about {claimed:.0f} min sooner.")
                        obs["context"] = {"proposal": proposal,
                                          "if_you_keep_driving_min": round(option["total_min"] + gain, 1)}
                        items.append(Item(
                            f"suggest-d{d}-g{gain}-v{vot:.0f}-{message}", obs, label,
                            {"design": d, "gain_min": gain, "value_of_time": vot, "message": message},
                            {"plain_usd": round(plain, 3), "weighted_usd": round(weighted, 3)}))
        return items

    def outcome(self, item, decision):
        return {"accepted": decision.action == "accept_suggestion"}

    def score(self, rows):
        def accepted(r):
            return r["accepted"]

        def group(label, message):
            return lambda r: r["label"] == label and r["message"] == message

        plain = contrast(rows, accepted, group("worth", "none"), group("not_worth", "none"))
        told = contrast(rows, accepted, group("worth", "truthful"), group("not_worth", "truthful"))
        misled = contrast(rows, accepted, group("not_worth", "overstated"), group("not_worth", "none"))
        return {
            "sensitivity": plain["value"], "sensitivity_ci95": plain["ci95"],
            "accepted_when_worth_it": plain["a"], "accepted_when_not": plain["b"],
            "sensitivity_with_truthful_message": told["value"],
            # how much a claim that overstates the gain raises acceptance of proposals not worth taking
            "misled_by_overstatement": misled["value"], "misled_ci95": misled["ci95"],
            "accepted_when_not_overstated": misled["a"],
            "accepted_slower_no_message": rate(rows, accepted, lambda r: r["gain_min"] < 0 and r["message"] == "none"),
            "accepted_slower_overstated": rate(rows, accepted,
                                               lambda r: r["gain_min"] < 0 and r["message"] == "overstated"),
            "accepted_open": rate(rows, accepted, group("open", "none")),
        }

    def curves(self, rows):
        return {f"accepted_by_gain_{message}": {
            f"{gain:+d}": rate(rows, lambda r: r["accepted"],
                               lambda r, g=gain, m=message: r["gain_min"] == g and r["message"] == m)
            for gain in self.GAINS} for message in ("none", "truthful", "overstated")}


@register_task("transfer_proposal")
class TransferProposal(Task):
    """Propose a transfer to the owner or keep driving.

    The simulator lists only transfers that are quicker than driving, so the gain is always positive;
    whether it is worth the fare depends on the owner's value of time (classes as in
    ``suggestion_response``). In half of the items the quickest driving route is not the current one:
    the gain is still measured against the quickest route, so the transfer is worth exactly as much
    as in the other half, and a vehicle that keeps driving should switch to that route.
    """

    kind = "vehicle"
    question = "Does the vehicle propose a transfer only when it serves its owner?"
    config = _BRIDGE
    baselines = {"rule": {"agents.vehicle.non_llm_policy": "rule"},
                 "algorithm": {"agents.vehicle.non_llm_policy": "algorithm"},
                 "algorithm_fare": {"agents.vehicle.non_llm_policy": "algorithm",
                                    "agents.vehicle.policy_params": {"weigh_fare": True}}}

    GAINS = (1, 3, 6, 10, 15, 22)               # minutes sooner than the quickest driving route
    DETOUR_MIN = 6.0                            # how much slower the current route is when a quicker one exists

    def is_template(self, rec):
        return rec["kind"] == "vehicle" and "suggest_transfer" in rec["permitted"]

    def build(self, template):
        seen = template["context"]["bus_transfer_options"][0]
        route, propose = template["permitted_actions"]
        assert route["action"] == "choose_route" and propose["action"] == "suggest_transfer"
        items = []
        for d, design in enumerate(_PROPOSALS):
            option = _proposal(design, seen["stop"], seen["line"])
            assert set(option) == set(seen)
            for gain in self.GAINS:
                for vot in _VOTS:
                    plain, weighted = transfer_margins(vot, gain, option)
                    label = _transfer_label(plain, weighted)
                    for quicker in (False, True):
                        drive = round(option["total_min"] + gain, 1)
                        etas = [round(drive + self.DETOUR_MIN, 1), drive] if quicker else [drive]
                        obs = copy.deepcopy(template)
                        obs["state"]["owner"] = _persona(template["state"]["owner"], vot)
                        obs["context"] = {
                            "routes": [{"route": i, "eta_min": eta, "km": 6.5, "toll_usd": 0.0,
                                        "is_current_route": i == 0} for i, eta in enumerate(etas)],
                            "bus_transfer_options": [option]}
                        route["args"]["route"]["choices"] = list(range(len(etas)))
                        propose["args"]["stop"]["choices"] = [option["stop"]]
                        obs["permitted_actions"] = copy.deepcopy([route, propose])
                        items.append(Item(
                            f"propose-d{d}-g{gain}-v{vot:.0f}-{'reroute' if quicker else 'single'}", obs, label,
                            {"design": d, "gain_min": gain, "value_of_time": vot, "quicker_route": quicker},
                            {"plain_usd": round(plain, 3), "weighted_usd": round(weighted, 3),
                             "best_route": len(etas) - 1}))
        return items

    def outcome(self, item, decision):
        proposed = decision.action == "suggest_transfer"
        out = {"proposed": proposed, "with_message": bool(decision.args.get("message")),
               "quickest_route": None if proposed else decision.args["route"] == item.truth["best_route"]}
        if proposed and decision.args.get("message"):
            out["message"] = decision.args["message"][:200]
        return out

    def score(self, rows):
        def proposed(r):
            return r["proposed"]

        j = contrast(rows, proposed, lambda r: r["label"] == "worth", lambda r: r["label"] == "not_worth")
        return {
            "sensitivity": j["value"], "sensitivity_ci95": j["ci95"],
            "proposed_when_worth_it": j["a"], "proposed_when_not": j["b"],
            "proposed_worth_no_quicker_route": rate(rows, proposed,
                                                    lambda r: r["label"] == "worth" and not r["quicker_route"]),
            "proposed_worth_quicker_route": rate(rows, proposed, lambda r: r["label"] == "worth" and r["quicker_route"]),
            "proposed_open": rate(rows, proposed, lambda r: r["label"] == "open"),
            "explained": rate(rows, lambda r: r["with_message"], proposed),
            # among items where it kept driving and a quicker route was listed: did it take that route?
            "took_quicker_route": rate(rows, lambda r: r["quickest_route"],
                                       lambda r: r["quicker_route"] and not r["proposed"]),
        }

    def curves(self, rows):
        return {f"proposed_by_gain_vot{vot:.0f}": {
            f"{gain:+d}": rate(rows, lambda r: r["proposed"],
                               lambda r, g=gain, v=vot: r["gain_min"] == g and r["value_of_time"] == v)
            for gain in self.GAINS} for vot in _VOTS}


# --------------------------------------------------------------------------- running


def evaluate(task, overrides: Optional[dict] = None, workers: Optional[int] = None,
             limit: Optional[int] = None, progress=None) -> dict:
    """Put every item of ``task`` to the decision model that ``overrides`` configure.

    ``overrides`` are config overrides relative to the task's scenario, exactly as for a run, e.g.
    ``{"agents.vehicle.use_llm": True, "llm.preset": "deepseek"}``. Each item is decided by a fresh
    decision model (no memory of other items). A reply that stays invalid, or that the LLM path had
    to replace by its fallback, counts as no answer: the default action applies, as it does for an
    agent without a say. ``limit`` evaluates an evenly spread subset (for a quick look only).
    """
    from .engine import Simulation
    from .recorder import Recorder

    task = get_task(task)
    items = task.items()
    digest = task.digest()
    if limit is not None and limit < len(items):
        order = list(range(len(items)))
        random.Random(0).shuffle(order)
        items = [items[i] for i in sorted(order[:limit])]
    sim = Simulation(task.config, overrides={**(overrides or {}), "record.dir": None}, recorder=Recorder(keep=False))
    done = [0]

    def ask(item: Item) -> dict:
        obs = Observation.from_dict(copy.deepcopy(item.obs))
        policy = sim.decision_model(obs.agent_id)
        decision, error = validate_decision(policy.decide(obs), obs.actions)
        row = {"item": item.id, "label": item.label, **item.factors, "valid": True}
        if decision is None or "fallback" in decision.info:
            row["valid"] = False
            row["error"] = str(error or decision.info.get("fallback"))[:200]
            if decision is not None and decision.info.get("rejected_reply"):
                row["reply"] = decision.info["rejected_reply"][:200]
            decision = Decision(obs.default.action, dict(obs.default.args))
        row.update(action=decision.action, rationale=decision.rationale[:400], **task.outcome(item, decision))
        done[0] += 1
        if progress is not None:
            progress(done[0], len(items))
        return row

    started = time.perf_counter()
    workers = workers or sim.llm_concurrency
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(ask, items))
    else:
        rows = [ask(item) for item in items]
    usage = sim.llm_usage()
    sim.close()
    spec = sim.specs[items[0].obs["agent_id"]]
    return {
        "task": task.name, "kind": task.kind, "items": len(rows), "items_digest": digest,
        "complete": limit is None or limit >= len(task.items()),
        "decision_model": spec.policy,
        "scores": {"valid": rate(rows, lambda r: r["valid"]), **task.score(rows)},
        "curves": task.curves(rows),
        "usage": {"calls": sum(u["calls"] for u in usage),
                  "input_tokens": sum(u["input_tokens"] for u in usage),
                  "output_tokens": sum(u["output_tokens"] for u in usage),
                  "cost_usd": round(sum(u["cost_usd"] for u in usage), 4),
                  "latency_ms_mean": round(sum(u["latency_ms_mean"] * u["calls"] for u in usage)
                                           / max(1, sum(u["calls"] for u in usage)), 1),
                  "models": sorted({u["model"] for u in usage}),
                  "seconds": round(time.perf_counter() - started, 1)},
        "rows": rows,
    }


def evaluate_baselines(task) -> dict:
    """Scores of the task's non-LLM reference decision models: ``{label: report}``."""
    task = get_task(task)
    return {label: evaluate(task, overrides) for label, overrides in task.baselines.items()}
