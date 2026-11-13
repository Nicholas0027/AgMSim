"""Travellers: the human layer.

A traveller has one trip need, or a day plan (a chain of activities joined by
trips, see :mod:`agmsim.plans`). Depending on their agency they choose how to
travel (own agentic vehicle, bus, walking), answer suggestions made by their
vehicle, give up waiting at a bus stop, re-plan after a failed attempt, and --
with a day plan -- change the rest of the plan when the day does not go as
they reckoned.
"""
from __future__ import annotations

import math

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..demand import generate_travelers
from ..entities import TState
from ..observation import km, minutes
from ..plans import REVIEW_GAIN_USD, hhmm, option_value, preferences, project
from .base import Kind, register_kind


# Told to a language model only when the scenario has day plans.
PLAN_PROMPT = (
    " You have a plan for the day: activities at fixed places, joined by trips. state.preferences says what an "
    "hour is worth to you in USD: at an activity, travelling (negative), and being late for an activity that "
    "must start by a given time (negative). Done for its usual length (typical_min) an activity is worth that "
    "length times the hourly rate times its priority (1 unless stated); a minute more or less than usual "
    "changes this by about the hourly rate per hour, cutting it short costs more the shorter it gets, and "
    "running over adds less and less. outside_min is time spent on the activity before or after the period "
    "shown. An optional activity that you skip is worth nothing, and one you keep but cannot do (arriving "
    "after it closes) is worse than skipping it."
)
PLAN_TRIGGERS = {
    "plan_review": "you are at an activity and, as traffic stands now, the rest of your day will not go as you "
                   "reckoned; keep the plan or change it. context.options lists what each choice leads to: when "
                   "you would leave, how long you would have been here, when you would reach each later "
                   "activity, how long you could stay and how late you would be, and the travel time and cost.",
}
PLAN_ACTIONS = {
    "keep_plan": "Carry on with the plan as it stands.",
    "leave_at": "End the current activity at the given time instead of the planned time.",
    "skip_activity": "Drop an optional activity from the rest of your plan.",
}


@register_kind("traveler")
class TravelerKind(Kind):
    name = "traveler"
    side = "human"
    title = "Traveller"
    role_prompt = (
        "You are role-playing one traveller. Make the choices this particular person would plausibly make, "
        "not the choices an optimiser would make. state.persona describes you: age group, trip purpose, how much "
        "you value your time (USD per hour), roughly how long you are willing to wait (patience, minutes) and "
        "whether your mobility is limited. Let these characteristics drive your choices. If you own a car it is "
        "an agentic vehicle that drives you and may suggest changes of plan."
    )
    triggers = {
        "need": "your trip need has just arisen; choose how to travel",
        "suggestion": "your vehicle proposes a change of plan while you are on the way; accept or decline",
        "wait_check": "you have been waiting at a bus stop; keep waiting or give up",
        "replan": "your earlier attempt ended without transport; choose again",
    }
    actions = {
        "choose_service": "Decide how to make the trip, using one of the listed options.",
        "accept_suggestion": "Accept your vehicle's proposal.",
        "decline_suggestion": "Decline the proposal and keep to the current plan.",
        "keep_waiting": "Keep waiting.",
        "give_up_waiting": "Leave the bus stop.",
    }
    agency = AgencySpec(
        "traveler",
        {
            "choose_service": "choose among the available travel options when a trip need arises",
            "respond": "accept or decline a change of plan proposed by their vehicle",
            "cancel": "give up while waiting at a bus stop",
            "replan": "choose again after a failed attempt",
            "adapt_plan": "change the rest of their day plan (when to leave an activity, dropping an optional one)",
        },
        {
            "L0": (),
            "L1": ("choose_service",),
            "L2": ("choose_service", "respond"),
            "L3": ("choose_service", "respond", "cancel"),
            "L4": ("choose_service", "respond", "cancel", "replan", "adapt_plan"),
        },
    )

    # ------------------------------------------------------------------ setup

    def setup(self):
        sim, w = self.sim, self.world
        w.travelers = generate_travelers(sim.cfg, w.net, sim.streams.get("demand"), w.dt, w.clock0, w.pois)
        self._needs: dict[int, list] = {}
        self._planners = [tr for tr in w.travelers.values() if tr.plan is not None]
        self.horizon_s = sim.n_ticks * w.dt
        self.review_regret, self.review_best = 0.0, 0      # how good the chosen plan edits were (see _apply_review)
        self.review_edit = {True: [0, 0], False: [0, 0]}   # a change was worth making? -> [reviews, plan changed]
        for tr in w.travelers.values():
            if tr.plan is None:
                self._needs.setdefault(tr.t_need, []).append(tr)
            else:
                tr.state, tr.origin = TState.AT_ACTIVITY, tr.plan.activities[0].node
        if "ondemand" in sim.kinds:         # only then is calling a car part of what a language model is told
            self.role_prompt = type(self).role_prompt + (
                " You can also call an on-demand car (the option 'ride'): it has to come to you first, which takes "
                "a while, and you pay its fare.")
            self.triggers = {**type(self).triggers, "wait_check": "you have been waiting, at a bus stop or for the "
                             "car you called; keep waiting or give up"}
            self.actions = {**type(self).actions, "give_up_waiting": "Stop waiting: leave the bus stop, or cancel the car."}
        if self._planners:                  # only then are plans part of what a language model is told
            self.role_prompt = self.role_prompt + PLAN_PROMPT
            self.triggers = {**self.triggers, **PLAN_TRIGGERS}
            self.actions = {**self.actions, **PLAN_ACTIONS}
        return list(w.travelers)

    def begin_tick(self):
        w = self.world
        for tr in self._needs.get(w.tick, ()):
            tr.state = TState.DECIDING
        for tr in self._planners:
            if tr.state == TState.AT_ACTIVITY:
                if tr.plan.expected is None:        # the start of the run: what the person reckons with
                    tr.plan.expected = self._expected(tr)
                if not self._review_due(tr) and w.time_s >= tr.plan.leave_s:
                    self._begin_leg(tr)

    # ------------------------------------------------------------------ day plans (see agmsim.plans)

    def _begin_leg(self, tr) -> None:
        """The current activity ends: the trip to the next one becomes an ordinary trip need."""
        w, p = self.world, tr.plan
        here = p.activities[p.at]
        here.left_s = w.time_s
        ahead = p.ahead()
        if not ahead:                       # nothing left to do today
            tr.state = TState.ARRIVED
            return
        goal = p.activities[ahead[0]]
        if goal.node == here.node:
            self._start_activity(tr, ahead[0], w.tick)
            return
        route = w.net.route(here.node, goal.node)
        tr.origin, tr.destination, tr.t_need = here.node, goal.node, w.tick
        tr.trip_m, tr.trip_s = route.length, route.time
        tr.mode = tr.t_depart = tr.t_arrival = tr.bus = tr.suggestion = tr.next_check = tr.wait_start = None
        tr.replan_pending = False
        p.leg = {"to": ahead[0], "depart": w.tick, "depart_s": w.time_s, "paid": tr.paid,
                 "waited": tr.waited_ticks, "cancels": tr.cancels}
        tr.state = TState.DECIDING

    def trip_done(self, tr, now: int) -> None:
        """A trip has ended. For a person with a day plan it was one leg: the next activity starts."""
        w, p = self.world, tr.plan
        at = tr.poi if p is None else p.activities[p.leg["to"]].poi if p.leg is not None else None
        if at is not None:                  # the trip went to a point of interest
            poi = w.pois[at]
            poi.visits += 1
            poi.found_closed += int(poi.close_s is not None and now * w.dt >= poi.close_s)
        if p is None or p.leg is None:
            return
        leg, p.leg = p.leg, None
        p.legs.append({"to": leg["to"], "mode": tr.mode, "depart": leg["depart"], "arrive": now,
                       "depart_s": leg["depart_s"], "arrive_s": now * self.world.dt, "paid": tr.paid - leg["paid"],
                       "waited": tr.waited_ticks - leg["waited"], "cancels": tr.cancels - leg["cancels"]})
        self._start_activity(tr, leg["to"], now)

    def _start_activity(self, tr, k: int, now: int) -> None:
        w, p = self.world, tr.plan
        a = p.activities[k]
        p.at, a.arrived_s, tr.origin = k, now * w.dt, a.node
        if k == len(p.activities) - 1:
            tr.state = TState.ARRIVED       # the last activity: the day's travel is over
            return
        tr.state, p.leave_s, p.next_check = TState.AT_ACTIVITY, a.leaves(a.arrived_s, self.horizon_s), now

    def _estimator(self, tr):
        """Travel time and cost between two nodes as they stand now, by the means this person would use."""
        sim, w, cfg, persona = self.sim, self.world, self.cfg, tr.persona
        v = w.vehicles.get(tr.car) if tr.car else None
        by_car = v is not None and v.link is None and v.passenger is None and v.node == tr.origin
        memo: dict = {}

        def estimate(a: int, b: int):
            if a == b:
                return 0.0, 0.0
            if (a, b) not in memo:
                metres = w.net.length(a, b)
                if by_car:
                    vcfg = sim.cfg.agents.vehicle
                    usd = (metres / 1000.0 * vcfg.operating_cost_per_km + vcfg.parking_fee
                           + w.route_toll(w.net.route(a, b, w.advisory).links))
                    memo[a, b] = (w.net.time(a, b, True), usd)
                else:
                    best = (metres / persona.walk_speed, 0.0)       # on foot, if nothing better
                    plan = sim.kinds["bus"].plan(a, b, persona) if "bus" in sim.kinds else None
                    if plan is not None:
                        total = plan["access_s"] + plan["wait_s"] + plan["ride_s"] + plan["egress_s"]
                        if total < best[0] or not cfg.walk or persona.mobility_limited or metres > persona.max_walk_m:
                            best = (total, plan["fare"])
                    memo[a, b] = best
            return memo[a, b]

        return estimate

    def _expected(self, tr) -> dict:
        """When the person now reckons to reach each later activity."""
        w, p = self.world, tr.plan
        rows, _, _ = project(p, max(w.time_s, p.leave_s), self._estimator(tr), self.horizon_s)
        return {k: arrive for k, arrive, _, _ in rows}

    def _review_due(self, tr) -> bool:
        """Has the rest of the day drifted from what the person reckons with? If so, prepare the review."""
        w, cfg, p = self.world, self.cfg, tr.plan
        if not self.allows(tr.id, "adapt_plan") or (w.tick < p.next_check and not p.follow_up):
            return False
        p.next_check = w.tick + w.ticks(cfg.plan_check_seconds)
        follow_up, p.follow_up = p.follow_up, False
        now, leave = w.time_s, max(w.time_s, p.leave_s)
        if not follow_up and leave - now > cfg.plan_review_horizon_seconds:
            return False
        estimate = self._estimator(tr)
        projected = project(p, leave, estimate, self.horizon_s)
        if not projected[0]:
            return False
        k, drift = max(((k, arrive - p.expected.get(k, arrive)) for k, arrive, _, _ in projected[0]),
                       key=lambda item: abs(item[1]))
        if not follow_up and abs(drift) < cfg.plan_review_threshold_seconds:
            return False
        if follow_up:
            reason, p.notice = p.notice or "you have just changed your plan and may adjust it further", ""
        else:
            reason = (f"as traffic stands now you would reach '{p.activities[k].type}' about "
                      f"{abs(drift) / 60.0:.0f} min {'later' if drift > 0 else 'earlier'} than you reckoned")
        p.pending = self._review(tr, leave, estimate, projected, reason)
        return True

    def place_closed(self, poi) -> None:
        """A place closes earlier than it said: what people planned to do there ends then, and those who may
        adapt their plan take a look at it."""
        w = self.world
        for tr in self._planners:
            p = tr.plan
            for k in [p.at, *p.ahead()]:
                a = p.activities[k]
                if a.poi == poi.id and a.left_s is None:
                    a.close_s = poi.close_s
                    if self.allows(tr.id, "adapt_plan"):
                        p.follow_up = True
                        p.notice = f"'{a.type}' at {poi.name} closes at {hhmm(w.clock0 + poi.close_s)} today, earlier than planned"

    def _plan_row(self, a, k: int) -> dict:
        c0 = self.world.clock0
        row = {"activity": k, "type": a.type, "typical_min": minutes(a.typical_s)}
        if a.poi is not None:
            row["place"] = self.world.pois[a.poi].name
        if a.priority != 1.0:
            row["priority"] = a.priority
        if a.outside_s > 0:
            row["outside_min"] = minutes(a.outside_s)
        if a.optional:
            row["optional"] = True
        for key, value in (("must_start_by", a.latest_start_s), ("opens", a.open_s), ("closes", a.close_s),
                           ("planned_end", a.end_s)):
            if value is not None:
                row[key] = hhmm(c0 + value)
        if a.end_s is None and a.duration_s is not None:
            row["planned_stay_min"] = minutes(a.duration_s)
        return row

    def _review(self, tr, leave: float, estimate, projected, reason: str) -> dict:
        """Context and permitted actions of a plan review: every edit, with what it leads to."""
        w, p = self.world, tr.plan
        acts, now, c0 = p.activities, w.time_s, w.clock0
        here = acts[p.at]

        def option(choice: dict, leave_s: float, outcome=None) -> dict:
            rows, travel, usd = outcome or project(p, leave_s, estimate, self.horizon_s,
                                                   frozenset((choice["activity"],)) if "activity" in choice else frozenset())
            then = []
            for k, arrive, there, late in rows:
                row = {"activity": k, "type": acts[k].type, "arrive": hhmm(c0 + arrive), "minutes_there": minutes(there)}
                if acts[k].latest_start_s is not None:
                    row["late_min"] = minutes(late)
                then.append(row)
            return {"choice": choice, "leave_at": hhmm(c0 + leave_s),
                    "minutes_here": minutes(here.stay(here.arrived_s, leave_s)[0]), "then": then,
                    "travel_min": minutes(travel), "travel_cost_usd": round(usd, 2)}

        options = [option({"action": "keep_plan"}, leave, projected)]
        # other times to leave: now, then on the clock's five-minute marks (ten-minute marks further out)
        times, taken = {}, {hhmm(c0 + leave)}
        candidates = [now]
        mark = (math.floor((c0 + now) / 300.0) + 1) * 300.0 - c0
        while mark <= leave + 1800.0:
            near = leave - 1800.0 <= mark <= leave
            if near or round(c0 + mark) % 600 == 0:
                candidates.append(mark)
            mark += 300.0
        for t in candidates:
            label = hhmm(c0 + t)
            if label not in taken:
                taken.add(label)
                times[label] = t
                options.append(option({"action": "leave_at", "time": label}, t))
        skippable = [k for k in p.ahead()[:-1] if acts[k].optional]
        for k in skippable:
            options.append(option({"action": "skip_activity", "activity": k}, leave))
        context = {"reason": reason, "now_at": {**self._plan_row(here, p.at), "since": hhmm(c0 + here.arrived_s),
                                                "planned_end": hhmm(c0 + leave)},
                   "plan": [self._plan_row(acts[k], k) for k in p.ahead()], "options": options}
        actions = [self.spec("keep_plan")]
        if times:
            actions.append(self.spec("leave_at", Param("time", "string", tuple(times))))
        if skippable:
            actions.append(self.spec("skip_activity", Param("activity", "integer", tuple(skippable))))
        return {"context": context, "actions": actions, "times": times}

    def _apply_review(self, tr, rec) -> str:
        sim, w, p = self.sim, self.world, tr.plan
        action, args = rec.decision.action, rec.decision.args
        p.pending, p.reviews = None, p.reviews + 1
        # How well does the choice serve the person? Worth of each listed option in their own terms.
        options = rec.obs.context["options"]
        values = [option_value(rec.obs.state["preferences"], rec.obs.context, o) for o in options]
        chosen = next(i for i, o in enumerate(options)
                      if o["choice"] == {"action": action, **args})
        regret = max(values) - values[chosen]
        self.review_regret, self.review_best = self.review_regret + regret, self.review_best + int(regret <= 0.01)
        tally = self.review_edit[max(values) - values[0] >= REVIEW_GAIN_USD]
        tally[0], tally[1] = tally[0] + 1, tally[1] + int(action != "keep_plan")
        result = "keeps the plan"
        if action == "leave_at":
            p.leave_s, p.edits = max(w.time_s, rec.obs.aux["times"][args["time"]]), p.edits + 1
            sim.emit("plan_changed", agent=tr.id, change="leave_at", time=args["time"])
            result = f"will leave at {args['time']}"
        elif action == "skip_activity":
            a = p.activities[args["activity"]]
            a.skipped, p.edits, p.follow_up = True, p.edits + 1, True
            sim.emit("plan_changed", agent=tr.id, change="skip_activity", activity=a.type)
            result = f"drops '{a.type}'"
        p.expected = self._expected(tr)
        return result

    # ------------------------------------------------------------------ options

    def options(self, tr):
        """Travel options open to this traveller: display rows plus engine-side details."""
        sim = self.sim
        rows, detail = [], {}
        p = tr.persona
        car = sim.kinds["vehicle"].car_option(tr) if "vehicle" in sim.kinds else None
        if car is not None:
            rows.append({"option": "car", "wait_min": 0.0, "travel_min": minutes(car["time_s"]),
                         "price_usd": car["price"]})
            detail["car"] = car
        if "bus" in sim.kinds:
            plan = sim.kinds["bus"].plan(tr.origin, tr.destination, p)
            if plan is not None:
                rows.append({"option": "bus", "wait_min": minutes(plan["access_s"] + plan["wait_s"]),
                             "travel_min": minutes(plan["ride_s"] + plan["egress_s"]), "price_usd": plan["fare"],
                             "walk_min": minutes(plan["access_s"] + plan["egress_s"])})
                detail["bus"] = {"time_s": plan["access_s"] + plan["wait_s"] + plan["ride_s"] + plan["egress_s"],
                                 "plan": plan}
        if "ondemand" in sim.kinds:
            quote = sim.kinds["ondemand"].quote(tr)
            if quote is not None:
                rows.append({"option": "ride", "wait_min": minutes(quote["wait_s"]), "travel_min": minutes(quote["ride_s"]),
                             "price_usd": quote["fare"]})
                detail["ride"] = {"time_s": quote["wait_s"] + quote["ride_s"], "quote": quote}
        if self.cfg.walk and not p.mobility_limited and tr.trip_m <= p.max_walk_m:
            walk_s = tr.trip_m / (p.walk_speed * self.world.walk_factor)
            rows.append({"option": "walk", "wait_min": 0.0, "travel_min": minutes(walk_s), "price_usd": 0.0})
            detail["walk"] = {"time_s": walk_s}
        return rows, detail

    def _state(self, tr, trigger: str = "") -> dict:
        w = self.world
        state = {"status": tr.state.value, "persona": tr.persona.to_dict(), "owns_vehicle": tr.car is not None}
        told = self.goal(tr.id)             # a traveller is told who they are; a scenario may add what to pursue
        if told:
            state["objective"] = told
        if tr.plan is not None:
            state["preferences"] = preferences(tr.persona.value_of_time, self.sim.cfg.scoring)
        if trigger == "plan_review":
            return state
        state["trip"] = {"origin_node": tr.origin, "destination_node": tr.destination, "distance_km": km(tr.trip_m)}
        at = tr.poi
        if tr.plan is not None and tr.plan.leg is not None:
            goal = tr.plan.activities[tr.plan.leg["to"]]
            at = goal.poi
            state["trip"]["to_activity"] = goal.type
            if goal.latest_start_s is not None:
                state["trip"]["must_start_by"] = hhmm(w.clock0 + goal.latest_start_s)
        if at is not None:                  # where they are going is a place with a name and, perhaps, hours
            poi = w.pois[at]
            state["trip"]["destination_place"] = {"name": poi.name, "type": poi.type}
            if poi.close_s is not None:
                state["trip"]["destination_place"]["closes"] = hhmm(w.clock0 + poi.close_s)
        state["minutes_since_need"] = minutes((w.tick - tr.t_need) * w.dt)
        return state

    # ------------------------------------------------------------------ decision points

    def due(self):
        w, tick = self.world, self.world.tick
        points = []
        for tr in w.travelers.values():
            if tr.state == TState.DECIDING:
                points.append((tr.id, "replan" if tr.replan_pending else "need"))
            elif tr.state == TState.AT_ACTIVITY:
                if tr.plan.pending is not None:
                    points.append((tr.id, "plan_review"))
            elif tr.state == TState.IN_CAR and tr.suggestion is not None:
                points.append((tr.id, "suggestion"))
            elif (tr.state in (TState.AT_STOP, TState.WAITING_RIDE) and tr.next_check is not None and tick >= tr.next_check
                  and self.allows(tr.id, "cancel")):
                points.append((tr.id, "wait_check"))
        return points

    def authority(self, agent_id, trigger):
        need = {"need": "choose_service", "suggestion": "respond", "wait_check": "cancel", "replan": "replan",
                "plan_review": "adapt_plan"}
        return self.allows(agent_id, need[trigger])

    def observe(self, agent_id, trigger):
        sim, w = self.sim, self.world
        tr = w.travelers[agent_id]
        state = self._state(tr, trigger)
        if trigger == "plan_review":
            review = tr.plan.pending
            return self.obs(tr.id, trigger, state, review["context"], review["actions"], Decision("keep_plan"),
                            {"times": review["times"]})
        if trigger in ("need", "replan"):
            rows, detail = self.options(tr)
            if trigger == "replan" and tr.mode == "ride":      # they gave up on a car: calling again would only queue anew
                rows = [r for r in rows if r["option"] != "ride"]
            if not rows:
                return None
            names = [r["option"] for r in rows]
            if trigger == "need" and self.cfg.default_mode in names:
                default = self.cfg.default_mode
            else:
                default = min(names, key=lambda n: (detail[n]["time_s"], n))
            context = {"options": rows}
            if trigger == "replan":
                context["reason"] = "your previous attempt ended without transport"
            return self.obs(tr.id, trigger, state, context,
                            [self.spec("choose_service", Param("option", "string", tuple(names)))],
                            Decision("choose_service", {"option": default}), {"options": detail})

        if trigger == "suggestion":
            s = tr.suggestion
            proposal = {"type": "park and continue by bus", **s["option"]}
            if s["message"]:
                proposal["message_from_vehicle"] = s["message"]
            context = {"proposal": proposal, "if_you_keep_driving_min": s["keep_driving_min"]}
            return self.obs(tr.id, trigger, state, context,
                            [self.spec("accept_suggestion"), self.spec("decline_suggestion")],
                            Decision("accept_suggestion"))

        if trigger == "wait_check" and tr.state == TState.WAITING_RIDE:      # waiting for the car they called
            rows, _ = self.options(tr)
            status, eta = sim.kinds["ondemand"].pickup_eta(tr)
            context = {"waited_min": minutes((w.tick - tr.wait_start) * w.dt), "your_request": status,
                       "pickup_eta_min": minutes(eta) if eta < float("inf") else None,
                       "alternatives": [r for r in rows if r["option"] != "ride"]}
            return self.obs(tr.id, trigger, state, context,
                            [self.spec("keep_waiting"), self.spec("give_up_waiting")], Decision("keep_waiting"))
        if trigger == "wait_check":
            rows, _ = self.options(tr)
            waited = (w.tick - tr.wait_start) * w.dt
            context = {"waited_min": minutes(waited),
                       "next_bus_eta_min": minutes(sim.kinds["bus"].next_bus_eta(tr.bus["line"], tr.bus["board"])),
                       "alternatives": [r for r in rows if r["option"] != "bus"]}
            return self.obs(tr.id, trigger, state, context,
                            [self.spec("keep_waiting"), self.spec("give_up_waiting")], Decision("keep_waiting"))
        raise ValueError(f"unknown traveller trigger '{trigger}'")

    def nothing_to_decide(self, agent_id, trigger):
        tr = self.world.travelers[agent_id]
        tr.state, tr.replan_pending = TState.ABANDONED, False
        self.sim.emit("abandoned", agent=tr.id, reason="no travel option available")

    # ------------------------------------------------------------------ applying decisions

    def apply(self, rec):
        sim, w = self.sim, self.world
        tr = w.travelers[rec.agent]
        action, args = rec.decision.action, rec.decision.args
        if rec.trigger == "plan_review":
            rec.result = self._apply_review(tr, rec)
        elif action == "choose_service":
            self.start(tr, args["option"], rec.obs.aux["options"])
            rec.result = f"travels by {args['option']}"
        elif action in ("accept_suggestion", "decline_suggestion"):
            rec.result = sim.kinds["vehicle"].suggestion_answered(tr, action == "accept_suggestion")
        elif action == "keep_waiting":
            tr.next_check = w.tick + w.ticks(self.cfg.wait_check_seconds)
        elif action == "give_up_waiting" and tr.state == TState.WAITING_RIDE:
            tr.cancels += 1
            sim.kinds["ondemand"].cancel(tr)
            self.failed(tr, "gave up waiting for a car", "ride")
            rec.result = "cancelled the ride request"
        elif action == "give_up_waiting":
            tr.cancels += 1
            sim.kinds["bus"].leave_stop(tr)
            sim.emit("left_stop", agent=tr.id)
            self.failed(tr, "gave up waiting for the bus")
            rec.result = "left the bus stop"

    def start(self, tr, option: str, detail: dict) -> None:
        """Begin travelling by ``option``."""
        sim, w = self.sim, self.world
        tr.mode, tr.replan_pending = option, False
        if tr.t_depart is None:
            tr.t_depart = w.tick
        if option == "car":
            tr.state = TState.IN_CAR
            sim.kinds["vehicle"].depart(tr)
        elif option == "bus":
            plan = detail["bus"]["plan"]
            tr.bus, tr.paid = plan, tr.paid + plan["fare"]
            tr.state, tr.phase_start = TState.TO_STOP, w.tick
            tr.phase_end = w.tick + (w.ticks(plan["access_s"]) if plan["access_s"] > 0 else 0)
        elif option == "ride":                     # call an on-demand car and wait where they are
            sim.kinds["ondemand"].request_ride(tr, detail["ride"]["quote"])
            tr.state, tr.wait_start = TState.WAITING_RIDE, w.tick
            tr.next_check = w.tick + w.ticks(self.cfg.wait_check_seconds)
        else:
            tr.state, tr.phase_start = TState.WALKING, w.tick
            tr.phase_end = w.tick + w.ticks(detail["walk"]["time_s"])
        sim.emit("mode_chosen", agent=tr.id, mode=option)

    def failed(self, tr, reason: str, tried: str = "bus") -> None:
        """An attempt by bus or on-demand car ended without transport: re-plan, or fall back, or abandon."""
        sim, w = self.sim, self.world
        if tr.wait_start is not None:
            tr.waited_ticks += w.tick - tr.wait_start
            tr.wait_start = None
        tr.next_check = None
        if self.allows(tr.id, "replan"):
            tr.state, tr.replan_pending = TState.DECIDING, True
            return
        rows, detail = self.options(tr)
        names = [r["option"] for r in rows if r["option"] != tried]
        if not names:
            tr.state = TState.ABANDONED
            sim.emit("abandoned", agent=tr.id, reason=reason)
            return
        best = min(names, key=lambda n: (detail[n]["time_s"], n))
        sim.forced(tr.id, self.name, "replan", Decision("choose_service", {"option": best}),
                   "no authority to re-plan: fastest remaining option")
        self.start(tr, best, detail)

    # ------------------------------------------------------------------ physics

    def physics(self):
        sim, w = self.sim, self.world
        now = w.tick + 1
        for tr in w.travelers.values():
            if tr.state in (TState.WALKING, TState.EGRESS) and now >= tr.phase_end:
                tr.state, tr.t_arrival = TState.ARRIVED, now
                sim.emit("arrived", agent=tr.id, mode=tr.mode, tick=now)
                sim.feedback(tr.id, {"event": "trip_completed", "mode": tr.mode,
                                     "travel_min": minutes((now - tr.t_need) * w.dt), "paid_usd": round(tr.paid, 2)})
                self.trip_done(tr, now)
            elif tr.state == TState.TO_STOP and now >= tr.phase_end:
                tr.state, tr.wait_start = TState.AT_STOP, now
                tr.next_check = now + w.ticks(self.cfg.wait_check_seconds)
                sim.kinds["bus"].join_stop(tr)

    def xy(self, tr):
        net = self.world.net

        def lerp(a, b):
            span = max(1, (tr.phase_end or 0) - (tr.phase_start or 0))
            f = min(1.0, max(0.0, (self.world.tick - (tr.phase_start or 0)) / span))
            (ax, ay), (bx, by) = net.xy(a), net.xy(b)
            return ax + (bx - ax) * f, ay + (by - ay) * f

        if tr.state == TState.WALKING:
            return lerp(tr.origin, tr.destination)
        if tr.bus:
            stops = self.world.lines[tr.bus["line"]]["stops"]
            if tr.state == TState.TO_STOP:
                return lerp(tr.origin, stops[tr.bus["board"]]) if tr.mode == "bus" else net.xy(stops[tr.bus["board"]])
            if tr.state == TState.AT_STOP:
                return net.xy(stops[tr.bus["board"]])
            if tr.state == TState.EGRESS:
                return lerp(stops[tr.bus["alight"]], tr.destination)
        return net.xy(tr.origin)

    def frame(self):
        shown = (TState.DECIDING, TState.TO_STOP, TState.AT_STOP, TState.EGRESS, TState.WALKING, TState.AT_ACTIVITY,
                 TState.WAITING_RIDE)
        out = []
        for tr in self.world.travelers.values():
            if tr.state in shown:
                x, y = self.xy(tr)
                out.append([tr.id, round(x, 1), round(y, 1), tr.state.value])
        return out
