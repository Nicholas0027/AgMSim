"""On-demand cars: a fleet that carries travellers on request.

A traveller calls a car; the service proposes the request to the nearest free
car; the car drives to the traveller, carries them to their destination and
is paid the fare. The car -- not a human driver -- is the agent. Its levels
follow the same reading of Table I in arXiv:2507.04996 as the private car's:

* L0 non-agentic: takes every request the service proposes, follows the
  route fixed when it sets off, and waits where it dropped its passenger;
* L1 context-aware: re-routes around congestion and incidents;
* L2 dialogic: weighs a proposed request against its objective and may turn
  it down (the service then offers it to the next nearest free car);
* L4 reflective: also decides, while free, where to wait for coming requests.

Not carried over from the earlier ride-sourcing version (``archive/``): cars
that set their own price (its L3), claim requests themselves or manage charging.
"""
from __future__ import annotations

import math

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..entities import Cab, CState, RideRequest, TState
from ..network import current_weight
from ..observation import OBJECTIVES, km, minutes
from .base import Kind, advance, known_eta, known_path, path_to, register_kind, route_memory, update_flows

FREE = (CState.IDLE, CState.REPOSITIONING)


@register_kind("ondemand")
class OnDemandKind(Kind):
    name = "ondemand"
    side = "machine"
    title = "On-demand car"
    drives = True
    role_prompt = (
        "You are the on-board decision-making agent of an on-demand car: a driverless car of a ride service. "
        "Travellers call a car and the service proposes each request to the nearest free car. You carry one "
        "traveller at a time from where they are to where they want to go, and you are paid the fare. Driving "
        "costs money per km, with or without a passenger. A traveller who waits too long may give up."
    )
    triggers = {
        "proposal": "the service proposes a ride request to you",
        "route": "you are driving and may reconsider how to get there",
        "idle": "you are free and may decide where to wait for the next request",
    }
    actions = {
        "accept_request": "Take the proposed request: drive to the traveller and carry them.",
        "decline_request": "Turn the request down; the service offers it to the next nearest free car.",
        "choose_route": "Follow one of the listed routes (route 0 is your current route).",
        "stay": "Stay where you are.",
        "reposition": "Drive, empty, to one of the listed places to be nearer to coming requests.",
    }
    agency = AgencySpec(
        "ondemand",
        {
            "reroute": "adapt its route to traffic conditions (context awareness)",
            "decline": "weigh a proposed request against its objective and turn it down (goal adjustment)",
            "reposition": "decide, while free, where to wait for coming requests (own goals)",
        },
        {"L0": (), "L1": ("reroute",), "L2": ("reroute", "decline"), "L3": ("reroute", "decline"),
         "L4": ("reroute", "decline", "reposition")},
    )

    # ------------------------------------------------------------------ setup

    def setup(self):
        sim, w, cfg = self.sim, self.world, self.cfg
        rng = sim.streams.get("ondemand_fleet")
        places = [int(n) for n in cfg.depots] or sorted(w.net.nodes)
        width = max(2, len(str(cfg.fleet)))
        for i in range(int(cfg.fleet)):
            cab = Cab(f"D{i + 1:0{width}d}", places[rng.randrange(len(places))])
            w.cabs[cab.id] = cab
        self._recent: list = []                 # (tick, node) of the latest requests
        self.requests = self.cancelled = self.pickups = self.pickup_wait_ticks = 0
        return list(w.cabs)

    def fare(self, trip_m: float, trip_s: float) -> float:
        cfg = self.cfg
        return round(cfg.base_fare + cfg.fare_per_km * trip_m / 1000.0 + cfg.fare_per_minute * trip_s / 60.0, 2)

    def informed(self, cab) -> bool:
        """Does the car navigate by what is known of traffic (its own knowledge, or a public advisory)?"""
        return self.world.advisory or (self.sim.traffic.can_know(self.name) and self.allows(cab.id, "reroute"))

    def on_the_road(self):
        return list(self.world.cabs.values())

    def _available(self, cab) -> bool:
        return cab.state in FREE and cab.proposal is None

    def _remaining(self, cab) -> tuple:
        return tuple(cab.path[1:] if cab.link is not None else cab.path)

    # ------------------------------------------------------------------ services used by travellers

    def pickup_estimate(self, node: int) -> float:
        """Seconds until a car could be at ``node``: the nearest free one, or the soonest to finish its ride."""
        w = self.world
        free = [w.eta(c, node, True) for c in w.cabs.values() if self._available(c)]
        if free:
            return min(free)
        best = math.inf
        for cab in w.cabs.values():
            done = w.anchor(cab)[2] + w.net.route_time(self._remaining(cab), current_weight)
            end = cab.target if cab.target is not None else cab.node
            if cab.state == CState.TO_PICKUP:
                req = w.requests[cab.request]
                done, end = done + w.net.time(req.origin, req.destination, True), req.destination
            best = min(best, done + w.net.time(end, node, True))
        return best

    def quote(self, tr):
        """The 'call a car' option as the traveller sees it, or None if there is no service."""
        w = self.world
        if not w.cabs:
            return None
        return {"wait_s": self.pickup_estimate(tr.origin), "ride_s": w.net.time(tr.origin, tr.destination, True),
                "fare": self.fare(tr.trip_m, tr.trip_s)}

    def request_ride(self, tr, quote: dict) -> None:
        w = self.world
        self.requests += 1
        req = RideRequest(f"Q{self.requests:04d}", tr.id, tr.origin, tr.destination, w.tick, quote["fare"], tr.trip_m)
        w.requests[req.id], tr.ride = req, req.id
        self._recent.append((w.tick, tr.origin))
        self.sim.emit("ride_requested", agent=tr.id, request=req.id)

    def pickup_eta(self, tr) -> tuple[str, float]:
        """Where the traveller's request stands, and how long until a car could be there."""
        w = self.world
        req = w.requests[tr.ride]
        if req.state == "assigned":
            return "a car is on its way", w.eta(w.cabs[req.car], req.origin, True)
        return "looking for a free car", self.pickup_estimate(req.origin)

    def cancel(self, tr) -> None:
        """The traveller gives up waiting."""
        w = self.world
        req, tr.ride = w.requests.get(tr.ride), None
        if req is None or req.state not in ("open", "proposed", "assigned"):
            return
        if req.offered is not None:
            w.cabs[req.offered].proposal = None
        if req.car is not None:
            self._release(w.cabs[req.car])
        req.state, req.offered = "cancelled", None
        self.cancelled += 1
        self.sim.emit("ride_cancelled", agent=tr.id, request=req.id)

    # ------------------------------------------------------------------ the service's dispatch rule

    def begin_tick(self):
        """Propose every open request, oldest first, to the nearest free car that has not turned it down."""
        w = self.world
        waiting = sorted((r for r in w.requests.values() if r.state == "open"), key=lambda r: (r.opened, r.id))
        for req in waiting:
            if len(req.declined_by) >= len(w.cabs):
                req.declined_by.clear()             # everybody said no: it goes round again
            free = [c for c in w.cabs.values() if self._available(c) and c.id not in req.declined_by]
            if not free:
                continue
            cab = min(free, key=lambda c: (w.eta(c, req.origin, True), c.id))
            req.state, req.offered, cab.proposal = "proposed", cab.id, req.id

    # ------------------------------------------------------------------ decision points

    def due(self):
        w, tick = self.world, self.world.tick
        points = []
        for cab in w.cabs.values():
            if cab.proposal is not None:
                points.append((cab.id, "proposal"))
            elif cab.state != CState.IDLE:
                if cab.path and (cab.route_pending or tick >= cab.next_route_check) and self.allows(cab.id, "reroute"):
                    points.append((cab.id, "route"))
            elif tick >= cab.next_idle_check and self.allows(cab.id, "reposition"):
                points.append((cab.id, "idle"))
        return points

    def authority(self, agent_id, trigger):
        return self.allows(agent_id, "decline") if trigger == "proposal" else True

    def _state(self, cab) -> dict:
        w = self.world
        location = {"node": cab.node} if cab.link is None else {
            "between_nodes": [w.net.links[cab.link].u, w.net.links[cab.link].v]}
        return {"status": cab.state.value, "location": location, "objective": OBJECTIVES[self.cfg.objective],
                "driving_cost_usd_per_km": self.cfg.operating_cost_per_km,
                "rides_so_far": cab.trips, "earned_usd": round(cab.revenue, 2)}

    def _places(self, cab) -> tuple:
        """Where requests have come from lately, away from where the car is; and how it looks where the car is."""
        w, cfg = self.world, self.cfg
        since = w.tick - w.ticks(cfg.demand_window_seconds)
        self._recent = [item for item in self._recent if item[0] >= since]
        calls: dict = {}
        for _, node in self._recent:
            calls[node] = calls.get(node, 0) + 1
        free: dict = {}
        for other in w.cabs.values():
            if other is not cab and self._available(other) and other.link is None:
                free[other.node] = free.get(other.node, 0) + 1
        ranked = sorted((n for n in calls if n != cab.node), key=lambda n: (-calls[n], n))[:3]
        return [{"node": n, "recent_requests": calls[n], "free_cars_there": free.get(n, 0),
                 "drive_min": minutes(w.eta(cab, n, True))} for n in ranked], calls.get(cab.node, 0), free.get(cab.node, 0)

    def observe(self, agent_id, trigger):
        w, cfg = self.world, self.cfg
        cab = w.cabs[agent_id]
        state = self._state(cab)
        if trigger == "proposal":
            req = w.requests[cab.proposal]
            pickup_m = w.distance(cab, req.origin)
            context = {
                "request": {"pickup_min": minutes(known_eta(self.sim, self, cab, req.origin)), "pickup_km": km(pickup_m),
                            "trip_min": minutes(w.net.time(req.origin, req.destination, True)), "trip_km": km(req.trip_m),
                            "fare_usd": req.fare, "traveller_has_waited_min": minutes((w.tick - req.opened) * w.dt)},
                "your_driving_cost_usd": round(cfg.operating_cost_per_km * (pickup_m + req.trip_m) / 1000.0, 2),
                "other_free_cars": sum(1 for c in w.cabs.values() if c is not cab and self._available(c)),
                "requests_waiting_for_a_car": sum(1 for r in w.requests.values() if r.state == "open"),
            }
            actions = [self.spec("accept_request")]
            if self.allows(cab.id, "decline"):
                actions.append(self.spec("decline_request"))
            return self.obs(cab.id, trigger, state, context, actions, Decision("accept_request"))

        if trigger == "route":
            anchor, _, lead, _ = w.anchor(cab)
            remaining = self._remaining(cab)
            weight, seen, reports = self.sim.traffic.knowledge(self.name, cab)
            current_s = w.net.route_time(remaining, weight)
            best = w.net._dijkstra(anchor, weight, target=cab.target)[0].get(cab.target, math.inf)
            if anchor == cab.target or best >= current_s - cfg.min_saving_seconds:
                return None
            routes = w.net.alternatives(anchor, cab.target, cfg.route_alternatives, weight, first=remaining)
            if len(routes) < 2:
                return None
            rows = [{"route": i, "eta_min": minutes(lead + r.time), "km": km(r.length), "is_current_route": i == 0}
                    for i, r in enumerate(routes)]
            context = {"routes": rows}
            basis = self.sim.traffic.explain(self.name, cab, routes, seen, reports)
            if basis:
                context["traffic_knowledge"] = basis
            return self.obs(cab.id, trigger, state, context,
                            [self.spec("choose_route", Param("route", "integer", tuple(range(len(routes)))))],
                            Decision("choose_route", {"route": 0}), {"routes": routes})

        places, here, free_here = self._places(cab)
        if not places:
            return None
        context = {"recent_means_last_min": minutes(cfg.demand_window_seconds),
                   "here": {"recent_requests": here, "other_free_cars_here": free_here}, "places": places}
        return self.obs(cab.id, trigger, state, context,
                        [self.spec("stay"), self.spec("reposition", Param("node", "integer", tuple(p["node"] for p in places)))],
                        Decision("stay"))

    def remember(self, rec):
        return route_memory(rec) if rec.decision.action == "choose_route" else super().remember(rec)

    def nothing_to_decide(self, agent_id, trigger):
        w, cab = self.world, self.world.cabs[agent_id]
        if trigger == "route":
            cab.route_pending, cab.next_route_check = False, w.tick + w.ticks(self.cfg.reroute_check_seconds)
        else:
            cab.next_idle_check = w.tick + w.ticks(self.cfg.idle_check_seconds)

    # ------------------------------------------------------------------ applying decisions

    def apply(self, rec):
        w, sim, cfg = self.world, self.sim, self.cfg
        cab = w.cabs[rec.agent]
        action, args = rec.decision.action, rec.decision.args
        if rec.trigger == "proposal":
            req, cab.proposal = w.requests.get(cab.proposal), None
            if req is None or req.state != "proposed" or req.offered != cab.id:
                rec.result = "the traveller had already given up"
                return
            req.offered = None
            if action == "decline_request":
                req.state, cab.declined = "open", cab.declined + 1
                req.declined_by.add(cab.id)
                sim.emit("ride_declined", agent=cab.id, request=req.id)
                rec.result = "declined"
                return
            req.state, req.car = "assigned", cab.id
            cab.state, cab.request, cab.target = CState.TO_PICKUP, req.id, req.origin
            cab.path = known_path(sim, self, cab, req.origin)
            cab.next_route_check, cab.route_pending = w.tick, False
            rec.result = f"on its way to {req.traveler}"
        elif action == "choose_route":
            cab.route_pending, cab.next_route_check = False, w.tick + w.ticks(cfg.reroute_check_seconds)
            if args["route"] != 0:
                cab.path = ([cab.link] if cab.link is not None else []) + list(rec.obs.aux["routes"][args["route"]].links)
                cab.reroutes += 1
                rec.result = "switched route"
        elif action == "reposition":
            cab.state, cab.target, cab.repositions = CState.REPOSITIONING, args["node"], cab.repositions + 1
            cab.path = known_path(sim, self, cab, args["node"])
            cab.next_route_check = w.tick + w.ticks(cfg.reroute_check_seconds)
            rec.result = f"moves to node {args['node']}"
        else:
            cab.next_idle_check = w.tick + w.ticks(cfg.idle_check_seconds)

    def conditions_changed(self) -> None:
        """An incident or a public advisory changed what is known: reconsider routes."""
        w = self.world
        for cab in w.cabs.values():
            if cab.state != CState.IDLE and cab.path:
                if w.advisory and not self.allows(cab.id, "reroute"):
                    cab.path = path_to(w, cab, cab.target, True)
                cab.route_pending = True

    # ------------------------------------------------------------------ physics

    def _free(self, cab) -> None:
        cab.state, cab.request, cab.target, cab.path, cab.route_pending = CState.IDLE, None, None, [], False
        cab.next_idle_check = self.world.tick + self.world.ticks(self.cfg.idle_check_seconds)

    def _release(self, cab) -> None:
        """The car's job is gone (the traveller gave up): it finishes the road it is on and is free again."""
        if cab.link is None:
            self._free(cab)
        else:
            cab.state, cab.request = CState.REPOSITIONING, None
            cab.path, cab.target = [cab.link], self.world.net.links[cab.link].v

    def physics(self):
        w, sim = self.world, self.sim
        if "vehicle" not in sim.kinds:
            update_flows(sim)
        now = w.tick + 1
        for cab in w.cabs.values():
            if cab.state == CState.IDLE:
                continue
            arrived = True
            if cab.path:
                moved, arrived = advance(w.net, cab, w.dt, w.car_factor)
                cab.s_moving += w.dt
                if cab.state == CState.OCCUPIED:
                    cab.m_loaded += moved
                else:
                    cab.m_empty += moved
            if arrived:
                self._arrived(cab, now)

    def _arrived(self, cab, now: int) -> None:
        w, sim = self.world, self.sim
        if cab.state == CState.REPOSITIONING:
            self._free(cab)
            return
        req = w.requests[cab.request]
        tr = w.travelers[req.traveler]
        if cab.state == CState.TO_PICKUP:
            req.state, req.picked = "on_board", now
            tr.state, tr.waited_ticks, tr.wait_start, tr.next_check = TState.IN_RIDE, tr.waited_ticks + now - tr.wait_start, None, None
            self.pickups, self.pickup_wait_ticks = self.pickups + 1, self.pickup_wait_ticks + now - req.opened
            cab.state, cab.target = CState.OCCUPIED, req.destination
            cab.path = known_path(self.sim, self, cab, req.destination)
            cab.next_route_check, cab.route_pending = now, False
            sim.emit("picked_up", agent=tr.id, car=cab.id, tick=now)
            return
        req.state, tr.paid, cab.revenue, cab.trips = "done", tr.paid + req.fare, cab.revenue + req.fare, cab.trips + 1
        tr.state, tr.t_arrival, tr.ride = TState.ARRIVED, now, None
        sim.emit("arrived", agent=tr.id, mode=tr.mode, tick=now)
        sim.feedback(tr.id, {"event": "trip_completed", "mode": tr.mode,
                             "travel_min": minutes((now - tr.t_need) * w.dt), "paid_usd": round(tr.paid, 2)})
        sim.feedback(cab.id, {"event": "ride_completed", "fare_usd": req.fare})
        sim.kinds["traveler"].trip_done(tr, now)
        self._free(cab)

    def frame(self):
        out = []
        for cab in self.world.cabs.values():
            x, y = self.world.xy(cab)
            out.append([cab.id, round(x, 1), round(y, 1), cab.state.value])
        return out
