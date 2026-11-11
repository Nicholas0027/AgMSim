"""Agentic vehicles (AgVs) providing on-demand rides.

The vehicle is the agent -- not a human driver. Its agency follows the level
structure of Table I in arXiv:2507.04996, simplified to decision authority in
a ride service:

* L0 non-agentic: executes the platform's assignments and routes;
* L1 context-aware: adapts its route to traffic conditions;
* L2 dialogic: weighs a proposed task against its objective (may decline) and
  talks to the rider in natural language;
* L3 adaptive: negotiates (adjusts the price of its offers);
* L4 reflective: forms its own goals -- selects its own requests, repositions,
  manages its own charging by dealing with charging stations.
"""
from __future__ import annotations

import math

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..demand import place_fleet
from ..entities import AVAILABLE_STATES, Request, RState, TState, VState
from ..mechanism import Claim
from ..network import current_weight, free_flow_weight
from ..observation import OBJECTIVES, km, minutes
from .base import Kind, advance, path_to, register_kind

MOVING = (VState.TO_PICKUP, VState.OCCUPIED, VState.REPOSITIONING, VState.TO_CHARGER)


@register_kind("vehicle")
class VehicleKind(Kind):
    name = "vehicle"
    side = "machine"
    title = "Agentic vehicle (AgV)"
    role_prompt = (
        "You are the on-board decision-making agent of an agentic vehicle (AgV) that provides on-demand rides. "
        "No human driver is involved: you are the vehicle. A platform may propose ride requests to you; depending "
        "on the authority you have been given you may also decline proposals, claim open requests yourself, "
        "reposition while unassigned, choose your route, adjust your price, attach a short message for the rider, "
        "or decide when and where to charge. Driving costs money per km whether or not a rider is on board. "
        "Riders may decline an offer or cancel if the wait is long or the price is high."
    )
    triggers = {
        "proposal": "the platform proposes a ride request to you",
        "idle": "you are unassigned (parked or repositioning) and may reconsider what to do",
        "route": "you are driving and alternative routes to your destination are known",
        "charge": "your battery is low and you must choose a charging station",
    }
    actions = {
        "accept_request": "Accept the proposed request and commit to serving it.",
        "decline_request": "Decline the proposed request; the platform will look for another vehicle.",
        "claim_request": "Offer to serve one of the listed open requests.",
        "reposition": "Drive empty to one of the listed nodes to be better placed for future requests.",
        "stay": "Keep doing what you are doing (remain parked, or continue the current repositioning).",
        "choose_route": "Follow one of the listed routes to your current destination.",
        "charge": "Drive to one of the listed charging stations and charge.",
    }
    agency = AgencySpec(
        "vehicle",
        {
            "reroute": "choose its own route among alternatives (context awareness)",
            "decline": "decline a request proposed by the platform (goal adjustment)",
            "communicate": "attach a natural-language message to its offers (natural language use)",
            "price": "adjust the fare of its offers within the allowed multipliers (negotiation)",
            "reposition": "relocate on its own initiative while unassigned (goal formation)",
            "self_select": "claim open requests on its own initiative (goal formation)",
            "charge": "decide when and where to charge (self-management with external systems)",
        },
        {
            "L0": (),
            "L1": ("reroute",),
            "L2": ("reroute", "decline", "communicate"),
            "L3": ("reroute", "decline", "communicate", "price"),
            "L4": ("reroute", "decline", "communicate", "price", "reposition", "self_select", "charge"),
        },
        aliases={"replan": ("reroute", "reposition")},
    )

    # ------------------------------------------------------------------ setup

    def setup(self):
        sim, w = self.sim, self.world
        rng = sim.streams.get("fleet")
        w.vehicles = place_fleet(sim.cfg, w.net, rng)
        self.battery_on = bool(sim.cfg.agents.evcs.stations)
        b = self.cfg.battery
        for v in w.vehicles.values():
            v.soc = b.initial_soc_min + rng.random() * (b.initial_soc_max - b.initial_soc_min) if self.battery_on else 1.0
        self._accepts: list = []
        self._claims: list = []
        self._station_near: dict = {}
        return list(w.vehicles)

    # ------------------------------------------------------------------ knowledge and feasibility

    def known_weight(self, v):
        """Link travel times as known to this vehicle, given its information scope."""
        scope, w = self.cfg.observation, self.world
        if scope.link_conditions == "global" or w.advisory:
            return current_weight
        if scope.link_conditions == "none":
            return free_flow_weight
        x, y = w.xy(v)
        radius = scope.radius_m if scope.radius_m is not None else 800.0
        net = w.net

        def weight(link):
            mx, my = net.position(link.id, link.length / 2.0)
            return link.cur_time if (mx - x) ** 2 + (my - y) ** 2 <= radius * radius else link.free_time

        return weight

    def known_eta(self, v, node: int) -> float:
        dynamic = self.cfg.observation.link_conditions != "none" or self.world.advisory
        return self.world.eta(v, node, dynamic)

    def _visible(self, v, node: int) -> bool:
        radius = self.cfg.observation.radius_m
        if radius is None:
            return True
        x, y = self.world.xy(v)
        nx, ny = self.world.net.xy(node)
        return (nx - x) ** 2 + (ny - y) ** 2 <= radius * radius

    def _nearest_station_m(self, node: int) -> float:
        if node not in self._station_near:
            net = self.world.net
            self._station_near[node] = min(net.length(node, s.node) for s in self.world.stations.values())
        return self._station_near[node]

    def energy_ok(self, v, request) -> bool:
        """Can the vehicle serve the request and still reach a charging station?"""
        if not self.battery_on:
            return True
        b = self.cfg.battery
        metres = self.world.distance(v, request.origin) + request.trip_m + self._nearest_station_m(request.destination)
        return v.soc - metres / 1000.0 * b.consumption_kwh_per_km / b.capacity_kwh >= 0.05

    def visible_requests(self, v) -> list:
        return [r for r in self.world.open_requests()
                if v.id not in r.excluded and self._visible(v, r.origin) and self.energy_ok(v, r)]

    # ------------------------------------------------------------------ decision points

    def due(self):
        w, tick = self.world, self.world.tick
        low = self.cfg.battery.reserve_soc
        points = []
        for v in w.vehicles.values():
            if v.proposal is not None:
                points.append((v.id, "proposal"))
            elif v.route_pending and v.state in MOVING and v.path:
                points.append((v.id, "route"))
            elif v.state in AVAILABLE_STATES and self.battery_on and v.soc < low:
                points.append((v.id, "charge"))
            elif v.state in AVAILABLE_STATES and tick >= v.next_check:
                a = self.sim.agency[v.id]
                idle = v.state == VState.IDLE
                if a.allows("self_select") or (idle and (a.allows("reposition") or (self.battery_on and a.allows("charge")))):
                    points.append((v.id, "idle"))
        return points

    def authority(self, agent_id, trigger):
        need = {"proposal": "decline", "route": "reroute", "charge": "charge"}
        return True if trigger == "idle" else self.allows(agent_id, need[trigger])

    # ------------------------------------------------------------------ observations

    def _state(self, v) -> dict:
        w = self.world
        if v.link is None:
            location = {"node": v.node}
        else:
            link = w.net.links[v.link]
            location = {"between_nodes": [link.u, link.v], "next_node": link.v}
        objective = self.sim.specs[v.id].objective
        state = {
            "status": v.state.value,
            "location": location,
            "objective": OBJECTIVES.get(objective, objective),
            "earnings_usd": round(v.earnings, 2),
            "trips_completed": v.trips,
            "operating_cost_usd_per_km": self.cfg.operating_cost_per_km,
        }
        if self.battery_on:
            b = self.cfg.battery
            state["battery"] = {"charge_pct": round(v.soc * 100),
                                "range_km": round(v.soc * b.capacity_kwh / b.consumption_kwh_per_km, 1)}
        return state

    def _market(self, v) -> dict:
        w = self.world
        market = {"open_requests_visible": len(self.visible_requests(v))}
        if self.cfg.observation.vehicles:
            market["other_available_vehicles_visible"] = sum(
                1 for o in w.available_vehicles() if o.id != v.id and self._visible(v, w.anchor(o)[0]))
        return market

    def _nearby_available(self, node: int, exclude: str) -> int:
        w, radius = self.world, self.cfg.observation.demand_radius_m
        return sum(1 for o in w.available_vehicles()
                   if o.id != exclude and w.net.euclid(w.anchor(o)[0], node) <= radius)

    def _offer_params(self, v) -> tuple:
        params = []
        if self.allows(v.id, "price"):
            params.append(Param("price_multiplier", "number", tuple(float(m) for m in self.cfg.price_multipliers),
                                required=False, description="multiplier applied to the listed fare (default 1.0)"))
        if self.allows(v.id, "communicate"):
            params.append(Param("message", "string", None, required=False,
                                description="short note shown to the rider together with your offer"))
        return tuple(params)

    def _request_row(self, v, request) -> dict:
        w = self.world
        return {
            "request_id": request.id,
            "pickup_node": request.origin,
            "dropoff_node": request.destination,
            "pickup_eta_min": minutes(self.known_eta(v, request.origin)),
            "pickup_km": km(w.distance(v, request.origin)),
            "trip_km": km(request.trip_m),
            "trip_min": minutes(request.trip_s),
            "fare_usd": self.sim.mechanism.fare(request.trip_m, request.trip_s),
            "rider_waited_min": minutes((w.tick - request.t_created) * w.dt),
        }

    def _station_rows(self, v) -> list:
        w = self.world
        rows = [{
            "station": s.id, "node": s.node,
            "eta_min": minutes(self.known_eta(v, s.node)), "km": km(w.distance(v, s.node)),
            "price_usd_per_kwh": s.price, "chargers": s.chargers,
            "chargers_in_use": len(s.plugged), "vehicles_queued": len(s.queue),
        } for s in w.stations.values()]
        rows.sort(key=lambda r: (r["eta_min"], r["station"]))
        return rows

    def observe(self, agent_id, trigger):
        w, scope = self.world, self.cfg.observation
        v = w.vehicles[agent_id]
        state = self._state(v)

        if trigger == "proposal":
            request = w.requests[v.proposal]
            context = {"proposal": self._request_row(v, request), "market": self._market(v)}
            return self.obs(v.id, trigger, state, context,
                            [self.spec("accept_request", *self._offer_params(v)), self.spec("decline_request")],
                            Decision("accept_request"))

        if trigger == "charge":
            rows = self._station_rows(v)
            return self.obs(v.id, trigger, state, {"stations": rows},
                            [self.spec("charge", Param("station", "string", tuple(r["station"] for r in rows)))],
                            Decision("charge", {"station": rows[0]["station"]}))

        if trigger == "idle":
            actions = [self.spec("stay")]
            context = {"minutes_unassigned": minutes((w.tick - v.idle_since) * w.dt)}
            idle = v.state == VState.IDLE
            if self.allows(v.id, "self_select"):
                rows = sorted((self._request_row(v, r) for r in self.visible_requests(v)),
                              key=lambda row: (row["pickup_eta_min"], row["request_id"]))[: scope.max_requests]
                context["open_requests"] = rows
                if rows:
                    ids = tuple(row["request_id"] for row in rows)
                    actions.append(self.spec("claim_request", Param("request_id", "string", ids), *self._offer_params(v)))
            if idle and self.allows(v.id, "reposition"):
                here = w.anchor(v)[0]
                heat = w.demand_heat(w.ticks(scope.demand_window_seconds), scope.demand_radius_m)
                nodes = [n for n in w.net.nodes if n != here and self._visible(v, n)]
                nodes.sort(key=lambda n: (-heat[n], self.known_eta(v, n), n))
                rows = []
                for node in nodes[: scope.reposition_candidates]:
                    row = {"node": node, "eta_min": minutes(self.known_eta(v, node)),
                           "km": km(w.distance(v, node)), "recent_requests_nearby": heat[node]}
                    if scope.vehicles:
                        row["other_available_vehicles_nearby"] = self._nearby_available(node, v.id)
                    rows.append(row)
                here_row = {"node": here, "recent_requests_nearby": heat[here]}
                if scope.vehicles:
                    here_row["other_available_vehicles_nearby"] = self._nearby_available(here, v.id)
                context["here"] = here_row
                context["reposition_candidates"] = rows
                if rows:
                    actions.append(self.spec("reposition", Param("node", "integer", tuple(r["node"] for r in rows))))
            if (idle and self.battery_on and self.allows(v.id, "charge")
                    and v.soc < self.cfg.battery.target_soc - 0.1):
                rows = self._station_rows(v)
                context["stations"] = rows
                actions.append(self.spec("charge", Param("station", "string", tuple(r["station"] for r in rows))))
            if len(actions) == 1:
                return None
            context["market"] = self._market(v)
            return self.obs(v.id, trigger, state, context, actions, Decision("stay"))

        if trigger == "route":
            if v.target is None:
                return None
            anchor = w.anchor(v)[0]
            remaining = tuple(v.path[1:] if v.link is not None else v.path)
            routes = w.net.alternatives(anchor, v.target, scope.route_alternatives, self.known_weight(v), first=remaining)
            if len(routes) < 2:
                return None
            rows = [{"route": i, "eta_min": minutes(r.time), "km": km(r.length),
                     "via_nodes": [w.net.links[lid].v for lid in r.links][:12], "is_current_route": i == 0}
                    for i, r in enumerate(routes)]
            purpose = {VState.TO_PICKUP: "pickup", VState.OCCUPIED: "dropoff",
                       VState.TO_CHARGER: "charging station"}.get(v.state, "reposition")
            context = {"destination_node": v.target, "purpose": purpose, "routes": rows}
            return self.obs(v.id, trigger, state, context,
                            [self.spec("choose_route", Param("route", "integer", tuple(range(len(routes)))))],
                            Decision("choose_route", {"route": 0}), {"routes": routes})
        raise ValueError(f"unknown vehicle trigger '{trigger}'")

    def nothing_to_decide(self, agent_id, trigger):
        v = self.world.vehicles[agent_id]
        if trigger == "route":
            v.route_pending = False
        else:
            v.next_check = self.world.tick + self.world.ticks(self.cfg.idle_check_seconds)

    # ------------------------------------------------------------------ applying decisions

    def apply(self, rec):
        w, sim = self.world, self.sim
        v = w.vehicles[rec.agent]
        action, args = rec.decision.action, rec.decision.args
        if rec.trigger == "idle":
            v.next_check = w.tick + w.ticks(self.cfg.idle_check_seconds)
        if action == "accept_request":
            self._accepts.append((v, v.proposal, args.get("price_multiplier", 1.0), args.get("message", "")))
        elif action == "decline_request":
            request = w.requests[v.proposal]
            v.proposal, v.declined = None, v.declined + 1
            if request.state == RState.PROPOSED:
                request.state, request.vehicle = RState.OPEN, None
                request.excluded.append(v.id)
            sim.emit("proposal_declined", agent=v.id, request=request.id)
            rec.result = "declined"
        elif action == "claim_request":
            request = w.requests[args["request_id"]]
            v.claims += 1
            multiplier = args.get("price_multiplier", 1.0) if self.allows(v.id, "price") else 1.0
            self._claims.append(Claim(v.id, request.id, multiplier, args.get("message", ""),
                                      sim.mechanism.eta(v, request.origin),
                                      round(sim.mechanism.fare(request.trip_m, request.trip_s) * multiplier, 2)))
        elif action == "reposition":
            node = args["node"]
            if not (v.link is None and v.node == node):
                v.state, v.target = VState.REPOSITIONING, node
                v.path = path_to(w, v, node, sim.mechanism.dynamic)
                self._flag_route(v)
                sim.emit("reposition", agent=v.id, node=node)
                rec.result = f"repositioning to node {node}"
        elif action == "choose_route":
            route = rec.obs.aux["routes"][args["route"]]
            v.route_pending = False
            if args["route"] != 0:
                v.path = ([v.link] if v.link is not None else []) + list(route.links)
                sim.emit("route_changed", agent=v.id)
                rec.result = "switched route"
        elif action == "charge":
            self.go_charge(v, args["station"])
            rec.result = f"heading to station {args['station']}"

    def go_charge(self, v, station_id: str) -> None:
        w = self.world
        station = w.stations[station_id]
        v.state, v.station, v.target = VState.TO_CHARGER, station_id, station.node
        v.path = path_to(w, v, station.node, self.sim.mechanism.dynamic)
        self._flag_route(v)
        self.sim.emit("charge_trip", agent=v.id, station=station_id)

    # ------------------------------------------------------------------ ride-service protocol

    def create_request(self, tr, quote: dict) -> Request:
        w = self.world
        request = Request(w.new_request_id(), tr.id, tr.origin, tr.destination, w.tick, tr.trip_m, tr.trip_s, quote=quote)
        w.requests[request.id] = request
        w.request_log.append((w.tick, tr.origin))
        tr.request_id, tr.attempts = request.id, tr.attempts + 1
        self.sim.emit("request", agent=tr.id, request=request.id)
        for v in w.vehicles.values():                 # self-selecting vehicles notice new work promptly
            if v.state in AVAILABLE_STATES and self.allows(v.id, "self_select"):
                v.next_check = min(v.next_check, w.tick + 1)
        return request

    def _free(self, v, when: int) -> None:
        v.state, v.request_id, v.passenger, v.station, v.target = VState.IDLE, None, None, None, None
        v.path = v.path[:1] if v.link is not None else []
        v.route_pending = False
        v.idle_since, v.next_check = when, when

    def _commit(self, v, request, multiplier: float, message: str, via: str) -> None:
        """The vehicle is committed to the request; the traveller confirms if they have that authority."""
        w, sim = self.world, self.sim
        if not self.allows(v.id, "price"):
            multiplier = 1.0
        if not self.allows(v.id, "communicate"):
            message = ""
        request.vehicle, request.multiplier, request.message, request.via = v.id, multiplier, message, via
        request.price = round(sim.mechanism.fare(request.trip_m, request.trip_s) * multiplier, 2)
        request.eta_s = sim.mechanism.eta(v, request.origin)
        traveler = w.travelers[request.traveler]
        v.request_id = request.id
        if sim.agency[traveler.id].allows("confirm_offer"):
            request.state, traveler.state, v.state = RState.OFFERED, TState.OFFERED, VState.RESERVED
            v.path = v.path[:1] if v.link is not None else []
            sim.emit("offer", agent=v.id, request=request.id, price=request.price, eta_s=round(request.eta_s, 1))
        else:
            sim.forced(traveler.id, "traveler", "offer", Decision("accept_offer"),
                       "no authority to decline offers")
            self._assign(v, request)

    def _assign(self, v, request) -> None:
        w, sim = self.world, self.sim
        traveler = w.travelers[request.traveler]
        request.state, request.t_assigned = RState.ASSIGNED, w.tick
        traveler.state = TState.ASSIGNED
        v.state, v.target, v.request_id = VState.TO_PICKUP, request.origin, request.id
        v.path = path_to(w, v, request.origin, sim.mechanism.dynamic)
        self._flag_route(v)
        sim.emit("assigned", agent=v.id, request=request.id, traveler=traveler.id, via=request.via)

    def offer_accepted(self, request) -> str:
        if request.state != RState.OFFERED:
            return "offer no longer valid"
        self._assign(self.world.vehicles[request.vehicle], request)
        return "offer accepted"

    def offer_declined(self, request) -> str:
        w, sim = self.world, self.sim
        if request.state != RState.OFFERED:
            return "offer no longer valid"
        v = w.vehicles[request.vehicle]
        request.state, request.vehicle = RState.OPEN, None
        request.excluded.append(v.id)
        w.travelers[request.traveler].state = TState.SEARCHING
        self._free(v, w.tick + 1)
        sim.emit("offer_declined", agent=request.traveler, request=request.id, vehicle=v.id)
        sim.feedback(v.id, {"event": "offer_declined_by_rider", "request_id": request.id,
                            "price_usd": request.price, "pickup_eta_min": minutes(request.eta_s)})
        return "offer declined"

    def cancel(self, request) -> None:
        """The traveller withdraws the request."""
        w, sim = self.world, self.sim
        if request.vehicle is not None:
            v = w.vehicles[request.vehicle]
            if request.state == RState.PROPOSED:
                v.proposal = None
            elif request.state in (RState.OFFERED, RState.ASSIGNED):
                if request.state == RState.ASSIGNED:
                    fee = sim.cfg.mechanism.pricing.cancel_fee
                    w.travelers[request.traveler].paid += fee
                    v.earnings += fee * (1.0 - sim.cfg.mechanism.pricing.commission)
                self._free(v, w.tick + 1)
                sim.feedback(v.id, {"event": "rider_cancelled", "request_id": request.id})
        request.state, request.closed_by = RState.CANCELLED, "traveler"
        sim.emit("cancelled", agent=request.traveler, request=request.id)

    def resolve(self):
        w, sim = self.world, self.sim
        mech = sim.mechanism
        for v, rid, multiplier, message in self._accepts:
            request = w.requests[rid]
            v.proposal = None
            if request.state == RState.PROPOSED and request.vehicle == v.id:
                self._commit(v, request, multiplier, message, "dispatch")
        self._accepts = []

        live = [c for c in self._claims if w.requests[c.request].state == RState.OPEN
                and w.vehicles[c.vehicle].state in AVAILABLE_STATES]
        self._claims = []
        winners, losers = mech.resolve_claims(live, sim.streams.get("mechanism"))
        for claim in winners:
            v = w.vehicles[claim.vehicle]
            v.claims_won += 1
            self._commit(v, w.requests[claim.request], claim.multiplier, claim.message, "claim")
        for claim in losers:
            w.vehicles[claim.vehicle].next_check = w.tick + 1
            sim.emit("claim_lost", agent=claim.vehicle, request=claim.request)
            sim.feedback(claim.vehicle, {"event": "claim_lost", "request_id": claim.request})

        m = sim.cfg.mechanism
        if not m.central_matching or w.tick % w.ticks(m.match_interval_seconds) != 0:
            return
        low = self.cfg.battery.reserve_soc
        vehicles = [v for v in w.available_vehicles()
                    if (m.dispatch_self_selecting or not self.allows(v.id, "self_select"))
                    and not (self.battery_on and v.soc < low)]
        requests = w.open_requests()
        if not vehicles or not requests:
            return

        def eta(v, request):
            if v.id in request.excluded or not self.energy_ok(v, request):
                return math.inf
            return mech.eta(v, request.origin)

        for request, v in mech.matcher(requests, vehicles, eta, m.max_pickup_seconds):
            if self.allows(v.id, "decline"):
                request.state, request.vehicle, v.proposal = RState.PROPOSED, v.id, request.id
                sim.emit("proposal", agent=v.id, request=request.id)
            else:
                sim.forced(v.id, self.name, "proposal", Decision("accept_request"),
                           "no authority to decline proposals")
                self._commit(v, request, 1.0, "", "dispatch")

    # ------------------------------------------------------------------ routing

    def _flag_route(self, v) -> None:
        """Mark that the vehicle should be asked about its route, if it has that authority and a real choice."""
        v.route_pending = False
        trigger = self.cfg.route_trigger
        if trigger == "never" or not v.path or not self.allows(v.id, "reroute"):
            return
        if trigger == "always":
            v.route_pending = True
            return
        w = self.world
        anchor = w.anchor(v)[0]
        if anchor == v.target:
            return
        weight = self.known_weight(v)
        remaining = v.path[1:] if v.link is not None else v.path
        best = w.net._dijkstra(anchor, weight, target=v.target)[0].get(v.target, math.inf)
        v.route_pending = best < w.net.route_time(remaining, weight) - 1.0

    def conditions_changed(self) -> None:
        """Link conditions (or what is known about them) changed: re-plan platform routes, flag route choices."""
        w, sim = self.world, self.sim
        for v in w.vehicles.values():
            if v.state in MOVING and v.path and v.target is not None:
                if sim.mechanism.dynamic:
                    v.path = path_to(w, v, v.target, True)
                self._flag_route(v)

    # ------------------------------------------------------------------ physics

    def physics(self):
        w, sim = self.world, self.sim
        now, dt = w.tick + 1, w.dt
        b = self.cfg.battery
        for v in w.vehicles.values():
            s = v.state
            if s == VState.IDLE or s == VState.RESERVED:
                v.s_idle += dt
            elif s == VState.TO_PICKUP:
                v.s_pickup += dt
            elif s == VState.OCCUPIED:
                v.s_loaded += dt
            elif s == VState.REPOSITIONING:
                v.s_reposition += dt
            elif s in (VState.BOARDING, VState.ALIGHTING):
                v.s_dwell += dt
            else:
                v.s_charge += dt

            if s in (VState.BOARDING, VState.ALIGHTING):
                if now >= v.dwell_until:
                    self._dwell_done(v, now)
            elif s in (VState.QUEUED, VState.CHARGING):
                continue
            elif v.path:
                moved, arrived = advance(w.net, v, dt, w.car_factor)
                if s == VState.TO_PICKUP:
                    v.m_pickup += moved
                elif s == VState.OCCUPIED:
                    v.m_loaded += moved
                elif s == VState.TO_CHARGER:
                    v.m_charge += moved
                else:
                    v.m_reposition += moved
                if self.battery_on:
                    v.soc -= moved / 1000.0 * b.consumption_kwh_per_km / b.capacity_kwh
                    if v.soc <= 0.0:
                        v.soc, v.depleted = 0.0, v.depleted + 1
                if arrived:
                    self._arrived(v, now)
            elif s in (VState.TO_PICKUP, VState.TO_CHARGER, VState.REPOSITIONING):
                self._arrived(v, now)

        timeout = sim.cfg.mechanism.request_timeout_seconds
        for request in list(w.requests.values()):
            if request.state == RState.OPEN and (now - request.t_created) * dt >= timeout:
                request.state, request.closed_by = RState.EXPIRED, "timeout"
                sim.emit("expired", agent=request.traveler, request=request.id)
                sim.kinds["traveler"].failed(w.travelers[request.traveler], "request expired")

    def _arrived(self, v, now: int) -> None:
        w, sim = self.world, self.sim
        if v.state == VState.TO_PICKUP:
            request = w.requests[v.request_id]
            traveler = w.travelers[request.traveler]
            v.state, v.passenger = VState.BOARDING, traveler.id
            v.dwell_until = now + w.ticks(self.cfg.boarding_seconds)
            request.state, request.t_pickup = RState.ONBOARD, now
            traveler.state, traveler.t_board = TState.RIDING, now
            if traveler.wait_start is not None:
                traveler.waited_ticks += now - traveler.wait_start
                traveler.wait_start = None
            traveler.next_check = None
            sim.emit("pickup", agent=v.id, request=request.id, traveler=traveler.id, tick=now)
        elif v.state == VState.OCCUPIED:
            v.state = VState.ALIGHTING
            v.dwell_until = now + w.ticks(self.cfg.alighting_seconds)
        elif v.state == VState.REPOSITIONING:
            self._free(v, now)
        elif v.state == VState.TO_CHARGER:
            v.state, v.target = VState.QUEUED, None
            w.stations[v.station].queue.append(v.id)
            sim.emit("station_arrival", agent=v.id, station=v.station)

    def _dwell_done(self, v, now: int) -> None:
        w, sim = self.world, self.sim
        request = w.requests[v.request_id]
        if v.state == VState.BOARDING:
            v.state, v.target = VState.OCCUPIED, request.destination
            v.path = path_to(w, v, request.destination, sim.mechanism.dynamic)
            self._flag_route(v)
            if not v.path:
                self._arrived(v, now)
            return
        traveler = w.travelers[request.traveler]
        income = request.price * (1.0 - sim.cfg.mechanism.pricing.commission)
        request.state, request.t_dropoff = RState.COMPLETED, now
        traveler.state, traveler.t_arrival = TState.ARRIVED, now
        traveler.paid += request.price
        v.earnings, v.trips = v.earnings + income, v.trips + 1
        sim.emit("dropoff", agent=v.id, request=request.id, traveler=traveler.id, tick=now, fare=request.price)
        sim.feedback(v.id, {"event": "trip_completed", "request_id": request.id, "income_usd": round(income, 2)})
        sim.feedback(traveler.id, {"event": "trip_completed", "mode": "ride",
                                   "waited_min": minutes(traveler.waited_ticks * w.dt),
                                   "paid_usd": round(traveler.paid, 2)})
        self._free(v, now)

    def charged(self, v, now: int) -> None:
        """Called by the charging-station kind when a session ends."""
        self._free(v, now)

    def frame(self):
        out = []
        for v in self.world.vehicles.values():
            x, y = self.world.xy(v)
            out.append([v.id, round(x, 1), round(y, 1), v.state.value, round(v.soc * 100) if self.battery_on else None])
        return out
