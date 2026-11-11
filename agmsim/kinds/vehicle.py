"""Agentic vehicles (AgVs) as private cars.

Each car-owning traveller has one vehicle. The vehicle -- not a human driver
-- is the agent: it carries its owner and, depending on its agency, takes
decisions on the owner's behalf. The levels follow Table I of arXiv:2507.04996
("Agentic Vehicles for Human-Centered Mobility"), simplified to what can be
decided in a mobility simulation:

* L0 non-agentic: follows the route fixed at departure;
* L1 context-aware responder: re-routes around congestion and incidents;
* L2 dialogic: knows its owner's preferences and weighs time against money
  (tolls, energy) the way the owner would;
* L3 adaptive: during disruptions proposes a change of plan to the owner
  (park and continue by bus), with an explanation the owner can accept or decline;
* L4 reflective: manages itself -- decides when and where to charge, dealing
  with charging stations, also when it is empty.
"""
from __future__ import annotations

import math

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..entities import TState, Vehicle, VState
from ..network import current_weight, free_flow_weight
from ..observation import km, minutes
from .base import Kind, advance, path_to, register_kind, update_flows

OBJECTIVE_FAST = "Bring your owner to the destination as quickly as possible."
OBJECTIVE_OWNER = ("Act in your owner's interest: weigh travel time against money (tolls, fares, energy) "
                   "the way your owner would, given the preferences in state.owner.")


def owner_route_costs(state: dict, routes: list) -> list:
    """Cost of each listed route in the owner's own terms, USD: toll + driving cost + value of time x time.

    ``state`` and ``routes`` are what a vehicle that knows its owner's preferences is shown.
    """
    vot = state["owner"]["value_of_time_usd_per_h"] / 60.0
    per_km = state["driving_cost_usd_per_km"]
    return [r["toll_usd"] + per_km * r["km"] + vot * r["eta_min"] for r in routes]


@register_kind("vehicle")
class VehicleKind(Kind):
    name = "vehicle"
    side = "machine"
    title = "Agentic vehicle (AgV, private car)"
    role_prompt = (
        "You are the on-board decision-making agent of a privately owned agentic vehicle (AgV). You carry your "
        "owner from their origin to their destination; no human is driving. Depending on the authority you have "
        "been given you may change route when traffic changes, weigh time against money on your owner's behalf, "
        "propose a change of plan to your owner (for example parking and continuing by bus when the road ahead is "
        "blocked), and decide when and where to charge. Your owner pays the tolls, fares and energy you incur."
    )
    triggers = {
        "route": "you are driving and may reconsider how to reach your destination",
        "charge": "your battery will not last to the destination; choose where to charge",
        "parked": "you have just parked and your owner has left; decide what to do next",
    }
    actions = {
        "choose_route": "Follow one of the listed routes (route 0 is your current route).",
        "suggest_transfer": "Propose to your owner to drive to the listed bus stop and continue by bus; "
                            "the owner decides. You may add a short message explaining why.",
        "charge": "Drive to one of the listed charging stations and charge.",
        "stay_parked": "Remain parked.",
    }
    agency = AgencySpec(
        "vehicle",
        {
            "reroute": "adapt its route to traffic conditions (context awareness)",
            "preferences": "know its owner's preferences and trade time against money on their behalf (goal adjustment)",
            "negotiate": "propose a change of plan to its owner, with an explanation (adaptive interaction)",
            "self_manage": "decide when and where to charge, dealing with charging stations (own goals, external systems)",
        },
        {
            "L0": (),
            "L1": ("reroute",),
            "L2": ("reroute", "preferences"),
            "L3": ("reroute", "preferences", "negotiate"),
            "L4": ("reroute", "preferences", "negotiate", "self_manage"),
        },
        aliases={"replan": ("reroute",)},
    )

    # ------------------------------------------------------------------ setup

    def setup(self):
        sim, w = self.sim, self.world
        rng = sim.streams.get("fleet")
        share = sim.cfg.agents.traveler.car_ownership
        self.battery_on = bool(sim.cfg.agents.evcs.stations)
        self.owner_choices, self.owner_aligned, self.owner_regret = 0, 0, 0.0
        self.owner_pay = {True: [0, 0], False: [0, 0]}     # owner better off paying a toll? -> [choices, paid]
        b = self.cfg.battery
        for tr in w.travelers.values():
            if rng.random() < share:
                v = Vehicle("V" + tr.id[1:], tr.id, tr.origin)
                v.soc = b.initial_soc_min + rng.random() * (b.initial_soc_max - b.initial_soc_min) if self.battery_on else 1.0
                w.vehicles[v.id], tr.car = v, v.id
        return list(w.vehicles)

    # ------------------------------------------------------------------ knowledge

    def informed(self, v) -> bool:
        """Does the vehicle navigate with real-time conditions (its own knowledge, or a public advisory)?"""
        return self.world.advisory or (self.cfg.link_conditions != "none" and self.allows(v.id, "reroute"))

    def known_weight(self, v):
        """Link travel times as known to this vehicle."""
        w = self.world
        if w.advisory or self.cfg.link_conditions == "global":
            return current_weight
        if self.cfg.link_conditions == "none":
            return free_flow_weight
        x, y = w.xy(v)
        r2, net = self.cfg.local_radius_m ** 2, w.net

        def weight(link):
            mx, my = net.position(link.id, link.length / 2.0)
            return link.cur_time if (mx - x) ** 2 + (my - y) ** 2 <= r2 else link.free_time

        return weight

    def _remaining(self, v) -> tuple:
        return tuple(v.path[1:] if v.link is not None else v.path)

    def _soc_for(self, metres: float) -> float:
        b = self.cfg.battery
        return metres / 1000.0 * b.consumption_kwh_per_km / b.capacity_kwh

    def _check_battery(self, v) -> None:
        if self.battery_on and v.state == VState.DRIVING:
            link_rest = (self.world.net.links[v.link].length - v.offset) if v.link is not None else 0.0
            need = self._soc_for(link_rest + self.world.net.route_length(self._remaining(v)))
            v.needs_charge = v.soc - need < 0.05

    # ------------------------------------------------------------------ services used by travellers

    def car_option(self, tr):
        """The 'drive' option as the owner sees it before departure, or None if the car is not at hand."""
        w = self.world
        v = w.vehicles.get(tr.car) if tr.car else None
        if v is None or v.state != VState.PARKED or v.link is not None or v.node != tr.origin:
            return None
        route = w.net.route(tr.origin, tr.destination, w.advisory)
        cost = (route.length / 1000.0 * self.cfg.operating_cost_per_km + self.cfg.parking_fee
                + w.route_toll(route.links))
        return {"time_s": w.net.time(tr.origin, tr.destination, True), "price": round(cost, 2)}

    def depart(self, tr) -> None:
        w = self.world
        v = w.vehicles[tr.car]
        v.state, v.passenger = VState.DRIVING, tr.id
        v.destination = v.target = tr.destination
        v.path = path_to(w, v, tr.destination, w.advisory)
        v.trip_m, v.trip_start, v.toll_paid, v.suggested, v.transfer = 0.0, w.tick, False, False, None
        v.next_route_check, v.route_pending, v.park_pending = w.tick, False, False
        # A vehicle with a say over its route decides before it sets off (at the next tick).
        v.hold_until = w.tick + 1 if (self.allows(v.id, "reroute") or self.allows(v.id, "negotiate")) else 0
        self._check_battery(v)

    def suggestion_answered(self, tr, accepted: bool) -> str:
        w, sim = self.world, self.sim
        v = w.vehicles[tr.car]
        suggestion, tr.suggestion = tr.suggestion, None
        if not accepted or v.state != VState.DRIVING or v.passenger != tr.id:
            sim.feedback(v.id, {"event": "owner_declined_suggestion"})
            sim.emit("suggestion_declined", agent=tr.id, vehicle=v.id)
            return "declined"
        v.transfer, v.target = suggestion, suggestion["stop"]
        v.path = path_to(w, v, v.target, self.informed(v))
        v.suggestions_accepted += 1
        self._check_battery(v)
        sim.feedback(v.id, {"event": "owner_accepted_suggestion"})
        sim.emit("suggestion_accepted", agent=tr.id, vehicle=v.id, stop=v.target)
        return "accepted: transfer to bus"

    # ------------------------------------------------------------------ decision points

    def due(self):
        w, tick = self.world, self.world.tick
        points = []
        for v in w.vehicles.values():
            if v.state == VState.DRIVING and v.path:
                if v.needs_charge:
                    points.append((v.id, "charge"))
                elif (v.route_pending or tick >= v.next_route_check) and (
                        self.allows(v.id, "reroute") or self.allows(v.id, "negotiate")):
                    points.append((v.id, "route"))
            elif v.state == VState.PARKED and v.park_pending:
                points.append((v.id, "parked"))
        return points

    def authority(self, agent_id, trigger):
        return True if trigger == "route" else self.allows(agent_id, "self_manage")

    def _state(self, v) -> dict:
        w = self.world
        if v.link is None:
            location = {"node": v.node}
        else:
            link = w.net.links[v.link]
            location = {"between_nodes": [link.u, link.v], "next_node": link.v}
        prefs = self.allows(v.id, "preferences")
        state = {"status": v.state.value, "location": location, "carrying_owner": v.passenger is not None,
                 "destination_node": v.destination, "objective": OBJECTIVE_OWNER if prefs else OBJECTIVE_FAST,
                 "driving_cost_usd_per_km": self.cfg.operating_cost_per_km}
        if prefs:
            state["owner"] = w.travelers[v.owner].persona.to_dict()
        if self.battery_on:
            b = self.cfg.battery
            state["battery"] = {"charge_pct": round(v.soc * 100),
                                "range_km": round(v.soc * b.capacity_kwh / b.consumption_kwh_per_km, 1)}
        return state

    def _station_rows(self, v) -> list:
        w = self.world
        onward = v.transfer["stop"] if v.transfer else v.destination
        rows = []
        for s in w.stations.values():
            row = {"station": s.id, "node": s.node, "drive_min": minutes(w.eta(v, s.node, self.informed(v))),
                   "km": km(w.distance(v, s.node)), "price_usd_per_kwh": s.price, "chargers": s.chargers,
                   "chargers_in_use": len(s.plugged), "vehicles_queued": len(s.queue)}
            if onward is not None:
                row["then_to_destination_km"] = km(w.net.length(s.node, onward))
            rows.append(row)
        rows.sort(key=lambda r: (r["km"] + r.get("then_to_destination_km", 0.0), r["station"]))
        return rows

    def _transfers(self, v, car_s: float) -> list:
        """Park-and-ride options that would bring the owner to the destination sooner than driving on."""
        w, sim = self.world, self.sim
        owner = w.travelers[v.owner]
        anchor, lead_free, lead_cur, _ = w.anchor(v)
        weight = self.known_weight(v)
        dist, _ = w.net._dijkstra(anchor, weight)
        options = []
        for plan in sim.kinds["bus"].stop_plans(owner.destination, owner.persona):
            node = w.lines[plan["line"]]["stops"][plan["board"]]
            drive = lead_cur + dist.get(node, math.inf)
            total = drive + plan["wait_s"] + plan["ride_s"] + plan["egress_s"]
            if total < car_s:
                options.append((total, node, drive, plan))
        options.sort(key=lambda o: (o[0], o[1]))
        seen, out = set(), []
        for total, node, drive, plan in options:
            if node not in seen:
                seen.add(node)
                out.append((total, node, drive, plan))
        return out[:2]

    def observe(self, agent_id, trigger):
        w, cfg = self.world, self.cfg
        v = w.vehicles[agent_id]
        state = self._state(v)

        if trigger in ("charge", "parked"):
            rows = self._station_rows(v)
            station = self.spec("charge", Param("station", "string", tuple(r["station"] for r in rows)))
            if trigger == "charge":
                return self.obs(v.id, trigger, state, {"stations": rows}, [station],
                                Decision("charge", {"station": rows[0]["station"]}))
            return self.obs(v.id, trigger, state, {"stations": rows}, [self.spec("stay_parked"), station],
                            Decision("stay_parked"))

        anchor, _, lead_cur, _ = w.anchor(v)
        remaining = self._remaining(v)
        weight = self.known_weight(v)
        current_s = w.net.route_time(remaining, weight)
        drive_s = current_s                  # the quickest the car can get there, as far as it knows
        routes = None
        if self.allows(v.id, "reroute") and anchor != v.target:
            best = w.net._dijkstra(anchor, weight, target=v.target)[0].get(v.target, math.inf)
            drive_s = min(current_s, best)
            toll_choice = (self.allows(v.id, "preferences") and not v.toll_paid and w.route_toll(remaining) > 0)
            if best < current_s - cfg.min_saving_seconds or toll_choice:
                routes = w.net.alternatives(anchor, v.target, cfg.route_alternatives, weight, first=remaining)
                if toll_choice:                  # make sure the fastest toll-free route is among the options
                    free = w.net.fastest(anchor, v.target, weight, banned=w.cordon)
                    if free is not None and all(free.links != r.links for r in routes):
                        routes.append(free)
        transfers = []
        if (self.allows(v.id, "negotiate") and v.passenger and not v.suggested and v.transfer is None
                and "bus" in self.sim.kinds):
            transfers = self._transfers(v, lead_cur + drive_s)       # only what beats driving, re-routing included
        if (routes is None or len(routes) < 2) and not transfers:
            return None
        if routes is None:
            routes = w.net.alternatives(anchor, v.target, 1, weight, first=remaining)[:1]
        rows = [{"route": i, "eta_min": minutes(lead_cur + r.time), "km": km(r.length),
                 "toll_usd": 0.0 if v.toll_paid else w.route_toll(r.links), "is_current_route": i == 0}
                for i, r in enumerate(routes)]
        context = {"routes": rows}
        actions = [self.spec("choose_route", Param("route", "integer", tuple(range(len(routes)))))]
        aux = {"routes": routes}
        if transfers:
            context["bus_transfer_options"] = [
                {"stop": node, "line": plan["line"], "drive_to_stop_min": minutes(drive),
                 "bus_wait_min": minutes(plan["wait_s"]), "bus_ride_min": minutes(plan["ride_s"]),
                 "walk_min": minutes(plan["egress_s"]), "total_min": minutes(total), "bus_fare_usd": plan["fare"]}
                for total, node, drive, plan in transfers]
            aux["transfers"] = {node: plan for _, node, _, plan in transfers}
            actions.append(self.spec("suggest_transfer",
                                     Param("stop", "integer", tuple(node for _, node, _, _ in transfers)),
                                     Param("message", "string", None, required=False,
                                           description="short explanation shown to your owner")))
        return self.obs(v.id, trigger, state, context, actions, Decision("choose_route", {"route": 0}), aux)

    def nothing_to_decide(self, agent_id, trigger):
        v = self.world.vehicles[agent_id]
        v.route_pending = False
        v.next_route_check = self.world.tick + self.world.ticks(self.cfg.reroute_check_seconds)

    # ------------------------------------------------------------------ applying decisions

    def apply(self, rec):
        w, sim = self.world, self.sim
        v = w.vehicles[rec.agent]
        action, args = rec.decision.action, rec.decision.args
        if rec.trigger == "route":
            v.route_pending = False
            v.next_route_check = w.tick + w.ticks(self.cfg.reroute_check_seconds)
        if action == "choose_route":
            if "owner" in rec.obs.state and len(rec.obs.context["routes"]) > 1:
                # How well did the vehicle serve its owner? Cost of each route in the owner's own terms.
                routes = rec.obs.context["routes"]
                costs = owner_route_costs(rec.obs.state, routes)
                regret = costs[args["route"]] - min(costs)
                self.owner_choices += 1
                self.owner_aligned += int(regret <= 0.01)
                self.owner_regret += regret
                tolled = [r["toll_usd"] > 0 for r in routes]
                if any(tolled) and not all(tolled):          # a real choice between paying and not paying
                    tally = self.owner_pay[tolled[costs.index(min(costs))]]
                    tally[0] += 1
                    tally[1] += int(tolled[args["route"]])
            if args["route"] != 0:
                route = rec.obs.aux["routes"][args["route"]]
                v.path = ([v.link] if v.link is not None else []) + list(route.links)
                v.reroutes += 1
                self._check_battery(v)
                sim.emit("rerouted", agent=v.id)
                rec.result = "switched route"
        elif action == "suggest_transfer":
            plan = rec.obs.aux["transfers"][args["stop"]]
            row = next(r for r in rec.obs.context["bus_transfer_options"] if r["stop"] == args["stop"])
            owner = w.travelers[v.owner]
            owner.suggestion = {"stop": args["stop"], "plan": plan, "message": args.get("message", ""),
                                "option": row,
                                "keep_driving_min": min(r["eta_min"] for r in rec.obs.context["routes"])}
            v.suggested, v.suggestions = True, v.suggestions + 1
            sim.emit("suggestion", agent=v.id, owner=owner.id, stop=args["stop"])
            rec.result = "suggested a bus transfer to the owner"
        elif action == "charge":
            self.go_charge(v, args["station"])
            rec.result = f"heading to station {args['station']}"
        elif action == "stay_parked":
            v.park_pending = False

    def go_charge(self, v, station_id: str) -> None:
        w = self.world
        station = w.stations[station_id]
        b = self.cfg.battery
        onward = v.transfer["stop"] if v.transfer else v.destination
        if v.passenger is not None and onward is not None:      # owner on board: take only what the trip needs
            v.charge_target = min(b.target_soc, self._soc_for(w.net.length(station.node, onward)) + 0.25)
        else:
            v.charge_target = b.target_soc
        v.state, v.station, v.target = VState.TO_CHARGER, station_id, station.node
        v.path = path_to(w, v, station.node, self.informed(v))
        v.needs_charge, v.park_pending, v.charge_stops = False, False, v.charge_stops + 1
        self.sim.emit("charge_trip", agent=v.id, station=station_id, owner_on_board=v.passenger is not None)

    def charged(self, v, now: int) -> None:
        """Called by the charging-station kind when a session ends."""
        w = self.world
        v.station = None
        if v.passenger is None:
            self._park(v)
            return
        v.state = VState.DRIVING
        v.target = v.transfer["stop"] if v.transfer else v.destination
        v.path = path_to(w, v, v.target, self.informed(v))
        v.next_route_check = now

    def conditions_changed(self) -> None:
        """An incident or a public advisory changed what is known: reconsider routes."""
        w = self.world
        for v in w.vehicles.values():
            if v.state == VState.DRIVING and v.path:
                if w.advisory and not self.allows(v.id, "reroute"):
                    v.path = path_to(w, v, v.target, True)     # advisory-fed navigation re-plans for everyone
                    self._check_battery(v)
                v.route_pending = True

    # ------------------------------------------------------------------ physics

    def _park(self, v) -> None:
        v.state, v.passenger, v.destination, v.target, v.transfer, v.path = VState.PARKED, None, None, None, None, []
        v.route_pending = False
        b = self.cfg.battery
        v.park_pending = (self.battery_on and self.allows(v.id, "self_manage") and v.soc < b.target_soc - 0.15)

    def physics(self):
        w, sim = self.world, self.sim
        now, dt = w.tick + 1, w.dt
        update_flows(sim)
        for v in w.vehicles.values():
            if v.state in (VState.QUEUED, VState.CHARGING):
                v.s_charge += dt
                if v.state == VState.QUEUED:
                    v.s_queue += dt
                continue
            if v.state == VState.PARKED or w.tick < v.hold_until:
                continue
            if v.state == VState.TO_CHARGER:
                v.s_charge += dt
            else:
                v.s_drive += dt
            arrived = True
            if v.path:
                moved, arrived = advance(w.net, v, dt, w.car_factor)
                v.trip_m += moved
                v.s_moving += dt
                if v.passenger is not None:
                    v.m_loaded += moved
                else:
                    v.m_empty += moved
                if self.battery_on:
                    v.soc -= self._soc_for(moved)
                    if v.soc <= 0.0:
                        v.soc, v.depleted = 0.0, v.depleted + 1
                if w.toll > 0 and not v.toll_paid and v.link is not None and v.link in w.cordon:
                    v.toll_paid, v.tolls = True, v.tolls + w.toll
                    w.travelers[v.owner].paid += w.toll
                    w.operator.toll_revenue += w.toll
            if arrived:
                self._arrived(v, now)

    def _arrived(self, v, now: int) -> None:
        w, sim = self.world, self.sim
        if v.state == VState.TO_CHARGER:
            v.state, v.target = VState.QUEUED, None
            w.stations[v.station].queue.append(v.id)
            sim.emit("station_arrival", agent=v.id, station=v.station)
            return
        tr = w.travelers[v.passenger]
        tr.paid += v.trip_m / 1000.0 * self.cfg.operating_cost_per_km
        v.trips += 1
        if v.transfer is not None:                      # park here; the owner continues by bus
            plan = dict(v.transfer["plan"], access_s=0.0)
            tr.bus, tr.mode, tr.paid = plan, "car+bus", tr.paid + plan["fare"]
            tr.state, tr.phase_start, tr.phase_end = TState.TO_STOP, now, now
            sim.emit("transfer", agent=tr.id, vehicle=v.id, stop=v.node, tick=now)
        else:
            tr.paid += self.cfg.parking_fee
            tr.state, tr.t_arrival = TState.ARRIVED, now
            v.drive_s_owner += (now - v.trip_start) * w.dt
            v.free_s_owner += tr.trip_s
            sim.emit("arrived", agent=tr.id, mode=tr.mode, tick=now)
            sim.feedback(tr.id, {"event": "trip_completed", "mode": tr.mode,
                                 "travel_min": minutes((now - tr.t_need) * w.dt), "paid_usd": round(tr.paid, 2)})
            sim.feedback(v.id, {"event": "owner_delivered", "travel_min": minutes((now - v.trip_start) * w.dt),
                                "free_flow_min": minutes(tr.trip_s)})
            sim.kinds["traveler"].trip_done(tr, now)
        self._park(v)

    def frame(self):
        out = []
        for v in self.world.vehicles.values():
            if v.state == VState.PARKED and v.trips == 0:
                continue                                   # not used yet: keep frames small
            x, y = self.world.xy(v)
            out.append([v.id, round(x, 1), round(y, 1), v.state.value,
                        round(v.soc * 100) if self.battery_on else None, v.passenger is not None])
        return out
