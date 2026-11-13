"""Electric-vehicle charging stations (EVCS) as agents.

A non-agentic station (L0) sells energy at a fixed tariff, first come first
served. With agency the station itself decides: it may adapt its price to the
situation (context awareness, L1) and choose which queued vehicle to serve
next (goal adjustment, L2). Vehicles that manage their own charging see the
prices and queues, so station and vehicle decisions interact.
"""
from __future__ import annotations

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..entities import Station, VState
from ..observation import minutes
from .base import Kind, register_kind

OBJECTIVE = "Maximise your revenue from energy sales while keeping the queue at your station acceptable."


@register_kind("evcs")
class EvcsKind(Kind):
    name = "evcs"
    side = "machine"
    title = "Charging station (EVCS)"
    objective = OBJECTIVE
    role_prompt = (
        "You are the decision-making agent of an electric-vehicle charging station used by private agentic "
        "vehicles. Vehicles that manage their own charging choose where to charge by looking at distance, price and queue, so a high price "
        "earns more per kWh but may send vehicles to another station, and a long queue keeps vehicles away."
    )
    triggers = {
        "review": "periodic review of your energy price",
        "queue": "a charger is free and several vehicles are waiting; choose who charges next",
    }
    actions = {
        "keep_price": "Leave the price unchanged.",
        "set_price": "Set the price to the base tariff times the chosen multiplier.",
        "serve": "Let the chosen waiting vehicle use the free charger.",
    }
    agency = AgencySpec(
        "evcs",
        {
            "price": "adapt its energy price to the situation (context awareness)",
            "prioritize": "choose which waiting vehicle is served next (goal adjustment)",
        },
        {"L0": (), "L1": ("price",), "L2": ("price", "prioritize"),
         "L3": ("price", "prioritize"), "L4": ("price", "prioritize")},
    )

    def setup(self):
        w, cfg = self.world, self.cfg
        for spec in cfg.stations:
            node = int(spec["node"])
            if node not in w.net.nodes:
                raise ValueError(f"charging station at unknown node {node}")
            sid = str(spec.get("id", f"S{len(w.stations) + 1}"))
            w.stations[sid] = Station(sid, node, int(spec.get("chargers", 2)), float(spec.get("power_kw", 50.0)),
                                      float(cfg.price_per_kwh), next_review=w.ticks(cfg.review_seconds))
        self._decided: set = set()
        self._arrival: dict = {}
        self._period_busy: dict = {sid: 0 for sid in w.stations}
        return list(w.stations)

    # ------------------------------------------------------------------ decision points

    def due(self):
        w, points = self.world, []
        for s in w.stations.values():
            if s.choose_pending and len(s.queue) >= 2:
                points.append((s.id, "queue"))
            elif w.tick >= s.next_review and self.allows(s.id, "price"):
                points.append((s.id, "review"))
        return points

    def authority(self, agent_id, trigger):
        return self.allows(agent_id, "prioritize" if trigger == "queue" else "price")

    def observe(self, agent_id, trigger):
        w, cfg = self.world, self.cfg
        s = w.stations[agent_id]
        state = {"station": s.id, "node": s.node, "chargers": s.chargers, "power_kw": s.power_kw,
                 "base_price_usd_per_kwh": s.base_price, "current_price_usd_per_kwh": s.price,
                 "objective": self.goal(agent_id)}
        if trigger == "queue":
            rows = [{"vehicle_id": vid, "battery_pct": round(w.vehicles[vid].soc * 100),
                     "waited_min": minutes((w.tick - self._arrival.get(vid, w.tick)) * w.dt)} for vid in s.queue]
            return self.obs(s.id, trigger, state, {"queue": rows},
                            [self.spec("serve", Param("vehicle_id", "string", tuple(s.queue)))],
                            Decision("serve", {"vehicle_id": s.queue[0]}))
        period = max(1, w.ticks(cfg.review_seconds))
        context = {
            "chargers_in_use": len(s.plugged),
            "vehicles_queued": len(s.queue),
            "utilisation_since_last_review_pct": round(100.0 * self._period_busy[s.id] / (period * s.chargers)),
            "energy_sold_kwh": round(s.kwh, 1),
            "revenue_usd": round(s.revenue, 2),
            "vehicles_low_on_battery": sum(1 for v in w.vehicles.values() if v.soc < 0.35),
            "vehicles_heading_here": sum(1 for v in w.vehicles.values() if v.station == s.id and v.id not in s.queue
                                         and v.id not in s.plugged),
            "other_stations": [{"station": o.id, "price_usd_per_kwh": o.price, "chargers_in_use": len(o.plugged),
                                "vehicles_queued": len(o.queue)} for o in w.stations.values() if o.id != s.id],
        }
        multipliers = tuple(float(m) for m in cfg.price_multipliers)
        return self.obs(s.id, trigger, state, context,
                        [self.spec("keep_price"), self.spec("set_price", Param("price_multiplier", "number", multipliers))],
                        Decision("keep_price"))

    def apply(self, rec):
        w = self.world
        s = w.stations[rec.agent]
        action, args = rec.decision.action, rec.decision.args
        if rec.trigger == "review":
            s.next_review = w.tick + w.ticks(self.cfg.review_seconds)
            self._period_busy[s.id] = 0
            if action == "set_price" and abs(args["price_multiplier"] - s.multiplier) > 1e-9:
                s.multiplier = args["price_multiplier"]
                self.sim.emit("price_set", agent=s.id, price=s.price)
                rec.result = f"price {s.price} USD/kWh"
        elif action == "serve":
            vid = args["vehicle_id"]
            if vid in s.queue:
                s.queue.remove(vid)
                s.queue.insert(0, vid)
            s.choose_pending = False
            self._decided.add(s.id)

    # ------------------------------------------------------------------ physics

    def physics(self):
        w, sim = self.world, self.sim
        now = w.tick + 1
        b = sim.cfg.agents.vehicle.battery
        vehicles = sim.kinds.get("vehicle")
        for s in w.stations.values():
            for vid in s.queue:
                self._arrival.setdefault(vid, w.tick)
            for vid in list(s.plugged):
                v = w.vehicles[vid]
                energy = min(s.power_kw * w.dt / 3600.0, max(0.0, v.charge_target - v.soc) * b.capacity_kwh)
                cost = energy * s.price
                v.soc += energy / b.capacity_kwh
                v.kwh_charged, v.energy_cost = v.kwh_charged + energy, v.energy_cost + cost
                s.kwh, s.revenue = s.kwh + energy, s.revenue + cost
                if v.soc >= v.charge_target - 1e-9:
                    s.plugged.remove(vid)
                    self._arrival.pop(vid, None)
                    sim.emit("charged", agent=vid, station=s.id)
                    sim.feedback(vid, {"event": "charging_finished", "station": s.id,
                                       "energy_cost_usd_so_far": round(v.energy_cost, 2)})
                    vehicles.charged(v, now)
            while s.queue and len(s.plugged) < s.chargers:
                if len(s.queue) >= 2 and self.allows(s.id, "prioritize") and s.id not in self._decided:
                    s.choose_pending = True      # the station chooses at the start of the next tick
                    break
                self._decided.discard(s.id)
                vid = s.queue.pop(0)
                s.plugged.append(vid)
                s.sessions += 1
                w.vehicles[vid].state = VState.CHARGING
                sim.emit("plugged", agent=vid, station=s.id)
            s.busy_charger_ticks += len(s.plugged)
            self._period_busy[s.id] += len(s.plugged)

    def frame(self):
        return [[s.id, s.node, s.price, len(s.plugged), len(s.queue), s.chargers] for s in self.world.stations.values()]
