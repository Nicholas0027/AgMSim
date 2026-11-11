"""The orchestrator: the network operator as an agent (kind ``operator``).

One orchestrator oversees the road network: it sees where traffic is slow and
where vehicles queue, and deals with cars, buses and traffic lights. Non-agentic
(L0) it does nothing at run time. With agency it decides, on incidents and at
regular reviews:

* L1 (context awareness): whether to publish real-time traffic advisories,
  which make current link conditions public knowledge -- even non-agentic
  vehicles then navigate around incidents;
* L2 (goal adjustment): whether to switch on transit priority (buses bypass
  congestion, cars are slightly slower, and buses pass traffic lights on red),
  what to charge for entering the central cordon, and what to ask of the
  traffic lights: a direction to favour everywhere, or green for one phase at
  one junction. Traffic lights act on this only if they cooperate (their L2).

This is a deliberately small action set; richer controls (closures, ramp
metering) fit the same interface.
"""
from __future__ import annotations

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..entities import OperatorState, TState, VState
from .base import Kind, register_kind
from .signal import PHASES

OBJECTIVE = ("Keep people moving: limit congestion and delays caused by incidents and keep public transport "
             "reliable, without burdening travellers more than necessary.")


@register_kind("operator")
class OperatorKind(Kind):
    name = "operator"
    side = "machine"
    title = "Network operator"
    role_prompt = (
        "You are the decision-making agent of the road-network operator. You can publish real-time traffic "
        "advisories, which let every vehicle route around congestion and incidents; switch transit priority on "
        "or off, which lets buses bypass congestion and makes cars slightly slower; and set the charge for driving into the "
        "central cordon, which discourages car traffic there."
    )
    # told to a language model only when the network has traffic lights
    signals_prompt = (" You also see where vehicles queue at traffic lights and can ask the lights to favour a "
                      "direction everywhere, or ask one junction to give green to one of its phases; lights act "
                      "on this if they cooperate.")
    triggers = {"review": "an incident changed link conditions, or the periodic review is due"}
    actions = {
        "no_change": "Leave the current settings as they are.",
        "publish_advisory": "Publish real-time link conditions to all vehicles.",
        "withdraw_advisory": "Stop publishing real-time link conditions.",
        "enable_bus_priority": "Switch transit priority on.",
        "disable_bus_priority": "Switch transit priority off.",
        "set_toll": "Set the charge for entering the central cordon to the chosen amount.",
        "favour_direction": "Ask all traffic lights to give the chosen direction longer greens; 'none' ends it.",
        "guide_signal": "Ask the traffic lights at one junction to give green to the chosen phase now.",
    }
    agency = AgencySpec(
        "operator",
        {
            "advise": "publish or withdraw real-time traffic advisories (context awareness)",
            "prioritize": "switch transit priority on or off (goal adjustment)",
            "toll": "set the cordon charge (goal adjustment)",
            "signals": "ask traffic lights to favour a direction or to serve a queue (coordination)",
        },
        {"L0": (), "L1": ("advise",), "L2": ("advise", "prioritize", "toll", "signals"),
         "L3": ("advise", "prioritize", "toll", "signals"), "L4": ("advise", "prioritize", "toll", "signals")},
    )

    def setup(self):
        w, cfg = self.world, self.cfg
        w.operator = OperatorState(next_review=w.ticks(cfg.review_seconds))
        if cfg.cordon_radius_m > 0:
            net = w.net
            centre = cfg.cordon_center
            if centre is None:
                xs = [x for x, _ in net.nodes.values()]
                ys = [y for _, y in net.nodes.values()]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                centre = min(net.nodes, key=lambda n: ((net.nodes[n][0] - cx) ** 2 + (net.nodes[n][1] - cy) ** 2, n))
            inside = {n for n in net.nodes if net.euclid(centre, n) <= cfg.cordon_radius_m}
            w.cordon = frozenset(l.id for l in net.links if l.v in inside)
        op = w.operator
        op.toll, op.advisory, op.bus_priority = float(cfg.initial_toll), bool(cfg.initial_advisory), bool(cfg.initial_bus_priority)
        if op.bus_priority:
            w.bus_factor, w.car_factor = cfg.bus_priority_gain, cfg.car_priority_penalty
        if w.signals:
            self.role_prompt = type(self).role_prompt + self.signals_prompt
        self._waited = {name: 0 for name in PHASES}      # vehicle ticks held at lights, as of the last review
        return [op.id]

    def due(self):
        op, w = self.world.operator, self.world
        if (op.incident_pending or w.tick >= op.next_review) and self.authority(op.id, "review"):
            return [(op.id, "review")]
        return []

    def authority(self, agent_id, trigger):
        return any(self.allows(agent_id, p) for p in ("advise", "prioritize", "toll", "signals"))

    def observe(self, agent_id, trigger):
        w = self.world
        op, net = w.operator, w.net
        incidents = [{"link": [l.u, l.v], "speed_pct_of_normal": round(l.factor * 100)}
                     for l in net.links if abs(l.factor - 1.0) > 1e-9]
        driving = [v for v in w.vehicles.values() if v.state in (VState.DRIVING, VState.TO_CHARGER)]

        def speed_pct(links):
            links = list(links)
            return round(100.0 * sum(l.flow * l.factor * l.ambient for l in links) / len(links)) if links else 100

        state = {"objective": OBJECTIVE, "advisory_published": op.advisory, "bus_priority_on": op.bus_priority}
        context = {
            "reason": "link conditions changed" if op.incident_pending else "periodic review",
            "incidents": incidents[:12],
            "cars_driving": len(driving),
            "network_speed_pct_of_free_flow": speed_pct(net.links),
            "travellers_waiting_for_a_bus": sum(1 for t in w.travelers.values() if t.state == TState.AT_STOP),
            "travellers_on_buses": sum(len(b.onboard) for b in w.buses.values()),
            "travellers_in_cars": sum(1 for v in driving if v.passenger is not None),
            # where it is slow: the roads on which traffic moves worst
            "slowest_roads": [{"from_node": l.u, "to_node": l.v, "speed_pct_of_free_flow": round(100 * l.flow * l.factor * l.ambient)}
                              for l in sorted((l for l in net.links if l.flow * l.factor < 0.9),
                                              key=lambda l: (l.flow * l.factor, l.id))[:5]],
        }
        actions, aux = [self.spec("no_change")], {}
        if self.allows(op.id, "advise"):
            actions.append(self.spec("withdraw_advisory" if op.advisory else "publish_advisory"))
        if self.allows(op.id, "prioritize") and w.buses:
            actions.append(self.spec("disable_bus_priority" if op.bus_priority else "enable_bus_priority"))
        if self.allows(op.id, "toll") and w.cordon:
            state["cordon_toll_usd"] = op.toll
            context["cordon"] = {
                "speed_pct_of_free_flow": speed_pct(net.links[i] for i in w.cordon),
                "cars_inside": sum(1 for v in driving if v.link in w.cordon),
                "toll_revenue_usd": round(op.toll_revenue, 2),
            }
            choices = tuple(float(c) for c in self.cfg.toll_choices)
            actions.append(self.spec("set_toll", Param("usd", "number", choices)))
        if self.allows(op.id, "signals") and w.signals:
            lights = self.sim.kinds["signal"]
            total, at_red = {name: 0 for name in PHASES}, []
            for sig in w.signals.values():
                for i, (held, _) in enumerate(lights.demand(sig)):
                    total[sig.names[i]] += sig.wait_by_phase[i]
                    if held and i != sig.phase and lights.allows(sig.id, "coordinate"):
                        at_red.append((held, sig.node, sig.names[i]))
            at_red.sort(key=lambda row: (-row[0], row[1]))
            # a count at one instant swings with the phases, so the picture is what accumulated since the last review
            waited = {name: round((total[name] - self._waited[name]) * w.dt) for name in PHASES}
            aux["waited"] = total
            context["traffic_lights"] = {
                "vehicle_seconds_waited_since_last_review": waited, "favoured_direction": op.favoured,
                "longest_queues_at_red": [{"junction": node, "phase": name, "waiting": held} for held, node, name in at_red[:4]]}
            actions.append(self.spec("favour_direction", Param("direction", "string", (*PHASES, "none"))))
            if at_red:
                actions.append(self.spec("guide_signal", Param("node", "integer", tuple(sorted({row[1] for row in at_red[:4]}))),
                                         Param("phase", "string", PHASES)))
        return self.obs(op.id, trigger, state, context, actions, Decision("no_change"), aux)

    def apply(self, rec):
        w, sim = self.world, self.sim
        op = w.operator
        action = rec.decision.action
        op.next_review = w.tick + w.ticks(self.cfg.review_seconds)
        op.incident_pending = False
        self._waited = rec.obs.aux.get("waited", self._waited) if rec.obs is not None else self._waited
        if action in ("publish_advisory", "withdraw_advisory"):
            op.advisory = action == "publish_advisory"
            op.advisories += int(op.advisory)
            sim.emit("advisory", agent=op.id, published=op.advisory)
            sim.conditions_changed(notify_operator=False)
        elif action in ("enable_bus_priority", "disable_bus_priority"):
            op.bus_priority = action == "enable_bus_priority"
            w.bus_factor = self.cfg.bus_priority_gain if op.bus_priority else 1.0
            w.car_factor = self.cfg.car_priority_penalty if op.bus_priority else 1.0
            sim.emit("bus_priority", agent=op.id, on=op.bus_priority)
        elif action == "set_toll" and abs(rec.decision.args["usd"] - op.toll) > 1e-9:
            op.toll, op.toll_changes = rec.decision.args["usd"], op.toll_changes + 1
            sim.emit("toll_set", agent=op.id, usd=op.toll)
        elif action == "favour_direction":
            direction = None if rec.decision.args["direction"] == "none" else rec.decision.args["direction"]
            if direction != op.favoured:
                op.favoured, op.signal_requests = direction, op.signal_requests + 1
                sim.emit("signals_favour", agent=op.id, direction=direction)
        elif action == "guide_signal":
            sig = next(s for s in w.signals.values() if s.node == rec.decision.args["node"])
            sig.request = {"phase": sig.names.index(rec.decision.args["phase"]),
                           "until": w.tick + w.ticks(sim.cfg.agents.signal.request_seconds)}
            sig.next_decision = w.tick                    # the junction looks at the request at once
            op.signal_requests += 1
            sim.emit("signal_request", agent=op.id, node=sig.node, phase=rec.decision.args["phase"])
        rec.result = action

    def physics(self):
        if self.world.operator.bus_priority:
            self.world.operator.priority_ticks += 1

    def frame(self):
        op = self.world.operator
        return [[op.id, op.advisory, op.bus_priority, op.toll]]
