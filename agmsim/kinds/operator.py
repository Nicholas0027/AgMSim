"""The network operator as an agent.

One operator oversees the road network. Non-agentic (L0) it does nothing at
run time. With agency it decides, on incidents and at regular reviews:

* L1 (context awareness): whether to publish real-time traffic advisories,
  which make current link conditions public knowledge -- even non-agentic
  vehicles then navigate around incidents;
* L2 (goal adjustment): whether to switch on transit priority (buses bypass
  congestion, cars are slightly slower) and what to charge for entering the
  central cordon.

This is a deliberately small action set; richer controls (signals, closures,
ramp metering) fit the same interface.
"""
from __future__ import annotations

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..entities import OperatorState, TState, VState
from .base import Kind, register_kind

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
    triggers = {"review": "an incident changed link conditions, or the periodic review is due"}
    actions = {
        "no_change": "Leave the current settings as they are.",
        "publish_advisory": "Publish real-time link conditions to all vehicles.",
        "withdraw_advisory": "Stop publishing real-time link conditions.",
        "enable_bus_priority": "Switch transit priority on.",
        "disable_bus_priority": "Switch transit priority off.",
        "set_toll": "Set the charge for entering the central cordon to the chosen amount.",
    }
    agency = AgencySpec(
        "operator",
        {
            "advise": "publish or withdraw real-time traffic advisories (context awareness)",
            "prioritize": "switch transit priority on or off (goal adjustment)",
            "toll": "set the cordon charge (goal adjustment)",
        },
        {"L0": (), "L1": ("advise",), "L2": ("advise", "prioritize", "toll"),
         "L3": ("advise", "prioritize", "toll"), "L4": ("advise", "prioritize", "toll")},
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
        return [op.id]

    def due(self):
        op, w = self.world.operator, self.world
        if (op.incident_pending or w.tick >= op.next_review) and self.authority(op.id, "review"):
            return [(op.id, "review")]
        return []

    def authority(self, agent_id, trigger):
        return any(self.allows(agent_id, p) for p in ("advise", "prioritize", "toll"))

    def observe(self, agent_id, trigger):
        w = self.world
        op, net = w.operator, w.net
        incidents = [{"link": [l.u, l.v], "speed_pct_of_normal": round(l.factor * 100)}
                     for l in net.links if abs(l.factor - 1.0) > 1e-9]
        driving = [v for v in w.vehicles.values() if v.state in (VState.DRIVING, VState.TO_CHARGER)]

        def speed_pct(links):
            links = list(links)
            return round(100.0 * sum(l.flow * l.factor for l in links) / len(links)) if links else 100

        state = {"objective": OBJECTIVE, "advisory_published": op.advisory, "bus_priority_on": op.bus_priority}
        context = {
            "reason": "link conditions changed" if op.incident_pending else "periodic review",
            "incidents": incidents[:12],
            "cars_driving": len(driving),
            "network_speed_pct_of_free_flow": speed_pct(net.links),
            "travellers_waiting_for_a_bus": sum(1 for t in w.travelers.values() if t.state == TState.AT_STOP),
            "travellers_on_buses": sum(len(b.onboard) for b in w.buses.values()),
            "travellers_in_cars": sum(1 for v in driving if v.passenger is not None),
        }
        actions = [self.spec("no_change")]
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
        return self.obs(op.id, trigger, state, context, actions, Decision("no_change"))

    def apply(self, rec):
        w, sim = self.world, self.sim
        op = w.operator
        action = rec.decision.action
        op.next_review = w.tick + w.ticks(self.cfg.review_seconds)
        op.incident_pending = False
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
        rec.result = action

    def physics(self):
        if self.world.operator.bus_priority:
            self.world.operator.priority_ticks += 1

    def frame(self):
        op = self.world.operator
        return [[op.id, op.advisory, op.bus_priority, op.toll]]
