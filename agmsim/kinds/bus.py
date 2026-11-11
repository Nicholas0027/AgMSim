"""Buses on fixed circular lines.

A non-agentic bus (L0) leaves each stop as soon as boarding is done. With
agency the bus itself decides at the stop: it may hold (for an approaching
passenger, or to even out headways -- context awareness, L1) and it may skip
the next stop when nobody on board needs it (goal adjustment, L2).
"""
from __future__ import annotations

from ..actions import Decision, Param
from ..agency import AgencySpec
from ..entities import Bus, BState, TState
from .base import Kind, advance, register_kind

OBJECTIVE = ("Serve passengers well: keep headways regular, do not leave passengers behind, "
             "and limit the delay imposed on passengers already on board.")


@register_kind("bus")
class BusKind(Kind):
    name = "bus"
    side = "machine"
    title = "Bus"
    role_prompt = (
        "You are the on-board decision-making agent of a bus running a fixed circular line. At each stop, once "
        "boarding is finished, you decide whether to depart, to hold a little longer, or to depart and skip the "
        "next stop. Holding helps passengers who are about to reach the stop and evens out the spacing between "
        "buses, but delays everyone already on board."
    )
    triggers = {"at_stop": "boarding at a stop is finished; decide how to proceed"}
    actions = {
        "depart": "Leave the stop now and serve the next stop.",
        "hold": "Stay at the stop for the given number of seconds, then reconsider.",
        "depart_skip_next": "Leave now and drive past the next stop without stopping.",
    }
    agency = AgencySpec(
        "bus",
        {
            "hold": "hold at a stop beyond the scheduled dwell (context awareness)",
            "skip": "skip the next stop when no passenger on board needs it (goal adjustment)",
        },
        {"L0": (), "L1": ("hold",), "L2": ("hold", "skip"), "L3": ("hold", "skip"), "L4": ("hold", "skip")},
    )

    # ------------------------------------------------------------------ setup

    def setup(self):
        w, cfg = self.world, self.cfg
        net = w.net
        self.queues: dict[tuple[str, int], list] = {}
        ids = []
        for spec in cfg.lines:
            stops = [int(s) for s in spec["stops"]]
            if len(stops) < 2 or any(s not in net.nodes for s in stops):
                raise ValueError(f"bus line {spec.get('id')} needs at least two valid stop nodes")
            line = str(spec["id"])
            n_bus = int(spec.get("buses", 1))
            legs = [net.time(stops[i], stops[(i + 1) % len(stops)]) / cfg.speed_factor for i in range(len(stops))]
            cycle = sum(legs) + len(stops) * cfg.dwell_seconds
            w.lines[line] = {"stops": stops, "legs_s": legs, "headway_s": cycle / n_bus,
                             "capacity": int(spec.get("capacity", 30))}
            for k in range(n_bus):
                idx = int(k * len(stops) / n_bus)
                bus = Bus(id=f"B{len(w.buses) + 1:02d}", line=line, node=stops[idx], next_stop=idx,
                          capacity=w.lines[line]["capacity"], dwell_until=w.ticks(cfg.dwell_seconds))
                w.buses[bus.id] = bus
                ids.append(bus.id)
        return ids

    # ------------------------------------------------------------------ services used by travellers

    def _itinerary(self, line: str, board: int, destination: int, persona, access_m: float):
        """Itinerary boarding ``line`` at stop index ``board``, or None if it does not serve the destination."""
        w, cfg, net = self.world, self.cfg, self.world.net
        info = w.lines[line]
        stops = info["stops"]
        limit = 300.0 if persona.mobility_limited else cfg.max_access_m
        alight = min(range(len(stops)), key=lambda i: (net.length(stops[i], destination), i))
        egress_m = net.length(stops[alight], destination)
        if board == alight or access_m > limit or egress_m > limit:
            return None
        ride, i = 0.0, board
        while i != alight:
            ride += info["legs_s"][i] + cfg.dwell_seconds
            i = (i + 1) % len(stops)
        return {"line": line, "board": board, "alight": alight,
                "access_s": round(access_m / persona.walk_speed, 1), "wait_s": round(info["headway_s"] / 2.0, 1),
                "ride_s": round(ride, 1), "egress_s": round(egress_m / persona.walk_speed, 1), "fare": cfg.fare}

    def plan(self, origin: int, destination: int, persona):
        """Best walk-bus-walk itinerary between two nodes, or None if no line serves the trip."""
        w, net = self.world, self.world.net
        best = None
        for line, info in w.lines.items():
            stops = info["stops"]
            board = min(range(len(stops)), key=lambda i: (net.length(origin, stops[i]), i))
            plan = self._itinerary(line, board, destination, persona, net.length(origin, stops[board]))
            if plan is not None:
                total = plan["access_s"] + plan["wait_s"] + plan["ride_s"] + plan["egress_s"]
                if best is None or total < best[0]:
                    best = (total, plan)
        return None if best is None else best[1]

    def stop_plans(self, destination: int, persona) -> list:
        """Itineraries to ``destination`` from every stop (for park-and-ride)."""
        plans = []
        for line, info in self.world.lines.items():
            for board in range(len(info["stops"])):
                plan = self._itinerary(line, board, destination, persona, 0.0)
                if plan is not None:
                    plans.append(plan)
        return plans

    def join_stop(self, tr) -> None:
        self.queues.setdefault((tr.bus["line"], tr.bus["board"]), []).append(tr.id)

    def leave_stop(self, tr) -> None:
        queue = self.queues.get((tr.bus["line"], tr.bus["board"]), [])
        if tr.id in queue:
            queue.remove(tr.id)

    def next_bus_eta(self, line: str, stop: int) -> float:
        """Rough seconds until the next bus of ``line`` reaches stop index ``stop``."""
        w, info = self.world, self.world.lines[line]
        n = len(info["stops"])
        best = float("inf")
        for b in w.buses.values():
            if b.line != line:
                continue
            if b.state == BState.DWELLING:
                eta, i = max(0.0, (b.dwell_until - w.tick) * w.dt), b.next_stop
                if i != stop:
                    eta += info["legs_s"][i]
                    i = (i + 1) % n
            else:
                eta, i = w.eta(b, info["stops"][b.next_stop]) / self.cfg.speed_factor, b.next_stop
            while i != stop:
                eta += self.cfg.dwell_seconds + info["legs_s"][i]
                i = (i + 1) % n
            best = min(best, eta)
        return best

    # ------------------------------------------------------------------ decision points

    def due(self):
        return [(b.id, "at_stop") for b in self.world.buses.values() if b.asked]

    def authority(self, agent_id, trigger):
        return self.allows(agent_id, "hold") or self.allows(agent_id, "skip")

    def observe(self, agent_id, trigger):
        w, cfg = self.world, self.cfg
        b = w.buses[agent_id]
        info = w.lines[b.line]
        stops, n = info["stops"], len(info["stops"])
        nxt = (b.next_stop + 1) % n
        last = w.stop_departures.get((b.line, b.next_stop))
        approaching = sorted(
            round((tr.phase_end - w.tick) * w.dt)
            for tr in w.travelers.values()
            if tr.state == TState.TO_STOP and tr.bus["line"] == b.line and tr.bus["board"] == b.next_stop
            and (tr.phase_end - w.tick) * w.dt <= cfg.max_hold_seconds
        )
        alight_next = sum(1 for tid in b.onboard if w.travelers[tid].bus["alight"] == nxt)
        state = {"line": b.line, "stop_node": stops[b.next_stop], "passengers_on_board": len(b.onboard),
                 "capacity": b.capacity, "objective": OBJECTIVE}
        context = {
            "seconds_since_previous_bus_left_this_stop": None if last is None else round((w.tick - last) * w.dt),
            "target_headway_s": round(info["headway_s"]),
            "already_held_here_s": round(b.held_s),
            "passengers_still_waiting_here": len(self.queues.get((b.line, b.next_stop), [])),
            "passengers_walking_to_this_stop_arrive_in_s": approaching,
            "passengers_waiting_at_next_stop": len(self.queues.get((b.line, nxt), [])),
            "passengers_on_board_alighting_at_next_stop": alight_next,
        }
        actions = [self.spec("depart")]
        if self.allows(b.id, "hold"):
            choices = tuple(int(c) for c in cfg.hold_choices if b.held_s + c <= cfg.max_hold_seconds)
            if choices:
                actions.append(self.spec("hold", Param("seconds", "integer", choices)))
        if self.allows(b.id, "skip") and n > 2 and alight_next == 0:
            actions.append(self.spec("depart_skip_next"))
        return self.obs(b.id, trigger, state, context, actions, Decision("depart"))

    def apply(self, rec):
        w = self.world
        b = w.buses[rec.agent]
        action = rec.decision.action
        b.asked = False
        if action == "hold":
            seconds = rec.decision.args["seconds"]
            b.held_s, b.holds = b.held_s + seconds, b.holds + 1
            b.dwell_until = w.tick + w.ticks(seconds)
            self.sim.emit("bus_hold", agent=b.id, seconds=seconds)
            rec.result = f"holding {seconds} s"
        else:
            if action == "depart_skip_next":
                b.skip_next = True
            self._depart(b, w.tick)

    # ------------------------------------------------------------------ physics

    def _depart(self, b, when: int) -> None:
        w, info = self.world, self.world.lines[b.line]
        n = len(info["stops"])
        b.left_behind += len(self.queues.get((b.line, b.next_stop), []))
        w.stop_departures[(b.line, b.next_stop)] = when
        step = 1
        if b.skip_next:
            step, b.skip_next, b.skips = 2, False, b.skips + 1
            self.sim.emit("bus_skip", agent=b.id, stop=info["stops"][(b.next_stop + 1) % n])
        b.next_stop = (b.next_stop + step) % n
        b.path = list(w.net.route(b.node, info["stops"][b.next_stop], w.advisory).links)
        b.state, b.held_s = BState.MOVING, 0.0

    def _board(self, b, now: int) -> int:
        w = self.world
        queue = self.queues.get((b.line, b.next_stop), [])
        boarded = 0
        while queue and len(b.onboard) < b.capacity:
            tr = w.travelers[queue.pop(0)]
            tr.state, tr.t_board = TState.ON_BUS, now
            if tr.wait_start is not None:
                tr.waited_ticks += now - tr.wait_start
                tr.wait_start = None
            tr.next_check = None
            b.onboard.append(tr.id)
            boarded += 1
        b.boardings += boarded
        return boarded

    def physics(self):
        w, cfg = self.world, self.cfg
        now = w.tick + 1
        for b in w.buses.values():
            if b.state == BState.MOVING:
                priority = w.operator is not None and w.operator.bus_priority    # priority = bus lanes
                moved, arrived = advance(w.net, b, w.dt, cfg.speed_factor * w.bus_factor, own_lane=priority)
                b.m_total += moved
                if not arrived:
                    continue
                b.state = BState.DWELLING
                alighted = 0
                for tid in list(b.onboard):
                    tr = w.travelers[tid]
                    if tr.bus["alight"] == b.next_stop:
                        b.onboard.remove(tid)
                        tr.state, tr.phase_start = TState.EGRESS, now
                        tr.phase_end = now + (w.ticks(tr.bus["egress_s"]) if tr.bus["egress_s"] > 0 else 0)
                        alighted += 1
                boarded = self._board(b, now)
                b.dwell_until = now + w.ticks(cfg.dwell_seconds + cfg.dwell_per_passenger_seconds * (alighted + boarded))
            else:
                self._board(b, now)
                if now >= b.dwell_until and not b.asked:
                    if self.authority(b.id, "at_stop"):
                        b.asked = True          # the bus decides at the start of the next tick
                    else:
                        self._depart(b, now)

    def frame(self):
        out = []
        for b in self.world.buses.values():
            x, y = self.world.xy(b)
            out.append([b.id, round(x, 1), round(y, 1), b.state.value, len(b.onboard), b.line])
        return out
