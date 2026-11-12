"""Does a language model use what a car remembers? The decision points of one run, put to it one by one.

    .venv/bin/python experiments/insitu_memory.py --llm deepseek --answer reason
    .venv/bin/python experiments/insitu_memory.py --llm deepseek deepseek-pro --answer reason --reflect

In ``crossing.yaml`` one of two bridges is blocked, and a car sees only what is near and keeps no road in
mind. Driven by the built-in algorithm, such a car goes back and forth: it leaves the blocked bridge when
it sees it, and returns to it once it is out of sight. Its decision points are of two kinds:

* **leave**: the car is on a route over the blocked bridge and sees the jam. Right: change route.
* **return**: the car is on its way round, and the route that looks quickest goes back over the blocked
  bridge. Right: do not take it. A return is *first* if the car has not yet been caught out on this trip,
  and *burned* if its memory holds the outcome of an earlier return (``route_slower_than_reckoned``).

The points are taken from runs driven by the algorithm (so they cost nothing to produce and are the same
for every model) and are asked without a run around them. The memory a car had at that point is part of
its observation; ``--no-memory`` takes it out, ``--no-roads`` leaves out the junctions each route passes.

The table goes to ``experiments/results/insitu_memory__<presets>.md``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agmsim.engine import Simulation
from agmsim.observation import Observation, clock_seconds

from suite import ANSWERS, LLM_NAMES

HERE = Path(__file__).parent
SCENARIO = str(HERE / "scenarios" / "crossing.yaml")
JAMMED_MIN = 15.0          # a route reckoned this long is one over the blocked bridge, seen


def points(seeds, roads: bool) -> list:
    """Decision points of the cars in algorithm-driven runs, each with its kind and what is right there."""
    out = []
    for seed in seeds:
        sim = Simulation(SCENARIO, {"simulation.seed": seed, "agents.vehicle.observation.route_roads": True})
        event = next(e for e in sim.cfg.interventions if e["type"] == "link_condition")
        blocked = {tuple(pair) for u, v in event["links"] for pair in ((u, v), (v, u))}
        clock0 = sim.world.clock0
        for n, rec in enumerate(sim.run().records):
            if not (rec.get("type") == "decision" and rec["kind"] == "vehicle" and rec["trigger"] == "route" and "obs" in rec):
                continue
            obs = rec["obs"]
            now = clock_seconds(obs["time"]) - clock0
            if not event["at_seconds"] <= now < event["until_seconds"]:
                continue
            routes = obs["context"]["routes"]
            over = [any(pair in blocked for pair in zip(r["via_nodes"], r["via_nodes"][1:])) for r in routes]
            quickest = min(routes, key=lambda r: (r["eta_min"], r["route"]))["route"]
            if over[0] and routes[0]["eta_min"] >= JAMMED_MIN and not all(over):
                kind, right = "leave", [r["route"] for r in routes if not over[r["route"]]]
            elif not over[0] and over[quickest]:
                burned = any(item.get("outcome") == "route_slower_than_reckoned" for item in obs.get("memory", ()))
                kind, right = ("return, burned" if burned else "return, first"), [r["route"] for r in routes if not over[r["route"]]]
            else:
                continue
            if not roads:               # the same point as a car without route identity is shown it
                obs = json.loads(json.dumps(obs))
                for row in obs["context"]["routes"]:
                    row.pop("via_nodes", None)
                for item in obs.get("memory", ()):
                    for key in ("left_route", "took_route", "route", "quickest_other_route"):
                        if isinstance(item.get(key), dict):
                            item[key].pop("via_nodes", None)
            out.append({"id": f"seed{seed}-{n}", "kind": kind, "obs": obs, "right": right, "agent": rec["agent"]})
    return out


def ask(items: list, preset: str, params: dict, workers: int) -> tuple:
    """Put every point to the model; returns the rows and what it cost."""
    sim = Simulation(SCENARIO, {"record.dir": None, "agents.vehicle.use_llm": True, "llm.preset": preset,
                                "agents.vehicle.llm_policy_params": params})
    policy = sim.decision_model(next(iter(sim.world.vehicles)))

    def one(item):
        decision = policy.decide(Observation.from_dict(item["obs"]))
        return {**{k: item[k] for k in ("id", "kind")}, "route": decision.args.get("route"),
                "ok": decision.args.get("route") in item["right"], "fallback": "fallback" in decision.info,
                "rationale": decision.rationale}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, items))
    usage = sim.llm_usage()
    sim.close()
    return rows, sum(u["cost_usd"] for u in usage), sum(u["calls"] for u in usage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", nargs="+", default=["deepseek"])
    parser.add_argument("--answer", default="reason", choices=list(ANSWERS))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--reflect", action="store_true", help="tell the model to read its memory first")
    parser.add_argument("--no-memory", action="store_true", help="take the car's memory out of what it is shown")
    parser.add_argument("--no-roads", action="store_true", help="leave out the junctions each route passes")
    parser.add_argument("--leave", type=int, default=40, help="how many 'leave' points to ask (all 'return' points are asked)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    items = points(range(1, args.seeds + 1), roads=not args.no_roads)
    leave = [it for it in items if it["kind"] == "leave"]
    step = max(1, len(leave) // max(1, args.leave))
    items = [it for it in items if it["kind"] != "leave"] + leave[::step][:args.leave]
    kinds = ["leave", "return, first", "return, burned"]
    shown = ("no memory" if args.no_memory else "memory") + (", no route identity" if args.no_roads else ", routes named by their junctions") \
        + (", told to read its memory first" if args.reflect else "")
    counts = ", ".join(str(sum(it["kind"] == k for it in items)) + " " + k for k in kinds)
    lines = [f"### insitu_memory\n\nWhat the car is shown: {shown}. Answer format: {args.answer}. "
             f"Points: {counts} "
             f"(seeds 1 to {args.seeds}, incident period only).\n",
             "| model | " + " | ".join(f"right at '{k}'" for k in kinds) + " | fallbacks | calls | cost (USD) |",
             "| - |" + " - |" * (len(kinds) + 3)]
    everything = {}
    for preset in args.llm:
        params = {"rationale": ANSWERS[args.answer][1], "reflect": args.reflect}
        if args.no_memory:
            params["memory"] = 0
        started = time.time()
        rows, cost, calls = ask(items, preset, params, args.workers)
        cells = []
        for k in kinds:
            mine = [r for r in rows if r["kind"] == k]
            cells.append(f"{sum(r['ok'] for r in mine)} of {len(mine)}" if mine else "–")
        lines.append(f"| {LLM_NAMES.get(preset, preset)} | " + " | ".join(cells)
                     + f" | {sum(r['fallback'] for r in rows)} | {calls} | {cost:.3f} |")
        everything[preset] = rows
        print(lines[-1], f"({time.time() - started:.0f} s)", flush=True)
    text = "\n".join(lines) + "\n"
    stem = "insitu_memory__" + "+".join(args.llm) + ("__reflect" if args.reflect else "") \
        + ("__nomemory" if args.no_memory else "") + ("__noroads" if args.no_roads else "")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{stem}.md").write_text(text, encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(everything, ensure_ascii=False), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
