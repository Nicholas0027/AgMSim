"""Does naming the junctions of each route change a decision it has nothing to do with?

    .venv/bin/python experiments/insitu_identity.py --llm deepseek --answer reason
    .venv/bin/python experiments/insitu_identity.py --llm deepseek-thinking --answer explain --retest

``observation.route_roads`` adds to every route a car is offered the junctions it passes. A car that
remembers its own decisions needs them to recognise a route it has left (``insitu_memory.py``). Here the
question is the other one: at decision points whose right answer does not depend on which route is which,
does a model decide differently when the junctions are shown?

Two sets of decision points, taken from runs driven by the algorithm (so they are the same for every model):

* **toll**: the pay-or-not choices of the ``llm_alignment`` scenario. Right: the route that costs the
  owner least (toll, driving cost and time at the owner's value of time).
* **transfer**: the points of ``bottleneck.yaml`` at which a vehicle (L3) may propose to its owner to
  continue by bus. Classed as in the benchmark's ``transfer_proposal``: worth proposing, not worth it, or
  open (which depends on how onerous the owner finds waiting and walking; not scored).

Every point is put to the model twice: without the junctions and with them. Nothing else differs (the
car's memory is shown both times, with or without the junctions of the routes it names). ``--retest``
asks the version without junctions once more: how many answers change between two askings of the very
same question is what a change between the two versions has to be compared with.

The tables go to ``experiments/results/insitu_identity__<presets>.md``; every asking is kept in
``insitu_identity.json`` and is not repeated by a later call with the same settings (so ``--retest`` can
be added afterwards).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from agmsim.benchmark import Item, Task, _transfer_label, evaluate, transfer_margins
from agmsim.engine import Simulation

from insitu import RoutePointsOfARun
from suite import ANSWERS, LLM_NAMES, is_local

HERE = Path(__file__).parent
ROADS = {"agents.vehicle.observation.route_roads": True}
BRIDGE = str(HERE / "scenarios" / "bottleneck.yaml")
BRIDGE_BASE = {"agents.vehicle.agency": "L3", "record.dir": None}
ROUTE_KEYS = ("left_route", "took_route", "route", "quickest_other_route")
STORE = HERE / "results" / "insitu_identity.json"      # every asking, so that none is paid for twice
LABELS = {"pay": "该付费", "free": "不该付费", "worth": "值得提议", "not_worth": "不值得", "open": "两可"}


def shown(item: Item, named: bool) -> Item:
    """The item as a car with (``named``) or without route identity is shown it."""
    if named:
        return item
    obs = json.loads(json.dumps(item.obs))
    for row in obs["context"]["routes"]:
        row.pop("via_nodes", None)
    for entry in obs.get("memory", ()):
        for key in ROUTE_KEYS:
            if isinstance(entry.get(key), dict):
                entry[key].pop("via_nodes", None)
    return Item(item.id, obs, item.label, item.factors, item.truth)


class TollPoints(RoutePointsOfARun):
    """Pay the toll or not: the route choices of the ``llm_alignment`` scenario."""

    name = "toll"
    title = "收费选路（llm_alignment 场景，车 L2）"
    right = "对车主最划算的路线"

    def __init__(self, seeds, named: bool):
        super().__init__(seeds, ROADS)
        self.named, self._items = named, None

    def items(self):
        if self._items is None:
            self._items = [shown(item, self.named) for item in super().items()]
        return self._items

    def outcome(self, item, decision):
        out = super().outcome(item, decision)
        return {**out, "answer": f"route {out['route']}", "ok": out["correct"], "stayed": out["route"] == 0,
                "leaning": out["paid"]}


class TransferPoints(Task):
    """Propose a transfer to the bus or drive on: the vehicles of ``bottleneck.yaml`` at L3."""

    name = "transfer"
    kind = "vehicle"
    title = "提议换乘（bottleneck 场景，车 L3）"
    right = "值得时提议、不值得时不提议（介于两者之间的不计分）"
    config = Simulation(BRIDGE, overrides=BRIDGE_BASE).cfg.to_dict()

    def __init__(self, seeds, named: bool):
        self.seeds, self.named, self._items = list(seeds), named, None

    def items(self):
        if self._items is not None:
            return self._items
        out = []
        for seed in self.seeds:
            result = Simulation(BRIDGE, overrides={**BRIDGE_BASE, **ROADS, "simulation.seed": seed}).run()
            for n, rec in enumerate(result.records):
                if not (rec.get("type") == "decision" and rec["kind"] == "vehicle" and "obs" in rec
                        and "suggest_transfer" in rec["permitted"]):
                    continue
                obs = rec["obs"]
                drive = min(r["eta_min"] for r in obs["context"]["routes"])
                vot = obs["state"]["owner"]["value_of_time_usd_per_h"]
                classes = {_transfer_label(*transfer_margins(vot, drive - option["total_min"], option))
                           for option in obs["context"]["bus_transfer_options"]}
                label = "worth" if "worth" in classes else "open" if "open" in classes else "not_worth"
                out.append(shown(Item(f"seed{seed}-{n}", obs, label, {"routes": len(obs["context"]["routes"])}), self.named))
        self._items = out
        return out

    def digest(self):
        return f"seeds{self.seeds[0]}-{self.seeds[-1]}"

    def outcome(self, item, decision):
        proposed = decision.action == "suggest_transfer"
        route = None if proposed else decision.args["route"]
        return {"answer": f"stop {decision.args['stop']}" if proposed else f"route {route}",
                "ok": None if item.label == "open" else proposed == (item.label == "worth"),
                "stayed": None if proposed else route == 0, "leaning": proposed}

    def score(self, rows):
        return {}


TASKS = {"toll": (TollPoints, "选了收费路线"), "transfer": (TransferPoints, "提议了换乘")}


def share(rows, key) -> str:
    rows = [r for r in rows if r[key] is not None]
    return f"{100 * sum(r[key] for r in rows) / len(rows):.0f}%" if rows else "–"


def paired(a: list, b: list) -> dict:
    """What differs between two askings of the same points (rows in the same order)."""
    assert [r["item"] for r in a] == [r["item"] for r in b]
    scored = [(x, y) for x, y in zip(a, b) if x["ok"] is not None]
    gained = sum(y["ok"] and not x["ok"] for x, y in scored)
    lost = sum(x["ok"] and not y["ok"] for x, y in scored)
    n = max(1, len(scored))
    half = 1.96 * math.sqrt(max(0.0, gained + lost - (gained - lost) ** 2 / n)) / n
    return {"changed": sum(x["answer"] != y["answer"] for x, y in zip(a, b)), "gained": gained, "lost": lost,
            "diff_pp": 100 * (gained - lost) / n, "half_pp": 100 * half}


def keep(key: str, arm: str, run: dict) -> None:
    """Add one asking to the store on disk (read again first: another call may have added to it meanwhile)."""
    store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
    store.setdefault(key, {})[arm] = run
    STORE.parent.mkdir(exist_ok=True)
    STORE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", nargs="+", default=["deepseek"])
    parser.add_argument("--answer", default="reason", choices=list(ANSWERS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--seeds", type=int, default=3, help="runs the toll points are taken from (transfer: 1)")
    parser.add_argument("--points", type=int, default=150, help="at most this many points of each set, evenly spread")
    parser.add_argument("--retest", action="store_true", help="ask the version without junctions a second time")
    parser.add_argument("--budget", type=float, default=0.10, help="USD at most for one asking of one set")
    args = parser.parse_args()

    store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
    blocks, spent = [], 0.0
    for name in args.tasks:
        cls, leaning = TASKS[name]
        seeds = range(1, (args.seeds if name == "toll" else 1) + 1)
        tasks = {"bare": cls(seeds, named=False), "named": cls(seeds, named=True)}
        tasks["again"] = tasks["bare"]
        rows_out, asked, scored = [], 0, 0
        for preset in args.llm:
            overrides = {"agents.vehicle.use_llm": True, "llm.preset": preset, "llm.budget_usd": args.budget,
                         "llm.max_concurrency": 8 if is_local(preset) else 24,
                         "agents.vehicle.llm_policy_params": {"rationale": ANSWERS[args.answer][1]}}
            key = f"{name}|{preset}|{args.answer}|seeds{len(seeds)}|points{args.points}"
            runs = store.setdefault(key, {})
            for arm in ("bare", "named") + (("again",) if args.retest else ()):
                if arm in runs:                     # asked before: not asked (or paid for) again
                    continue
                started = time.time()
                report = evaluate(tasks[arm], overrides, limit=args.points)
                invalid = sum(not r["valid"] for r in report["rows"])
                print(f"  {name} / {preset} / {arm}: {len(report['rows'])} points, {invalid} without a valid reply, "
                      f"{report['usage']['cost_usd']:.3f} USD, {time.time() - started:.0f} s", flush=True)
                spent += report["usage"]["cost_usd"]
                if invalid > 0.1 * len(report["rows"]):
                    raise SystemExit("too many points without a valid reply (budget reached, or the model is not answering); "
                                     "nothing of this asking was kept")
                runs[arm] = {"rows": report["rows"], "usage": report["usage"]}
                keep(key, arm, runs[arm])
            a, b = runs["bare"]["rows"], runs["named"]["rows"]
            d = paired(a, b)
            asked, scored = len(a), sum(r["ok"] is not None for r in a)
            tokens = [round(runs[k]["usage"]["input_tokens"] / max(1, runs[k]["usage"]["calls"])) for k in ("bare", "named")]
            star = "*" if abs(d["diff_pp"]) > d["half_pp"] else ""
            rows_out.append([
                f"{LLM_NAMES.get(preset, preset)}（{ANSWERS[args.answer][0]}）", f"{share(a, 'ok')} → {share(b, 'ok')}",
                f"{d['diff_pp']:+.1f} ± {d['half_pp']:.1f}{star}（+{d['gained']} / −{d['lost']}）", str(d["changed"]),
                str(paired(a, runs["again"]["rows"])["changed"]) if "again" in runs else "–",
                f"{share(a, 'leaning')} → {share(b, 'leaning')}", f"{share(a, 'stayed')} → {share(b, 'stayed')}",
                str(sum(not r["valid"] for run in runs.values() for r in run["rows"])), f"{tokens[0]} → {tokens[1]}",
                f"{sum(run['usage']['cost_usd'] for run in runs.values()):.3f}"])
        head = ["模型", "判断对了：不带 → 带路口", "差值(百分点)", "答案变了", "同一问法再问一次时变了", f"{leaning}：不带 → 带",
                "留在当前路线：不带 → 带", "无效回复", "输入 token/次：不带 → 带", "花费(美元)"]
        lines = ["| " + " | ".join(head) + " |", "| - |" + " - |" * (len(head) - 1)] + ["| " + " | ".join(r) + " |" for r in rows_out]
        labels = {}
        for item in tasks["named"].items():
            labels[item.label] = labels.get(item.label, 0) + 1
        blocks.append(f"#### {cls.title}\n\n" + "\n".join(lines) + "\n\n"
                      f"每个模型问 {asked} 个决策点（计分 {scored} 个），一次不带途经路口，一次带。对的做法：{cls.right}。"
                      f"决策点取自{'种子 1' if len(seeds) == 1 else f'种子 1 到 {len(seeds)}'} 的算法运行，共 {len(tasks['named'].items())} 个"
                      f"（{'、'.join(f'{LABELS[k]} {v}' for k, v in sorted(labels.items()))}），超过 {args.points} 个时均匀抽取。"
                      "差值是逐点配对的，± 为 95% 置信区间半宽，`*` 表示区间不含 0；括号内是改对和改错的点数。"
                      "花费含表中各次提问（按列表价估算）。\n")
    text = ("### insitu_identity\n\n**问题**：给每条路线加上途经路口（`observation.route_roads`），会不会改变与它无关的决策？\n\n"
            + "\n".join(blocks) + f"\n测于 {time.strftime('%Y-%m-%d')}。\n")
    stem = "insitu_identity__" + "+".join(args.llm) + ("" if set(args.tasks) == set(TASKS) else "__" + "+".join(args.tasks))
    (HERE / "results" / f"{stem}.md").write_text(text, encoding="utf-8")
    print(f"\n{text}\nSpent in this call: {spent:.3f} USD")


if __name__ == "__main__":
    main()
