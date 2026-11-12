"""Does telling a car which route is which change a decision it has nothing to do with?

    .venv/bin/python experiments/insitu_identity.py --llm deepseek --answer reason
    .venv/bin/python experiments/insitu_identity.py --llm qwen-local --answer explain --retest --names

A car that remembers its own decisions has to be able to recognise a route it has left
(``insitu_memory.py``). Two settings give routes an identity: ``observation.route_roads`` lists the
junctions each route passes, ``observation.route_names`` gives each way a short name that it keeps. Here
the question is the other one: at decision points whose right answer does not depend on which route is
which, does a model decide differently when it is shown one of them?

Two sets of decision points, taken from runs driven by the algorithm (so they are the same for every model):

* **toll**: the pay-or-not choices of the ``llm_alignment`` scenario. Right: the route that costs the
  owner least (toll, driving cost and time at the owner's value of time).
* **transfer**: the points of ``bottleneck.yaml`` at which a vehicle (L3) may propose to its owner to
  continue by bus. Classed as in the benchmark's ``transfer_proposal``: worth proposing, not worth it, or
  open (which depends on how onerous the owner finds waiting and walking; not scored).

Every point is put to the model without route identity and with the junctions; ``--names`` adds the
version with names. Nothing else differs (the car's memory is shown every time, its routes in the same
form). ``--retest`` asks the version without identity once more: how many answers change between two
askings of the very same question is what a change between two versions has to be compared with.
``--reordered`` asks the version without identity with the fields of each route written in another order:
the same content, so what changes there is what any change of wording does to this model.

The tables go to ``experiments/results/insitu_identity__<presets>.md``; every asking is kept in
``insitu_identity.json`` and is not repeated by a later call with the same settings (so ``--retest`` and
``--names`` can be added afterwards).
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
from insitu_memory import IDENTITY, as_shown
from suite import ANSWERS, LLM_NAMES, is_local

HERE = Path(__file__).parent
BRIDGE = str(HERE / "scenarios" / "bottleneck.yaml")
BRIDGE_BASE = {"agents.vehicle.agency": "L3", "record.dir": None}
STORE = HERE / "results" / "insitu_identity.json"      # every asking, so that none is paid for twice
LABELS = {"pay": "该付费", "free": "不该付费", "worth": "值得提议", "not_worth": "不值得", "open": "两可"}
ARMS = {"bare": "none", "named": "junctions", "labelled": "names", "again": "none",      # asking -> what is shown
        "reordered": "reordered"}


def shown(item: Item, identity: str) -> Item:
    """The item as a car is shown it that recognises routes by their junctions, by names, or not at all."""
    return Item(item.id, as_shown(item.obs, identity), item.label, item.factors, item.truth)


class TollPoints(RoutePointsOfARun):
    """Pay the toll or not: the route choices of the ``llm_alignment`` scenario."""

    name = "toll"
    title = "收费选路（llm_alignment 场景，车 L2）"
    right = "对车主最划算的路线"

    def __init__(self, seeds, identity: str):
        super().__init__(seeds, IDENTITY)
        self.identity, self._items = identity, None

    def items(self):
        if self._items is None:
            self._items = [shown(item, self.identity) for item in super().items()]
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

    def __init__(self, seeds, identity: str):
        self.seeds, self.identity, self._items = list(seeds), identity, None

    def items(self):
        if self._items is not None:
            return self._items
        out = []
        for seed in self.seeds:
            result = Simulation(BRIDGE, overrides={**BRIDGE_BASE, **IDENTITY, "simulation.seed": seed}).run()
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
                out.append(shown(Item(f"seed{seed}-{n}", obs, label, {"routes": len(obs["context"]["routes"])}),
                                 self.identity))
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


def against(runs: dict, arm: str) -> tuple:
    """Two table cells for one version against the one without identity: the paired difference, the answers changed."""
    if arm not in runs:
        return "–", "–"
    d = paired(runs["bare"]["rows"], runs[arm]["rows"])
    star = "*" if abs(d["diff_pp"]) > d["half_pp"] else ""
    return f"{d['diff_pp']:+.1f} ± {d['half_pp']:.1f}{star}（+{d['gained']} / −{d['lost']}）", str(d["changed"])


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
    parser.add_argument("--names", action="store_true", help="also ask with the name of each route's way")
    parser.add_argument("--retest", action="store_true", help="ask the version without identity a second time")
    parser.add_argument("--reordered", action="store_true",
                        help="ask the version without identity with the fields of each route in another order")
    parser.add_argument("--budget", type=float, default=0.10, help="USD at most for one asking of one set")
    args = parser.parse_args()

    store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
    wanted = ["bare", "named"] + ["labelled"] * args.names + ["again"] * args.retest + ["reordered"] * args.reordered
    blocks, spent = [], 0.0
    for name in args.tasks:
        cls, leaning = TASKS[name]
        seeds = range(1, (args.seeds if name == "toll" else 1) + 1)
        tasks = {identity: cls(seeds, identity) for identity in set(ARMS.values())}
        rows_out, asked, scored = [], 0, 0
        for preset in args.llm:
            overrides = {"agents.vehicle.use_llm": True, "llm.preset": preset, "llm.budget_usd": args.budget,
                         "llm.max_concurrency": 8 if is_local(preset) else 24,
                         "agents.vehicle.llm_policy_params": {"rationale": ANSWERS[args.answer][1]}}
            key = f"{name}|{preset}|{args.answer}|seeds{len(seeds)}|points{args.points}"
            runs = store.setdefault(key, {})
            for arm in wanted:
                if arm in runs:                     # asked before: not asked (or paid for) again
                    continue
                started = time.time()
                report = evaluate(tasks[ARMS[arm]], overrides, limit=args.points)
                invalid = sum(not r["valid"] for r in report["rows"])
                print(f"  {name} / {preset} / {arm}: {len(report['rows'])} points, {invalid} without a valid reply, "
                      f"{report['usage']['cost_usd']:.3f} USD, {time.time() - started:.0f} s", flush=True)
                spent += report["usage"]["cost_usd"]
                if invalid > 0.1 * len(report["rows"]):
                    raise SystemExit("too many points without a valid reply (budget reached, or the model is not answering); "
                                     "nothing of this asking was kept")
                runs[arm] = {"rows": report["rows"], "usage": report["usage"]}
                keep(key, arm, runs[arm])
            versions = [arm for arm in ("bare", "named", "labelled") if arm in runs]
            a = runs["bare"]["rows"]
            asked, scored = len(a), sum(r["ok"] is not None for r in a)

            def across(what, runs=runs, versions=versions):
                return " → ".join(what(runs[arm]) for arm in versions)

            rows_out.append([
                f"{LLM_NAMES.get(preset, preset)}（{ANSWERS[args.answer][0]}）",
                across(lambda run: share(run["rows"], "ok")), *against(runs, "named"), *against(runs, "labelled"),
                against(runs, "again")[1], against(runs, "reordered")[1],
                across(lambda run: share(run["rows"], "leaning")), across(lambda run: share(run["rows"], "stayed")),
                str(sum(not r["valid"] for run in runs.values() for r in run["rows"])),
                across(lambda run: str(round(run["usage"]["input_tokens"] / max(1, run["usage"]["calls"])))),
                f"{sum(run['usage']['cost_usd'] for run in runs.values()):.3f}"])
        order = "不带 → 带路口 → 带代号"
        head = ["模型", f"判断对了（{order}）", "带路口：配对差值(百分点)", "带路口：答案变了", "带代号：配对差值(百分点)",
                "带代号：答案变了", "同一问法再问一次时变了", "内容不变只调字段顺序时变了", leaning, "留在当前路线", "无效回复", "输入 token/次", "花费(美元)"]
        lines = ["| " + " | ".join(head) + " |", "| - |" + " - |" * (len(head) - 1)] + ["| " + " | ".join(r) + " |" for r in rows_out]
        labels = {}
        for item in tasks["none"].items():
            labels[item.label] = labels.get(item.label, 0) + 1
        blocks.append(f"#### {cls.title}\n\n" + "\n".join(lines) + "\n\n"
                      f"每个模型问 {asked} 个决策点（计分 {scored} 个）。各列里用 → 隔开的数依次是{order}（没问过的版本不列）。"
                      f"对的做法：{cls.right}。"
                      f"决策点取自{'种子 1' if len(seeds) == 1 else f'种子 1 到 {len(seeds)}'} 的算法运行，共 {len(tasks['none'].items())} 个"
                      f"（{'、'.join(f'{LABELS[k]} {v}' for k, v in sorted(labels.items()))}），超过 {args.points} 个时均匀抽取。"
                      "差值和“答案变了”都是与不带路线身份的那一版逐点配对比较，± 为 95% 置信区间半宽，`*` 表示区间不含 0；"
                      "括号内是改对和改错的点数。花费含表中各次提问（按列表价估算）。\n")
    text = ("### insitu_identity\n\n**问题**：让车认得出路线（途经路口 `observation.route_roads`，或代号 `observation.route_names`），"
            "会不会改变与它无关的决策？\n\n" + "\n".join(blocks) + f"\n测于 {time.strftime('%Y-%m-%d')}。\n")
    stem = "insitu_identity__" + "+".join(args.llm) + ("" if set(args.tasks) == set(TASKS) else "__" + "+".join(args.tasks))
    (HERE / "results" / f"{stem}.md").write_text(text, encoding="utf-8")
    print(f"\n{text}\nSpent in this call: {spent:.3f} USD")


if __name__ == "__main__":
    main()
