"""Does the benchmark agree with a closed-loop run? Score a run's own decision points one by one.

    .venv/bin/python experiments/insitu.py --llm deepseek --answer explain reason
    .venv/bin/python experiments/insitu.py --llm deepseek-thinking --answer explain

The ``llm_alignment`` experiment lets a language model drive the vehicles through a whole run. Here
the route choices that arise in that scenario (taken from runs driven by the algorithm) are put to
the model one at a time, with no run around them and no memory, and scored like the
``route_delegation`` task. If the two agree, what a closed-loop run measures is the model's
judgement at single decision points -- which is what the benchmark measures directly, on items
whose difficulty is controlled.

The table goes to ``experiments/results/insitu_alignment__<presets>__<answer formats>.md``.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from agmsim.benchmark import Item, RouteDelegation, evaluate
from agmsim.config import LLM_PRESETS
from agmsim.engine import Simulation
from agmsim.kinds.vehicle import owner_route_costs

from suite import ANSWERS, EXPERIMENTS, LLM_NAMES, is_local

HERE = Path(__file__).parent
SCENARIO = str(HERE / "scenarios" / EXPERIMENTS["llm_alignment"]["scenario"])
BASE = {**EXPERIMENTS["llm_alignment"]["base"], "record.dir": None}


class RoutePointsOfARun(RouteDelegation):
    """The toll-or-not choices that occur in the ``llm_alignment`` scenario, scored like ``route_delegation``."""

    name = "route_points_of_a_run"
    config = Simulation(SCENARIO, overrides=BASE).cfg.to_dict()

    def __init__(self, seeds, overrides=None):
        self.seeds = list(seeds)
        self.overrides = dict(overrides or {})      # on top of the scenario, for the runs the points are taken from

    def items(self):
        out = []
        for seed in self.seeds:
            result = Simulation(SCENARIO, overrides={**BASE, **self.overrides, "simulation.seed": seed}).run()
            for n, rec in enumerate(result.records):
                if not (rec.get("type") == "decision" and rec["kind"] == "vehicle" and rec["trigger"] == "route"
                        and "obs" in rec and "owner" in rec["obs"]["state"]):
                    continue
                obs = rec["obs"]
                routes = obs["context"]["routes"]
                tolled = [r["toll_usd"] > 0 for r in routes]
                if all(tolled) or not any(tolled):
                    continue
                costs = owner_route_costs(obs["state"], routes)
                best = costs.index(min(costs))
                quick_toll = min((r for r in routes if r["toll_usd"] > 0), key=lambda r: r["eta_min"])
                quick_free = min((r for r in routes if r["toll_usd"] == 0), key=lambda r: r["eta_min"])
                saved = quick_free["eta_min"] - quick_toll["eta_min"]
                extra = quick_toll["toll_usd"] + obs["state"]["driving_cost_usd_per_km"] * (quick_toll["km"] - quick_free["km"])
                ratio = math.inf if saved <= 0 or extra <= 0 else \
                    obs["state"]["owner"]["value_of_time_usd_per_h"] / (60.0 * extra / saved)
                away = abs(math.log(ratio)) if 0 < ratio < math.inf else math.inf
                tier = "hard" if away < math.log(1.6) else "medium" if away < math.log(2.5) else "easy"
                out.append(Item(f"seed{seed}-{n}", obs, "pay" if tolled[best] else "free",
                                {"design": seed, "ratio": "", "tier": tier, "current": "toll" if tolled[0] else "free",
                                 "extra_routes": len(routes) - 2, "margin_usd": round(sorted(costs)[1] - costs[best], 3)},
                                {"costs": costs, "roles": ["toll" if t else "free" for t in tolled], "best": best}))
        return out

    def digest(self):
        return f"seeds{self.seeds[0]}-{self.seeds[-1]}"

    def curves(self, rows):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", nargs="+", default=[], choices=[p for p in LLM_PRESETS if p != "mock"])
    parser.add_argument("--answer", nargs="+", default=["explain"], choices=list(ANSWERS))
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    task = RoutePointsOfARun(range(1, args.seeds + 1))
    items = task.items()
    margins = sorted(i.factors["margin_usd"] for i in items)
    subjects = [("规则（只求快）", {"agents.vehicle.non_llm_policy": "rule"}),
                ("算法", {"agents.vehicle.non_llm_policy": "algorithm"})]
    for preset in args.llm:
        for answer in args.answer:
            subjects.append((f"{LLM_NAMES.get(preset, preset)}（{ANSWERS[answer][0]}）",
                             {"agents.vehicle.use_llm": True, "llm.preset": preset,
                              "llm.max_concurrency": 8 if is_local(preset) else 24,
                              "agents.vehicle.llm_policy_params": {"rationale": ANSWERS[answer][1]}}))
    lines = ["| 决策模型 | 与车主利益一致率 | 灵敏度 | 该付时付了 | 不该付时付了 | 易 | 中 | 难 | 估算费用(美元) |",
             "| - | - | - | - | - | - | - | - | - |"]
    for label, overrides in subjects:
        report = evaluate(task, overrides)
        s = report["scores"]
        print(f"  {label}: accuracy {s['accuracy']}, sensitivity {s['sensitivity']}", flush=True)
        lines.append(f"| {label} | {100 * s['accuracy']:.1f}% | {s['sensitivity']:+.3f} | "
                     f"{100 * s['paid_when_worth_it']:.0f}% | {100 * s['paid_when_not']:.0f}% | "
                     + " | ".join(f"{100 * s[f'accuracy_{t}']:.0f}%" for t in ("easy", "medium", "hard"))
                     + f" | {report['usage']['cost_usd']:.3f} |")
    tiers = {t: sum(i.factors["tier"] == t for i in items) for t in ("easy", "medium", "hard")}
    text = ("### insitu_alignment\n\n**问题**：把闭环实验 llm_alignment 里出现的选路决策点逐个拿出来单独打分，结果与闭环一致吗？\n\n"
            + "\n".join(lines) + "\n\n"
            f"{args.seeds} 个种子的算法运行中出现的 {len(items)} 个“付不付费”决策点（该付 "
            f"{sum(i.label == 'pay' for i in items)} 个），逐个交给模型，不带历史。"
            f"其中 {sum(i.factors['extra_routes'] > 0 for i in items)} 个列出三条以上路线；"
            f"最优与次优路线的差额中位数为 {margins[len(margins) // 2]:.2f} 美元。"
            f"按车主时间价值离临界值的远近，易 {tiers['easy']}、中 {tiers['medium']}、难 {tiers['hard']} 个"
            f"（分界为 2.5 倍和 1.6 倍）。测于 {time.strftime('%Y-%m-%d')}。\n")
    stem = "insitu_alignment" + (f"__{'+'.join(args.llm)}__{'+'.join(args.answer)}" if args.llm else "")
    (HERE / "results" / f"{stem}.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
