"""Experiments on the testbed (the packaged scenario ``montreal``).

    .venv/bin/python experiments/suite.py                          # every experiment that needs no LLM, 3 seeds
    .venv/bin/python experiments/suite.py vehicle_agency --seeds 10

An experiment is a set of arms: overrides of the testbed's settings, with everything else left as it is.
Every arm runs on the same seeds, i.e. on identical travellers, and results are reported as paired
differences from the first arm. Tables go to ``experiments/results/<name>.md``, raw rows to ``<name>.json``.

An arm whose label contains ``{llm}`` needs a language model: it is run once per ``--llm`` preset and
``--answer`` format (``choice``: the action only; ``explain``: the action, then a reason -- the default;
``reason``: the reasoning first, then the action). Arms on a hosted model run ``--llm-workers`` at a
time, arms on a local server ``--local-workers`` at a time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agmsim.config import LLM_PRESETS, TESTBED
from agmsim.experiment import paired_markdown, run_arms

HERE = Path(__file__).parent
LLM_NAMES = {"qwen-local": "Qwen2.5-7B", "deepseek": "DeepSeek-Flash", "deepseek-thinking": "DeepSeek-Flash（思考）",
             "deepseek-pro": "DeepSeek-V4-Pro"}
ANSWERS = {"choice": ("只给选择", False), "explain": ("先选择后说明", True), "reason": ("先推理后选择", "first")}
LLM_KINDS = ("traveler", "vehicle")      # kinds that the LLM experiments put on a language model

# the vehicles' level, from following the default route to arranging their own charging
VEHICLE_LEVELS = {
    "车 L0（照默认路线开）": {"agents.vehicle.agency": "L0"},
    "车 L1（可以改道）": {"agents.vehicle.agency": "L1"},
    "车 L2（还知道车主的偏好）": {"agents.vehicle.agency": "L2"},
    "车 L3（还可以提议换乘）": {"agents.vehicle.agency": "L3"},
    "车 L4（还自己安排充电）": {"agents.vehicle.agency": "L4"},
    "车 L4 · 不主动去充电": {"agents.vehicle.agency": "L4", "agents.vehicle.policy_params": {"top_up_below_pct": 0}},
}
LEVEL_METRICS = {"outcomes.wait_time_s_mean": "平均等待(秒)", "outcomes.vmt_miles": "VMT(英里)", "outcomes.vht_hours": "VHT(小时)",
                 "outcomes.traffic_speed_mph": "平均车速(英里/时)", "travelers.trip_time_s_mean": "平均出行时间(秒)",
                 "travelers.generalised_cost_usd_mean": "广义成本(美元)", "vehicles.reroutes": "改道次数",
                 "vehicles.tolls_usd": "收费总额(美元)", "vehicles.suggestions": "换乘提议", "vehicles.charge_stops": "充电出行次数",
                 "vehicles.empty_km_share": "空驶占比", "vehicles.charge_queue_s_mean": "每次充电排队(秒)",
                 "bus.boardings": "公交上车人次", "operator.advisories_published": "协调者通告"}

EXPERIMENTS = {
    "vehicle_agency": {
        "question": "蒙特利尔测试床，其他都不变，只把私家车的 agency 从 L0 调到 L4，系统表现怎么变？",
        "arms": {label: ov for label, ov in VEHICLE_LEVELS.items() if len(ov) == 1},
        "metrics": LEVEL_METRICS,
    },
    "vehicle_agency_alone": {
        "question": "同一个测试床，其他主体都不作为（协调者不发通告，市中心收费固定为 2 美元，出行者听车的），只改车的等级呢？",
        "base": {"agents.traveler.agency": "L0", "agents.bus.agency": "L0", "agents.evcs.agency": "L0",
                 "agents.ondemand.agency": "L0", "agents.signal.agency": "L0",
                 "agents.operator.agency": "L0", "agents.operator.initial_toll": 2.0},
        "arms": {label: ov for label, ov in VEHICLE_LEVELS.items() if len(ov) == 1},
        "metrics": LEVEL_METRICS,
    },
}


def is_local(preset: str) -> bool:
    url = LLM_PRESETS[preset].get("base_url") or ""
    return "127.0.0.1" in url or "localhost" in url


def run_experiment(name: str, seeds: int, workers: int, llms=("qwen-local",), llm_workers: int = 8,
                   answers=("explain",), local_workers: int = 4) -> str:
    exp = EXPERIMENTS[name]
    base = exp.get("base", {})
    groups: dict[int, dict] = {}          # arms that may run this many at a time
    order = []
    for label, ov in exp["arms"].items():
        if "{llm}" not in label:
            groups.setdefault(workers, {})[label] = {**base, **ov}
            order.append(label)
            continue
        for preset in llms:
            for answer in answers:
                model = LLM_NAMES.get(preset, preset) + (f"（{ANSWERS[answer][0]}）" if tuple(answers) != ("explain",) else "")
                full = label.format(llm=model)
                extra = {"llm.preset": preset}
                if answer != "explain":            # on top of what the arm itself passes to the LLM decision model
                    extra.update({f"agents.{k}.llm_policy_params": {**{**base, **ov}.get(f"agents.{k}.llm_policy_params", {}),
                                                                    "rationale": ANSWERS[answer][1]} for k in LLM_KINDS})
                groups.setdefault(local_workers if is_local(preset) else llm_workers, {})[full] = {**base, **ov, **extra}
                order.append(full)
    total = len(order) * seeds
    started, done, rows = time.time(), 0, []
    for group_workers, arms in sorted(groups.items(), reverse=True):
        def progress(n, _total, offset=done):
            print(f"\r  {name}: {offset + n}/{total} runs", end="", file=sys.stderr, flush=True)

        rows += run_arms(exp.get("scenario", TESTBED), arms, list(range(1, seeds + 1)), workers=group_workers,
                         progress=progress)
        done = len(rows)
    print(file=sys.stderr)
    rows.sort(key=lambda row: order.index(row["arm"]))
    table = paired_markdown(rows, order[0], exp["metrics"])
    text = (f"### {name}\n\n**问题**：{exp['question']}\n\n{table}\n\n"
            f"{seeds} 个种子；括号内为与基准组的逐种子配对差值 ± 95% 置信区间半宽，`*` 表示区间不含 0。"
            f"用时 {time.time() - started:.0f} 秒。\n")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    default = not exp.get("llm") or (tuple(llms) == ("qwen-local",) and tuple(answers) == ("explain",))
    suffix = {("explain",): "", ("choice",): "__bare"}.get(tuple(answers), "__" + "+".join(answers))
    stem = name if default else f"{name}__{'+'.join(llms)}{suffix}"
    (out / f"{stem}.md").write_text(text, encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="experiments to run (default: all that need no LLM)")
    parser.add_argument("--seeds", type=int, default=3, help="3 while developing; more for results that are reported")
    parser.add_argument("--workers", type=int, default=6, help="runs at a time for arms without a language model")
    parser.add_argument("--llm", nargs="+", default=["qwen-local"], choices=[p for p in LLM_PRESETS if p != "mock"],
                        help="language model preset(s) for the {llm} arms")
    parser.add_argument("--llm-workers", type=int, default=8, help="runs at a time on a hosted model")
    parser.add_argument("--local-workers", type=int, default=4, help="runs at a time on a local model server")
    parser.add_argument("--answer", nargs="+", default=None, choices=list(ANSWERS),
                        help="answer format(s) for the {llm} arms (default: explain)")
    parser.add_argument("--no-rationale", action="store_true",
                        help="same as --answer choice: no reason with each choice (fewer output tokens, faster)")
    args = parser.parse_args()
    answers = args.answer or (["choice"] if args.no_rationale else ["explain"])
    names = args.names or [n for n, e in EXPERIMENTS.items() if not e.get("llm")]
    for name in names:
        if name not in EXPERIMENTS:
            parser.error(f"unknown experiment '{name}'. Available: {', '.join(EXPERIMENTS)}")
        print(run_experiment(name, args.seeds, args.workers, args.llm, args.llm_workers, answers, args.local_workers))


if __name__ == "__main__":
    main()
