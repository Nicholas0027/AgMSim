"""Designed experiments: each isolates one kind of agency and asks whether it changes outcomes.

    .venv/bin/python experiments/suite.py                 # all non-LLM experiments, 10 seeds
    .venv/bin/python experiments/suite.py toll --seeds 20
    .venv/bin/python experiments/suite.py llm_alignment --seeds 3                 # local Qwen (see README)
    .venv/bin/python experiments/suite.py llm_alignment --seeds 3 --llm deepseek  # hosted; runs in parallel
    .venv/bin/python experiments/suite.py llm_alignment --llm qwen-local deepseek # both models as arms
    .venv/bin/python experiments/suite.py llm_alignment --llm deepseek --answer explain reason

In the experiments that need a language model, an arm whose label contains ``{llm}`` is run once per
``--llm`` preset and ``--answer`` format (``choice``: the action only; ``explain``: the action, then a
reason -- the default; ``reason``: the reasoning first, then the action). Arms on a hosted model run
``--llm-workers`` at a time, arms on a local server ``--local-workers`` at a time (mlx_lm.server
batches simultaneous requests).

Every arm of an experiment runs on the same seeds, i.e. on identical travellers, and results are
reported as paired differences from the baseline arm (first arm). Tables go to
``experiments/results/<name>.md`` and raw rows to ``<name>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agmsim.config import LLM_PRESETS
from agmsim.experiment import paired_markdown, run_arms

HERE = Path(__file__).parent
LLM_NAMES = {"qwen-local": "Qwen2.5-7B", "deepseek": "DeepSeek-Flash", "deepseek-thinking": "DeepSeek-Flash（思考）",
             "deepseek-pro": "DeepSeek-V4-Pro"}
ANSWERS = {"choice": ("只给选择", False), "explain": ("先选择后说明", True), "reason": ("先推理后选择", "first")}
LLM_KINDS = ("traveler", "vehicle")      # kinds that the LLM experiments put on a language model

PLAN_METRICS = {"plans.score_usd_mean": "计划得分(美元)", "plans.late_min_mean": "人均迟到(分)",
                "plans.people_late": "迟到人数", "plans.activities_skipped": "主动取消的活动",
                "plans.reviews": "复查次数", "plans.edits": "改动次数",
                "plans.changed_when_worth_it": "该改时改了", "plans.changed_when_not": "不该改时改了",
                "plans.review_sensitivity": "灵敏度", "plans.review_regret_usd_mean": "每次复查少得(美元)",
                "travelers.trip_time_s_mean": "平均出行时间(秒)"}

# the three events of places.yaml, so that arms can have them one at a time
EVENTS = {
    "rain": {"type": "weather", "label": "heavy rain", "at_seconds": 1200, "until_seconds": 3600, "speed_factor": 0.75,
             "walk_factor": 0.85, "discomfort": 1.8},
    "match": {"type": "demand_surge", "poi": "stadium", "at_seconds": 1800, "until_seconds": 2700, "trips": 120},
    "block": {"type": "link_condition", "at_seconds": 2100, "until_seconds": 3900, "factor": 0.1, "links": [[14, 15], [15, 21]]},
}

# the vehicle algorithm's use of its memory: for 10 minutes after a change of route, change again only to save 5 minutes
SETTLE = {"settle_min": 10, "resettle_saving_min": 5}

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
                 "vehicles.empty_km_share": "空驶占比", "operator.advisories_published": "协调者通告"}

EXPERIMENTS = {
    "vehicle_agency": {
        "question": "同一座城，其他都不变，只把车的 agency 从 L0 调到 L4，系统表现怎么变？",
        "scenario": "city_multimodal",                # a packaged scenario, by its name
        "arms": VEHICLE_LEVELS,
        "metrics": LEVEL_METRICS,
    },
    "vehicle_agency_alone": {
        "question": "同一座城，其他主体都不作为（协调者不发通告，收费固定，公交优先常开，出行者听车的），只改车的等级呢？",
        "scenario": "city_multimodal",
        "base": {"agents.traveler.agency": "L0", "agents.bus.agency": "L0", "agents.evcs.agency": "L0",
                 "agents.operator.agency": "L0", "agents.operator.initial_toll": 0.6,
                 "agents.operator.initial_bus_priority": True},
        "arms": VEHICLE_LEVELS,
        "metrics": LEVEL_METRICS,
    },
    "incident": {
        "question": "事故发生时，是车有 agency 管用，还是运营方有 agency 管用？",
        "scenario": "incident.yaml",
        "arms": {
            "车 L0 · 运营方 L0": {},
            "车 L1 · 运营方 L0": {"agents.vehicle.agency": "L1"},
            "车 L0 · 运营方 L1": {"agents.operator.agency": "L1"},
            "车 L1 · 运营方 L1": {"agents.vehicle.agency": "L1", "agents.operator.agency": "L1"},
        },
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)", "travelers.trip_time_s_p90": "90 分位(秒)",
                    "vehicles.travel_time_ratio": "车程/自由流", "vehicles.reroutes": "改道次数",
                    "operator.advisories_published": "通告次数"},
    },
    "penetration": {
        "question": "会改道的车（L1）占比从 0 到 100%，装了的和没装的各自受益多少？",
        "scenario": "incident.yaml",
        "arms": {f"L1 占 {int(s * 100)}%": ({"agents.vehicle.mix": [{"share": s, "agency": "L1"}]} if s else {})
                 for s in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)",
                    "vehicles.travel_time_ratio_by_level.L0": "L0 车：车程/自由流",
                    "vehicles.travel_time_ratio_by_level.L1": "L1 车：车程/自由流",
                    "vehicles.reroutes": "改道次数"},
    },
    "toll": {
        "question": "有固定区域收费时，知道车主偏好（L2）的车会不会替不同车主做出不同选择？",
        "scenario": "toll.yaml",
        "arms": {
            "车 L1（只求快）": {},
            "车 L2（算法，按车主时间价值）": {"agents.vehicle.agency": "L2"},
            "车 L2（规则，仍只求快）": {"agents.vehicle.agency": "L2", "agents.vehicle.non_llm_policy": "rule"},
        },
        "metrics": {"vehicles.tolls_usd": "收费总额(美元)", "travelers.trip_time_s_mean": "平均出行时间(秒)",
                    "travelers.generalised_cost_usd_mean_low_vot": "低时间价值者广义成本",
                    "travelers.generalised_cost_usd_mean_high_vot": "高时间价值者广义成本",
                    "vehicles.owner_alignment_rate": "与车主利益一致率", "vehicles.owner_regret_usd_mean": "平均多花(美元)"},
    },
    "bottleneck": {
        "question": "过桥拥堵而公交有专用道时，车能否通过提议换乘（L3）改善出行？车主有没有回应权有何影响？",
        "scenario": "bottleneck.yaml",
        "arms": {
            "车 L2": {},
            "车 L3 · 车主默认听从": {"agents.vehicle.agency": "L3"},
            "车 L3 · 车主自行判断": {"agents.vehicle.agency": "L3", "agents.traveler.agency": "L2"},
        },
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)", "travelers.trip_time_s_p90": "90 分位(秒)",
                    "vehicles.suggestions": "提议次数", "vehicles.suggestions_accepted": "被接受",
                    "bus.boardings": "公交上车人次", "travelers.completion_rate": "完成率"},
    },
    "charging": {
        "question": "充电紧张时，车自己选站（L4）和充电站动态定价（L1）各自起什么作用？",
        "scenario": "charging.yaml",
        "arms": {
            "车 L3 · 站 L0": {},
            "车 L4 · 站 L0": {"agents.vehicle.agency": "L4"},
            "车 L3 · 站 L1": {"agents.evcs.agency": "L1"},
            "车 L4 · 站 L1": {"agents.vehicle.agency": "L4", "agents.evcs.agency": "L1"},
        },
        "metrics": {"vehicles.charge_queue_s_mean": "每次充电排队(秒)", "travelers.trip_time_s_mean": "平均出行时间(秒)",
                    "evcs.sessions_gini": "各站负荷不均(基尼)", "evcs.revenue_usd": "充电站收入(美元)",
                    "evcs.price_usd_per_kwh_mean_paid": "平均成交电价"},
    },
    "self_charging": {
        "question": "L4 的车在车主下车后自己去充电，会带来多少额外交通？",
        "scenario": "charging.yaml",
        "arms": {f"电量低于 {p}% 就去充": {"agents.vehicle.agency": "L4",
                                      "agents.vehicle.policy_params": {"top_up_below_pct": p}}
                 for p in (0, 15, 25, 100)},
        "metrics": {"vehicles.vkt_km": "车公里", "vehicles.empty_km_share": "空驶占比",
                    "travelers.trip_time_s_mean": "平均出行时间(秒)", "vehicles.charge_stops": "充电出行次数"},
    },
    "signals": {
        "question": "信号灯自己有 agency 管用，还是协调者指挥管用？两者需要同时有吗？",
        "scenario": "signals.yaml",
        "arms": {
            "灯 L0（固定配时）": {"agents.signal.agency": "L0"},
            "灯 L1（按本路口排队）": {},
            "灯 L2（固定配时，听协调者）· 协调者 L2": {"agents.signal.agency": "L2", "agents.signal.non_llm_policy": "rule",
                                         "agents.operator.agency": "L2"},
            "灯 L0 · 协调者 L2": {"agents.signal.agency": "L0", "agents.operator.agency": "L2"},
            "灯 L2（按排队，听协调者）· 协调者 L2": {"agents.signal.agency": "L2", "agents.operator.agency": "L2"},
        },
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)", "outcomes.traffic_speed_mph": "平均车速(英里/时)",
                    "outcomes.vht_hours": "VHT(小时)", "signals.wait_s_mean": "每次过灯等待(秒)",
                    "signals.stopped_share": "被红灯拦下的比例", "signals.switches": "切换次数",
                    "operator.signal_requests": "协调者指令数"},
    },
    "ondemand": {
        "question": "按需车有了 agency（改道、拒单、主动调度）之后，乘客等车和车队效率怎么变？",
        "scenario": "ondemand.yaml",
        "arms": {
            "车 L0（接所有派单，原地等）": {},
            "车 L1（按路况改道）": {"agents.ondemand.agency": "L1"},
            "车 L2 · 逐利（每分钟收入低就拒单）": {"agents.ondemand.agency": "L2", "agents.ondemand.objective": "profit"},
            "车 L2 · 服务（有拒单权但不用）": {"agents.ondemand.agency": "L2", "agents.ondemand.objective": "service"},
            "车 L4 · 服务（空闲时主动调度）": {"agents.ondemand.agency": "L4", "agents.ondemand.objective": "service"},
        },
        "metrics": {"outcomes.wait_time_s_mean": "平均等车(秒)", "travelers.completion_rate": "出行完成率",
                    "ondemand.cancelled": "乘客放弃叫车", "ondemand.declined": "拒单次数",
                    "ondemand.repositions": "调度次数", "ondemand.empty_km_share": "空驶里程占比",
                    "ondemand.profit_usd": "车队利润(美元)", "outcomes.vmt_miles": "VMT(英里)"},
    },
    "places": {
        "question": "下雨、散场、封路各自影响多大？同时发生时是简单相加吗？",
        "scenario": "places.yaml",
        "arms": {
            "平静的一天（无事件）": {"events": []},
            "只下雨": {"events": [EVENTS["rain"]]},
            "只有散场": {"events": [EVENTS["match"]]},
            "只封路": {"events": [EVENTS["block"]]},
            "三件事同时发生": {},
        },
        "metrics": {"outcomes.wait_time_s_mean": "平均等待(秒)", "travelers.trip_time_s_mean": "平均出行时间(秒)",
                    "outcomes.traffic_speed_mph": "平均车速(英里/时)", "travelers.mode_share.walk": "步行占比",
                    "travelers.mode_share.ride": "叫车占比", "ondemand.cancelled": "放弃叫车", "outcomes.vmt_miles": "VMT(英里)"},
    },
    "knowledge": {
        "question": "同样会改道的车（L1），看得多远、记不记得、互不互通，各自值多少？",
        "scenario": "crossing.yaml",
        "arms": {
            "只看得到附近": {},
            "什么都看不到": {"agents.vehicle.observation.traffic": "none"},
            "附近 + 记忆 10 分钟": {"agents.vehicle.memory.traffic_seconds": 600},
            "附近 + 车车互通": {"agents.vehicle.interaction.share_traffic": True},
            "附近 + 记忆 + 互通": {"agents.vehicle.memory.traffic_seconds": 600,
                             "agents.vehicle.interaction.share_traffic": True},
            "全网路况都知道": {"agents.vehicle.observation.traffic": "global"},
        },
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)", "travelers.trip_time_s_p90": "90 分位(秒)",
                    "vehicles.reroutes": "改道次数", "outcomes.vmt_miles": "VMT(英里)",
                    "knowledge.reports_heard": "听到的路况条数", "knowledge.share_of_agents_that_heard": "听到过的车占比"},
    },
    "knowledge_near": {
        "question": "车只看得到 450 米（上桥前看不到桥上）时，记忆和互通还能补回来多少？",
        "scenario": "crossing.yaml",
        "arms": {
            "只看 450 米": {"agents.vehicle.observation.radius_m": 450},
            "450 米 + 记忆 10 分钟": {"agents.vehicle.observation.radius_m": 450, "agents.vehicle.memory.traffic_seconds": 600},
            "450 米 + 车车互通": {"agents.vehicle.observation.radius_m": 450, "agents.vehicle.interaction.share_traffic": True},
            "450 米 + 记忆 + 互通": {"agents.vehicle.observation.radius_m": 450, "agents.vehicle.memory.traffic_seconds": 600,
                               "agents.vehicle.interaction.share_traffic": True},
            "全网路况都知道": {"agents.vehicle.observation.traffic": "global"},
        },
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)", "travelers.trip_time_s_p90": "90 分位(秒)",
                    "vehicles.reroutes": "改道次数", "outcomes.vmt_miles": "VMT(英里)",
                    "knowledge.reports_heard": "听到的路况条数", "knowledge.share_of_agents_that_heard": "听到过的车占比"},
    },
    "decision_memory": {
        "question": "车不记得路况时，只记得自己最近做过的决策，能不能少走冤枉路？记几条才够？",
        "scenario": "crossing.yaml",
        "arms": {
            "算法 · 不用记忆": {},
            **{f"会用记忆的算法 · 记得最近 {n} 条": {"agents.vehicle.policy_params": SETTLE, "agents.vehicle.memory.decisions": n}
               for n in (0, 1, 2, 4, 8)},
            "参照：算法 · 记路况 10 分钟": {"agents.vehicle.memory.traffic_seconds": 600},
        },
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)", "travelers.trip_time_s_p90": "90 分位(秒)",
                    "vehicles.reroutes": "改道次数", "vehicles.vehicles_with_3plus_reroutes": "改道 3 次以上的车",
                    "outcomes.vmt_miles": "VMT(英里)"},
    },
    "day_plan": {
        "question": "一天的计划被突发事故打乱时，当天就改计划（而不是像 MATSim 那样靠反复重演来学）能挽回多少？",
        "scenario": "evening.yaml",
        "arms": {
            "照计划执行": {},
            "当天调整 · 算法": {"agents.traveler.agency": ["adapt_plan"], "agents.traveler.non_llm_policy": "utility"},
            "无事故 · 照计划执行": {"interventions": []},
        },
        "metrics": PLAN_METRICS,
    },
    # ---- experiments that need a language model -------------------------------------------------
    "llm_day_plan": {
        "question": "当天改计划这件事交给 LLM 扮演的出行者，改得对不对？",
        "scenario": "evening.yaml",
        "llm": True,
        "arms": {
            "照计划执行": {},
            "当天调整 · 算法": {"agents.traveler.agency": ["adapt_plan"], "agents.traveler.non_llm_policy": "utility"},
            "当天调整 · {llm}": {"agents.traveler.agency": ["adapt_plan"], "agents.traveler.use_llm": True},
        },
        "metrics": {**PLAN_METRICS, "llm.calls": "LLM 调用", "llm.fallbacks": "回退"},
    },
    "llm_alignment": {
        "question": "同样是 L2，LLM 驱动的车替车主做的选择，与车主自身利益一致吗？",
        "scenario": "toll.yaml",
        "llm": True,
        "base": {"agents.traveler.count": 60, "simulation.duration_seconds": 2400, "demand.end_seconds": 1200,
                 "agents.vehicle.agency": "L2"},
        "arms": {
            "算法（按定义 100% 一致）": {},
            "规则（只求快）": {"agents.vehicle.non_llm_policy": "rule"},
            "{llm}": {"agents.vehicle.use_llm": True},
        },
        "metrics": {"vehicles.owner_choices": "选择次数", "vehicles.owner_alignment_rate": "与车主利益一致率",
                    "vehicles.owner_sensitivity": "灵敏度", "vehicles.owner_regret_usd_mean": "平均多花(美元)",
                    "vehicles.tolls_usd": "收费总额(美元)", "llm.calls": "LLM 调用", "llm.fallbacks": "回退"},
    },
    "llm_decision_memory": {
        "question": "LLM 开的车不记得路况时，让它看到自己最近的决策和结果（以及当时的理由），它会不会不再来回折返？",
        "scenario": "crossing.yaml",
        "llm": True,
        "arms": {
            "算法 · 不用记忆": {},
            "会用记忆的算法 · 记得最近 4 条": {"agents.vehicle.policy_params": SETTLE},
            "{llm} · 不记得": {"agents.vehicle.use_llm": True, "agents.vehicle.memory.decisions": 0},
            "{llm} · 记得做过什么和结果": {"agents.vehicle.use_llm": True},
            "{llm} · 同上，并被要求先回看记忆": {"agents.vehicle.use_llm": True,
                                      "agents.vehicle.llm_policy_params": {"reflect": True}},
            "{llm} · 还记得当时的理由，并被要求先回看记忆": {"agents.vehicle.use_llm": True, "agents.vehicle.memory.reasons": True,
                                            "agents.vehicle.llm_policy_params": {"reflect": True}},
            "{llm} · 路线带途经路口，记得做过什么和结果，并被要求先回看记忆": {
                "agents.vehicle.use_llm": True, "agents.vehicle.observation.route_roads": True,
                "agents.vehicle.llm_policy_params": {"reflect": True}},
            "{llm} · 路线带代号，记得做过什么和结果，并被要求先回看记忆": {
                "agents.vehicle.use_llm": True, "agents.vehicle.observation.route_names": True,
                "agents.vehicle.llm_policy_params": {"reflect": True}},
        },
        "metrics": {"travelers.trip_time_s_mean": "平均出行时间(秒)", "travelers.trip_time_s_p90": "90 分位(秒)",
                    "vehicles.reroutes": "改道次数", "vehicles.vehicles_with_3plus_reroutes": "改道 3 次以上的车",
                    "decisions.vehicle.decided": "被问到的决策", "llm.calls": "LLM 调用", "llm.fallbacks": "回退",
                    "llm.cost_usd": "花费(美元)"},
    },
    "llm_persuasion": {
        "question": "车提议换乘时附不附一句解释，会不会改变 LLM 扮演的车主的接受率？",
        "scenario": "bottleneck.yaml",
        "llm": True,
        "base": {"agents.vehicle.agency": "L3", "agents.traveler.agency": {"permissions": ["respond"]}},
        "arms": {
            "效用模型车主": {},
            "{llm} 车主 · 无解释": {"agents.traveler.use_llm": True, "agents.vehicle.policy_params": {"explain": False}},
            "{llm} 车主 · 有解释": {"agents.traveler.use_llm": True},
        },
        "metrics": {"vehicles.suggestions": "提议次数", "vehicles.suggestions_accepted": "被接受",
                    "travelers.trip_time_s_mean": "平均出行时间(秒)",
                    "travelers.generalised_cost_usd_mean": "广义成本(美元)", "llm.calls": "LLM 调用", "llm.fallbacks": "回退"},
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

        source = exp["scenario"] if not exp["scenario"].endswith(".yaml") else str(HERE / "scenarios" / exp["scenario"])
        rows += run_arms(source, arms, list(range(1, seeds + 1)), workers=group_workers, progress=progress)
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
