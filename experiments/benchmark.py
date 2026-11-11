"""Decision benchmark: controlled decision points with a known answer, the same for every model.

    .venv/bin/python experiments/benchmark.py                                 # reference models only
    .venv/bin/python experiments/benchmark.py --llm deepseek                  # + DeepSeek-Flash, three answer formats
    .venv/bin/python experiments/benchmark.py route_delegation --llm qwen-local --answer reason
    .venv/bin/python experiments/benchmark.py --report                        # rebuild the tables only

Tasks are defined in ``agmsim/benchmark.py``. A language model is evaluated in up to three answer
formats (``--answer``): ``choice`` (the action only), ``explain`` (the action, then one sentence of
reason -- the simulator's default) and ``reason`` (the reasoning first, then the action).

Each (task, model, format) result is stored with every item's answer in
``experiments/results/benchmark/``; the tables in ``experiments/results/benchmark.md`` are rebuilt
from all stored results that belong to the current item set, so models can be added one at a time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agmsim.benchmark import TASKS, evaluate, get_task, rate
from agmsim.config import LLM_PRESETS
from agmsim.llm import LLMError

from suite import ANSWERS, LLM_NAMES, is_local       # answer formats: name -> (label, value of ``rationale``)

HERE = Path(__file__).parent
STORE = HERE / "results" / "benchmark"
BASELINE_NAMES = {"rule": "规则（不看偏好）", "algorithm": "算法", "utility": "效用模型",
                  "algorithm_fare": "算法（计入票价）"}

TEXT = {
    "route_delegation": {
        "title": "按车主偏好选路（车，L2）",
        "question": "一条收费但更快的路线和一条免费但更慢的路线，车替车主选。每道题有一个临界时间价值（每省一小时要多花多少钱），"
                    "车主的时间价值设为临界值的 1/3 到 3 倍：高于临界值就该付费，低于就不该付。192 题，该付与不该付各半。"
                    "“易、中、难”按时间价值离临界值 3 倍、2 倍、1.3 倍划分；一半的题只列这两条路线，另一半多列两条更慢的收费路线。",
        "columns": {"sensitivity": "灵敏度", "paid_when_worth_it": "该付时付了", "paid_when_not": "不该付时付了",
                    "accuracy": "选对", "accuracy_easy": "易", "accuracy_medium": "中", "accuracy_hard": "难",
                    "accuracy_two_routes": "两条路线", "accuracy_four_routes": "四条路线",
                    "regret_usd_mean": "平均多花(美元)", "valid": "有效回复"},
        "curves": {"paid_by_ratio": "选择付费的比例，按“车主时间价值 ÷ 临界值”分列（理想：左三列 0%，右三列 100%）"},
    },
    "suggestion_response": {
        "title": "车主回应换乘提议（出行者）",
        "question": "车提议停车换乘公交，车主接受或拒绝。提议的各项时间和票价始终给出；变化的是比继续开车早到（或晚到）几分钟、"
                    "车主的时间价值，以及车的留言：没有、如实、或把节省的时间多说 8 分钟。即使把等车和步行算得更难熬仍然划算的记为“值得”，"
                    "只算时间和票价都不划算的记为“不值得”，两者之间不计分。171 题。",
        "columns": {"sensitivity": "灵敏度（无留言）", "accepted_when_worth_it": "值得时接受",
                    "accepted_when_not": "不值得时接受", "sensitivity_with_truthful_message": "灵敏度（如实留言）",
                    "misled_by_overstatement": "被夸大留言误导", "accepted_slower_no_message": "更慢也接受（无留言）",
                    "accepted_slower_overstated": "更慢也接受（夸大留言）", "valid": "有效回复"},
        "curves": {"accepted_by_gain_none": "接受比例，按早到分钟数分列（无留言）",
                   "accepted_by_gain_overstated": "接受比例，按早到分钟数分列（夸大留言）"},
    },
    "transfer_proposal": {
        "title": "车是否提议换乘（车，L3）",
        "question": "车可以继续开，也可以向车主提议停车换乘。仿真只列出比开车快的换乘，所以总能省时间；值不值票价取决于车主的时间价值"
                    "（分类同上）。一半的题里最快的开车路线不是当前路线。108 题。",
        "columns": {"sensitivity": "灵敏度", "proposed_when_worth_it": "值得时提议", "proposed_when_not": "不值得时提议",
                    "proposed_worth_no_quicker_route": "值得，当前路线最快", "proposed_worth_quicker_route": "值得，另有更快路线",
                    "took_quicker_route": "不提议时改走更快路线", "valid": "有效回复"},
        "curves": {"proposed_by_gain_vot8": "提议比例，按省时分钟数分列（车主时间价值 8 美元/小时）",
                   "proposed_by_gain_vot50": "提议比例，按省时分钟数分列（车主时间价值 50 美元/小时）"},
    },
}
SIGNED = {"sensitivity", "sensitivity_with_truthful_message", "misled_by_overstatement"}   # differences, not shares
INTERVALS = {"sensitivity": "sensitivity_ci95", "misled_by_overstatement": "misled_ci95"}


def cell(key: str, scores: dict) -> str:
    value = scores.get(key)
    if value is None:
        return "–"
    if key == "regret_usd_mean":
        return f"{value:.2f}"
    if key in SIGNED:
        half = scores.get(INTERVALS.get(key, ""))
        return f"{value:+.2f}" + (f" ± {half:.2f}" if half else "")
    return f"{100 * value:.0f}%"


def stored(task) -> tuple:
    """Stored model results of this task that belong to the current item set (in a stable order), and how many do not."""
    digest, out, stale = task.digest(), [], 0
    for path in sorted(STORE.glob(f"{task.name}__*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["items_digest"] != digest or not report.get("complete", True):
            stale += 1
            continue
        # scores are derived from the stored answers, so a score added later needs no new model run
        report["scores"] = {"valid": rate(report["rows"], lambda r: r["valid"]), **task.score(report["rows"])}
        report["curves"] = task.curves(report["rows"])
        out.append(report)
    order = {name: i for i, name in enumerate(LLM_NAMES)}
    out.sort(key=lambda r: (order.get(r["preset"], 99), list(ANSWERS).index(r["answer"])))
    return out, stale


def build_report() -> str:
    parts = ["# 决策基准结果\n",
             "每个任务把同一批受控的决策点交给每个决策模型。题目取自仿真里真实出现的观测（格式与仿真完全一致），"
             "数值按析因设计替换，标准答案由当事人自己的时间价值算出。\n",
             "**灵敏度** = 该行动对当事人有利时采取它的比例 − 不利时采取它的比例。完全不看偏好的模型（总是做、从不做、随机做）得 0，"
             "完全按偏好行事得 1。± 后是按题目数估计的 95% 区间半宽。\n"]
    summary, usage_rows = {}, []
    for name in TASKS.names():
        task, text = get_task(name), TEXT[name]
        reports = [(BASELINE_NAMES.get(label, label), evaluate(task, overrides))
                   for label, overrides in task.baselines.items()]
        models, stale = stored(task)
        for report in models:
            label = f"{LLM_NAMES.get(report['preset'], report['preset'])} · {ANSWERS[report['answer']][0]}"
            reports.append((label, report))
            u = report["usage"]
            usage_rows.append(f"| {text['title']} | {label} | {report['items']} | {u['seconds']:.0f} | "
                              f"{u['latency_ms_mean'] / 1000:.1f} | {u['output_tokens'] / max(1, u['calls']):.0f} | "
                              f"{u['cost_usd']:.3f} | {report.get('date', '')} |")
        columns = text["columns"]
        lines = [f"## {text['title']}\n", text["question"] + "\n",
                 "| 决策模型 | " + " | ".join(columns.values()) + " |", "| - |" + " - |" * len(columns)]
        for label, report in reports:
            lines.append(f"| {label} | " + " | ".join(cell(key, report["scores"]) for key in columns) + " |")
            summary.setdefault(label, {})[name] = report["scores"]
        for key, title in text["curves"].items():
            levels = list(reports[0][1]["curves"][key])
            lines += ["", f"{title}：\n", "| 决策模型 | " + " | ".join(levels) + " |", "| - |" + " - |" * len(levels)]
            for label, report in reports:
                values = report["curves"][key]
                lines.append(f"| {label} | " + " | ".join("–" if values[lv] is None else f"{100 * values[lv]:.0f}%"
                                                         for lv in levels) + " |")
        if stale:
            lines.append(f"\n（另有 {stale} 份已存结果属于旧版题目或只跑了部分题目，未列入。）")
        parts.append("\n".join(lines) + "\n")

    names = TASKS.names()
    head = ["## 汇总：各任务的灵敏度\n", "| 决策模型 | " + " | ".join(TEXT[n]["title"] for n in names)
            + " | 被夸大留言误导 |", "| - |" + " - |" * (len(names) + 1)]
    for label, by_task in summary.items():
        cells = [cell("sensitivity", by_task[n]) if n in by_task else "–" for n in names]
        misled = by_task.get("suggestion_response")
        cells.append(cell("misled_by_overstatement", misled) if misled else "–")
        head.append(f"| {label} | " + " | ".join(cells) + " |")
    parts.insert(3, "\n".join(head) + "\n")
    if usage_rows:
        parts.append("\n".join(["## 语言模型的用量\n",
                                "| 任务 | 模型 · 回答格式 | 题数 | 用时(秒) | 每题延迟(秒) | 每题输出 token | 估算费用(美元) | 日期 |",
                                "| - | - | - | - | - | - | - | - |", *usage_rows]) + "\n")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", help=f"tasks to run (default: all of {', '.join(TASKS.names())})")
    parser.add_argument("--llm", nargs="+", default=[], choices=[p for p in LLM_PRESETS if p != "mock"],
                        help="language model preset(s) to evaluate")
    parser.add_argument("--answer", nargs="+", default=list(ANSWERS), choices=list(ANSWERS),
                        help="answer format(s) for the language models")
    parser.add_argument("--workers", type=int, default=None,
                        help="simultaneous calls (default: 24 for a hosted model, 8 for a local server)")
    parser.add_argument("--limit", type=int, default=None, help="only this many items per task (not stored)")
    parser.add_argument("--report", action="store_true", help="only rebuild the tables from stored results")
    args = parser.parse_args()
    for name in args.tasks:
        if name not in TASKS.names():
            parser.error(f"unknown task '{name}'. Available: {', '.join(TASKS.names())}")
    STORE.mkdir(parents=True, exist_ok=True)

    if not args.report:
        for name in args.tasks or TASKS.names():
            task = get_task(name)
            for preset in args.llm:
                for answer in args.answer:
                    label = f"{name} · {preset} · {answer}"
                    overrides = {f"agents.{task.kind}.use_llm": True, "llm.preset": preset,
                                 "llm.max_concurrency": args.workers or (8 if is_local(preset) else 24),
                                 f"agents.{task.kind}.llm_policy_params": {"rationale": ANSWERS[answer][1]}}

                    def progress(n, total, label=label):
                        if sys.stderr.isatty():
                            print(f"\r  {label}: {n}/{total}", end="", file=sys.stderr, flush=True)

                    try:
                        report = evaluate(task, overrides, limit=args.limit, progress=progress)
                    except LLMError as exc:
                        raise SystemExit(f"\n{label}: {exc}")
                    if sys.stderr.isatty():
                        print(file=sys.stderr)
                    report.update(preset=preset, answer=answer, date=time.strftime("%Y-%m-%d"))
                    s, u = report["scores"], report["usage"]
                    print(f"  {label}: sensitivity {s['sensitivity']}, valid {s['valid']}, "
                          f"{u['seconds']:.0f} s, est. {u['cost_usd']:.3f} USD", flush=True)
                    if report["complete"]:
                        (STORE / f"{name}__{preset}__{answer}.json").write_text(
                            json.dumps(report, ensure_ascii=False), encoding="utf-8")

    text = build_report()
    (HERE / "results" / "benchmark.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
