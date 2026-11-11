"""How fast is a language model as a decision model?

    .venv/bin/python experiments/bench_llm.py --llm qwen-local deepseek
    .venv/bin/python experiments/bench_llm.py --llm deepseek --n 48 --concurrency 1 8 32

Real decision points are taken from a non-LLM run of a scenario (spread over all kinds of agent
and all triggers) and each is sent to the model exactly as the simulator would send it. Reported per
setting: time per decision, decisions per second, tokens, the share of replies that were a valid
permitted action, and estimated cost. A few untimed warm-up calls open the connections first.

The table goes to ``experiments/results/bench_llm.md``.
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agmsim.actions import validate_decision
from agmsim.config import LLM_PRESETS, LLMCfg
from agmsim.engine import Simulation
from agmsim.llm import LLMError, LLMRequest, build_llm_client
from agmsim.policies.llm_policy import (build_system_prompt, build_user_prompt, observation_from_prompt,
                                        parse_decision_text)

HERE = Path(__file__).parent


def collect_prompts(scenario: str, n: int, rationale: bool) -> list:
    """``n`` (system, user) prompt pairs from recorded decision points, spread over (kind, trigger)."""
    sim = Simulation(scenario, overrides={"record.dir": None})
    result = sim.run()
    groups: dict[tuple, list] = {}
    for rec in result.records:
        if rec.get("type") == "decision" and "obs" in rec:
            groups.setdefault((rec["kind"], rec["trigger"]), []).append(rec)
    quota = math.ceil(n / len(groups))
    picked = []
    for j in range(quota):                      # round-robin over groups, evenly spaced within a group
        for key in sorted(groups):
            items = groups[key]
            if j < min(quota, len(items)):
                picked.append(items[int(j * len(items) / min(quota, len(items)))])
    prompts = [(build_system_prompt(sim.kinds[rec["kind"]], rationale), build_user_prompt(rec["obs"], (), rationale))
               for rec in picked[:n]]
    return prompts, sorted(groups)


def measure(preset: str, prompts: list, concurrency: int, keep_alive: bool = True) -> dict:
    client = build_llm_client(LLMCfg(preset=preset, max_concurrency=concurrency,
                                     params={} if keep_alive else {"keep_alive": False}))

    def ask(prompt):
        system, user = prompt
        try:
            response = client.complete(LLMRequest(system, user))
        except LLMError as exc:
            return None, f"{type(exc).__name__}: {exc}"
        try:
            decision, error = validate_decision(parse_decision_text(response.text), observation_from_prompt(user).actions)
        except ValueError as exc:
            decision, error = None, str(exc)
        return response, None if decision is not None else error

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        warm = list(pool.map(ask, prompts[-concurrency:]))             # untimed: opens the connections
        failed = next((error for response, error in warm if response is None), None)
        if failed:
            raise SystemExit(f"{preset}: {failed}")
        started = time.perf_counter()
        results = list(pool.map(ask, prompts))
        wall = time.perf_counter() - started
    answered = [response for response, _ in results if response is not None]
    lat = sorted(r.latency_ms for r in answered)
    n_in = sum(r.input_tokens for r in answered)
    row = {
        "model": client.cfg.model, "concurrency": concurrency, "n": len(prompts), "wall_s": wall,
        "per_s": len(prompts) / wall,
        "lat_mean": sum(lat) / len(lat) if lat else float("nan"),
        "lat_p50": lat[len(lat) // 2] if lat else float("nan"),
        "lat_p95": lat[int(0.95 * (len(lat) - 1))] if lat else float("nan"),
        "in_tok": n_in / max(1, len(answered)),
        "out_tok": sum(r.output_tokens for r in answered) / max(1, len(answered)),
        "cached": sum(r.cached_input_tokens for r in answered) / max(1, n_in),
        "valid": sum(error is None for _, error in results) / len(results),
        "failed": len(results) - len(answered),
        "usd_per_1000": 1000 * sum(client.cost(r) for r in answered) / max(1, len(answered)),
        "first_error": next((error for _, error in results if error), ""),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", nargs="+", default=["qwen-local"], choices=[p for p in LLM_PRESETS if p != "mock"])
    parser.add_argument("--scenario", default="city_multimodal")
    parser.add_argument("--n", type=int, default=32, help="decision points per setting")
    parser.add_argument("--concurrency", type=int, nargs="+", default=None,
                        help="simultaneous calls to try (default: 1 and 8 for a local server; 1, 8 and 32 for a hosted one)")
    parser.add_argument("--no-keep-alive-check", action="store_true",
                        help="skip the extra row that opens a new connection per call")
    args = parser.parse_args()

    rows, kinds = [], []
    for rationale in (True, False):
        prompts, kinds = collect_prompts(args.scenario, args.n, rationale)
        for preset in args.llm:
            url = LLM_PRESETS[preset].get("base_url") or ""
            local = "127.0.0.1" in url or "localhost" in url
            for c in args.concurrency or ([1, 8] if local else [1, 8, 32]):
                row = measure(preset, prompts, c)
                rows.append({"preset": preset, "rationale": rationale, "note": "", **row})
                print(f"  {preset} rationale={rationale} concurrency={c}: {row['lat_mean']:.0f} ms/decision, "
                      f"{row['per_s']:.2f} decisions/s, valid {row['valid']:.0%}", flush=True)
            if rationale and not local and not args.no_keep_alive_check:
                row = measure(preset, prompts, 1, keep_alive=False)
                rows.append({"preset": preset, "rationale": rationale, "note": "新建连接", **row})
    lines = ["| 模型 | 附理由 | 并发 | 每次决策(毫秒) 均值 / 中位 / 95分位 | 每秒决策数 | 输入 / 输出 token | 缓存命中 | 有效回复 | 每千次决策(美元) |",
             "| - | - | - | - | - | - | - | - | - |"]
    for r in rows:
        note = f"（{r['note']}）" if r["note"] else ""
        lines.append(f"| {r['model']}{note} | {'是' if r['rationale'] else '否'} | {r['concurrency']} | "
                     f"{r['lat_mean']:.0f} / {r['lat_p50']:.0f} / {r['lat_p95']:.0f} | {r['per_s']:.2f} | "
                     f"{r['in_tok']:.0f} / {r['out_tok']:.0f} | {r['cached']:.0%} | {r['valid']:.0%} | "
                     f"{r['usd_per_1000']:.3f} |")
    errors = sorted({r["first_error"] for r in rows if r["first_error"]})
    text = ("### bench_llm\n\n" + "\n".join(lines) + "\n\n"
            f"场景 {args.scenario}，每组 {args.n} 个决策点，覆盖 {len(kinds)} 种（主体, 触发原因）组合；"
            f"测于 {time.strftime('%Y-%m-%d %H:%M')}。延迟为单次调用的端到端耗时；"
            "“每秒决策数”为该并发下的总吞吐。费用按预设中的估算单价，本地模型为 0。\n")
    if errors:
        text += "\n无效或失败回复的首个原因：\n" + "\n".join(f"- {e[:200]}" for e in errors) + "\n"
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "bench_llm.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
