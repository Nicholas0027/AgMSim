"""Command-line interface: ``agmsim run | matrix | replay | explain | prompt | list | serve``."""
from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, list_scenarios
from .llm import LLMError


def _print_summary(metrics: dict) -> None:
    def line(title, d, keys):
        parts = [f"{k}={d[k]}" for k in keys if k in d and d[k] is not None]
        if parts:
            print(f"  {title:<10} " + "  ".join(parts))

    if "travelers" in metrics:
        line("travellers", metrics["travelers"], ["trips", "completion_rate", "mode_share", "trip_time_s_mean",
                                                 "generalised_cost_usd_mean", "gave_up_waiting"])
    for name, keys in (("outcomes", ["wait_time_s_mean", "vmt_miles", "vht_hours", "traffic_speed_mph"]),
                       ("plans", ["people", "completed", "score_usd_mean", "people_late", "late_min_mean",
                                  "activities_skipped", "reviews", "edits", "review_sensitivity"]),
                       ("vehicles", ["trips", "vkt_km", "travel_time_ratio", "reroutes", "suggestions",
                                     "suggestions_accepted", "charge_stops", "tolls_usd"]),
                       ("bus", ["boardings", "holds", "stops_skipped", "left_behind"]),
                       ("evcs", ["sessions", "energy_kwh", "revenue_usd", "utilisation"]),
                       ("operator", ["advisories_published", "bus_priority_time_share", "toll_usd_final",
                                     "toll_revenue_usd"])):
        if name in metrics:
            line(name, metrics[name], keys)
    print("  decisions  " + "  ".join(
        f"{k}[{d['agency_level']},{d['policy']}]: asked {d['decided']}, default-by-rule {d['forced']}, "
        f"used authority {d['exercised']}" for k, d in metrics["decisions"].items()))
    llm = metrics["llm"]
    if llm["calls"] or llm["replayed"]:
        backends = ", ".join(f"{b['backend']}:{b['model'] or '-'}" for b in llm["backends"])
        print(f"  llm        calls={llm['calls']} replayed={llm['replayed']} tokens={llm['input_tokens']}/{llm['output_tokens']} "
              f"fallbacks={llm['fallbacks']} cost_usd={llm['cost_usd']} waited={llm['deciding_wall_s']}s [{backends}]")
        if any(b["backend"] == "mock" for b in llm["backends"]):
            print("             note: the mock backend is an offline stand-in, not a language model")


def cmd_run(args) -> int:
    from .engine import Simulation
    sim = Simulation(args.config, overrides=args.set)
    kinds = ", ".join(f"{name}×{sum(1 for s in sim.specs.values() if s.kind == name)}" for name in sim.kinds)
    print(f"{sim.cfg.name}: {kinds}; {sim.n_ticks} ticks of {sim.dt:g} s; seed {sim.cfg.simulation.seed}")

    def progress(tick, total):
        if not args.quiet and tick % max(1, total // 20) == 0:
            print(f"\r  {100 * tick // total:3d}%", end="", flush=True)

    result = sim.run(progress)
    if not args.quiet:
        print("\r", end="")
    _print_summary(result.metrics)
    print(f"  trace digest {result.trace_digest}" + (f"   saved to {result.run_dir}" if result.run_dir else ""))
    if args.json:
        print(json.dumps(result.metrics, indent=2))
    return 0


def cmd_matrix(args) -> int:
    from .experiment import LLM_2X2, format_table, run_matrix
    factors = {}
    if args.llm_2x2:
        factors.update(LLM_2X2)
    for item in args.factor or []:
        key, sep, values = item.partition("=")
        if not sep:
            raise ConfigError(f"--factor '{item}' must look like key.path=v1,v2,...")
        import yaml
        factors[key.strip()] = [yaml.safe_load(v) for v in values.split(",")]
    if not factors:
        raise ConfigError("give at least one --factor key=v1,v2 or use --llm-2x2")
    seeds = list(range(args.seed0, args.seed0 + args.seeds))

    def progress(done, total):
        print(f"\r  {done}/{total} runs", end="", flush=True)

    out = run_matrix(args.config, factors, seeds, overrides=args.set, workers=args.workers,
                     out_dir=args.out, progress=progress)
    print("\r" + format_table(out["summary"], args.metric or None))
    if args.out:
        print(f"saved to {args.out}/runs.csv and summary.json")
    return 0


def cmd_replay(args) -> int:
    from .replay import replay
    report = replay(args.run)
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] else 1


def cmd_explain(args) -> int:
    """Print the decision history of one agent from a trace."""
    from .recorder import read_trace
    shown = 0
    for rec in read_trace(args.run):
        if rec["type"] != "decision" or rec["agent"] != args.agent:
            continue
        d = rec["decision"]
        tag = "no authority -> default" if rec["source"] == "forced" else rec["source"]
        print(f"tick {rec['tick']:>5}  {rec['trigger']:<10} {d['action']} {json.dumps(d.get('args') or {})}  [{tag}]")
        if rec.get("permitted"):
            print(f"            permitted: {', '.join(rec['permitted'])}")
        for key in ("rationale",):
            if d.get(key):
                print(f"            rationale: {d[key]}")
        for key in ("error", "result", "note"):
            if rec.get(key):
                print(f"            {key}: {rec[key]}")
        if args.observations and rec.get("obs"):
            print("            context: " + json.dumps(rec["obs"]["context"], ensure_ascii=False))
        shown += 1
    if not shown:
        print(f"no decisions recorded for agent '{args.agent}'")
    return 0


def cmd_prompt(args) -> int:
    """Print what a language model is told for one kind of agent, or for every kind of the scenario."""
    from .prompts import told
    kinds = told(args.config, args.set)
    if args.kind and args.kind not in kinds:
        raise ValueError(f"the scenario has no '{args.kind}'; its kinds of agent: {', '.join(kinds)}")
    for name, info in kinds.items():
        if args.kind in (None, name):
            print(f"===== {name} =====")
            print(f"goal: {info['goal'] or '(none: the kind states no objective)'}")
            print(info["system"] if info["system"] is not None else
                  f"(decided by the policy '{info['llm_policy']}', which writes its own prompt)")
            print()
    return 0


def cmd_list(args) -> int:
    from . import kinds as _k, policies as _p  # noqa: F401
    from .agency import LEVEL_NAMES
    from .demand import DEMANDS
    from .kinds.base import KINDS
    from .llm import LLM_BACKENDS
    from .network import NETWORKS
    from .policies.base import available_policies
    print("scenarios:   " + ", ".join(list_scenarios()))
    print("networks:    " + ", ".join(NETWORKS.names()))
    print("demand:      " + ", ".join(DEMANDS.names()))
    print("llm backends: " + ", ".join(LLM_BACKENDS.names()))
    from .config import LLM_PRESETS
    print("llm presets:  " + ", ".join(f"{name} ({p.get('model') or p['backend']})" for name, p in LLM_PRESETS.items()))
    print("agency levels: " + "; ".join(f"{k} {v}" for k, v in LEVEL_NAMES.items()))
    for name in KINDS.names():
        kind = KINDS.get(name)
        print(f"\n{name} ({kind.title}; {kind.side})")
        print("  policies:    " + ", ".join(available_policies(name)))
        for level in sorted(kind.agency.levels):
            print(f"  {level}: " + (", ".join(kind.agency.levels[level]) or "(no authority: default actions only)"))
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    serve(args.host, args.port)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agmsim", description="AgMSim: agentic mobility simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run one simulation")
    p.add_argument("config", help="YAML file or packaged scenario name")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--json", action="store_true", help="print all metrics as JSON")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("matrix", help="run a factorial experiment over seeds")
    p.add_argument("config")
    p.add_argument("--factor", action="append", metavar="KEY=V1,V2", help="factor and its levels (repeatable)")
    p.add_argument("--llm-2x2", action="store_true", help="add the human x machine LLM switch as factors")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--metric", action="append", help="metric to show (repeatable)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", help="directory for runs.csv and summary.json")
    p.set_defaults(fn=cmd_matrix)

    p = sub.add_parser("replay", help="re-execute a recorded run from its decisions and verify it")
    p.add_argument("run", help="run directory")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("explain", help="show why an agent did what it did")
    p.add_argument("run")
    p.add_argument("agent")
    p.add_argument("--observations", action="store_true", help="also print what the agent saw")
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("prompt", help="show what a language model is told for a kind of agent")
    p.add_argument("config", help="YAML file or packaged scenario name")
    p.add_argument("kind", nargs="?", help="one kind of agent (default: every kind of the scenario)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.set_defaults(fn=cmd_prompt)

    p = sub.add_parser("list", help="list scenarios, agent kinds, agency levels and plugins")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("serve", help="start the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (ConfigError, LLMError, KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
