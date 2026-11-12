"""Core guarantees: configuration, agency enforcement, determinism, replay, the LLM decision path."""
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from agmsim import Config, ConfigError, Decision, Simulation, register_policy
from agmsim.actions import ActionSpec, Param, validate_decision
from agmsim.agency import AgencyError, parse_agency
from agmsim.entities import TState
from agmsim.kinds.base import KINDS
from agmsim.network import build_network
from agmsim.policies.llm_policy import parse_decision_text
from agmsim.replay import replay

QUIET = ["record.dir=null"]
SCENARIOS = Path(__file__).parent.parent / "experiments" / "scenarios"


def run(scenario="demo_3x3", overrides=(), **kwargs):
    sim = Simulation(scenario, overrides=[*QUIET, *overrides], **kwargs)
    return sim, sim.run()


# ---------------------------------------------------------------- configuration


def test_unknown_config_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"simulation": {"sed": 1}})


def test_two_class_shorthand_maps_to_kinds():
    cfg = Config.from_dict({"human_agents": {"use_llm": True, "can_replan": True},
                            "machine_agents": {"use_llm": True, "can_reroute": True},
                            "agents": {"bus": {"use_llm": False}}})
    assert cfg.agents.traveler.use_llm and cfg.agents.vehicle.use_llm and cfg.agents.evcs.use_llm
    assert cfg.agents.bus.use_llm is False                      # an explicit per-kind setting wins
    assert cfg.agents.traveler.agency == {"can_replan": True}
    assert cfg.agents.vehicle.agency == {"can_reroute": True}


def test_override_of_shorthand_reaches_every_machine_kind():
    cfg = Config.load("city_multimodal", ["machine_agents.use_llm=true", "llm.backend=mock"])
    assert all(getattr(cfg.agents, k).use_llm for k in ("vehicle", "bus", "evcs", "operator"))
    assert cfg.agents.traveler.use_llm is False


# ---------------------------------------------------------------- agency


def test_agency_levels_are_cumulative_and_parse_in_all_forms():
    spec = KINDS.get("vehicle").agency
    previous = set()
    for level in ("L0", "L1", "L2", "L3", "L4"):
        current = set(parse_agency(spec, level).permissions)
        assert previous <= current
        previous = current
    assert parse_agency(spec, "AgV-2").level == "L2"
    assert parse_agency(spec, {"level": "L1", "can_self_manage": True}).permissions == {"reroute", "self_manage"}
    assert parse_agency(spec, ["reroute", "preferences"]).level == "L2"
    with pytest.raises(AgencyError):
        parse_agency(spec, {"can_fly": True})


def test_decision_validation_rejects_what_is_not_permitted():
    specs = [ActionSpec("choose_route", "", (Param("route", "integer", (0, 1)),)), ActionSpec("stay_parked")]
    ok, err = validate_decision({"action": "choose_route", "route": "1"}, specs)
    assert err is None and ok.args == {"route": 1}
    assert validate_decision("charge", specs)[0] is None                       # action not on offer
    assert validate_decision(Decision("choose_route", {"route": 7}), specs)[0] is None
    ok, _ = validate_decision({"action": "stay_parked", "args": {"message": "hi"}}, specs)
    assert ok.args == {} and ok.info["dropped_args"] == ["message"]            # cannot invent parameters


def test_without_authority_the_decision_model_is_never_consulted():
    calls = []

    @register_policy("spy", role="vehicle")
    def spy(obs):
        calls.append(obs.trigger)
        return obs.default

    sim, res = run(overrides=["agents.vehicle.agency=L0", "agents.vehicle.non_llm_policy=spy"])
    assert calls == [] and res.metrics["decisions"]["vehicle"]["decided"] == 0
    assert res.metrics["vehicles"]["reroutes"] == 0
    calls.clear()
    sim, res = run(overrides=["agents.vehicle.agency=L1", "agents.vehicle.non_llm_policy=spy"])
    assert calls and set(calls) == {"route"}


def test_invalid_output_falls_back_to_the_default_action():
    @register_policy("rogue", role="vehicle")
    def rogue(obs):
        return {"action": "teleport", "args": {"node": 0}}

    sim, res = run(overrides=["agents.vehicle.non_llm_policy=rogue"])
    stats = res.metrics["decisions"]["vehicle"]
    assert stats["decided"] > 0 and stats["invalid"] == stats["decided"] and stats["exercised"] == 0
    assert res.metrics["vehicles"]["reroutes"] == 0


def test_agency_can_change_during_a_run():
    sim, res = run(overrides=['interventions=[{type: set_agency, at_seconds: 600, kind: vehicle, agency: L0}]'])
    assert all(p.level == "L0" for a, p in sim.agency.items() if sim.specs[a].kind == "vehicle")
    assert any(r["type"] == "event" and r["event"] == "agency_changed" for r in res.records)


# ---------------------------------------------------------------- dynamics


def test_every_trip_ends_in_exactly_one_outcome():
    sim, res = run("city_multimodal")
    m = res.metrics["travelers"]
    assert m["completed"] + m["abandoned"] + m["unfinished"] == m["trips"] == 400
    for tr in sim.world.travelers.values():
        if tr.state == TState.ARRIVED:
            assert tr.t_arrival >= tr.t_need and tr.mode in ("car", "bus", "walk", "car+bus")
    v = res.metrics["vehicles"]
    assert v["travel_time_ratio"] >= 1.0            # nobody beats free flow
    assert v["battery_depleted_ticks"] == 0


def test_congestion_slows_links_and_incidents_clear():
    from agmsim.config import NetworkCfg
    net = build_network(NetworkCfg(rows=2, cols=2))
    link = net.links[0]
    free = link.cur_time
    net.set_flows({0: 3}, spacing_m=7.0, scale=0.1)
    assert link.cur_time > free
    net.set_flows({}, spacing_m=7.0, scale=0.1)
    net.set_factor(0, 0.2)
    assert net.disrupted() and abs(link.cur_time - free / 0.2) < 1e-6
    net.set_factor(0, 1.0)
    assert not net.disrupted()


def test_same_seed_same_run_and_common_random_numbers():
    _, a = run()
    _, b = run()
    assert a.trace_digest == b.trace_digest
    _, c = run(overrides=["simulation.seed=43"])
    assert c.trace_digest != a.trace_digest
    # changing the vehicles' agency must not change who travels where and who owns a car
    s0, _ = run(overrides=["agents.vehicle.agency=L0"])
    s4, _ = run(overrides=["agents.vehicle.agency=L4"])
    trips = lambda s: [(t.id, t.origin, t.destination, t.t_need, t.car, t.persona.value_of_time)
                       for t in s.world.travelers.values()]
    assert trips(s0) == trips(s4)


def test_recorded_run_replays_identically(tmp_path):
    sim = Simulation("city_multimodal", overrides=[f"record.dir={tmp_path}", "record.run_name=r",
                                                   "human_agents.use_llm=true", "machine_agents.use_llm=true",
                                                   "llm.backend=mock", "llm.params={error_rate: 0.1}"])
    res = sim.run()
    assert res.metrics["llm"]["calls"] > 0 and res.metrics["llm"]["fallbacks"] > 0
    report = replay(tmp_path / "r")
    assert report["identical"] and report["unused_decisions"] == 0
    # tampering with one recorded decision is detected
    trace = (tmp_path / "r" / "trace.jsonl").read_text().splitlines()
    for i, line in enumerate(trace):
        rec = json.loads(line)
        if rec["type"] == "decision" and rec["kind"] == "traveler" and rec["decision"]["args"].get("option") == "car":
            rec["decision"]["args"]["option"] = "walk"
            trace[i] = json.dumps(rec)
            break
    (tmp_path / "r" / "trace.jsonl").write_text("\n".join(trace) + "\n")
    assert not replay(tmp_path / "r")["identical"]


# ---------------------------------------------------------------- the LLM decision path


def test_llm_switches_route_decisions_to_the_right_model():
    for human, machine in ((False, False), (True, False), (False, True), (True, True)):
        _, res = run(overrides=[f"human_agents.use_llm={human}", f"machine_agents.use_llm={machine}",
                                "llm.backend=mock"])
        d = res.metrics["decisions"]
        assert d["traveler"]["policy"] == ("llm" if human else "utility")
        assert d["vehicle"]["policy"] == ("llm" if machine else "algorithm")
        assert (res.metrics["llm"]["calls"] > 0) == (human or machine)


def test_reply_parsing_tolerates_reasoning_and_fences():
    text = '<think>{"action": "wrong"}</think>Sure!\n```json\n{"action": "choose_route", "args": {"route": 1, "message": null}, "rationale": "faster"}\n```'
    assert parse_decision_text(text) == {"action": "choose_route", "args": {"route": 1}, "rationale": "faster"}
    with pytest.raises(ValueError):
        parse_decision_text("I will keep going straight.")


def test_llm_budget_falls_back_to_the_non_llm_policy():
    _, res = run(overrides=["human_agents.use_llm=true", "llm.backend=mock", "llm.max_calls=5"])
    asked = [r for r in res.records if r.get("type") == "decision" and r["kind"] == "traveler" and r["source"] != "forced"]
    no_choice = [r for r in asked if r["decision"].get("rationale") == "the only option"]
    assert no_choice and all(len(r["permitted"]) == 1 and "fallback" not in r["info"] for r in no_choice)
    assert res.metrics["llm"]["calls"] == 5                 # a model is not consulted when there is nothing to choose
    assert res.metrics["llm"]["fallbacks"] == len(asked) - 5 - len(no_choice)
    assert res.metrics["travelers"]["completed"] > 0


def test_cassette_makes_llm_runs_repeatable_without_the_model(tmp_path):
    cassette = tmp_path / "cassette.jsonl"
    ov = ["human_agents.use_llm=true", "llm.backend=mock", f"llm.cassette={cassette}"]
    _, first = run(overrides=ov)
    _, second = run(overrides=[*ov, "llm.cassette_mode=replay"])
    assert first.metrics["llm"]["calls"] > 0
    assert second.metrics["llm"]["calls"] == 0 and second.metrics["llm"]["replayed"] == first.metrics["llm"]["calls"]
    assert second.trace_digest == first.trace_digest


def test_openai_compatible_backend_against_a_local_stub():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            user = body["messages"][-1]["content"]
            obs = json.loads(user[user.index("<observation>") + 13:user.index("</observation>")])
            first = obs["permitted_actions"][0]
            args = {k: v["choices"][0] for k, v in (first.get("args") or {}).items() if "choices" in v}
            reply = "<think>hmm</think>" + json.dumps({"action": first["action"], "args": args, "rationale": "stub"})
            payload = json.dumps({"model": "stub", "choices": [{"message": {"content": reply}}],
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 10}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _, res = run(overrides=["human_agents.use_llm=true", "llm.backend=openai_compatible", "llm.model=stub",
                                f"llm.base_url=http://127.0.0.1:{server.server_port}/v1",
                                "llm.pricing={input: 1.0, output: 2.0}"])
    finally:
        server.shutdown()
    llm = res.metrics["llm"]
    assert llm["calls"] == len(seen) > 0 and llm["fallbacks"] == 0
    assert llm["input_tokens"] == 100 * llm["calls"]
    assert abs(llm["cost_usd"] - llm["calls"] * (100 * 1.0 + 10 * 2.0) / 1e6) < 1e-9
    assert seen[0]["messages"][0]["role"] == "system" and "traveller" in seen[0]["messages"][0]["content"]


def test_llm_preset_fills_what_is_left_at_default_and_can_be_switched():
    from agmsim.config import LLMCfg
    eff = LLMCfg(preset="deepseek", max_tokens=200, params={"user_suffix": "!"}).resolved()
    assert (eff.model, eff.base_url, eff.api_key_env) == ("deepseek-flash", "https://api.deepseek.com", "DEEPSEEK_API_KEY")
    assert eff.max_tokens == 200 and eff.max_concurrency > 1
    assert eff.params == {"extra_body": {"thinking": {"type": "disabled"}}, "user_suffix": "!"}
    assert LLMCfg(preset="deepseek", model="other").resolved().model == "other"
    # the stored config keeps only the name, so an override replaces the whole preset
    cfg = Config.load(Config.load({"llm": {"preset": "qwen-local"}}), {"llm.preset": "deepseek"})
    assert cfg.llm.model == "" and cfg.llm.resolved().model == "deepseek-flash"
    with pytest.raises(ConfigError):
        Config.load({"llm": {"preset": "nope"}})


def test_env_file_supplies_a_missing_key_but_never_overrides(tmp_path):
    import os
    from agmsim.llm import load_env_file
    (tmp_path / ".env").write_text("# comment\nAGMSIM_T_NEW=abc\nexport AGMSIM_T_SET = 'from file'\nAGMSIM_T_EMPTY=\n")
    os.environ["AGMSIM_T_SET"] = "exported"
    try:
        assert load_env_file(tmp_path) == tmp_path / ".env"
        assert os.environ["AGMSIM_T_NEW"] == "abc" and os.environ["AGMSIM_T_SET"] == "exported"
        assert "AGMSIM_T_EMPTY" not in os.environ
    finally:
        for name in ("AGMSIM_T_NEW", "AGMSIM_T_SET", "AGMSIM_T_EMPTY"):
            os.environ.pop(name, None)


def test_http_client_reuses_its_connection_and_counts_cached_tokens():
    from agmsim.config import LLMCfg
    from agmsim.llm import LLMRequest, build_llm_client
    connections, drop = set(), []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            connections.add(self.client_address)
            payload = b"\n\n" + json.dumps({                     # hosted servers may send blank keep-alive lines first
                "model": "stub", "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 80}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            if drop:                                              # close without telling the client
                drop.pop()
                self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)      # kept-alive connections each hold a handler thread
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = build_llm_client(LLMCfg(backend="openai_compatible", model="stub",
                                         base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                         pricing={"input": 1.0, "cached_input": 0.1, "output": 2.0}))
        for _ in range(3):
            client.complete(LLMRequest("system", "user"))
        assert len(connections) == 1
        drop.append(True)
        client.complete(LLMRequest("system", "user"))            # the server drops the connection after this reply
        client.complete(LLMRequest("system", "user"))            # ... and the next call silently reconnects
        assert len(connections) == 2 and client.usage["calls"] == 5 and client.usage["errors"] == 0
    finally:
        client.close()
        server.shutdown()
    assert client.usage["cached_input_tokens"] == 5 * 80
    assert abs(client.usage["cost_usd"] - 5 * (20 * 1.0 + 80 * 0.1 + 10 * 2.0) / 1e6) < 1e-12


def test_concurrent_model_calls_and_traceless_runs_do_not_change_the_outcome():
    from agmsim.recorder import Recorder
    ov = ["human_agents.use_llm=true", "machine_agents.use_llm=true", "llm.backend=mock"]
    _, serial = run(overrides=[*ov, "llm.max_concurrency=1"])
    _, pooled = run(overrides=[*ov, "llm.max_concurrency=8"])
    _, bare = run(overrides=[*ov, "llm.max_concurrency=8"], recorder=Recorder(keep=False))
    assert serial.trace_digest == pooled.trace_digest == bare.trace_digest
    assert serial.records and not bare.records
    assert bare.metrics["travelers"] == serial.metrics["travelers"]


def test_rationale_can_be_left_out_of_the_prompts():
    from agmsim.policies.llm_policy import build_system_prompt, build_user_prompt
    sim, res = run(overrides=["human_agents.use_llm=true", "llm.backend=mock",
                              "agents.traveler.llm_policy_params={rationale: false}"])
    kind = sim.kinds["traveler"]
    obs = next(r["obs"] for r in res.records if r.get("type") == "decision" and r["kind"] == "traveler" and "obs" in r)
    assert "rationale" in build_system_prompt(kind) and "rationale" in build_user_prompt(obs)
    assert "rationale" not in build_system_prompt(kind, False) and "rationale" not in build_user_prompt(obs, (), False)
    assert res.metrics["llm"]["calls"] > 0 and res.metrics["llm"]["fallbacks"] == 0


# ---------------------------------------------------------------- plugins and interface


def test_example_plugin_scenario_runs_and_reports_custom_metric():
    from pathlib import Path
    scenario = Path(__file__).parent.parent / "examples" / "custom_scenario.yaml"
    sim = Simulation(str(scenario))
    res = sim.run()
    assert res.metrics["decisions"]["vehicle"]["policy"] == "toll_averse"
    assert res.metrics["decisions"]["bus"]["policy"] == "cautious_bus"
    assert "car_trips_rerouted_share" in res.metrics["custom"]
    assert res.metrics["bus"]["holds"] <= 5 * res.metrics["bus"]["buses"]


def test_policy_object_can_be_passed_directly():
    seen = []

    def watcher(obs):
        seen.append(obs.kind)
        return obs.default

    _, res = run("city_multimodal", policies={"operator": watcher})
    assert seen and set(seen) == {"operator"} and res.metrics["decisions"]["operator"]["exercised"] == 0


def test_web_interface_api():
    import time
    import urllib.request
    from http.server import ThreadingHTTPServer
    from agmsim.server import Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    try:
        info = call("/api/meta")
        assert {"traveler", "vehicle", "bus", "evcs", "operator"} <= set(info["kinds"])
        assert info["llm_presets"]["deepseek"]["key_env"] == "DEEPSEEK_API_KEY"
        run_id = call("/api/run", {"scenario": "demo_3x3", "overrides": ["agents.vehicle.agency=L2"]})["id"]
        for _ in range(100):
            view = call(f"/api/run/{run_id}?frames=0&decisions=0")
            if view["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert view["status"] == "done" and view["frames"] and view["header"]["kinds"]["vehicle"]["agency"]["level"] == "L2"
        assert "obs" in call(f"/api/run/{run_id}/decision/0")
    finally:
        server.shutdown()


# ---------------------------------------------------------------- designed experiments


def test_mix_gives_nested_sub_populations():
    def levels(share):
        sim = Simulation("demo_3x3", overrides=[*QUIET, "agents.vehicle.agency=L0",
                                                f"agents.vehicle.mix=[{{share: {share}, agency: L1}}]"])
        return {a for a, p in sim.agency.items() if sim.specs[a].kind == "vehicle" and p.level == "L1"}, sim

    quarter, sim = levels(0.25)
    half, _ = levels(0.5)
    n = sum(1 for s in sim.specs.values() if s.kind == "vehicle")
    assert len(quarter) == round(0.25 * n) and len(half) == round(0.5 * n)
    assert quarter < half                                    # raising the share only adds vehicles


def test_paired_differences_use_common_seeds():
    from agmsim.experiment import paired
    rows = [{"arm": "a", "seed": s, "m": 100.0 + 10 * s} for s in (1, 2, 3)]
    rows += [{"arm": "b", "seed": s, "m": 98.0 + 10 * s} for s in (1, 2, 3)]
    out = paired(rows, "a", "m")
    assert abs(out["b"]["diff"] + 2.0) < 1e-9 and out["b"]["ci95"] < 1e-9 and out["b"]["significant"]
    assert "diff" not in out["a"]


def test_designed_experiments_separate_the_arms():
    """Smoke test of two experiment designs on two seeds: the direction of the effect must show."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    from suite import EXPERIMENTS, HERE
    from agmsim.experiment import run_arms

    def mean(rows, arm, key):
        vals = [r[key] for r in rows if r["arm"] == arm]
        return sum(vals) / len(vals)

    exp = EXPERIMENTS["incident"]
    rows = run_arms(str(HERE / "scenarios" / exp["scenario"]), exp["arms"], [1, 2])
    base = mean(rows, "车 L0 · 运营方 L0", "travelers.trip_time_s_mean")
    assert mean(rows, "车 L1 · 运营方 L0", "travelers.trip_time_s_mean") < 0.5 * base
    assert mean(rows, "车 L0 · 运营方 L1", "travelers.trip_time_s_mean") < 0.5 * base

    exp = EXPERIMENTS["toll"]
    rows = run_arms(str(HERE / "scenarios" / exp["scenario"]), exp["arms"], [1, 2])
    assert mean(rows, "车 L2（算法，按车主时间价值）", "vehicles.owner_alignment_rate") == 1.0
    assert mean(rows, "车 L2（规则，仍只求快）", "vehicles.owner_alignment_rate") < 0.8
    assert (mean(rows, "车 L2（算法，按车主时间价值）", "vehicles.tolls_usd")
            < mean(rows, "车 L1（只求快）", "vehicles.tolls_usd"))


# ---------------------------------------------------------------- decision benchmark


def test_benchmark_references_bracket_the_scale():
    """Balanced item sets; a model that ignores preferences scores 0, one that follows them scores 1."""
    from agmsim.benchmark import TASKS, evaluate_baselines, get_task
    assert set(TASKS.names()) == {"route_delegation", "suggestion_response", "transfer_proposal"}

    items = get_task("route_delegation").items()
    assert len(items) == 192 and sum(i.label == "pay" for i in items) == 96
    assert len({i.id for i in items}) == 192
    route = {k: r["scores"] for k, r in evaluate_baselines("route_delegation").items()}
    assert route["rule"]["sensitivity"] == 0.0 and route["rule"]["accuracy"] == 0.5
    assert route["algorithm"]["sensitivity"] == 1.0 and route["algorithm"]["regret_usd_mean"] == 0.0

    answer = evaluate_baselines("suggestion_response")
    assert answer["rule"]["scores"]["sensitivity"] == 0.0                     # accepts everything
    assert answer["utility"]["scores"]["sensitivity"] == 1.0
    assert answer["utility"]["scores"]["misled_by_overstatement"] == 0.0      # it never reads the message
    assert answer["utility"]["scores"]["accepted_slower_overstated"] == 0.0

    propose = {k: r["scores"] for k, r in evaluate_baselines("transfer_proposal").items()}
    assert propose["rule"]["sensitivity"] == 0.0                              # never proposes
    assert 0.2 < propose["algorithm"]["sensitivity"] < 0.7                    # proposes on time alone
    assert propose["algorithm_fare"]["sensitivity"] == 1.0
    assert propose["algorithm"]["took_quicker_route"] == 1.0


def test_benchmark_items_have_the_simulators_observation_format():
    from agmsim import Observation
    from agmsim.benchmark import TASKS, get_task

    def shape(value):
        """Keys in order, recursively; lists are described by their first element."""
        if isinstance(value, dict):
            return [(k, shape(v)) for k, v in value.items()]
        if isinstance(value, list):
            return [shape(value[0])] if value else []
        return type(value).__name__

    for name in TASKS.names():
        task = get_task(name)
        template = task.template()
        for item in task.items()[::17]:
            seen = dict(template["context"])
            made = dict(item.obs["context"])
            if name == "suggestion_response":                                 # the message is optional in a run too
                for ctx in (seen, made):
                    ctx["proposal"] = {k: v for k, v in ctx["proposal"].items() if k != "message_from_vehicle"}
            assert shape(made) == shape(seen), item.id
            assert shape(item.obs["state"]) == shape(template["state"]), item.id
            obs = Observation.from_dict(item.obs)
            assert obs.to_dict()["permitted_actions"] == item.obs["permitted_actions"]
            assert validate_decision(obs.default, obs.actions)[0] is not None


def test_benchmark_scores_a_language_model_and_counts_missing_answers():
    from agmsim.benchmark import evaluate
    llm = {"agents.vehicle.use_llm": True, "llm.preset": "mock", "llm.max_concurrency": 4,
           "agents.vehicle.llm_policy_params": {"rationale": "first"}}
    report = evaluate("route_delegation", {**llm, "llm.params": {"explore": 0.0}})
    assert report["decision_model"] == "llm" and report["complete"] and report["usage"]["calls"] == 192
    assert report["scores"]["sensitivity"] == 1.0 and report["scores"]["valid"] == 1.0   # the mock answers like the algorithm
    assert all(set(r) >= {"item", "label", "ratio", "route", "paid", "correct", "rationale"} for r in report["rows"])

    # a model that never produces a usable reply gets no credit for its fallback: the default action applies
    silent = evaluate("route_delegation", {**llm, "llm.params": {"error_rate": 1.0}}, limit=48)
    assert silent["scores"]["valid"] == 0.0 and not silent["complete"] and silent["items"] == 48
    assert all(r["route"] == 0 and "error" in r for r in silent["rows"])
    assert silent["items_digest"] == report["items_digest"]


def test_reasoning_can_be_asked_for_before_the_choice():
    from agmsim.policies.llm_policy import build_system_prompt, build_user_prompt
    sim, res = run(overrides=["human_agents.use_llm=true", "llm.backend=mock",
                              "agents.traveler.llm_policy_params={rationale: first}"])
    kind = sim.kinds["traveler"]
    obs = next(r["obs"] for r in res.records if r.get("type") == "decision" and r["kind"] == "traveler" and "obs" in r)
    system, user = build_system_prompt(kind, "first"), build_user_prompt(obs, (), "first")
    assert system.index('"reasoning"') < system.index('"action": "<name') and "rationale" not in system
    assert user.rstrip().endswith('{"reasoning": "...", "action": ..., "args": {...}}')
    assert res.metrics["llm"]["calls"] > 0 and res.metrics["llm"]["fallbacks"] == 0
    parsed = parse_decision_text('{"reasoning": "toll 0.6 > time saved 0.2", "action": "choose_route", "args": {"route": 1}}')
    assert parsed == {"action": "choose_route", "args": {"route": 1}, "rationale": "toll 0.6 > time saved 0.2"}
    with pytest.raises(ValueError, match="rationale must be"):
        build_system_prompt(kind, "sometimes")


def test_transfer_is_weighed_against_the_quickest_route_and_optionally_the_fare():
    """A transfer is proposed only if it beats re-routing, and the owner is told the quickest driving time."""
    from agmsim import Observation
    from agmsim.benchmark import get_task
    from agmsim.policies.base import PolicyContext, create_policy
    import random

    def decide(item, **params):
        policy = create_policy("vehicle", "algorithm", params)
        policy.bind(PolicyContext(item.obs["agent_id"], "vehicle", random.Random(0)))
        return policy.decide(Observation.from_dict(item.obs))

    items = {i.id: i for i in get_task("transfer_proposal").items()}
    small = items["propose-d0-g1-v20-reroute"]            # bus 1 min quicker than the best route, 7 than the current
    assert decide(small).action == "choose_route" and decide(small).args["route"] == 1
    cheap_time = items["propose-d0-g6-v8-single"]         # 6 min at 8 USD/h is worth 0.80 USD; the fare is 1.50
    assert decide(cheap_time).action == "suggest_transfer"
    assert decide(cheap_time, weigh_fare=True).action == "choose_route"
    assert decide(items["propose-d0-g15-v50-single"], weigh_fare=True).action == "suggest_transfer"

    sim, res = run(str(SCENARIOS / "bottleneck.yaml"), ["agents.vehicle.agency=L3"])
    proposals = [r for r in res.records if r.get("type") == "decision" and r["decision"]["action"] == "suggest_transfer"]
    assert proposals
    for rec in proposals:
        ctx = rec["obs"]["context"]
        chosen = next(o for o in ctx["bus_transfer_options"] if o["stop"] == rec["decision"]["args"]["stop"])
        assert chosen["total_min"] < min(r["eta_min"] for r in ctx["routes"])
    told = [r["obs"]["context"] for r in res.records
            if r.get("type") == "decision" and r["trigger"] == "suggestion" and "obs" in r]
    assert all(c["proposal"]["total_min"] < c["if_you_keep_driving_min"] for c in told)


def test_owner_sensitivity_is_measured_inside_a_run():
    toll = ["agents.traveler.count=40", "simulation.duration_seconds=1800", "demand.end_seconds=900",
            "agents.vehicle.agency=L2"]
    _, follows = run(str(SCENARIOS / "toll.yaml"), toll)
    _, ignores = run(str(SCENARIOS / "toll.yaml"), [*toll, "agents.vehicle.non_llm_policy=rule"])
    assert follows.metrics["vehicles"]["owner_sensitivity"] == 1.0
    # the fastest-route rule pays whenever the tolled route is quicker, whoever the owner is; it is not
    # exactly 0 only because on a few trips the free route happens to be the quickest as well
    assert ignores.metrics["vehicles"]["owner_sensitivity"] < 0.3


# ---------------------------------------------------------------- day plans (MATSim's action mechanism)


def test_plan_score_follows_charypar_nagel():
    from agmsim.config import ScoringCfg
    from agmsim.plans import Activity, Plan, performing, rates, score

    r = rates(24.0, ScoringCfg())
    assert r["activity"] - r["travel"] == pytest.approx(24.0)          # an hour of travel instead of an activity: the value of time
    assert r["late"] == pytest.approx(3 * r["travel"])                 # MATSim's defaults: -18 against -6
    assert performing(3600, 3600, 10.0) == pytest.approx(10.0)         # worth rate x typical at the usual length
    slope = (performing(3601, 3600, 10.0) - performing(3599, 3600, 10.0)) / 2 * 3600
    assert slope == pytest.approx(10.0, rel=1e-3)                      # and one more hour there is worth the rate
    zero = 3600 / 2.718281828
    assert performing(zero, 3600, 10.0) == pytest.approx(0.0, abs=1e-6)
    assert performing(0.0, 3600, 10.0) == pytest.approx(-10.0)         # kept but not performed: worse than nothing
    assert performing(3600, 3600, 10.0, priority=3) == pytest.approx(30.0)     # priority: what doing it at all is worth,
    assert performing(7200, 3600, 10.0, 3) - performing(3600, 3600, 10.0, 3) == pytest.approx(
        performing(7200, 3600, 10.0) - performing(3600, 3600, 10.0))           # not how long one wants to stay
    plan = Plan([Activity("work", 0, end_s=600, typical_s=3600, outside_s=3000, arrived_s=0.0, left_s=600),
                 Activity("child", 1, duration_s=300, latest_start_s=900, typical_s=300, arrived_s=1200, left_s=1500),
                 Activity("errand", 2, typical_s=600, optional=True, skipped=True),
                 Activity("home", 3, typical_s=3600, arrived_s=2100)],
                legs=[{"depart_s": 600, "arrive_s": 1200}, {"depart_s": 1500, "arrive_s": 2100}])
    s = score(plan, r, 5700.0, paid=2.5)
    assert s["late_s"] == 300 and s["late"] == pytest.approx(r["late"] * 300 / 3600)
    assert s["travel"] == pytest.approx(r["travel"] * 1200 / 3600) and s["money"] == -2.5
    assert s["activities"] == pytest.approx(r["activity"] * (1.0 + 300 / 3600 + 1.0))   # all three at their usual length
    assert s["missed"] == 0


EVENING = str(SCENARIOS / "evening.yaml")
SMALLER = ["agents.traveler.count=120", "network.capacity_scale=0.1"]
ADAPT = ["agents.traveler.agency=[adapt_plan]"]


def test_day_plans_are_followed_like_matsim_plans():
    sim, result = run(EVENING, SMALLER)
    people = list(sim.world.travelers.values())
    plans = result.metrics["plans"]
    assert plans["people"] == plans["completed"] == 120 and plans["reviews"] == 0
    moves = lambda acts: [k for k in range(1, len(acts)) if acts[k].node != acts[k - 1].node]   # noqa: E731
    assert result.metrics["travelers"]["trips"] == sum(len(moves(t.plan.activities)) for t in people)
    for t in people:
        acts = t.plan.activities
        assert acts[0].left_s == pytest.approx(acts[0].end_s, abs=sim.dt)          # leaves when the plan says
        assert [leg["to"] for leg in t.plan.legs] == moves(acts)                   # one trip per change of place
        assert all(a.arrived_s is not None for a in acts)
        assert sim.world.vehicles[t.car].node == acts[-1].node                     # the car came along the whole chain
    assert plans["people_late"] > 0                                                # the incident makes followers late
    _, calm = run(EVENING, [*SMALLER, "interventions=[]"])
    assert calm.metrics["plans"]["score_usd_mean"] > plans["score_usd_mean"]


def test_plans_are_adapted_within_the_day_only_with_the_authority_and_only_when_the_day_drifts():
    sim, adapted = run(EVENING, [*SMALLER, *ADAPT, "agents.traveler.non_llm_policy=utility"])
    _, followed = run(EVENING, SMALLER)
    plans = adapted.metrics["plans"]
    assert plans["reviews"] > 20 and 0 < plans["edits"] < plans["reviews"]
    assert plans["review_regret_usd_mean"] < 0.5           # it keeps the plan unless an edit gains plan_gain_usd
    assert plans["score_usd_mean"] > followed.metrics["plans"]["score_usd_mean"]
    assert plans["late_min_mean"] < followed.metrics["plans"]["late_min_mean"]
    assert sim.event_counts["plan_changed"] == plans["edits"]
    _, eager = run(EVENING, [*SMALLER, *ADAPT, "agents.traveler.non_llm_policy=utility",
                             "agents.traveler.policy_params={plan_gain_usd: 0}"])
    assert eager.metrics["plans"]["review_best_rate"] == 1.0 and eager.metrics["plans"]["review_regret_usd_mean"] == 0.0
    assert plans["changed_when_worth_it"] == 1.0 and plans["changed_when_not"] == 0.0 and plans["review_sensitivity"] == 1.0
    _, kept = run(EVENING, [*SMALLER, *ADAPT])                         # may adapt, but the rule never does
    assert kept.metrics["plans"]["edits"] == 0 and kept.metrics["plans"]["review_regret_usd_mean"] > 0
    assert kept.metrics["plans"]["review_sensitivity"] == 0.0
    _, calm = run(EVENING, [*SMALLER, *ADAPT, "agents.traveler.non_llm_policy=utility", "interventions=[]",
                            "network.capacity_scale=0.3"])
    assert calm.metrics["plans"]["reviews"] == 0                       # nothing drifts: nobody is asked
    _, again = run(EVENING, [*SMALLER, *ADAPT, "agents.traveler.non_llm_policy=utility"])
    assert again.state_digest == adapted.state_digest and again.trace_digest == adapted.trace_digest


def test_plan_review_is_an_ordinary_observation():
    from agmsim.plans import option_value
    from agmsim.policies.llm_policy import build_system_prompt, build_user_prompt

    seen = []

    def watch(obs):
        if obs.trigger == "plan_review":
            seen.append(obs)
        return obs.default

    sim, _ = run(EVENING, [*SMALLER, *ADAPT], policies={"traveler": watch})
    obs = next(o for o in seen if o.has("skip_activity") and o.has("leave_at"))
    data = json.loads(json.dumps(obs.to_dict()))                       # what any decision model is shown
    options = data["context"]["options"]
    assert options[0]["choice"] == {"action": "keep_plan"} and data["time"] >= "16:30:00"
    assert {o["choice"].get("time") for o in options if o["choice"]["action"] == "leave_at"} == set(obs.choices("leave_at", "time"))
    skip = next(o for o in options if o["choice"]["action"] == "skip_activity")
    assert skip["choice"]["activity"] not in [row["activity"] for row in skip["then"]]
    assert all(isinstance(option_value(data["state"]["preferences"], data["context"], o), float) for o in options)
    prompt = build_system_prompt(sim.kinds["traveler"])
    assert "plan_review" in prompt and "leave_at" in prompt and "typical_min" in prompt
    assert "plan_review" in build_user_prompt(obs)
    plain, _ = run()
    assert "plan_review" not in build_system_prompt(plain.kinds["traveler"])       # single trips: nothing about plans


def test_web_interface_runs_without_a_server():
    """The online demo: the page's requests are answered in-process and the run is advanced in slices."""
    from agmsim import browser
    from agmsim.server import RUNS

    info = json.loads(browser.request("/api/meta"))
    assert info["online"] and info["llm_presets"]["deepseek"]["key_env"] == "DEEPSEEK_API_KEY"
    assert "evening_plans" in info["scenarios"]
    run_id = json.loads(browser.request("/api/run", json.dumps({"scenario": "demo_3x3", "overrides": []})))["id"]
    assert RUNS[run_id].thread is None
    slices = 0
    while browser.advance(0.01):
        slices += 1
        assert json.loads(browser.request(f"/api/run/{run_id}?frames=0&decisions=0"))["status"] in ("starting", "running")
    view = json.loads(browser.request(f"/api/run/{run_id}?frames=0&decisions=0"))
    assert slices > 2 and view["status"] == "done" and view["frames"] and view["metrics"]["travelers"]["completed"] > 0
    assert RUNS[run_id].sim.cfg.llm.max_concurrency == 1               # no threads in a browser tab
    assert "obs" in json.loads(browser.request(f"/api/run/{run_id}/decision/0"))
    again = json.loads(browser.request("/api/run", "{}"))["id"]
    assert again == run_id + 1 and list(RUNS) == [again]           # only the latest run is kept in a browser tab
    assert json.loads(browser.request(f"/api/run/{again}/stop", "{}")) == {"ok": True}
    assert not browser.advance(0.01) or not browser.advance(5.0)
    assert json.loads(browser.request("/api/nothing")) == {"error": "not found"}


def test_online_demo_takes_the_visitors_key_and_calls_the_model_through_the_browser(monkeypatch):
    """In the online demo the key is typed into the page, kept in memory, and the browser makes the calls."""
    import sys
    import types

    from agmsim import browser, llm
    from agmsim.server import RUNS

    sent = []

    class FakeRequest:                     # stands in for the browser's XMLHttpRequest
        status, responseText = 200, ""

        @classmethod
        def new(cls):
            return cls()

        def open(self, method, url, is_async):
            self.call = {"method": method, "url": url, "async": is_async, "headers": {}}

        def setRequestHeader(self, name, value):
            self.call["headers"][name] = value

        def send(self, body):
            self.call["body"] = json.loads(body)
            sent.append(self.call)
            key = self.call["headers"].get("Authorization")
            if key != "Bearer visitor-test-key":
                self.status, self.responseText = 401, '{"error": "bad key"}'
                return
            obs = json.loads(self.call["body"]["messages"][1]["content"].split("<observation>")[1].split("</observation>")[0])
            first = obs["permitted_actions"][0]
            args = {name: spec["choices"][0] for name, spec in (first.get("args") or {}).items() if spec.get("choices")}
            self.responseText = json.dumps({"choices": [{"message": {"content": json.dumps({"action": first["action"], "args": args})}}],
                                            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    monkeypatch.setitem(sys.modules, "js", types.SimpleNamespace(XMLHttpRequest=FakeRequest))
    monkeypatch.setattr(llm, "IN_BROWSER", True)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(llm, "load_env_file", lambda *a, **k: None)    # a browser tab has no .env file

    def run_with(key):
        if key is not None:
            assert json.loads(browser.request("/api/key", json.dumps({"env": "DEEPSEEK_API_KEY", "key": key}))) == {"ok": True}
        body = {"scenario": "demo_3x3", "overrides": ["agents.vehicle.use_llm=true", "llm.preset=deepseek"]}
        run_id = json.loads(browser.request("/api/run", json.dumps(body)))["id"]
        while browser.advance(1.0):
            pass
        return RUNS[run_id], json.loads(browser.request(f"/api/run/{run_id}?frames=0&decisions=0"))

    _, view = run_with("")                                             # nothing typed
    assert view["status"] == "error" and "type your own key" in view["error"]
    _, view = run_with("wrong-key")
    assert view["status"] == "error" and "HTTP 401" in view["error"]
    run, view = run_with("  visitor-test-key ")
    assert view["status"] == "done" and view["metrics"]["llm"]["calls"] == len([c for c in sent if c["headers"]["Authorization"].endswith("test-key")]) > 0
    assert all(c["url"] == "https://api.deepseek.com/chat/completions" and c["async"] is False for c in sent)
    shown = json.dumps(view) + json.dumps(run.recorder.header, default=str) + json.dumps(run.recorder.decisions, default=str)
    assert "visitor-test-key" not in shown                             # the key is in no config, trace or page data
    assert json.loads(browser.request("/api/key", json.dumps({"env": "path; rm", "key": "x"})))["ok"] is False


def test_frames_carry_what_the_charts_need_and_an_agents_day_can_be_asked_for():
    from agmsim import browser
    from agmsim.server import RUNS

    body = {"scenario": "evening_plans", "overrides": ["agents.traveler.count=120", "network.capacity_scale=0.1"]}
    run_id = json.loads(browser.request("/api/run", json.dumps(body)))["id"]
    while browser.advance(5.0):
        pass
    run = RUNS[run_id]
    frames = run.recorder.frames
    assert all(sum(f["stats"]["people"].values()) == 120 for f in frames)             # everybody is in exactly one stage
    assert frames[0]["stats"]["people"]["activity"] == 120 and frames[-1]["stats"]["people"]["done"] == 120
    busy = [f for f in frames if f.get("busy_links")]
    assert busy and all(0.15 <= v < 0.95 for f in busy for v in f["busy_links"].values())
    speeds = [f["stats"]["speed"] for f in frames if f["stats"]["speed"] is not None]
    assert 0.15 <= min(speeds) < 0.8 and max(speeds) <= 1.0 and frames[-1]["stats"]["on_road"] == 0
    changed = next(d["agent"] for d in run.recorder.decisions if d["trigger"] == "plan_review" and d["decision"]["action"] != "keep_plan")
    story = json.loads(browser.request(f"/api/run/{run_id}/agent/{changed}"))
    assert story["kind"] == "traveler" and any(d["trigger"] == "plan_review" and d["exercised"] for d in story["decisions"])
    line = story["timeline"]
    assert line[0]["label"] == "work" and line[-1]["label"] == "home" and line[-1]["end"] == story["horizon_s"]
    assert all(a["end"] == pytest.approx(b["start"], abs=2.1) for a, b in zip(line, line[1:]))   # no gaps in the day
    assert {i["kind"] for i in line} == {"activity", "travel"}
    car = json.loads(browser.request(f"/api/run/{run_id}/agent/{story['car']}"))
    assert car["kind"] == "vehicle" and car["owner"] == changed and "timeline" not in car
    assert json.loads(browser.request(f"/api/run/{run_id}/agent/nobody")) == {}


def test_outcome_variables_waiting_vmt_vht_and_traffic_speed():
    sim, res = run("city_multimodal")
    o, w = res.metrics["outcomes"], sim.world
    assert o["vmt_miles"] == pytest.approx(o["vkt_km"] / 1.609344, rel=1e-3) and o["vkt_km"] > res.metrics["vehicles"]["vkt_km"]   # buses count too
    assert o["traffic_speed_mph"] == pytest.approx(o["vmt_miles"] / o["vht_hours"], rel=1e-3)
    free = max(link.speed for link in w.net.links) * 3.6
    assert 5 < o["traffic_speed_kmh"] <= free                       # between a crawl and free flow
    assert all(v.s_moving <= sim.n_ticks * sim.dt for v in w.vehicles.values())
    waited = [t.waited_ticks * sim.dt for t in w.travelers.values() if t.state == TState.ARRIVED]
    assert o["wait_time_s_mean"] == pytest.approx(sum(waited) / len(waited), abs=1e-3) and 0 < o["wait_time_s_mean"] < res.metrics["travelers"]["bus_wait_s_mean"]
    _, cars = run()                                                   # no buses: nobody waits for transport
    assert cars.metrics["outcomes"]["wait_time_s_mean"] == 0.0 and cars.metrics["outcomes"]["vht_hours"] > 0


def test_assistant_turns_a_request_into_checked_settings():
    from agmsim import assistant, browser
    from agmsim.llm import LLMClient, LLMResponse, register_llm

    replies, seen = [], []

    @register_llm("scripted")
    class Scripted(LLMClient):                 # stands in for the language model behind the assistant
        backend = "scripted"

        def _complete(self, request):
            seen.append(request)
            return LLMResponse(replies.pop(0), "scripted", 10, 10)

    llm = {"backend": "scripted", "model": "x"}
    ask = lambda text, **extra: assistant.configure({"messages": [{"role": "user", "text": text}],   # noqa: E731
                                                     "scenario": "evening_plans", "llm": llm, **extra})
    replies.append('Sure. {"reply": "Vehicles now negotiate; 300 people.", "scenario": null, '
                   '"set": {"agents.vehicle.agency": "L3", "agents.traveler.count": 300}, "run": true}')
    out = ask("300 travellers and vehicles at level 3, then run it")
    assert out["set"] == {"agents.vehicle.agency": "L3", "agents.traveler.count": 300} and out["run"] and "error" not in out
    assert "evening_plans" in seen[0].system and "L3" in seen[0].system and "clock_start: 16:30" in seen[0].user
    # a proposal that does not hold is sent back once with the reason; the corrected one is used
    replies += ['{"reply": "x", "set": {"agents.bus.agency": "L1"}}',
                '{"reply": "This scenario has no buses, so I slowed the bridge instead.", "set": {"interventions": '
                '[{"type": "link_condition", "at_seconds": 3600, "until_seconds": 7200, "factor": 0.05, "links": [[7, 20]]}]}}']
    out = ask("let the buses hold at stops")
    assert out["set"]["interventions"][0]["links"] == [[7, 20]] and "has no bus" in seen[-1].user
    # nothing usable twice: nothing is changed, and the reason is reported
    replies += ['{"reply": "x", "set": {"llm.model": "other"}}', '{"reply": "x", "set": {"interventions": '
                '[{"type": "link_condition", "at_seconds": 0, "factor": 0.1, "links": [[0, 31]]}]}}']
    out = ask("use another model")
    assert out["set"] == {} and not out["run"] and "no road between" in out["error"]
    for bad in ({"agents.vehicle.agency": "L9"}, {"agents.vehicle.non_llm_policy": "magic"}, {"simulation.sed": 1},
                {"record.dir": "x"}, {"agents.traveler.mix": [{"agency": "L1"}]}):
        assert assistant.check("evening_plans", bad)
    assert assistant.check("evening_plans", {"agents.vehicle.mix": [{"share": 0.3, "agency": "L3"}], "network.capacity_scale": 0.1}) is None
    assert assistant.check("nowhere", {}).startswith("there is no scenario")
    assert "needs a language model" in assistant.configure({"messages": [{"role": "user", "text": "hi"}], "llm": {"backend": "mock"}})["error"]
    assert assistant.extract_json('<think>{"a": 1}</think> text {"reply": "say \\"hi\\" {now}", "set": {}} tail')["reply"] == 'say "hi" {now}'
    replies.append('{"reply": "ok", "set": {"simulation.seed": 5}}')            # the page reaches it through the same router
    via = json.loads(browser.request("/api/chat", json.dumps({"messages": [{"role": "user", "text": "seed 5"}],
                                                              "scenario": "demo_3x3", "llm": llm})))
    assert via["set"] == {"simulation.seed": 5} and via["scenario"] == "demo_3x3"


# ---------------------------------------------------------------- traffic lights and the orchestrator


def test_a_red_light_holds_movers_and_lets_them_through_in_arrival_order():
    from types import SimpleNamespace
    from agmsim.kinds.base import advance
    from agmsim.network import StopLine

    net = build_network(Config.from_dict({"network": {"type": "grid", "rows": 2, "cols": 3, "block_m": 100, "speed_mps": 10}}).network)
    first, second = net.link_between(0, 1).id, net.link_between(1, 2).id
    line = net.stop_lines[first] = StopLine(green=False)
    mover = lambda name, path: SimpleNamespace(id=name, node=0, link=None, offset=0.0, path=list(path), queued=False, go=False)   # noqa: E731
    cars = [mover(f"c{i}", [first, second]) for i in range(3)]
    for car in cars:
        assert advance(net, car, 20.0) == (100.0, False)                # reaches the stop line and stays on its link
    assert all(c.link == first and c.offset == 100.0 and c.queued for c in cars) and line.queue == cars
    assert advance(net, mover("home", [first]), 20.0)[1]                # a trip that ends at the junction needs no green
    line.green, line.credit = True, 1.0
    late = mover("late", [first, second])
    advance(net, late, 20.0)
    assert late.queued and line.queue[-1] is late and line.credit == 1.0   # green, but it queues behind those already waiting
    cars[0].queued, cars[0].go = False, True                            # what the light does when it lets the first one go
    line.queue.pop(0)
    assert advance(net, cars[0], 5.0)[0] == pytest.approx(50.0) and cars[0].link == second and not cars[0].go
    assert advance(net, cars[1], 5.0)[0] == 0.0 and cars[1].link == first    # the others still wait
    bus = mover("bus", [first, second])
    advance(net, bus, 15.0, preempt=True)
    assert bus.link == second and not bus.queued                        # priority: through on whatever the light shows
    free = mover("free", [second, net.link_between(2, 5).id])
    assert advance(net, free, 30.0)[0] == pytest.approx(200.0) and line.passed == 0   # no light at the end of that link


LIGHTS = ["agents.traveler.count=300", "simulation.duration_seconds=2400", "demand.end_seconds=1200"]


def test_lights_with_agency_cut_delay_and_the_orchestrator_needs_lights_that_listen(tmp_path):
    fixed_sim, fixed = run("city_signals", [*LIGHTS, "agents.signal.agency=L0"])
    _, own = run("city_signals", LIGHTS)
    listen_sim, listen = run("city_signals", [*LIGHTS, "agents.signal.agency=L2", "agents.signal.non_llm_policy=rule",
                                              "agents.operator.agency=L2"])
    _, deaf = run("city_signals", [*LIGHTS, "agents.signal.agency=L0", "agents.operator.agency=L2"])
    trip = lambda r: r.metrics["travelers"]["trip_time_s_mean"]         # noqa: E731
    lights = lambda r: r.metrics["signals"]                             # noqa: E731
    assert lights(fixed)["junctions"] == 32 and fixed.metrics["decisions"]["signal"]["decided"] == 0     # the 4 corners have none
    assert fixed.metrics["decisions"]["signal"]["forced"] == lights(fixed)["switches"] > 0                 # the fixed plan is a default
    assert own.metrics["decisions"]["signal"]["decided"] > 0 and lights(own)["wait_s_mean"] < 0.6 * lights(fixed)["wait_s_mean"]
    assert trip(own) < trip(listen) < trip(fixed)
    assert listen.metrics["operator"]["signal_requests"] > 0 and listen_sim.world.operator.favoured == "east-west"
    assert deaf.metrics["operator"]["signal_requests"] > 0 and trip(deaf) == trip(fixed)                   # nobody listens
    for sim, res in ((fixed_sim, fixed), (listen_sim, listen)):
        assert res.metrics["travelers"]["completion_rate"] == 1.0
        assert all(not line.queue for line in sim.world.net.stop_lines.values())                           # nobody left at a light
        assert not any(v.queued or v.go for v in sim.world.vehicles.values())
    assert lights(fixed)["crossings"] > 0 and 0 < lights(own)["stopped_share"] < lights(fixed)["stopped_share"] < 1
    # the orchestrator sees where it is slow and what waited at the lights; a light that listens sees what is asked of it
    seen = {}

    def watch(obs):
        seen.setdefault(obs.kind, obs)
        return obs.default

    run("city_signals", [*LIGHTS, "agents.signal.agency=L2", "agents.operator.agency=L2", "simulation.duration_seconds=900"],
        policies={"operator": watch, "signal": watch})
    assert "slowest_roads" in seen["operator"].context and seen["operator"].has("favour_direction")
    assert set(seen["operator"].context["traffic_lights"]["vehicle_seconds_waited_since_last_review"]) == {"east-west", "north-south"}
    assert {p["phase"] for p in seen["signal"].context["phases"]} == {"east-west", "north-south"}
    assert "orchestrator" in seen["signal"].context and "next_junctions" in seen["signal"].context
    # same seed, same run; and the recorded decisions replay to the same state
    _, again = run("city_signals", LIGHTS)
    assert again.state_digest == own.state_digest and again.trace_digest == own.trace_digest
    saved = Simulation("city_signals", overrides=[f"record.dir={tmp_path}", "record.run_name=lights", *LIGHTS,
                                                  "agents.signal.agency=L2", "agents.operator.agency=L2"]).run()
    assert replay(saved.run_dir)["identical"]
    frame = next(r for r in saved.records if r["type"] == "frame" and r["tick"] > 600)
    header = next(r for r in saved.records if r["type"] == "header")
    assert len(frame["signal"]) == 32 and all(row[2] in (-1, 0, 1) for row in frame["signal"])
    assert set(header["signals"][frame["signal"][0][0]]) == {"node", "names", "from"}
    _, none = run("city_signals", [*LIGHTS, "agents.signal.junctions=[]"])
    assert "signals" not in none.metrics and trip(none) < trip(own)                                        # no junction delay at all


# ---------------------------------------------------------------- on-demand cars

CABS = ["agents.traveler.count=150", "agents.ondemand.fleet=25", "simulation.duration_seconds=3000"]


def test_on_demand_cars_carry_travellers_who_call_them(tmp_path):
    sim, base = run("city_ondemand", CABS)
    w, od = sim.world, base.metrics["ondemand"]
    assert "vehicle" not in sim.kinds and od["cars"] == 25 and od["rides"] > 100
    reqs = list(w.requests.values())
    assert od["requests"] == len(reqs) == od["rides"] + od["cancelled"] + sum(r.state not in ("done", "cancelled") for r in reqs)
    done = [r for r in reqs if r.state == "done"]
    assert all(w.travelers[r.traveler].mode == "ride" and w.travelers[r.traveler].state == TState.ARRIVED for r in done)
    assert sum(r.fare for r in done) == pytest.approx(od["revenue_usd"], abs=0.01) == pytest.approx(sum(c.revenue for c in w.cabs.values()), abs=0.01)
    assert base.metrics["travelers"]["mode_share"]["ride"] > 0.5 and base.metrics["outcomes"]["wait_time_s_mean"] > 30
    assert od["pickup_wait_s_mean"] > 0 and 0 < od["empty_km_share"] < 1 and od["declined"] == od["repositions"] == 0
    assert base.metrics["decisions"]["ondemand"]["decided"] == 0 and base.metrics["decisions"]["ondemand"]["forced"] >= od["rides"]
    assert base.metrics["outcomes"]["vkt_km"] == pytest.approx(od["vkt_km"], abs=0.05)             # the only vehicles here
    assert base.metrics["outcomes"]["traffic_speed_kmh"] < 11 * 3.6                               # they slow each other down
    assert all(c.proposal is None and not c.queued for c in w.cabs.values())

    # L2 with a profit objective turns poorly paid requests down; they go to another car
    picky_sim, picky = run("city_ondemand", [*CABS, "agents.ondemand.agency=L2", "agents.ondemand.objective=profit"])
    assert picky.metrics["ondemand"]["declined"] > 0
    turned_down = [r for r in picky_sim.world.requests.values() if r.declined_by and r.state == "done"]
    assert turned_down and all(r.car not in r.declined_by for r in turned_down)
    _, serving = run("city_ondemand", [*CABS, "agents.ondemand.agency=L2", "agents.ondemand.objective=service"])
    assert serving.metrics["ondemand"]["declined"] == 0
    # L4 also moves while free
    _, roaming = run("city_ondemand", [*CABS, "agents.ondemand.agency=L4", "agents.ondemand.objective=service"])
    assert roaming.metrics["ondemand"]["repositions"] > 0 and roaming.metrics["ondemand"]["empty_km_share"] > od["empty_km_share"]

    # too few cars: people give up, walk or stay; nobody is left hanging and no car is tied to a dead request
    few_sim, few = run("city_ondemand", ["agents.traveler.count=150", "agents.ondemand.fleet=3", "simulation.duration_seconds=3000"])
    fw = few_sim.world
    assert few.metrics["ondemand"]["cancelled"] > 0 and few.metrics["travelers"]["gave_up_waiting"] == few.metrics["ondemand"]["cancelled"]
    assert all(t.ride is None or fw.requests[t.ride].state in ("open", "proposed", "assigned", "on_board") for t in fw.travelers.values())
    assert all(c.request is None or fw.requests[c.request].state in ("assigned", "on_board") for c in fw.cabs.values())
    assert few.metrics["travelers"]["mode_share"].get("walk", 0) > base.metrics["travelers"]["mode_share"].get("walk", 0)
    assert few.metrics["outcomes"]["wait_time_s_mean"] > base.metrics["outcomes"]["wait_time_s_mean"]

    # what a car and a waiting traveller are shown
    seen = {}

    def watch(obs):
        seen.setdefault((obs.kind, obs.trigger), obs)
        return obs.default

    run("city_ondemand", ["agents.traveler.count=150", "agents.ondemand.fleet=6", "agents.ondemand.agency=L4",
                          "simulation.duration_seconds=2400"], policies={"ondemand": watch, "traveler": watch})
    proposal = seen[("ondemand", "proposal")]
    assert {"pickup_min", "trip_km", "fare_usd", "traveller_has_waited_min"} <= set(proposal.context["request"])
    assert proposal.has("decline_request") and "your_driving_cost_usd" in proposal.context
    assert seen[("ondemand", "idle")].has("reposition") and seen[("ondemand", "idle")].context["places"]
    wait = seen[("traveler", "wait_check")]
    assert wait.context["your_request"] in ("looking for a free car", "a car is on its way") and "pickup_eta_min" in wait.context
    assert any(o["option"] == "ride" for o in seen[("traveler", "need")].context["options"])

    # same seed, same run; recorded decisions replay to the same state; cars stop at traffic lights too
    _, again = run("city_ondemand", CABS)
    assert again.state_digest == base.state_digest and again.trace_digest == base.trace_digest
    saved = Simulation("city_ondemand", overrides=[f"record.dir={tmp_path}", "record.run_name=cabs", *CABS,
                                                   "agents.ondemand.agency=L4"]).run()
    assert replay(saved.run_dir)["identical"]
    frame = next(r for r in saved.records if r["type"] == "frame" and r["tick"] > 900)
    assert len(frame["ondemand"]) == 25 and {row[3] for row in frame["ondemand"]} <= {"idle", "to_pickup", "occupied", "repositioning"}
    lit_sim, lit = run("city_ondemand", [*CABS, "agents.signal.junctions=all"])
    assert lit.metrics["signals"]["crossings"] > 0 and lit.metrics["outcomes"]["wait_time_s_mean"] > base.metrics["outcomes"]["wait_time_s_mean"]
    mixed_sim, mixed = run("city_multimodal", ["agents.ondemand.fleet=15"])                     # next to private cars, buses, charging
    assert mixed.metrics["ondemand"]["rides"] > 0 and {"car", "bus", "ride"} <= set(mixed.metrics["travelers"]["mode_share"])


# ---------------------------------------------------------------- places (POIs) and external events


def test_trips_go_to_places_that_fit_their_purpose_and_events_change_the_day():
    from agmsim.pois import PURPOSE_PLACE

    sim, day = run("town_places")
    w = sim.world
    other, _ = run("town_places", ["simulation.seed=9"])
    assert {p.id: p.node for p in w.pois.values()} == {p.id: p.node for p in other.world.pois.values()}     # places do not move with the seed
    assert len({p.node for p in w.pois.values()}) == len(w.pois) == 16 and w.pois["stadium"].node == 21
    kinds = {p.id: p.type for p in w.pois.values()}
    regular = [t for t in w.travelers.values() if t.poi]
    surge = [t for t in w.travelers.values() if not t.poi and t.origin == 21 and 1800 <= t.t_need * sim.dt < 2700]
    assert all(PURPOSE_PLACE[t.persona.purpose] == kinds[t.poi] and t.destination == w.pois[t.poi].node for t in regular)
    assert len(w.travelers) == 440 and len(regular) >= 315 and len(surge) >= 120       # (a trip from the only clinic keeps its node)
    places = day.metrics["pois"]
    assert places["visits"] == sum(t.state == TState.ARRIVED for t in regular) == sum(places["visits_by_type"].values())
    assert places["found_closed"] == sum(p.found_closed for p in w.pois.values()) > 0                       # the shops shut at 18:30

    events = sim.cfg.interventions
    only = lambda kind: ["events=" + json.dumps([e for e in events if e["type"] == kind])]              # noqa: E731
    _, quiet = run("town_places", ["events=[]"])
    _, rain = run("town_places", only("weather"))
    walk = lambda r: r.metrics["travelers"]["mode_share"].get("walk", 0.0)                             # noqa: E731
    wait = lambda r: r.metrics["outcomes"]["wait_time_s_mean"]                                         # noqa: E731
    assert quiet.metrics["travelers"]["trips"] == 320 and day.metrics["travelers"]["trips"] == 440
    assert walk(rain) < walk(quiet) and rain.metrics["outcomes"]["traffic_speed_mph"] < quiet.metrics["outcomes"]["traffic_speed_mph"]
    assert wait(day) > wait(rain) > wait(quiet)

    # everybody sees the weather while it lasts, and a traveller sees the place they are going to
    seen = []
    run("town_places", only("weather"), policies={"traveler": lambda obs: seen.append(obs) or obs.default})
    wet = [o for o in seen if "environment" in o.context]
    assert wet and all(o.context["environment"]["weather"] == "heavy rain" and 1200 <= o.tick * 2 < 3600 for o in wet)
    assert any("environment" not in o.context for o in seen)
    place = next(o for o in seen if o.trigger == "need").state["trip"]["destination_place"]
    assert set(place) >= {"name", "type"}
    header = next(r for r in day.records if r["type"] == "header")
    assert len(header["pois"]) == 16 and {"id", "name", "type", "node", "closes"} <= set(header["pois"][0])
    frames = [r for r in day.records if r["type"] == "frame"]
    assert any("environment" in f for f in frames) and "environment" not in frames[0]
    _, again = run("town_places")
    assert again.state_digest == day.state_digest
    with pytest.raises(ConfigError, match="external event"):
        Config.from_dict({"events": [{"type": "earthquake", "at_seconds": 0}]})
    with pytest.raises(ConfigError, match="pois"):
        Config.from_dict({"pois": [{"type": "shop"}]})


def test_a_place_that_closes_early_makes_people_look_at_their_plan():
    scenario = {
        "simulation": {"time_step_seconds": 2, "duration_seconds": 7200, "seed": 4, "clock_start": "16:00"},
        "network": {"type": "grid", "rows": 4, "cols": 4, "block_m": 400, "speed_mps": 10, "capacity_scale": 0.5},
        "pois": [{"id": "market", "type": "shop", "node": 5, "name": "Market", "closes": "18:30"}],
        "demand": {"pattern": "uniform", "start_seconds": 1800, "end_seconds": 3000, "min_trip_m": 800,
                   "plan": [{"type": "work", "typical_min": 480},
                            {"type": "errand", "at": {"poi": "shop"}, "duration_min": 20, "optional": True, "priority": 2},
                            {"type": "home", "typical_min": 720}]},
        "events": [{"type": "poi_closure", "poi": "market", "at_seconds": 2400}],            # the market shuts at 16:40 today
        "agents": {"traveler": {"count": 60, "car_ownership": 1.0, "non_llm_policy": "utility", "agency": ["adapt_plan"],
                                "walk": False},
                   "vehicle": {"agency": "L0"}},
    }
    sim, adapted = run(scenario)
    plans = [t.plan for t in sim.world.travelers.values()]
    assert all(p.activities[1].poi == "market" and p.activities[1].node == 5 for p in plans)        # the errand is at the market
    assert sim.world.pois["market"].close_s == 2400 and sim.event_counts["poi_closed"] == 1
    late = [p for p in plans if p.activities[0].end_s > 2400]                                       # had not left work yet
    assert late and all(p.activities[1].close_s == 2400 for p in late)
    assert adapted.metrics["plans"]["reviews"] >= len(late) and adapted.metrics["plans"]["activities_skipped"] > 0
    seen = []
    run(scenario, policies={"traveler": lambda obs: seen.append(obs) or obs.default})
    told = [o for o in seen if o.trigger == "plan_review" and "closes at 16:40 today" in o.context["reason"]]
    at_work = next(o for o in told if o.context["now_at"]["type"] == "work")                        # still to go there
    assert at_work.context["plan"][0]["place"] == "Market" and at_work.context["plan"][0]["closes"] == "16:40"
    assert at_work.has("skip_activity")
    _, followed = run(scenario, ["agents.traveler.agency=L0"])                                      # nobody may change the plan
    assert followed.metrics["plans"]["reviews"] == 0 and followed.metrics["plans"]["activities_skipped"] == 0
    assert followed.metrics["pois"]["found_closed"] > adapted.metrics["pois"]["found_closed"]
    assert adapted.metrics["plans"]["score_usd_mean"] > followed.metrics["plans"]["score_usd_mean"]


# ---------------------------------------------------------------- agency mechanisms: observation, memory, interaction


def test_mechanism_settings_are_written_per_kind_and_checked():
    cfg = Config.load(SCENARIOS / "crossing.yaml", ["agents.vehicle.memory.traffic_seconds=300",
                                                    "agents.vehicle.interaction.share_traffic=true"])
    v = cfg.agents.vehicle
    assert (v.observation.traffic, v.observation.radius_m) == ("local", 900)
    assert v.memory.traffic_seconds == 300 and v.memory.decisions == 4 and v.interaction.share_traffic
    # the earlier names of the vehicle's knowledge of traffic are still understood
    old = Config.from_dict({"agents": {"vehicle": {"link_conditions": "none", "local_radius_m": 500}}})
    assert (old.agents.vehicle.observation.traffic, old.agents.vehicle.observation.radius_m) == ("none", 500)
    with pytest.raises(ConfigError, match="observation.traffic"):
        Config.load("demo_3x3", ["agents.vehicle.observation.traffic=nearby"])
    with pytest.raises(ConfigError, match="find their own way"):          # a bus does not choose its roads
        Config.load("city_multimodal", ["agents.bus.interaction.share_traffic=true"])
    with pytest.raises(ConfigError, match="negative"):
        Config.load("demo_3x3", ["agents.traveler.memory.decisions=-1"])
    sim = Simulation("demo_3x3", overrides=QUIET)
    header = sim.recorder.records[0]["kinds"]
    assert header["vehicle"]["mechanisms"]["observation"] == {"traffic": "global"}
    assert header["traveler"]["mechanisms"] == {"memory": {"decisions": 4}}


def test_memory_reaches_every_decision_model_with_the_observation():
    """What an agent remembers is kept by the framework: a function, a rule and a language model get the same."""
    seen = []

    def watcher(obs):
        seen.append((obs.agent_id, obs.memory))
        return obs.default

    ov = ["agents.vehicle.agency=L1", "agents.vehicle.memory.decisions=2"]
    run(SCENARIOS / "incident.yaml", ov, policies={"vehicle": watcher})
    lengths = {len(memory) for _, memory in seen}
    assert lengths == {0, 1, 2}                               # never more than the two latest entries
    first = next(memory for _, memory in seen if memory)[0]
    assert first["did"] in ("changed route", "kept route") and first["trigger"] == "route" and "at" in first
    assert set(first) == {"at", "trigger", "did", "route", "quickest_other_route"}  # the routes by what they were,
    assert first["did"] == "kept route" and "args" not in first                     # not by their numbers
    seen.clear()
    run(SCENARIOS / "incident.yaml", ["agents.vehicle.agency=L1", "agents.vehicle.memory.decisions=0"],
        policies={"vehicle": watcher})
    assert seen and all(memory == () for _, memory in seen)
    # outcomes are remembered too, and a language model reads the memory as part of its observation
    from agmsim.policies.llm_policy import build_user_prompt
    _, res = run(SCENARIOS / "incident.yaml", ["agents.vehicle.agency=L1", "agents.vehicle.use_llm=true", "llm.backend=mock"])
    shown = [r["obs"] for r in res.records if r.get("type") == "decision" and r["kind"] == "vehicle" and "obs" in r]
    remembered = next(o for o in shown if o.get("memory"))
    assert '"memory":[{"at"' in build_user_prompt(remembered)
    assert '"memory"' not in build_user_prompt(remembered, memory=0)
    travellers, _ = run()
    assert any(item.get("outcome") == "trip_completed" for t in travellers.world.travelers
               for item in travellers.memories.of(t))


def test_what_a_car_knows_follows_from_what_it_sees_remembers_and_is_told():
    """One bridge is blocked. Cars that see only what is near find out at the bridge; memory keeps them from
    going back to it, and cars that talk to each other find out before they get there."""
    V = "agents.vehicle."

    def crossing(*overrides):
        return run(SCENARIOS / "crossing.yaml", list(overrides))[1]

    def basis(res):
        """What the cars' estimates rested on, wherever they did not see for themselves."""
        found = [r["obs"]["context"]["traffic_knowledge"] for r in res.records
                 if r.get("type") == "decision" and r["kind"] == "vehicle" and "traffic_knowledge" in r["obs"]["context"]]
        return found, [row for b in found for row in b.get("reports_about_roads_you_do_not_see", ())]

    runs = {"near": crossing(), "blind": crossing(V + "observation.traffic=none"),
            "everything": crossing(V + "observation.traffic=global"),
            "remembering": crossing(V + "memory.traffic_seconds=600"),
            "told": crossing(V + "memory.traffic_seconds=600", V + "interaction.share_traffic=true")}
    m = {name: res.metrics for name, res in runs.items()}
    time = {name: metrics["travelers"]["trip_time_s_mean"] for name, metrics in m.items()}
    assert time["blind"] > time["near"] > time["remembering"] > time["told"] > time["everything"]
    assert m["blind"]["vehicles"]["reroutes"] == 0            # nothing to go on: the level alone does nothing
    assert m["remembering"]["vehicles"]["reroutes"] < m["near"]["vehicles"]["reroutes"]   # no going back to the blocked bridge
    assert "knowledge" not in m["near"] and m["told"]["knowledge"]["reports_heard"] > 0
    assert 0 < m["told"]["knowledge"]["share_of_agents_that_heard"] <= 1
    # a report is about a road the car does not see itself, and says how old it is and where it came from
    assert basis(runs["everything"]) == ([], [])              # a car that knows every road is told nothing more
    found, rows = basis(runs["near"])                         # one that sees what is near is told how far it sees
    assert found and not rows and all(b["you_see_roads_within_km"] == 0.9 and "roads_you_do_not_see" in b for b in found)
    found, rows = basis(runs["remembering"])
    assert rows and {row["source"] for row in rows} == {"seen earlier"}
    found, rows = basis(runs["told"])
    assert "told by another vehicle" in {row["source"] for row in rows}
    assert all(0 < row["speed_pct_of_normal"] <= 100 and row["age_min"] <= 10 for row in rows)
    assert any(row["speed_pct_of_normal"] == 100 for row in rows)      # "the bridge is clear again" gets around too
    assert all(b["you_see_roads_within_km"] == 0.9 for b in found)


def test_without_memory_a_report_does_not_outlive_the_exchange_it_came_in():
    from agmsim.mechanisms import PERIOD_SECONDS
    V = "agents.vehicle."
    sim = Simulation(SCENARIOS / "crossing.yaml", overrides=[*QUIET, V + "interaction.share_traffic=true"])
    period = sim.world.ticks(PERIOD_SECONDS)
    while sim.step():
        if sim.world.tick > 600 and sim.traffic.reports and sim.world.tick % period == 1:
            newest = max(report[1] for store in sim.traffic.reports.values() for report in store.values())
            assert all(report[1] == newest for v in sim.kinds["vehicle"].on_the_road()
                       for report in sim.traffic.reports.get(v.id, {}).values())
    assert sim.traffic.metrics()["reports_heard"] > 0


def test_a_decision_can_be_remembered_with_the_reason_given_for_it():
    seen = []

    def watcher(obs):
        seen.append(obs.memory)
        best = min(obs.context["routes"], key=lambda r: (r["eta_min"], r["route"]))
        return Decision("choose_route", {"route": best["route"]}, "the quickest as far as I can tell " + "x" * 400)

    run(SCENARIOS / "crossing.yaml", ["agents.vehicle.memory.reasons=true"], policies={"vehicle": watcher})
    items = [item for memory in seen for item in memory if "did" in item]
    assert items and all(item["why"].startswith("the quickest") and len(item["why"]) == 280 for item in items)
    changed = next(item for item in items if item["did"] == "changed route")
    assert set(changed) == {"at", "trigger", "did", "left_route", "took_route", "why"}
    assert changed["left_route"]["eta_min"] > changed["took_route"]["eta_min"]
    seen.clear()
    run(SCENARIOS / "crossing.yaml", [], policies={"vehicle": watcher})
    assert seen and not any("why" in item for memory in seen for item in memory)


def test_a_car_that_recalls_its_last_change_of_route_stops_going_back_and_forth():
    """A car that sees only what is near and keeps no road in mind returns to the blocked bridge again and
    again. The algorithm's use of its memory of decisions stops that -- if the car has such a memory."""
    V = "agents.vehicle."
    settle = V + "policy_params={settle_min: 10, resettle_saving_min: 5}"
    _, plain = run(SCENARIOS / "crossing.yaml")
    _, blank = run(SCENARIOS / "crossing.yaml", [settle, V + "memory.decisions=0"])
    _, recalls = run(SCENARIOS / "crossing.yaml", [settle, V + "memory.decisions=4"])
    assert blank.state_digest == plain.state_digest               # nothing remembered: nothing to go on
    p, r = plain.metrics, recalls.metrics
    assert p["vehicles"]["vehicles_with_3plus_reroutes"] > 10 > r["vehicles"]["vehicles_with_3plus_reroutes"]
    assert r["vehicles"]["reroutes"] < p["vehicles"]["reroutes"]
    assert r["travelers"]["trip_time_s_mean"] < p["travelers"]["trip_time_s_mean"]
    held = [x for x in recalls.records if x.get("type") == "decision" and x["kind"] == "vehicle"
            and "changed route" in x["decision"].get("rationale", "")]
    assert held and all(x["decision"]["args"] == {"route": 0} and x["obs"]["memory"] for x in held)


def test_a_car_is_told_when_the_route_it_chose_turns_out_much_slower_than_it_reckoned():
    """What came of a decision goes into memory when it shows: here, at the next look at the route."""
    sim, res = run(SCENARIOS / "crossing.yaml")
    shown = [r["obs"] for r in res.records if r.get("type") == "decision" and r["kind"] == "vehicle" and "obs" in r]
    told = [item for o in shown for item in o.get("memory", ()) if item.get("outcome") == "route_slower_than_reckoned"]
    assert told and all(item["now_min_to_go"] - item["reckoned_min_to_go"] >= 5 and "route_chosen_at" in item for item in told)
    # it is the choice just before that is meant: the entry names the time of a remembered change of route
    o = next(o for o in shown if any(item.get("outcome") for item in o.get("memory", ())) and len(o["memory"]) >= 2)
    k = next(i for i, item in enumerate(o["memory"]) if item.get("outcome"))
    assert k > 0 and o["memory"][k - 1]["did"] == "changed route" and o["memory"][k - 1]["at"] == o["memory"][k]["route_chosen_at"]
    # cars that know every road are not caught out that way
    _, clear = run(SCENARIOS / "crossing.yaml", ["agents.vehicle.observation.traffic=global"])
    assert not any(item.get("outcome") == "route_slower_than_reckoned" for r in clear.records
                   if r.get("type") == "decision" and r["kind"] == "vehicle" and "obs" in r for item in r["obs"].get("memory", ()))


def test_a_language_model_can_be_told_to_read_its_memory_first():
    from agmsim.policies.llm_policy import REFLECT
    ov = ["agents.vehicle.agency=L1", "agents.vehicle.use_llm=true", "llm.backend=mock"]
    plain = Simulation(SCENARIOS / "incident.yaml", overrides=[*QUIET, *ov])
    told = Simulation(SCENARIOS / "incident.yaml", overrides=[*QUIET, *ov, "agents.vehicle.llm_policy_params={reflect: true}"])
    agent = next(iter(plain.world.vehicles))
    assert REFLECT not in plain.policies[agent].system and told.policies[agent].system.endswith(REFLECT)
    plain.close(), told.close()


def test_routes_can_be_shown_and_remembered_by_the_junctions_they_pass():
    """A route's number changes from one look to the next (route 0 is the route the car is on); the junctions
    it passes do not, so a car can tell later whether an offered route is one it left."""
    on = "agents.vehicle.observation.route_roads=true"
    sim, res = run(SCENARIOS / "crossing.yaml", [on])
    net = sim.world.net
    shown = [r["obs"] for r in res.records if r.get("type") == "decision" and r["kind"] == "vehicle" and "obs" in r]
    for o in shown[:40]:
        start = o["state"]["location"].get("next_node", o["state"]["location"].get("node"))
        for row in o["context"]["routes"]:
            via = row["via_nodes"]
            assert via[0] == start and via[-1] == o["state"]["destination_node"]
            assert all(net.link_between(a, b) is not None for a, b in zip(via, via[1:]))     # a path along roads
            assert abs(sum(net.link_between(a, b).length for a, b in zip(via, via[1:])) / 1000 - row["km"]) < 0.4
    changed = next(item for o in shown for item in o.get("memory", ()) if item.get("did") == "changed route")
    assert changed["left_route"]["via_nodes"] != changed["took_route"]["via_nodes"]
    # a car that goes back is offered the very road it left: the blocked bridge 3 -> 20 is in both
    back = next(o for o in shown if o.get("memory") and o["memory"][-1].get("did") == "changed route"
                and [3, 20] in [o["memory"][-1]["left_route"]["via_nodes"][i:i + 2] for i in range(8)]
                and any([3, 20] in [row["via_nodes"][i:i + 2] for i in range(len(row["via_nodes"]))]
                        for row in o["context"]["routes"][1:]))
    assert back["context"]["routes"][0]["is_current_route"]
    # off by default, and the run itself is the same either way
    _, plain = run(SCENARIOS / "crossing.yaml")
    assert plain.state_digest == res.state_digest
    assert not any("via_nodes" in row for r in plain.records if r.get("type") == "decision" and r["kind"] == "vehicle"
                   and "obs" in r for row in r["obs"]["context"]["routes"])
    with pytest.raises(ConfigError, match="find their own way"):
        Config.load("city_multimodal", ["agents.bus.observation.route_roads=true"])


def test_routes_can_be_shown_and_remembered_by_the_name_of_their_way():
    """A short name instead of the junctions: the same name for the same way at every decision, so that a
    remembered route can be recognised; routes that share most of their roads share a name."""
    names, roads = "agents.vehicle.observation.route_names=true", "agents.vehicle.observation.route_roads=true"
    sim, res = run(SCENARIOS / "crossing.yaml", [names, roads])
    shown = [r for r in res.records if r.get("type") == "decision" and r["kind"] == "vehicle" and "obs" in r]
    bridge = {(3, 20), (20, 3)}                       # the blocked one
    on, seen, left, returns = {}, {}, {}, 0
    for r in shown:
        o, car = r["obs"], r["agent"]
        rows = o["context"]["routes"]
        assert list(rows[0])[:2] == ["route", "name"] and "your memory" in o["context"]["route_names"]
        if car in on:
            assert rows[0]["name"] == on[car]         # the route it is on keeps the name it chose
        over = [bool(bridge & set(zip(row["via_nodes"], row["via_nodes"][1:]))) for row in rows]
        assert not ({row["name"] for row, o_ in zip(rows, over) if o_} & {row["name"] for row, o_ in zip(rows, over) if not o_})
        for row in rows:                              # the very same roads offered again have the same name
            assert seen.setdefault((car, tuple(row["via_nodes"])), row["name"]) == row["name"]
        quickest = min(rows, key=lambda row: (row["eta_min"], row["route"]))
        if not over[0] and over[quickest["route"]] and any("outcome" in item for item in o.get("memory", ())):
            returns += 1                              # a way back after having been caught out: it has the name the car left
            assert quickest["name"] in left[car]
        chosen = r["decision"]["args"]["route"]
        if chosen:
            left.setdefault(car, set()).add(rows[0]["name"])
        on[car] = rows[chosen]["name"]
    assert returns > 20
    changed = next(item for r in shown for item in r["obs"].get("memory", ()) if item.get("did") == "changed route")
    assert list(changed["left_route"])[0] == "name" and changed["left_route"]["name"] != changed["took_route"]["name"]
    assert max(len({name for (car, _), name in seen.items() if car == c}) for c in on) <= 4      # a few ways, not a name per route
    # off by default, the run itself is the same either way, and only cars have ways
    _, plain = run(SCENARIOS / "crossing.yaml")
    assert plain.state_digest == res.state_digest
    assert not any("name" in row or "route_names" in r["obs"]["context"] for r in plain.records
                   if r.get("type") == "decision" and r["kind"] == "vehicle" and "obs" in r for row in r["obs"]["context"]["routes"])
    assert sim._mechanisms("vehicle")["observation"]["route_names"] is True
    with pytest.raises(ConfigError, match="find their own way"):
        Config.load("city_multimodal", ["agents.bus.observation.route_names=true"])


def test_a_new_destination_starts_new_names():
    from types import SimpleNamespace
    from agmsim.mechanisms import RouteNames
    sim, _ = run(SCENARIOS / "crossing.yaml")
    net = sim.world.net
    names = RouteNames(sim)
    a, b = net.route(0, 39).links, net.route(16, 39).links
    route = lambda links: SimpleNamespace(links=tuple(links))
    assert names.give("car", [route(a), route(b)], 39) == ["A", "B"]
    assert names.give("car", [route(b[1:]), route(a)], 39) == ["B", "A"]            # further along, in another order
    assert names.give("car", [route(a[:1] + b[1:])], 39) == ["B"]                    # mostly B's roads: B's way
    assert names.give("car", [route(a)], 12) == ["C"]                                # the same roads to another place


def test_the_testbed_is_a_simplified_montreal_with_every_kind_of_agent():
    from agmsim.config import list_scenarios
    from agmsim.montreal import NODE, NODES, ROADS
    assert list_scenarios()[0] == "montreal"                     # what the page opens with
    sim, res = run("montreal", ["simulation.duration_seconds=1200"])
    net = sim.world.net
    assert len(net.nodes) == len(NODES) == 37 and len(net.links) == 2 * len(ROADS) == 152
    assert net.names[NODE["downtown"]] == "Downtown" and sim._header()["network"]["names"]["0"] == "Downtown"
    # an island: the South Shore is reached over the crossings only
    crossings = {"Champlain Bridge", "Victoria Bridge", "Jacques-Cartier Bridge", "Lafontaine Tunnel", "Mercier Bridge"}
    south = {NODE[name] for name in ("longueuil", "brossard", "st_lambert", "boucherville", "kahnawake")}
    laval = {NODE[name] for name in ("laval_w", "laval_c", "laval_e")}
    into_south = {net.road_names[(link.u, link.v)] for link in net.links if link.v in south and link.u not in south | laval}
    assert into_south == crossings
    assert 20_000 < net.length(NODE["west_island"], NODE["downtown"]) < 32_000
    assert set(sim.kinds) == {"traveler", "vehicle", "ondemand", "bus", "evcs", "signal", "operator"}
    assert res.metrics["travelers"]["trips"] > 0 and res.metrics["outcomes"]["vmt_miles"] > 0

