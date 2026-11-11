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
