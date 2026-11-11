"""Configure a run by talking to it.

The assistant turns what a researcher writes ("300 travellers, let the vehicles decide with the
LLM at level 3, and slow the bridge down from 17:30") into a scenario choice and configuration
settings. A language model proposes them as JSON; the proposal is then checked the way any
configuration is (unknown keys, wrong values, kinds the scenario does not have, links that do not
exist), and one that does not pass is sent back to the model once, with the reason.

The model changes nothing by itself: :func:`configure` returns the proposal, the page shows it in
its form, and the researcher can edit it before pressing Run.
"""
from __future__ import annotations

import json
from typing import Optional

from .agency import AgencyError, parse_agency
from .config import LLM_PRESETS, Config, ConfigError, LLMCfg, list_scenarios
from .kinds.base import KINDS
from .llm import LLMError, LLMRequest, build_llm_client
from .network import build_network
from .policies.base import available_policies

#: What the assistant may set. Model access, recording and plug-ins are not among them.
ALLOWED = ("simulation.", "network.", "demand.", "agents.", "scoring.", "interventions")

KEYS = """simulation.seed (integer)
simulation.duration_seconds (number): length of the run
agents.traveler.count (integer): number of travellers
agents.traveler.car_ownership (0 to 1): share of travellers who own a private agentic vehicle
agents.traveler.default_mode (car | bus | walk): how a traveller without authority travels
agents.<kind>.agency (L0 | L1 | L2 | L3 | L4): what that kind of agent may decide; see the levels above
agents.<kind>.use_llm (true | false): true means the kind is decided by a language model
agents.<kind>.non_llm_policy (a built-in decision model of that kind, listed above): used when use_llm is false, and as the fallback of a language model
agents.<kind>.mix (a list such as [{"share": 0.3, "agency": "L1"}]): that share of the kind gets other settings than the rest
network.capacity_scale (number): road capacity; smaller means more congestion
network.congestion (true | false): whether traffic slows roads down at all
agents.vehicle.link_conditions (none | local | global): how much a vehicle knows about current traffic
agents.bus.fare (USD), agents.evcs.price_per_kwh (USD)
agents.signal.junctions ("all", or a list of node numbers; an empty list means no traffic lights), agents.signal.green_seconds (number): green per direction in the fixed-time plan
agents.operator.review_seconds (number): how often the orchestrator (the kind named operator) reviews the network
agents.operator.initial_toll (USD), agents.operator.initial_bus_priority (true | false), agents.operator.initial_advisory (true | false)
demand.start_seconds, demand.end_seconds (seconds from the start of the run): the period in which trips begin
interventions (a list that replaces the scenario's own): each entry is
  {"type": "link_condition", "at_seconds": <start>, "until_seconds": <end>, "factor": <speed multiplier such as 0.1>, "links": [[<node>, <node>], ...]} or
  {"type": "set_agency", "at_seconds": <when>, "kind": "<kind>", "agency": "<level>"}"""

ANSWER = ('{"reply": "<one or two sentences for the researcher: what you set, and anything you could not do>", '
          '"scenario": "<a scenario name, or null to keep the current one>", '
          '"set": {"<key>": <value>, ...}, "run": <true only if they asked to start the run>}')


def system_prompt() -> str:
    """Stable part of the prompt: what the simulator offers and how to answer."""
    lines = ["You help a researcher set up a run of AgMSim, a mobility simulator in which every kind of agent "
             "(travellers, their private agentic vehicles, buses, charging stations, traffic lights, and the "
             "orchestrator, whose kind is named operator) has a "
             "decision model and an agency level. Turn the researcher's request into settings.", "",
             "Answer with one JSON object and nothing else:", ANSWER, "",
             "Rules",
             "- Use only the keys listed under Keys, with values of the stated kind. Put only what should change into \"set\".",
             "- Only kinds that the scenario has can be set. When you switch scenario, work from that scenario's own "
             "clock, duration and interventions as listed under Scenarios.",
             "- Times in interventions are seconds from the start of the run. If the scenario has a clock (clock_start), "
             "convert times of day: seconds = time of day minus clock_start.",
             "- Do not invent node numbers. For an incident, reuse links that the scenario's interventions already use, "
             "unless the researcher names nodes.",
             "- For a relative request (more, less, twice as long), start from the values given under the current state.",
             "- If the request is unclear, or cannot be done with these keys, set nothing and say so in \"reply\".",
             "- Write \"reply\" in the language the researcher writes in.", "",
             "Scenarios"]
    for name in list_scenarios():
        cfg = Config.load(name)
        lines.append(f"- {name}: {cfg.description} Kinds: {', '.join(cfg.active_kinds())}. clock_start {cfg.simulation.clock_start}, "
                     f"duration {cfg.simulation.duration_seconds:g} s, {cfg.agents.traveler.count} travellers. "
                     f"Its interventions: {json.dumps(cfg.interventions)}")
    lines += ["", "Kinds of agent, their built-in decision models, and what each agency level adds"]
    for name in KINDS.names():
        kind = KINDS.get(name)
        models = [p for p in available_policies(name) if p not in ("llm", "remote")]
        lines.append(f"- {name} ({kind.title}); built-in decision models: {', '.join(models)}")
        for level, permissions in sorted(kind.agency.levels.items()):
            what = "; ".join(kind.agency.permissions[p] for p in permissions) or "follows the default behaviour"
            lines.append(f"    {level}: {what}")
    lines += ["", "Keys", KEYS]
    return "\n".join(lines)


def _state(scenario: str, settings: dict) -> str:
    """The moving part of the prompt: where the form stands now."""
    cfg = Config.load(scenario, dict(settings)) if settings else Config.load(scenario)
    sim = cfg.simulation
    agents = {k: {"agency": getattr(cfg.agents, k).agency, "decided_by": "llm" if getattr(cfg.agents, k).use_llm
                  else getattr(cfg.agents, k).non_llm_policy} for k in cfg.active_kinds()}
    return "\n".join([
        f"Current scenario: {scenario} ({cfg.description})",
        f"clock_start: {sim.clock_start}; duration_seconds: {sim.duration_seconds:g}; travellers: {cfg.agents.traveler.count}",
        f"Agents now: {json.dumps(agents)}",
        "Values now: " + json.dumps({
            "network.capacity_scale": cfg.network.capacity_scale, "network.congestion": cfg.network.congestion,
            "agents.traveler.car_ownership": cfg.agents.traveler.car_ownership,
            "agents.traveler.default_mode": cfg.agents.traveler.default_mode,
            "agents.vehicle.link_conditions": cfg.agents.vehicle.link_conditions,
            "agents.bus.fare": cfg.agents.bus.fare, "agents.evcs.price_per_kwh": cfg.agents.evcs.price_per_kwh,
            "agents.operator.initial_toll": cfg.agents.operator.initial_toll,
            "demand.start_seconds": cfg.demand_window()[0], "demand.end_seconds": cfg.demand_window()[1]}),
        f"Interventions now: {json.dumps(cfg.interventions)}",
        f"Settings changed so far: {json.dumps(settings)}"])


def extract_json(text: str) -> dict:
    """The first JSON object in a model reply (tolerates reasoning blocks, code fences and prose around it)."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    start = text.find("{")
    while start != -1:
        depth, in_string, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    raise ValueError("the answer was not a JSON object")


def check(scenario: str, settings: dict) -> Optional[str]:
    """Why these settings cannot be used, or None if they can."""
    if scenario not in list_scenarios():
        return f"there is no scenario '{scenario}'; scenarios: {', '.join(list_scenarios())}"
    for key in settings:
        if not str(key).startswith(ALLOWED):
            return f"'{key}' cannot be set here; keys start with: {', '.join(ALLOWED)}"
    try:
        cfg = Config.load(scenario, dict(settings))
    except (ConfigError, ValueError, TypeError) as exc:
        return str(exc)
    active = cfg.active_kinds()
    for key in settings:
        parts = str(key).split(".")
        if parts[0] == "agents" and len(parts) > 1 and parts[1] not in active:
            return f"scenario '{scenario}' has no {parts[1]}; its kinds are: {', '.join(active)}"
    for name in active:
        kcfg, kind = getattr(cfg.agents, name), KINDS.get(name)
        try:
            parse_agency(kind.agency, kcfg.agency)
            for part in kcfg.mix:
                if "share" not in part:
                    return f"each entry of agents.{name}.mix needs a 'share'"
                if "agency" in part:
                    parse_agency(kind.agency, part["agency"])
        except (AgencyError, ValueError, TypeError) as exc:
            return f"agents.{name}: {exc}"
        models = [p for p in available_policies(name) if p not in ("llm", "remote")]
        if kcfg.non_llm_policy not in models:
            return f"agents.{name}.non_llm_policy must be one of {models}"
    net = build_network(cfg.network, cfg.base_dir)
    for item in cfg.interventions:
        if not isinstance(item, dict) or item.get("type") not in ("link_condition", "set_agency"):
            return "each intervention needs \"type\": \"link_condition\" or \"set_agency\""
        if item["type"] == "link_condition":
            if not all(k in item for k in ("at_seconds", "factor", "links")):
                return "a link_condition intervention needs at_seconds, factor and links"
            for pair in item["links"]:
                if len(pair) != 2 or net.link_between(int(pair[0]), int(pair[1])) is None:
                    return f"there is no road between nodes {pair}"
        elif item.get("kind") not in active or "agency" not in item or "at_seconds" not in item:
            return "a set_agency intervention needs at_seconds, a kind the scenario has, and agency"
    return None


def _client(llm: dict):
    spec = {k: llm[k] for k in ("preset", "backend", "base_url", "model", "api_key_env") if llm.get(k)}
    if not spec or "mock" in (spec.get("preset"), spec.get("backend")):
        raise LLMError("the assistant needs a language model: choose one under Language model "
                       "(the offline stand-in cannot read a request)")
    preset = LLM_PRESETS.get(spec.get("preset") or "", {})
    spec["max_tokens"] = max(900, preset.get("max_tokens") or 0)
    return build_llm_client(LLMCfg(**spec))


def configure(body: dict) -> dict:
    """Answer one turn of the conversation.

    ``body``: ``messages`` (``[{"role": "user" | "assistant", "text": ...}, ...]``, the last one from
    the researcher), ``scenario`` (the one selected in the form), ``settings`` (what the form sets
    now, as ``{key: value}``) and ``llm`` (``{"preset": ...}`` or ``{"backend", "base_url", "model"}``).
    Returns ``reply``, ``scenario``, ``set`` (settings to change), ``run`` and, if the model could
    not be reached or gave nothing usable, ``error``.
    """
    scenario = body.get("scenario") or list_scenarios()[0]
    settings = {k: v for k, v in (body.get("settings") or {}).items() if str(k).startswith(ALLOWED)}
    messages = [m for m in body.get("messages") or [] if m.get("text")]
    nothing = {"scenario": scenario, "set": {}, "run": False}
    if not messages:
        return {**nothing, "reply": "Tell me what you would like to run."}
    try:
        client = _client(body.get("llm") or {})
        system = system_prompt()
        talk = "\n".join(f"{'Researcher' if m.get('role') == 'user' else 'Assistant'}: {m['text']}" for m in messages[-12:])
        user = f"{_state(scenario, settings)}\n\nConversation\n{talk}\n\nAnswer with the JSON object."
        error = None
        for _ in range(2):
            prompt = user if error is None else (f"{user}\n\nYour previous answer could not be used: {error}\n"
                                                 "Answer again with a corrected JSON object.")
            text = client.complete(LLMRequest(system, prompt, {"agent": "assistant"})).text
            try:
                answer = extract_json(text)
            except ValueError as exc:
                error = str(exc)
                continue
            chosen = answer.get("scenario") or scenario
            changes = answer.get("set") if isinstance(answer.get("set"), dict) else {}
            error = check(chosen, {**(settings if chosen == scenario else {}), **changes})
            if error is None:
                return {"reply": str(answer.get("reply") or "Done."), "scenario": chosen, "set": changes,
                        "run": bool(answer.get("run")), "usage": client.summary()}
        return {**nothing, "reply": f"I could not turn that into settings that work ({error}). Nothing was changed.",
                "error": error}
    except (LLMError, ConfigError) as exc:
        return {**nothing, "reply": str(exc), "error": str(exc)}
