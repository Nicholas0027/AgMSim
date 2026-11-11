"""The LLM decision path.

One generic policy serves every kind of agent. The system prompt is assembled
from what the agent's kind declares about itself (who it is, why it may be
asked, what each action means); the user prompt carries the observation as
JSON, exactly the structure a non-LLM policy receives. The model's reply is
parsed, validated against the permitted actions, re-asked once with the error
if invalid, and replaced by the agent's non-LLM policy if it stays invalid --
every such fallback is counted and recorded.
"""
from __future__ import annotations

import json
import re
from collections import deque

from ..actions import Decision, as_decision, validate_decision
from ..llm import LLMBudgetExceeded, LLMRequest, LLMTransientError
from ..observation import Observation
from .base import Policy, register_policy

OBS_OPEN, OBS_CLOSE = "<observation>", "</observation>"

PROTOCOL = """How you are asked to decide
- You are consulted only at decision points. Each time you receive an observation as JSON with: "trigger" (why you are asked now), "state" (your own state), "context" (what you can currently see) and "permitted_actions".
- "permitted_actions" is the complete list of what you may do right now. Each entry names an action and, where relevant, its arguments with the allowed values ("choices"). Arguments marked "required": false may be left out.
- Choose exactly one permitted action. Anything else is rejected.
- Distances are in km, durations in minutes unless a key says otherwise, money in USD.

Answer format
Reply with one JSON object and nothing else:
"""
ANSWER = '{"action": "<name of a permitted action>", "args": {"<argument>": <value>}, "rationale": "<one short sentence giving the main reason>"}'
ANSWER_BARE = '{"action": "<name of a permitted action>", "args": {"<argument>": <value>}}'
# The reasoning comes first, so that a model without a reasoning phase of its own works the options
# out before it commits to one (in ANSWER the reason is written after the choice is already made).
ANSWER_REASONED = ('{"reasoning": "<work out, in at most three short sentences and with the numbers you were given, '
                   'what each option means for your objective>", "action": "<name of a permitted action>", '
                   '"args": {"<argument>": <value>}}')
_ANSWERS = {True: ANSWER, False: ANSWER_BARE, "first": ANSWER_REASONED}
_REMINDERS = {True: '{"action": ..., "args": {...}, "rationale": "..."}', False: '{"action": ..., "args": {...}}',
              "first": '{"reasoning": "...", "action": ..., "args": {...}}'}


def rationale_mode(value):
    """Normalise the ``rationale`` setting: true (reason after the choice), false (none), "first" (reason before)."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("first", "before"):
            return "first"
        if text in ("true", "after", "yes"):
            return True
        if text in ("false", "none", "no"):
            return False
        raise ValueError(f"rationale must be true, false or 'first', got {value!r}")
    return bool(value)


def build_system_prompt(kind, rationale=True) -> str:
    """Stable system prompt for all agents of one kind (identical across calls, so providers can cache it)."""
    lines = [kind.role_prompt, "This is a transportation simulation used for research.", "",
             PROTOCOL + _ANSWERS[rationale_mode(rationale)], "", "Why you may be asked (trigger)"]
    lines += [f'- "{name}": {text}' for name, text in kind.triggers.items()]
    lines += ["", "What the actions mean"]
    lines += [f'- "{name}": {text}' for name, text in kind.actions.items()]
    return "\n".join(lines)


def build_user_prompt(obs, history=(), rationale=True) -> str:
    """``obs`` is an Observation or its dict form (as stored in a trace)."""
    data = obs.to_dict() if isinstance(obs, Observation) else obs
    parts = [OBS_OPEN, json.dumps(data, ensure_ascii=False, separators=(",", ":")), OBS_CLOSE]
    if history:
        parts.append("Your recent history (oldest first):")
        parts += [f"- {line}" for line in history]
    parts.append("Reply with one JSON object: " + _REMINDERS[rationale_mode(rationale)])
    return "\n".join(parts)


def observation_from_prompt(user_prompt: str) -> Observation:
    """Recover the observation from a user prompt (used by the mock backend)."""
    start, end = user_prompt.index(OBS_OPEN) + len(OBS_OPEN), user_prompt.index(OBS_CLOSE)
    return Observation.from_dict(json.loads(user_prompt[start:end]))


def parse_decision_text(text: str) -> dict:
    """Extract the decision object from a model reply (tolerates reasoning blocks, code fences, extra prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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
                    if isinstance(obj, dict) and "action" in obj:
                        if isinstance(obj.get("args"), dict):
                            obj["args"] = {k: v for k, v in obj["args"].items() if v is not None}
                        reasoning = obj.pop("reasoning", None)       # reasoning-first answers: keep it as the rationale
                        if reasoning and not obj.get("rationale"):
                            obj["rationale"] = reasoning if isinstance(reasoning, str) else json.dumps(reasoning)
                        return obj
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object with an 'action' field found in the reply")


@register_policy("llm", role="any")
class LLMPolicy(Policy):
    """LLM-driven agent: persona and objective in the observation, short memory, validated output.

    Parameters: ``memory`` (number of past decisions and outcomes shown, default 4),
    ``rationale`` (true, the default: a one-sentence reason after the choice; false: the choice
    only, which roughly halves the output tokens and so the time per decision; "first": the model
    works the options out *before* it names its choice, which costs more output tokens),
    ``system_prompt`` (replace the generated system prompt), ``extra_instructions`` (appended to
    it -- useful for prompt-sensitivity studies).
    """

    concurrent = True
    uses_llm = True

    def __init__(self, memory: int = 4, rationale=True, system_prompt: str | None = None,
                 extra_instructions: str = "", **params):
        super().__init__(**params)
        self.rationale = rationale_mode(rationale)
        self.history = deque(maxlen=max(0, int(memory)) or None) if memory else deque(maxlen=0)
        self._system_override = system_prompt
        self._extra = extra_instructions

    def bind(self, ctx):
        super().bind(ctx)
        self.llm = ctx.llm
        self.fallback = ctx.fallback_factory()
        self.system = self._system_override or build_system_prompt(ctx.extras["kind"], self.rationale)
        if self._extra:
            self.system += "\n\n" + self._extra
        self.retries = int(self.llm.cfg.retries_on_invalid)

    def decide(self, obs):
        if len(obs.actions) == 1 and all(p.choices is not None and len(p.choices) == 1 for p in obs.actions[0].params):
            only = obs.actions[0]               # one permitted action, one value per argument: nothing to choose
            return Decision(only.name, {p.name: p.choices[0] for p in only.params}, "the only option",
                            {"llm": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "replayed": 0}})
        user = build_user_prompt(obs, self.history, self.rationale)
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "replayed": 0}
        decision, error, reply = None, None, ""
        try:
            for _ in range(1 + self.retries):
                prompt = user if error is None else (
                    f"{user}\nYour previous reply was rejected: {error}\nReply again with one JSON object "
                    "that uses a permitted action.")
                response = self.llm.complete(LLMRequest(self.system, prompt, {"agent": obs.agent_id}))
                reply = response.text
                usage["calls"] += 1
                usage["replayed"] += int(response.cached)
                usage["input_tokens"] += response.input_tokens
                usage["output_tokens"] += response.output_tokens
                usage["latency_ms"] += response.latency_ms
                try:
                    decision, error = validate_decision(parse_decision_text(reply), obs.actions)
                except ValueError as exc:
                    decision, error = None, str(exc)
                if decision is not None:
                    break
        except LLMBudgetExceeded as exc:
            error = f"budget exhausted ({exc})"
        except LLMTransientError as exc:
            error = f"model unavailable ({exc})"
        usage["latency_ms"] = round(usage["latency_ms"], 1)
        info = {"llm": usage}
        if decision is None:
            chosen = as_decision(self.fallback.decide(obs))
            decision = Decision(chosen.action, dict(chosen.args), chosen.rationale, info)
            info["fallback"] = error
            info["rejected_reply"] = reply[:400]
        else:
            decision.info.update(info)
        self.history.append(f"{obs.to_dict()['time']} {obs.trigger}: chose {decision.action} "
                            f"{json.dumps(decision.args, ensure_ascii=False)}")
        return decision

    def feedback(self, event: dict) -> None:
        self.fallback.feedback(event)
        self.history.append("outcome: " + json.dumps(event, ensure_ascii=False))
