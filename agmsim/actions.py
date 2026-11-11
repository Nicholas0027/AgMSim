"""Actions, decisions and their validation.

Every decision model -- rule, optimisation, learned policy or LLM -- acts
through the same interface: it receives an observation listing the actions it
is permitted to take right now (:class:`ActionSpec`) and returns one
:class:`Decision`. The environment validates the decision against that list,
so no decision model can exceed the authority its agent was given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Argument vocabulary shared by all built-in actions (name -> JSON type).
#: Plugins that introduce new actions may add entries.
ARG_TYPES: dict[str, str] = {
    "option": "string",
    "request_id": "string",
    "node": "integer",
    "route": "integer",
    "price_multiplier": "number",
    "message": "string",
    "station": "string",
    "vehicle_id": "string",
    "seconds": "integer",
    "stop": "integer",
    "usd": "number",
    "time": "string",
    "activity": "integer",
}

MAX_MESSAGE_CHARS = 280


@dataclass(frozen=True)
class Param:
    name: str
    type: str = "string"                 # string | integer | number
    choices: Optional[tuple] = None      # allowed values; None means free-form
    required: bool = True
    description: str = ""

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"type": self.type, "required": self.required}
        if self.choices is not None:
            out["choices"] = list(self.choices)
        if self.description:
            out["description"] = self.description
        return out


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str = ""
    params: tuple = ()

    def param(self, name: str) -> Optional[Param]:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"action": self.name, "description": self.description}
        if self.params:
            out["args"] = {p.name: p.to_dict() for p in self.params}
        return out


@dataclass
class Decision:
    """One chosen action.

    ``rationale`` is free text recorded for explainability. ``info`` carries
    decision-model metadata (LLM usage, fallbacks, ...) into the trace.
    """

    action: str
    args: dict = field(default_factory=dict)
    rationale: str = ""
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"action": self.action, "args": dict(self.args)}
        if self.rationale:
            out["rationale"] = self.rationale
        return out

    def same_choice(self, other: "Decision") -> bool:
        """True if both decisions pick the same action with the same substantive arguments."""
        if self.action != other.action:
            return False
        keys = (set(self.args) | set(other.args)) - {"message"}
        for key in keys:
            a, b = self.args.get(key), other.args.get(key)
            if key == "price_multiplier":
                a, b = (1.0 if a is None else a), (1.0 if b is None else b)
            if a != b:
                return False
        return True


def as_decision(value: Any) -> Decision:
    """Normalise whatever a policy returned into a :class:`Decision`.

    Accepted: a ``Decision``; an action name; ``(action, args)``; or a dict
    such as ``{"action": "reposition", "args": {"node": 4}}`` or the flat form
    ``{"action": "reposition", "node": 4}``.
    """
    if isinstance(value, Decision):
        return value
    if isinstance(value, str):
        return Decision(value.strip())
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], str):
        return Decision(value[0].strip(), dict(value[1] or {}))
    if isinstance(value, dict):
        if "action" not in value:
            raise ValueError("decision is missing the 'action' field")
        action = str(value["action"]).strip()
        args = value.get("args", value.get("params"))
        args = dict(args) if isinstance(args, dict) else {}
        for key, val in value.items():
            if key not in ("action", "args", "params", "rationale", "info"):
                args.setdefault(key, val)
        rationale = value.get("rationale") or ""
        info = value.get("info") if isinstance(value.get("info"), dict) else {}
        return Decision(action, args, str(rationale), dict(info))
    raise ValueError(f"cannot interpret {type(value).__name__} as a decision")


def _coerce(param: Param, value: Any) -> Any:
    """Convert ``value`` to the parameter's type or raise ValueError."""
    if param.type == "integer":
        if isinstance(value, bool):
            raise ValueError("expected an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        raise ValueError("expected an integer")
    if param.type == "number":
        if isinstance(value, bool):
            raise ValueError("expected a number")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        raise ValueError("expected a number")
    if isinstance(value, (dict, list)):
        raise ValueError("expected a string")
    return str(value).strip()


def _match_choice(param: Param, value: Any) -> Any:
    for choice in param.choices or ():
        if param.type == "number":
            if abs(float(choice) - float(value)) < 1e-9:
                return choice
        elif param.type == "string":
            if str(choice) == value or str(choice).lower() == str(value).lower():
                return choice
        elif choice == value:
            return choice
    raise ValueError(f"must be one of {list(param.choices or ())}")


def validate_decision(decision: Any, specs: "list[ActionSpec] | tuple") -> tuple[Optional[Decision], Optional[str]]:
    """Check a decision against the permitted actions.

    Returns ``(decision, None)`` with arguments normalised, or ``(None,
    reason)``. Arguments that the permitted action does not define are dropped
    (and listed in ``info['dropped_args']``): an agent cannot widen its own
    authority by inventing parameters.
    """
    try:
        decision = as_decision(decision)
    except (ValueError, TypeError) as exc:
        return None, str(exc)
    spec = next((s for s in specs if s.name == decision.action), None)
    if spec is None:
        permitted = [s.name for s in specs]
        return None, f"action '{decision.action}' is not permitted here; permitted actions: {permitted}"
    args: dict[str, Any] = {}
    for param in spec.params:
        raw = decision.args.get(param.name)
        if raw is None or (isinstance(raw, str) and raw.strip() == "" and param.type != "string"):
            if param.required:
                return None, f"action '{spec.name}' requires argument '{param.name}'"
            continue
        try:
            value = _coerce(param, raw)
            if param.choices is not None:
                value = _match_choice(param, value)
        except (ValueError, TypeError) as exc:
            return None, f"argument '{param.name}' of '{spec.name}' is invalid: {exc}"
        if param.name == "message":
            value = value[:MAX_MESSAGE_CHARS]
            if not value:
                continue
        args[param.name] = value
    info = dict(decision.info)
    dropped = sorted(k for k, v in decision.args.items() if spec.param(k) is None and v not in (None, ""))
    if dropped:
        info["dropped_args"] = dropped
    return Decision(spec.name, args, decision.rationale, info), None


def decision_json_schema() -> dict:
    """JSON schema of a decision, usable with structured-output APIs.

    Every argument is present and nullable, and every object forbids extra
    properties, which is the common denominator accepted by strict
    structured-output implementations.
    """
    arg_props = {name: {"anyOf": [{"type": typ}, {"type": "null"}]} for name, typ in sorted(ARG_TYPES.items())}
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "args": {
                "type": "object",
                "properties": arg_props,
                "required": sorted(arg_props),
                "additionalProperties": False,
            },
            "rationale": {"type": "string"},
        },
        "required": ["action", "args", "rationale"],
        "additionalProperties": False,
    }
