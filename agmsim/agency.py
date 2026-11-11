"""Agency: what an agent is *permitted* to decide.

AgMSim follows the agentic-vehicle literature (Yu et al., "Agentic Vehicles
for Human-Centered Mobility", arXiv:2507.04996) in treating agency as a
continuum that is distinct from automation, and operationalises it as
*decision authority*: a set of permissions that determines which actions the
environment offers an agent at each decision point.

Agency is configured separately from the decision model. A vehicle driven by
an LLM can be confined to obeying dispatch (level 0); a vehicle driven by a
fixed rule can be allowed to form its own goals (level 4). At a decision point
where the agent holds no relevant permission, the mechanism's default action
is applied and the agent's decision model is not consulted.

Every kind of agent declares its own permissions and how they accumulate over
levels ``L0`` .. ``L4``. The level names follow Table I of the paper; the
mapping of permissions to levels is this simulator's simplified reading of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Level names after Table I of arXiv:2507.04996.
LEVEL_NAMES = {
    "L0": "Non-agentic",
    "L1": "Context-aware responder",
    "L2": "Dialogic agentic",
    "L3": "Adaptive agentic",
    "L4": "Ethical, social, reflective agent",
}


class AgencyError(ValueError):
    pass


@dataclass(frozen=True)
class AgencySpec:
    """The agency vocabulary of one kind of agent."""

    kind: str
    permissions: dict            # permission -> description
    levels: dict                 # "L0".."L4" -> tuple of permissions (cumulative)
    aliases: dict = None         # can_<alias> -> tuple of permissions

    def level_of(self, permissions: frozenset) -> str:
        """Lowest level whose permission set equals ``permissions``, or ``custom``."""
        for name in sorted(self.levels):
            if frozenset(self.levels[name]) == permissions:
                return name
        return "custom"


@dataclass(frozen=True)
class AgencyProfile:
    kind: str
    permissions: frozenset
    level: str

    def allows(self, permission: str) -> bool:
        return permission in self.permissions

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "level_name": LEVEL_NAMES.get(self.level, "custom"),
            "permissions": sorted(self.permissions),
        }


def _profile(spec: AgencySpec, permissions) -> AgencyProfile:
    permissions = frozenset(permissions)
    return AgencyProfile(spec.kind, permissions, spec.level_of(permissions))


def _apply_flags(spec: AgencySpec, permissions: set, flags: dict) -> AgencyProfile:
    for key, value in flags.items():
        if not key.startswith("can_"):
            raise AgencyError(
                f"unknown agency key '{key}' for {spec.kind} agents; use 'level', 'permissions' or can_<permission> flags"
            )
        name = key[4:]
        for target in (spec.aliases or {}).get(name, (name,)):
            if target not in spec.permissions:
                raise AgencyError(
                    f"unknown permission '{name}' for {spec.kind} agents. Valid: {sorted(spec.permissions)}"
                )
            if value:
                permissions.add(target)
            else:
                permissions.discard(target)
    return _profile(spec, permissions)


def parse_agency(spec: AgencySpec, value: Any, base: "AgencyProfile | None" = None) -> AgencyProfile:
    """Build a profile from a level name, a permission list, or a mapping.

    Accepted forms::

        "L3"                                     # also "AgV-3", "3", "none", "full"
        ["decline", "reposition"]
        {"level": "L2", "can_reposition": true}
        {"can_decline": true}                    # flags on top of ``base`` (default: L0)
    """
    if value is None:
        return base or _profile(spec, ())
    if isinstance(value, AgencyProfile):
        return value
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        key = str(value).strip().lower()
        if key in ("none", "minimal"):
            return _profile(spec, ())
        if key in ("full", "all"):
            return _profile(spec, spec.permissions)
        digits = key.replace("agv-", "").replace("l", "")
        name = f"L{digits}" if digits.isdigit() else None
        if name in spec.levels:
            return _profile(spec, spec.levels[name])
        raise AgencyError(f"unknown agency level '{value}' for {spec.kind} agents. Valid: {sorted(spec.levels)}, none, full")
    if isinstance(value, (list, tuple, set, frozenset)):
        unknown = sorted(set(value) - set(spec.permissions))
        if unknown:
            raise AgencyError(f"unknown permission(s) {unknown} for {spec.kind} agents. Valid: {sorted(spec.permissions)}")
        return _profile(spec, value)
    if isinstance(value, dict):
        value = dict(value)
        explicit = "level" in value or "preset" in value or "permissions" in value
        permissions: set = set() if explicit or base is None else set(base.permissions)
        level = value.pop("level", value.pop("preset", None))
        if level is not None:
            permissions |= set(parse_agency(spec, level).permissions)
        if "permissions" in value:
            permissions |= set(parse_agency(spec, list(value.pop("permissions"))).permissions)
        return _apply_flags(spec, permissions, value)
    raise AgencyError(f"cannot interpret agency specification {value!r}")
