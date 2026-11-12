"""The decision-model interface.

A *policy* is the decision model of one agent. It is the main plug-in point of
AgMSim: anything that can map an :class:`~agmsim.observation.Observation` to a
:class:`~agmsim.actions.Decision` can drive a traveller or a vehicle -- a few
``if`` statements, a discrete-choice model, an optimiser, a reinforcement-
learning policy, an LLM agent with memory and tools, or a remote service.

Minimal plug-in::

    from agmsim import register_policy, Decision

    @register_policy("eager", role="vehicle")
    def eager(obs):
        if obs.trigger == "proposal":
            return Decision("accept_request")
        return obs.default

and in the config: ``agents: {vehicle: {non_llm_policy: eager}}``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..registry import Registry

POLICIES = Registry("policy")


@dataclass
class PolicyContext:
    """Everything a policy is given when it is attached to an agent."""

    agent_id: str
    role: str                                           # the agent's kind: traveler, vehicle, bus, evcs, operator, ...
    rng: random.Random                                  # the agent's private random stream
    objective: str = ""
    llm: Any = None                                     # LLM client, for policies that declare ``uses_llm``
    fallback_factory: Optional[Callable[[], "Policy"]] = None   # builds the agent's non-LLM policy
    config: Any = None                                  # the run's Config (read-only)
    extras: dict = field(default_factory=dict)


class Policy:
    """Base class of all decision models. One instance serves one agent."""

    name = "policy"
    #: Set to True for I/O-bound models; the engine may then call ``decide`` from a worker thread
    #: so that the agents due in the same tick are queried in parallel.
    concurrent = False
    #: Set to True to receive an LLM client in ``ctx.llm``.
    uses_llm = False

    def __init__(self, **params: Any):
        self.params = params
        self.ctx: Optional[PolicyContext] = None

    def bind(self, ctx: PolicyContext) -> None:
        """Attach the policy to its agent. Called once, before the first decision."""
        self.ctx = ctx
        self.agent_id = ctx.agent_id
        self.role = ctx.role
        self.rng = ctx.rng

    def decide(self, obs):
        """Return the action to take: a Decision, an action name, or a dict."""
        raise NotImplementedError

    def feedback(self, event: dict) -> None:
        """Receive an outcome that concerns this agent (trip completed, offer declined, ...)."""

    def end_episode(self, summary: dict) -> None:
        """Called at the end of each episode; use it to learn across simulated days."""


class FunctionPolicy(Policy):
    """Wraps a plain function ``obs -> decision``."""

    def __init__(self, fn: Callable, **params: Any):
        super().__init__(**params)
        self.fn = fn
        self.name = getattr(fn, "__name__", "function")

    def decide(self, obs):
        return self.fn(obs)


def register_policy(name: str, role: str = "any"):
    """Register a policy class or function under ``name`` for agents of kind ``role``.

    ``role`` is an agent kind (``traveler``, ``vehicle``, ``bus``, ``evcs``,
    ``operator`` or a plugin kind) or ``any``.
    """

    def deco(obj):
        POLICIES.register(f"{role}:{name}", obj)
        if isinstance(obj, type) and getattr(obj, "name", "policy") == "policy":
            obj.name = name
        return obj

    return deco


def explain_policy(role: str, name: str) -> list[str]:
    """What a registered decision model does for agents of kind ``role``, in plain sentences (for the guide).

    A policy class says so in ``explained``: a list of sentences, or ``{kind: [...]}`` where one class
    serves several kinds. Without it, the first paragraph of its docstring is used.
    """
    obj = next((POLICIES.get(key) for key in (f"{role}:{name}", f"any:{name}") if key in POLICIES), None)
    if obj is None:
        return []
    told = getattr(obj, "explained", None)
    if isinstance(told, dict):
        told = told.get(role)
    if told:
        return list(told)
    doc = (obj.__doc__ or "").strip().split("\n\n")[0]
    return [" ".join(doc.split())] if doc else []


def available_policies(role: str) -> list[str]:
    names = set()
    for key in POLICIES.names():
        r, _, name = key.partition(":")
        if r in (role, "any"):
            names.add(name)
    return sorted(names)


def instantiate(obj: Any, params: Optional[dict] = None) -> Policy:
    """Turn a registered object (class, function or factory) into a fresh Policy instance."""
    params = dict(params or {})
    if isinstance(obj, Policy):
        raise TypeError("pass a Policy class or a factory, not an instance: every agent needs its own policy object")
    if isinstance(obj, type) and issubclass(obj, Policy):
        return obj(**params)
    if callable(obj):
        return FunctionPolicy(obj, **params)
    raise TypeError(f"cannot build a policy from {obj!r}")


def create_policy(role: str, name: str, params: Optional[dict] = None) -> Policy:
    for key in (f"{role}:{name}", f"any:{name}"):
        if key in POLICIES:
            policy = instantiate(POLICIES.get(key), params)
            if policy.name in ("policy", "function"):
                policy.name = name
            return policy
    raise KeyError(f"unknown {role} policy '{name}'. Available: {', '.join(available_policies(role)) or '(none)'}")
