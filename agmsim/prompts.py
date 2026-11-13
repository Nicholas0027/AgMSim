"""What a language model is told, read off a configuration without running it.

For every kind of agent of a scenario: the goal an agent of the kind is shown, and the system prompt a
language model gets before every decision. The run itself writes its prompts with the same function
(:func:`agmsim.policies.llm_policy.compose_system_prompt`), so what is reported here is what is sent.

Both can be set in a scenario, per kind::

    agents:
      vehicle:
        objective: "Get your owner there without using a bridge unless it saves ten minutes."
        llm_policy_params:
          extra_instructions: "Say in your reason which road you avoided."   # added at the end of the prompt
          # system_prompt: "..."                                            # or: your own prompt in place of the simulator's
"""
from __future__ import annotations

from .config import Config
from .recorder import Recorder


def told(config, overrides=None) -> dict:
    """``{kind: {...}}`` for every kind of agent the scenario has.

    * ``goal`` -- what an agent of the kind is told to pursue: a sentence in its state at every decision
      (empty when the kind states none, as for travellers, who are told who they are instead);
    * ``own_goal`` -- the kind's own sentence, in force when the scenario sets no ``objective``;
    * ``system`` -- the system prompt in force;
    * ``generated`` -- the prompt the simulator writes itself, without what the scenario replaces or adds.

    The two prompts are None for a kind whose language-model policy (``llm_policy``) is not the built-in
    one: such a policy writes its own. No model is set up or asked.
    """
    from .engine import Simulation
    from .policies.llm_policy import build_system_prompt, compose_system_prompt

    cfg = Config.load(config, overrides)
    quiet = cfg.to_dict()               # the kinds as the scenario sets them up, every one on its built-in model
    for kind in quiet["agents"].values():
        kind["use_llm"], kind["mix"] = False, []
    quiet["record"]["dir"] = None
    sim = Simulation(Config.from_dict(quiet, base_dir=cfg.base_dir), recorder=Recorder(None, keep=False))
    out = {}
    try:
        for name, kind in sim.kinds.items():
            kcfg = getattr(cfg.agents, name)
            agent = next((agent_id for agent_id, spec in sim.specs.items() if spec.kind == name), "")
            params = dict(kcfg.llm_policy_params or {})
            ours = kcfg.llm_policy == "llm"
            out[name] = {"goal": kind.goal(agent), "own_goal": kind.own_goal(agent), "llm_policy": kcfg.llm_policy,
                         "system": compose_system_prompt(kind, **params) if ours else None,
                         "generated": build_system_prompt(kind, params.get("rationale", True)) if ours else None}
    finally:
        sim.close()
    return out
