# Extending

Everything is looked up by name in a registry. Put your code in a module and list it under `plugins:` in the
scenario.

```python
from agmsim import Decision, register_policy

@register_policy("toll_averse", role="vehicle")
def toll_averse(obs):
    if obs.trigger == "route":
        best = min(obs.context["routes"], key=lambda r: (r["toll_usd"], r["eta_min"]))
        return Decision("choose_route", {"route": best["route"]}, "cheapest route first")
    return obs.default
```

| To add | Use | Contract |
| - | - | - |
| a decision model | `@register_policy(name, role=kind)` | a function or `Policy` subclass: `decide(obs) -> Decision`, optional `feedback(event)` |
| an LLM backend | `@register_llm(name)` | subclass `LLMClient`, implement `_complete(request)` |
| a remote agent, in any language | policy `remote` with `policy_params: {url: ...}` | POST `{"observation": ...}` → `{"action", "args", "rationale"}` |
| a new kind of agent | `@register_kind(name)` | subclass `Kind`: `setup`, `due`, `observe`, `apply`, `physics`, plus an `AgencySpec` |
| a network, demand pattern or metric | `@register_network`, `@register_demand`, `@register_metric` | see `agmsim/network.py`, `demand.py`, `metrics.py` |

`examples/custom_agents.py` and `examples/custom_scenario.yaml` are a working example;
`agmsim/kinds/evcs.py` (about 150 lines) is a compact template for a new kind.

## What a decision model receives

An observation has the same shape for every kind and every decision model:

| Field | Content |
| - | - |
| `trigger` | why the agent is asked now |
| `state` | its own state |
| `context` | what it can currently see |
| `memory` | its own latest decisions and what came of them, oldest first |
| `actions` | the complete list of what it may do, with the allowed values of each argument |
| `default` | what happens if it has no say |

It returns one permitted action with its arguments and, if it likes, a reason.
