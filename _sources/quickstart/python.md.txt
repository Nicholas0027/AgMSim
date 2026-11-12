# Python usage

## Run and read the result

```python
from agmsim import Simulation

result = Simulation("montreal").run()
result.metrics["outcomes"]        # waiting time, VMT, VHT, traffic speed
result.metrics["travelers"]       # trips, completion, time, mode shares
```

Overrides are given like on the command line, as a dictionary or as `key=value` strings:

```python
sim = Simulation("montreal", overrides={"agents.vehicle.agency": "L3", "simulation.seed": 5, "record.dir": None})
result = sim.run()
```

## Read decisions

Every decision is a record with what the agent saw, what it was permitted to do and what it chose:

```python
for rec in result.records:
    if rec.get("type") == "decision" and rec["kind"] == "vehicle":
        print(rec["agent"], rec["trigger"], rec["decision"]["action"], rec["decision"]["rationale"])
        break
```

## Arms over common seeds

```python
from agmsim.experiment import paired, run_arms

arms = {"L0": {"agents.vehicle.agency": "L0"}, "L1": {"agents.vehicle.agency": "L1"}}
rows = run_arms("montreal", arms, seeds=[1, 2, 3], workers=6)
paired(rows, baseline="L0", metric="outcomes.traffic_speed_mph")
```

Arms share seeds and therefore travellers, so differences are taken within a seed.

## Your own decision model

```python
from agmsim import Decision, register_policy

@register_policy("toll_averse", role="vehicle")
def toll_averse(obs):                       # obs.trigger, obs.state, obs.context, obs.actions, obs.default
    if obs.trigger == "route":
        best = min(obs.context["routes"], key=lambda r: (r["toll_usd"], r["eta_min"]))
        return Decision("choose_route", {"route": best["route"]}, "cheapest route first")
    return obs.default
```

Then `agents: {vehicle: {non_llm_policy: toll_averse}}`. More in [](../guide/extending.md).
