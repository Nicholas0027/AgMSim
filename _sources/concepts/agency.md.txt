# Agents and agency

## Three independent settings

For each kind of agent you set, independently of the other kinds:

- **the decision model**: who decides. A rule, a behavioural model, an optimiser, a language model or a remote
  service.
- **the agency**: what it may decide. A level from `L0` to `L4`, or any subset of permissions.
- **what it has to go on**: observation, memory and interaction, see [](mechanisms.md).

Agency is **decision authority**: the set of actions the environment offers an agent at a decision point. An
agent without authority there takes the default action and its decision model is not consulted. The same
language model can therefore be a level-0 or a level-4 vehicle, and a rule can be compared with a language
model under identical authority.

## The seven built-in kinds

| Kind | L0 (non-agentic) | What agency adds |
| - | - | - |
| `traveler` (human) | habitual mode, follows the vehicle, waits | L1 choose mode · L2 accept or decline the vehicle's suggestions · L3 give up waiting · L4 re-plan |
| `vehicle` (private car, one per car-owning traveller) | route fixed at departure | L1 re-route on traffic · L2 weigh time against money for its owner · L3 propose park-and-ride to the owner · L4 manage its own charging |
| `ondemand` (a fleet that carries travellers who call a car) | takes every request, fixed route, waits where it dropped its passenger | L1 re-route on traffic · L2 turn a request down · L4 decide, while free, where to wait |
| `bus` (transit vehicle) | leaves when boarding is done | L1 hold at stops · L2 skip a stop nobody needs |
| `evcs` (charging station) | fixed tariff, first come first served | L1 dynamic price · L2 choose who charges next |
| `signal` (traffic lights of a junction) | fixed-time plan | L1 keep or end a green by the queues at its own junction · L2 see the next junctions and act on what the orchestrator asks for |
| `operator` (the orchestrator) | no run-time control | L1 publish traffic advisories · L2 transit priority, a cordon charge, requests to traffic lights |

Level names are those of Table I in [arXiv:2507.04996](https://arxiv.org/abs/2507.04996). The mapping of
permissions to levels is this simulator's simplified reading of that table.

## Writing agency in a scenario

```yaml
agents:
  vehicle:
    agency: L3                                   # a level
  bus:
    agency: [hold]                               # or a list of permissions
  traveler:
    agency: L0
    mix: [{share: 0.3, agency: L4}]              # 30% of travellers at L4, the rest at L0
```

An event can change agency during a run (`type: set_agency`).

## Kinds interact

A level-4 vehicle chooses a station by the price a level-1 station set; an orchestrator's advisory lets even
level-0 vehicles avoid a crash. What one kind's agency is worth therefore depends on the others: on the
testbed, cars that re-route raise traffic speed by 4 mph when nobody else acts, and by almost nothing once
the orchestrator publishes an advisory. See [](../guide/experiments.md).
