# Configuration

A scenario is a YAML file; the testbed is `agmsim/scenarios/montreal.yaml`. Any value can be overridden on
the command line with `--set key=value`, in Python with `overrides`, or in an arm of an experiment. Unknown
keys are errors.

```yaml
name: montreal
simulation: {time_step_seconds: 2, duration_seconds: 21600, seed: 1, clock_start: "06:00"}
network: {type: montreal, capacity_scale: 0.015}
pois: [...]                              # named places, see "Places and events"
demand: {pattern: hotspots, end_seconds: 19800, min_trip_m: 3000, to_pois: true, params: {...}}
events: [...]
agents:
  traveler: {count: 900, car_ownership: 0.65, non_llm_policy: utility, agency: L4}
  vehicle:  {non_llm_policy: algorithm, agency: L1, battery: {...}}
  ondemand: {fleet: 40, non_llm_policy: algorithm, agency: L1}
  bus:      {agency: L1, lines: [{id: Orange, stops: [0, 3, 4, 12, 11, 24, 23, 8, 5], buses: 8, capacity: 120}]}
  evcs:     {agency: L1, stations: [{id: Downtown, node: 0, chargers: 3, power_kw: 100}]}
  signal:   {junctions: [0, 1, 2, 3, 4, 5, 12, 13, 15, 19], agency: L1}
  operator: {enabled: true, agency: L1, cordon_center: 0, cordon_radius_m: 2000}
llm: {preset: deepseek}
record: {dir: runs}
```

## Blocks

`simulation`
: `time_step_seconds`, `duration_seconds`, `seed`, `clock_start` (time of day at which the run starts).

`network`
: `type` (`montreal`; also `grid`, `bridge`, `file`), `capacity_scale` (below 1 when each simulated car stands
  for several real ones; smaller means more congestion), `congestion`.

`demand`
: `pattern` (`hotspots`, `cbd`, `uniform`, `corridor`), the period in which trips start (`start_seconds`,
  `end_seconds`), `min_trip_m`, `to_pois`, and `params` of the pattern (for `hotspots`: weights of origins
  and destinations by node).

`agents.<kind>`, for every kind
: `use_llm` and `non_llm_policy` (who decides), `agency` (a level, a list of permissions), `mix` (shares of
  the kind with other settings), `policy_params` and `llm_policy_params`, and `observation`, `memory`,
  `interaction` (see [](concepts/mechanisms.md)).

`agents.traveler`
: `count`, `car_ownership`, `default_mode`, `personas`.

`agents.vehicle`
: `battery` (capacity, consumption, initial and target state of charge), `reroute_check_seconds`,
  `route_alternatives`.

`agents.ondemand`
: `fleet`, `depots`, `objective` (`profit`, `service`, `balanced`), fares.

`agents.bus`
: `lines` (stops as node ids, number of vehicles, capacity), `fare`, `max_access_m`.

`agents.evcs`
: `stations` (node, chargers, power), `price_per_kwh`.

`agents.signal`
: `junctions` (a list of nodes, or `all`), `green_seconds`.

`agents.operator`
: `enabled`, `cordon_center`, `cordon_radius_m`, `initial_toll`, `initial_bus_priority`, `review_seconds`.

`llm`
: `preset`, or `backend`, `base_url`, `model`, `api_key_env`; `max_calls`, `budget_usd`, `cassette`,
  `max_concurrency`.

`record`
: `dir` (where a run is written; `null` keeps it in memory), `frames_every_seconds`.

The shorthand `human_agents: {use_llm: true}` is the traveller block, and `machine_agents: {use_llm: true}`
switches every machine kind. `agmsim list` prints the kinds, their levels and permissions, the decision
models and the presets.
