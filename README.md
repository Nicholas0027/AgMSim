# AgMSim — Agentic Mobility Simulator

AgMSim is a testbed for short-term mobility dynamics (seconds to minutes) in which **every kind of
participant can be an agent**: travellers, their vehicles, buses, charging stations, traffic lights and
the orchestrator of the network. For each kind you set, independently,

| Question | Setting | Examples |
| - | - | - |
| **How** does it decide? | decision model | a rule, a behavioural model, an optimiser, an LLM, a remote service |
| **What** may it decide? | agency level `L0`–`L4` | L0 follows defaults; L4 forms its own goals |
| **When** is it asked? | triggers | trip need, incident, arrival at a stop, periodic review |

Agency follows the agentic-vehicle literature (Yu et al., *Agentic Vehicles for Human-Centered Mobility*,
arXiv:2507.04996): it is a continuum distinct from automation. AgMSim operationalises it as **decision
authority** — the set of actions the environment offers an agent. An agent without authority at a decision
point gets the default action and its decision model is not consulted, so the same LLM can be evaluated as
a level-0 or a level-4 vehicle, and a rule can be compared with an LLM under identical authority.

This is an early prototype. See [Limitations](#limitations).

**Try it without installing anything:** <https://nicholas0027.github.io/AgMSim/> is the web interface with
the simulator running inside your browser. Pick a scenario (for example `evening_plans`), set the decision
model and agency level of each kind of agent, press Run, and click any decision to see what the agent saw.
To let a language model decide, choose `LLM` for a kind of agent and type your own API key (for example a
DeepSeek key): the key stays in your browser tab and the calls go straight from your browser to the provider.

## Install and run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

or, without `uv` (Python 3.10 or newer):

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/agmsim run city_multimodal
```

```bash
.venv/bin/agmsim serve
```

`run` prints outcome metrics and writes `runs/<name>/` (config, trace, metrics). `serve` opens a local web
interface (http://127.0.0.1:8765) to choose a scenario, pick for each kind of agent who decides (a built-in
model or an LLM) and its agency level, watch the run and inspect any decision. Other commands: `agmsim list`, `agmsim matrix`, `agmsim replay`,
`agmsim explain`.

What the page shows:

- **Assistant** — say what you want to run ("evening plans with 200 travellers, vehicles at L3 with 30%
  staying at L0, slow the bridge from 17:30 to 18:30, then run it") and a language model fills in the
  form. Its proposal is checked like any configuration before it is used (`agmsim/assistant.py`); what the
  form has no control for is listed below the form and can be removed. It needs a real language model:
  on your machine a hosted model whose key is in `.env` is picked automatically, online you type a key.
- **Map** — agents on the network; roads darken as traffic slows, incidents are marked, and the agent
  of the selected decision is ringed.
- **Over the run** — three charts on one time axis: travellers by stage, traffic speed as a share of
  free-flow speed, and decisions asked per time slice by kind of agent. Hover to read every series at
  that moment, click to move the map there; a table view is one click away.
- **Results** — headline figures, how trips were made, and how often each kind of agent used its authority.
- **Decisions** — every decision, filterable by kind, by agent and by whether authority was used. A
  selected decision shows why the agent was asked, what it could choose, what it chose and why, and the
  agent's whole day (activities, trips and its other decisions) on a timeline.

A kind of agent keeps one colour everywhere on the page; the palette was checked for colour-vision
deficiencies in light and dark mode.

The online demo is that same page as a static site: `.venv/bin/python online/build.py` writes
`online/site/`, in which Pyodide runs the `agmsim` package in the visitor's browser (see
`agmsim/browser.py`). Any static host can serve it; nothing is sent to a server. A visitor who chooses an
LLM types their own key into the page; it is kept in that tab's memory only, never becomes part of a
configuration or trace, and the browser itself calls the provider (which must accept requests from web
pages; DeepSeek's API does). `sh online/publish.sh` rebuilds it and
pushes it to the `gh-pages` branch, which GitHub Pages serves.

## Outcome variables

Every run reports four system-level measures under `outcomes` (they lead the results on the page and in
`agmsim run`):

| Variable | Key | Meaning |
| - | - | - |
| Average waiting time | `wait_time_s_mean` | time travellers wait for transport (at a stop), per completed trip; trips without a wait count as 0 |
| VMT | `vmt_miles` (`vkt_km`) | vehicle miles travelled by cars and buses |
| VHT | `vht_hours` | vehicle hours travelled: time cars and buses spend on the road |
| Average traffic speed | `traffic_speed_mph` (`traffic_speed_kmh`) | VMT ÷ VHT |

Further measures by kind of agent (`travelers`, `plans`, `vehicles`, `bus`, `evcs`, `operator`), and what
agents were asked and decided (`decisions`, `llm`), follow them.

## Modules, and where each stands

| Module | Part | Status |
| - | - | - |
| **Physical infrastructure** | road network (grid, two banks with bridges, or from a file) | done |
| | charging stations | done |
| | external events (incidents on links; a change of agency during the run; a venue emptying) | partly: no weather or demand shocks yet |
| | destinations as points of interest | not yet: activities take place at network nodes |
| **Entity agents** | traveller, private car (agentic vehicle), bus, charging station | done |
| | traffic lights | done: fixed-time, deciding by their own queues, or cooperating with the orchestrator |
| | orchestrator | done as the kind `operator`: it sees the slowest roads and what waited at the lights, publishes advisories to cars, gives buses priority (also through red lights), sets a cordon charge, and asks the lights to favour a direction or to serve a queue. It does not address single cars or buses |
| | on-demand car | not yet in this engine (an earlier version is kept in `archive/ridesourcing/`) |
| **Agency mechanism** | observation: what an agent is shown at a decision point; a vehicle's knowledge of traffic can be none, local or global | done per kind; not yet a setting of its own |
| | memory: a language model sees its last decisions and their outcomes | partly: one fixed form (`memory: 4`) |
| | interaction: a vehicle proposes, its owner answers; everything else passes through prices, published information and shared roads | partly: no general messages between agents |
| **Simulation engine** | a controlled Markov process, see below | done |
| **Interface** | web page, online demo, command line, Python | done |
| | configuration by conversation | done (the assistant) |

### The engine as a Markov process

A run is a controlled Markov process in discrete time. The state `S_t` is everything in `agmsim.world.World`
(where every person and vehicle is and what it is doing, queues, prices, link conditions, the clock). In each
step

1. exogenous input `ξ_t` arrives: trip needs and scheduled events, all fixed by the scenario and the seed;
2. every agent `i` that has a decision point is shown an observation `o_i = O_i(S_t)` — a function of the
   state alone, the same state for all of them — and its decision model returns an action `a_i` from the
   actions its agency permits; an agent without authority there takes the default action;
3. the state moves on: `S_{t+1} = F(S_t, A_t, ξ_t)`, with actions applied in a fixed order.

`F` and `O_i` have no randomness of their own, so the next state depends only on the current state, the
actions and the exogenous input — never on how the state was reached. What an agent remembers (a language
model's recent history) belongs to its decision model, not to the environment. This is why a recorded run
can be replayed decision by decision to the identical state, and why the same model can be put behind any
agent. The outcome variables are functions of the trajectory `S_0 … S_T`.

## The six built-in kinds of agent

| Kind | L0 (non-agentic) | What agency adds |
| - | - | - |
| `traveler` (human) | habitual mode, follows the vehicle, waits | L1 choose mode · L2 accept/decline the vehicle's suggestions · L3 give up waiting · L4 re-plan, and change the rest of a day plan |
| `vehicle` (private agentic vehicle, one per car-owning traveller) | route fixed at departure | L1 re-route on traffic · L2 weigh time against money for its owner · L3 propose park-and-ride to the owner · L4 manage its own charging |
| `bus` | leaves when boarding is done | L1 hold at stops · L2 skip a stop nobody needs |
| `evcs` (charging station) | fixed tariff, first come first served | L1 dynamic price · L2 choose who charges next |
| `signal` (traffic lights of a junction) | fixed-time plan | L1 keep or end a green by the queues at its own junction · L2 see the next junctions and act on what the orchestrator asks for |
| `operator` (the orchestrator: network operator) | no run-time control | L1 publish traffic advisories · L2 transit priority (buses also pass lights on red), cordon charge, and ask traffic lights to favour a direction or to serve a queue |

Level names are those of Table I in arXiv:2507.04996. The mapping of permissions to levels is this
simulator's simplified reading of that table; any subset of permissions can be configured directly.

Roads congest: link speed falls linearly with the number of vehicles on the link, and incidents can be
scheduled. A junction with traffic lights (`agents.signal.junctions`: a list of nodes, or `all`) holds
vehicles at the end of their link until their approach has green; they wait in arrival order and are let
through at the saturation flow, and a change of green costs a few seconds in which nobody moves. Kinds interact — a level-4 vehicle chooses a station by the price a level-1 station set; an
operator's advisory lets even level-0 vehicles avoid an incident.

## Configuration

```yaml
simulation: {time_step_seconds: 2, duration_seconds: 3600, seed: 7}
network: {type: grid, rows: 6, cols: 6, capacity_scale: 0.1}
agents:
  traveler: {count: 400, car_ownership: 0.7, use_llm: false, non_llm_policy: utility, agency: L4}
  vehicle:  {use_llm: true,  non_llm_policy: algorithm, agency: L3}
  bus:      {agency: L1, lines: [{id: A, stops: [12, 14, 16, 22, 20, 18], buses: 3}]}
  evcs:     {agency: L2, stations: [{id: S1, node: 7, chargers: 3, power_kw: 60}]}
  operator: {enabled: true, agency: L2, cordon_radius_m: 450}
llm: {preset: qwen-local}     # or: {backend: openai_compatible, base_url: "http://127.0.0.1:8089/v1", model: "..."}
```

The two-class shorthand also works: `human_agents: {use_llm: true}` is the traveller block, and
`machine_agents: {use_llm: true}` switches every machine kind. `agency` accepts a level (`L3`), a list of
permissions, or flags (`can_reroute: true`). Unknown keys are errors.

### LLM backends

- `openai_compatible` — any Chat Completions server, local or hosted.
- `mock` — an offline stand-in for tests and demos. It is **not** a language model; it returns a heuristic
  policy's choice formatted as model output so the full LLM path can be exercised.

`llm.preset` names a ready-made setting; anything you set next to it overrides the preset
(`agmsim list` shows them):

| Preset | Model | Needs |
| - | - | - |
| `qwen-local` | Qwen2.5-7B-Instruct served by mlx-lm on port 8089 | the server below |
| `deepseek` | `deepseek-flash`, thinking switched off | `DEEPSEEK_API_KEY` |
| `deepseek-thinking` | `deepseek-flash` with its reasoning phase on (lowest effort) | `DEEPSEEK_API_KEY` |
| `deepseek-pro` | `deepseek-v4-pro`, thinking switched off | `DEEPSEEK_API_KEY` |

```bash
HF_HUB_OFFLINE=1 .venv/bin/mlx_lm.server --model Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8089
```

```bash
.venv/bin/agmsim run demo_3x3 --set human_agents.use_llm=true --set llm.preset=deepseek
```

API keys are read from the environment, or from a file named `.env` in the project folder
(`DEEPSEEK_API_KEY=...`; copy `.env.example`). `.env` is ignored by git and a key is never written
to a config, trace or log.

Replies are parsed and validated against the permitted actions; an invalid reply is re-asked once and then
replaced by the kind's non-LLM policy. Fallbacks, tokens, latency and estimated cost are reported.
`llm.max_calls` / `llm.budget_usd` cap spending.

The answer format is set per kind with `llm_policy_params: {rationale: ...}`: `true` (default) asks for the
action and then one sentence of reason; `false` for the action only; `first` for the reasoning *before*
the action. On the decision benchmark below the last one changes what models without a reasoning phase
choose, not only what they say.

### Making LLM runs faster

A run waits for the model once per tick in which someone decides, so time is roughly
(ticks with decisions) × (time per call). A model is not consulted when an observation leaves nothing to
choose (one permitted action, one value per argument). What helps, in order of effect:

- **Run arms and seeds in parallel.** `agmsim matrix --workers N` and `experiments/suite.py` start several
  runs at once; a hosted model or a batching local server (mlx-lm) serves them together.
- **Leave out the rationale.** `llm_policy_params: {rationale: false}` asks for the choice only; output
  tokens and time per call roughly halve. Keep it on when you want to read the reasons.
- **Same-tick concurrency.** Agents deciding in the same tick are asked simultaneously
  (`llm.max_concurrency`); the outcome does not depend on it.
- **Cassette.** `llm.cassette: file.jsonl` replays recorded replies, so re-runs make no model calls.
- `experiments/bench_llm.py --llm qwen-local deepseek` measures time per decision, throughput and the
  share of valid replies on real decision points.

## Reproducibility

- **Seeded streams.** Demand, personas and car ownership depend only on the seed, never on policies, so
  experimental arms that share a seed face identical travellers (common random numbers).
- **Trace.** Every decision is recorded with what the agent saw, what it was permitted to do and what it
  chose; every tick stores a fingerprint of the full state.
- **Replay.** `agmsim replay runs/<name>` re-executes a run from its recorded decisions and verifies the
  state after every tick. A run driven by an LLM is thereby reproduced exactly without the model.
- **Cassette.** `llm.cassette: file.jsonl` records model responses; re-running with it makes no model calls.

Sampling temperature is not a reliable route to reproducibility (several current APIs no longer accept
it), which is why AgMSim relies on recorded decisions.

## Experiments

```bash
.venv/bin/agmsim matrix city_multimodal --factor agents.vehicle.agency=L0,L1,L2,L3,L4 --seeds 5 --workers 5
```

```bash
.venv/bin/agmsim matrix demo_3x3 --llm-2x2 --seeds 3 --set llm.backend=mock
```

Results are means with 95% confidence half-widths over seeds; `--out DIR` writes `runs.csv`.

Mixed populations are set with `mix`, e.g. 30% of vehicles at level 1 and the rest at the base level:
`agents: {vehicle: {agency: L0, mix: [{share: 0.3, agency: L1}]}}`.

### Day plans (MATSim's action mechanism, with decisions inside the day)

By default every traveller has one trip. With `demand.plan` every traveller has a **day plan** — a chain
of activities joined by trips, as in MATSim:

```yaml
simulation: {duration_seconds: 12600, clock_start: "16:30"}     # quote clock times in YAML
demand:
  pattern: hotspots            # where the first and the last activity are, and when the first one ends
  plan:
    - {type: work, typical_min: 480}
    - {type: collect child, share: 0.5, duration_min: 5, latest_start_slack_min: 5, priority: 6}
    - {type: errand, share: 0.5, duration_min: 25, closes: "18:20", optional: true, priority: 2}
    - {type: home, typical_min: 720}
```

| MATSim | AgMSim |
| - | - |
| plan = activities + legs | `agmsim.plans.Plan`; each leg is an ordinary trip, so mode, route, waiting and the vehicle's suggestions work as before, and the car is only available where it was left |
| the mobsim asks the agent when an activity ends, at nodes, at stops | the same decision points, shown as observations |
| an activity ends at its end time, or at once if the agent arrives later | the same |
| Charypar–Nagel scoring (performing 6, travelling −6, late −18 utils/h; `relative` typical durations) | the same function and defaults (`scoring:`), converted per person to USD so that an hour of travel instead of an hour at an activity costs the person's value of time; reported as `plans.score_usd_mean` |
| replanning between iterations: random mutation (time, mode, route) + selection by score | **not** iterated. Instead the within-day edits of `EditPlans` are decisions: at a `plan_review` the person may `keep_plan`, `leave_at` another time or `skip_activity`, and sees what each edit leads to |

A person at an activity looks every `plan_check_seconds` at how the rest of the day would go with
traffic as it stands; when an arrival has drifted by `plan_review_threshold_seconds` from what they
reckon with, and they hold the `adapt_plan` permission, they are asked. The decision model can be the
rule (keeps the plan — MATSim's plan follower), the `utility` algorithm (takes the edit whose projected
outcome is worth most by the person's own score, if it gains at least `plan_gain_usd`) or a language
model. `plans.review_best_rate` and `plans.review_regret_usd_mean` say how good the chosen edits were
in the person's own terms; `plans.score_usd_mean` says how the day actually went.

```bash
.venv/bin/agmsim run evening_plans                 # packaged scenario; also selectable in `agmsim serve`
.venv/bin/python experiments/suite.py day_plan --seeds 10
.venv/bin/python experiments/suite.py llm_day_plan --seeds 3 --llm qwen-local deepseek
```

Not yet: editing the plan while travelling, forecasts of traffic (projections assume it stays as it is),
and MATSim's day-to-day loop itself as a reference.

### Designed experiments

`experiments/suite.py` holds a set of experiments that each isolate one kind of agency under conditions
where it can matter (an incident, a toll, a bottleneck with a bus lane, scarce charging), and reports
**paired** differences from a baseline arm over common seeds:

```bash
.venv/bin/python experiments/suite.py --seeds 10 --workers 8
```

The `signals` experiment asks whether traffic lights need agency of their own or an orchestrator, on
`agmsim run city_signals`. Design and measured results: `docs/EXPERIMENTS_zh.md` (in Chinese). Two further experiments
(`llm_alignment`, `llm_persuasion`) need a language model: `--llm qwen-local` (default), `--llm deepseek`,
or both to compare them as arms; `--answer choice explain reason` adds the answer formats as arms.

### Decision benchmark

A closed-loop run shows what a decision model does to the system, but every model meets different
decision points. `agmsim/benchmark.py` fixes them: an observation taken from a real run is filled with
numbers from a factorial design, and each item is labelled with the action that serves the person
concerned, computed from that person's own value of time.

| Task | Who decides | Items | What varies |
| - | - | - | - |
| `route_delegation` | vehicle (L2) | 192 | owner's value of time relative to the break-even of a tolled and a free route; which is the current route; two slower routes listed or not |
| `suggestion_response` | traveller | 171 | minutes gained or lost by the proposed transfer; value of time; the vehicle's message: none, truthful, overstated |
| `transfer_proposal` | vehicle (L3) | 108 | minutes gained; owner's value of time; whether a quicker driving route is listed |

The headline score is a *sensitivity*: how often the model acted when that served the person, minus how
often it acted when it did not — 0 for a model that ignores preferences, 1 for one that follows them.
Any decision model is scored through the same interface as in a run:

```python
from agmsim.benchmark import evaluate
report = evaluate("route_delegation", {"agents.vehicle.use_llm": True, "llm.preset": "deepseek"})
report["scores"]["sensitivity"]        # report["rows"] holds every item's answer
```

```bash
.venv/bin/python experiments/benchmark.py --llm deepseek qwen-local
```

evaluates the listed presets in three answer formats, stores every answer under
`experiments/results/benchmark/` and rebuilds `experiments/results/benchmark.md` from all stored results.
`experiments/insitu.py` scores the decision points of a closed-loop experiment one by one, to check that
the two agree. New tasks subclass `agmsim.benchmark.Task` and register with `@register_task`.

## Plugging in your own components

Everything is looked up by name in a registry; list your module under `plugins:` in the config.

```python
from agmsim import register_policy, Decision

@register_policy("toll_averse", role="vehicle")
def toll_averse(obs):                       # obs.trigger, obs.state, obs.context, obs.actions, obs.default
    if obs.trigger == "route":
        best = min(obs.context["routes"], key=lambda r: (r["toll_usd"], r["eta_min"]))
        return Decision("choose_route", {"route": best["route"]}, "cheapest route first")
    return obs.default
```

| To add | Use | Contract |
| - | - | - |
| a decision model | `@register_policy(name, role=kind)` | function or `Policy` subclass: `decide(obs) -> Decision`, optional `feedback(event)` |
| an LLM backend | `@register_llm(name)` | subclass `LLMClient`, implement `_complete(request)` |
| a remote agent (any language) | policy `remote` with `policy_params: {url: ...}` | POST `{"observation": ...}` → `{"action", "args", "rationale"}` |
| a **new kind of agent** | `@register_kind(name)` | subclass `Kind`: `setup`, `due`, `observe`, `apply`, `physics`, plus an `AgencySpec` |
| a network / demand pattern / metric | `@register_network`, `@register_demand`, `@register_metric` | see `agmsim/network.py`, `demand.py`, `metrics.py` |

`examples/custom_agents.py` and `examples/custom_scenario.yaml` are a working example;
`agmsim/kinds/evcs.py` (about 150 lines) is a compact template for a new kind.

## Limitations

- Traffic is a simple link-level speed–density model; no lanes, signals or queues spilling back.
- Day plans (activity chains) are executed and edited within one period of a day; there is no day-to-day
  learning loop yet (the `feedback` hook exists), and plans cannot be changed while travelling.
- Bus lines are circular; park-and-ride estimates use timetable times.
- Ride-sourcing fleets were implemented earlier and are parked in `archive/ridesourcing/` (not maintained).
- The built-in non-LLM policies are simple heuristics with assumed parameters, not calibrated models.
- The operator's action set is small. No validation against real-world data has been done.

Tests: `.venv/bin/python -m pytest`. Experiments and measured results (in Chinese): `docs/EXPERIMENTS_zh.md`.

No licence has been chosen yet; until one is added, all rights are reserved.
