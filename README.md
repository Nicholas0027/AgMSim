# AgMSim — Agentic Mobility Simulator

AgMSim is a testbed for short-term mobility dynamics (seconds to minutes) in which **every kind of
participant can be an agent**: travellers, their private cars, on-demand cars, buses, charging stations,
traffic lights and the orchestrator of the network. For each kind you set, independently,

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

## The testbed: a simplified Montreal

Experiments are run on one network, in the way much of transport research uses Sioux Falls: `montreal`, a
simplified Montreal of 25 nodes and 51 two-way roads (`agmsim/montreal.py`), about 21 km by 17 km. It keeps
what shapes traffic in the city: an island reached over seven bridges and one tunnel, a centre on its south side, the ring of
expressways (A-40 Metropolitan, A-15 Decarie, Ville-Marie, A-25, A-20) and the arterials between them, with
the South Shore and Laval at the far ends of the crossings. The scenario is a weekday morning, 06:00 to
12:00 with its peak around 08:00, with every kind of agent the simulator has: 900 travellers who live where people live and go to named
places (downtown offices, McGill, the airport, hospitals, markets), their cars, 40 on-demand cars, four
transit lines on their own right of way, traffic lights in the inner city, seven charging stations, an
orchestrator, and a crash on the Decarie expressway at 07:30. A run with the built-in decision models takes about
15 seconds.

```bash
agmsim run montreal                                   # the testbed as it is
agmsim run montreal --set agents.vehicle.agency=L0    # change one kind's level, keep everything else
```

Positions, lengths, speeds and lanes are approximate values by class of road. It is an abstraction to
experiment on, not a calibrated model of the city. It is the one scenario of the package: an experiment
changes settings of the testbed, it does not bring a network of its own.

**Project site, with the simulator in the page:** <https://nicholas0027.github.io/agenticmobility/> (source in
`website/`, published with `sh website/publish.sh` to the repository `agenticmobility`; building it needs
`uv pip install --python .venv/bin/python -e ".[docs]"` once). The simulator runs inside your browser. It opens on the testbed (`montreal`); set the decision
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
.venv/bin/agmsim run montreal
```

```bash
.venv/bin/agmsim serve
```

`run` prints outcome metrics and writes `runs/<name>/` (config, trace, metrics). `serve` opens a local web
interface (http://127.0.0.1:8765) on the testbed: pick for each kind of agent who decides (a built-in
model or an LLM) and its agency level, watch the run and inspect any decision. Other commands: `agmsim list`, `agmsim matrix`, `agmsim replay`,
`agmsim explain`.

What the page shows:

- **Guide** (top right) — everything that is in the simulation, one entry each: the run, the trips, the outcome
  measures, the road network, places, events, transit lines, charging sites, junctions with lights, and every
  kind of agent. Click an entry to see its icon, what it is, what it may do (its functions, with those switched
  on at its level marked), all its settings, and what it is doing in the run: how often it was asked and why,
  what it chose, and its state at the moment shown on the map. A kind's name in the Agents table opens its entry.
- **Agents** — who decides for each kind and how much it may decide. The page opens with a language model
  (DeepSeek Flash) deciding for travellers, private cars, on-demand cars, charging stations and the orchestrator;
  traffic lights and transit vehicles, which decide every few seconds, start on their built-in models. Three
  buttons switch every kind at once. A language model takes about a second per decision and the run waits for it:
  about half an hour for the testbed as the page opens, a few seconds with built-in models only. (On the command
  line and in experiments the testbed runs on built-in models unless you say otherwise.)
- **Events** — the scenario's own events can be switched off, and the weather set to rain or snow for the run.
- **Assistant** — say what you want to run ("evening plans with 200 travellers, vehicles at L3 with 30%
  staying at L0, slow the bridge from 17:30 to 18:30, then run it") and a language model fills in the
  form. Its proposal is checked like any configuration before it is used (`agmsim/assistant.py`); what the
  form has no control for is listed below the form and can be removed. It needs a real language model:
  on your machine a hosted model whose key is in `.env` is picked automatically, online you type a key.
- **What they know** — for every kind, how many of its own decisions it recalls; for cars, how much of
  traffic they see, how long they keep a slow road in mind, and whether they tell each other.
- **Map** — agents on the network; roads darken as traffic slows, incidents are marked, and the agent
  of the selected decision is ringed. For a car that sees only what is near or that talks to others, two
  circles show how far it sees and how far it is heard.
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
pages; DeepSeek's API does). `sh website/publish.sh` rebuilds the project site with the simulator in it and
pushes it to the `gh-pages` branch of the repository `agenticmobility`, which GitHub Pages serves.

## Outcome variables

Every run reports four system-level measures under `outcomes` (they lead the results on the page and in
`agmsim run`):

| Variable | Key | Meaning |
| - | - | - |
| Average waiting time | `wait_time_s_mean` | time travellers wait for transport (at a stop, or for a car they called), per completed trip; trips without a wait count as 0 |
| VMT | `vmt_miles` (`vkt_km`) | vehicle miles travelled by private cars, on-demand cars and buses |
| VHT | `vht_hours` | vehicle hours travelled: time cars and buses spend on the road |
| Average traffic speed | `traffic_speed_mph` (`traffic_speed_kmh`) | VMT ÷ VHT |

Further measures by kind of agent (`travelers`, `plans`, `vehicles`, `bus`, `evcs`, `operator`), and what
agents were asked and decided (`decisions`, `llm`), follow them.

## Modules, and where each stands

| Module | Part | Status |
| - | - | - |
| **Physical infrastructure** | road network (the Montreal testbed; also a grid, two banks with bridges, or from a file) | done |
| | charging stations | done |
| | external events (a road blocked or slowed; weather; a surge of trips from or to a place; a place closing early; a change of agency during the run) | done; not yet: outages of charging stations or bus lines |
| | destinations as points of interest | done: named places with a type and opening hours; trips go to a place that fits their purpose, and activities of a day plan can take place at one |
| **Entity agents** | traveller, private car (agentic vehicle), bus, charging station | done |
| | traffic lights | done: fixed-time, deciding by their own queues, or cooperating with the orchestrator |
| | orchestrator | done as the kind `operator`: it sees the slowest roads and what waited at the lights, publishes advisories to cars, gives buses priority (also through red lights), sets a cordon charge, and asks the lights to favour a direction or to serve a queue. It does not address single cars or buses |
| | on-demand car | done: travellers call a car, the service proposes each request to the nearest free car, cars re-route, turn requests down and move while free according to their level. Not carried over from the earlier version in `archive/ridesourcing/`: cars setting their own price, claiming requests themselves, charging |
| **Agency mechanism** | observation: what an agent perceives of the current state | done: a setting of each kind. Cars see no traffic, the roads around them, or every road; what the other kinds are shown follows from their role and level |
| | memory: what an agent carries from the past | done: a setting of each kind. Every agent is shown its own latest decisions and what came of them (if wanted, with the reason given at the time), whatever decides for it; cars keep slow roads in mind for a set time |
| | interaction: what agents exchange | done: cars in range of each other pass on what they know about slow roads, next to the channels the levels open (a car proposes to its owner, the orchestrator asks the lights, advisories, prices, ride proposals). Not yet: free-text messages between any two agents |
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
actions and the exogenous input — never on how the state was reached. What an agent carries from the past
(its latest decisions and outcomes, the road reports it holds) is part of the state, kept by the framework
under rules of its own (`agmsim/mechanisms.py`) and handed to the decision model with the observation; a
decision model needs no state of its own. This is why a recorded run can be replayed decision by decision
to the identical state, and why the same model can be put behind any agent. The outcome variables are
functions of the trajectory `S_0 … S_T`.

### What an agent has to go on: observation, memory, interaction

The agency level says what a kind of agent may *do*. Three more settings of each kind say what it has to
go on when it decides:

```yaml
agents:
  vehicle:
    agency: L1
    observation: {traffic: local, radius_m: 900, route_names: false, route_roads: false}   # traffic: none | local | global
    memory: {decisions: 4, reasons: false, traffic_seconds: 600}
    interaction: {share_traffic: true, range_m: 500}
```

- **Observation** — what it perceives now. `traffic` is how much of current road conditions a car sees
  for itself: nothing, the roads within `radius_m`, or every road. A route's number does not say which
  route it is (route 0 is always the route the car is on), so a car cannot tell from it whether an offered
  route is one it remembers. Two settings let it: with `route_names` every route has the short name of its
  *way*, which it keeps from one decision to the next (routes that share at least half their length are
  one way and share the name); with `route_roads` every route lists the junctions it passes.
- **Memory** — what it carries from the past. `decisions` is how many of its own latest decisions and
  outcomes it is shown at every decision point, and with `reasons` each decision comes with the reason
  its decision model gave at the time (these two apply to every kind); `traffic_seconds` is how long a
  slow road that a car saw or was told about stays with it.
- **Interaction** — what it exchanges. With `share_traffic`, cars within `range_m` of each other pass on
  what they know about slow roads, including that a road is clear again.

A car reckons a road it sees at its current speed, a road it holds a report about at the reported speed,
and every other road as free. When it learns of a slow road on the route it is following it reconsiders
that route at once. A car that does not see every road is told so in its observation: how far it sees,
that roads beyond are reckoned as clear, and the reports its estimates rest on, each with its age and
whether the car saw the road itself or was told. Without memory a report lasts until the next exchange
(10 s), so a car knows what the cars in range see now and nothing of what they heard; with memory, news
travels from car to car. The traffic settings belong to the kinds that find their own way (`vehicle`,
`ondemand`); writing them for another kind is an error.

A remembered decision is put in words that still mean something later (each kind says how: a choice of
route is remembered by what the routes were, not by their numbers), and what came of a decision is added
when it shows: a car that finds the route it chose much slower than it reckoned is told so at its next
look at the route.

All three are kept by the framework and not by a decision model, so a rule, an algorithm, a language
model and a remote service behind the same kind are handed the same observation and the same memory.
Whether a decision model *uses* its memory is another matter: the vehicle `algorithm` does with
`policy_params: {settle_min: 10}` (after a change of route it changes again only for a large saving),
and a language model can be told to read its memory first with `llm_policy_params: {reflect: true}`.

## The seven built-in kinds of agent

| Kind | L0 (non-agentic) | What agency adds |
| - | - | - |
| `traveler` (human) | habitual mode, follows the vehicle, waits | L1 choose mode · L2 accept/decline the vehicle's suggestions · L3 give up waiting · L4 re-plan, and change the rest of a day plan |
| `vehicle` (private agentic vehicle, one per car-owning traveller) | route fixed at departure | L1 re-route on traffic · L2 weigh time against money for its owner · L3 propose park-and-ride to the owner · L4 manage its own charging |
| `ondemand` (on-demand car: a fleet that carries travellers who call a car) | takes every request the service proposes, fixed route, waits where it dropped its passenger | L1 re-route on traffic · L2 weigh a proposed request against its objective and turn it down · L4 decide, while free, where to wait |
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
.venv/bin/agmsim run montreal --set human_agents.use_llm=true --set llm.preset=deepseek
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
.venv/bin/agmsim matrix montreal --factor agents.vehicle.agency=L0,L1,L2,L3,L4 --seeds 3 --workers 5
```

```bash
.venv/bin/agmsim matrix montreal --llm-2x2 --seeds 3 --set llm.backend=mock
```

Results are means with 95% confidence half-widths over seeds; `--out DIR` writes `runs.csv`.

Mixed populations are set with `mix`, e.g. 30% of vehicles at level 1 and the rest at the base level:
`agents: {vehicle: {agency: L0, mix: [{share: 0.3, agency: L1}]}}`.

### Places and external events

Two parts of the physical setting are given in a scenario next to the network: the **places** people go
to, and the **events** that happen to the town during the run.

```yaml
pois:                                    # points of interest; the same for every seed
  - {type: work, count: 6}               # six offices, put down at free nodes
  - {type: shop, count: 4, closes: "18:30"}
  - {id: stadium, type: leisure, node: 21, name: Stadium, weight: 3}
demand: {pattern: uniform, to_pois: true}   # a trip goes to a place that fits its purpose
events:                                  # (the list is also accepted under its older name, interventions)
  - {type: weather, label: heavy rain, at_seconds: 1200, until_seconds: 3600, speed_factor: 0.75,
     walk_factor: 0.85, discomfort: 1.8}
  - {type: demand_surge, poi: stadium, at_seconds: 1800, until_seconds: 2700, trips: 120}
  - {type: link_condition, at_seconds: 2100, until_seconds: 3900, factor: 0.1, links: [[14, 15], [15, 21]]}
  - {type: poi_closure, poi: shop2, at_seconds: 2400}
```

- **Places** (`agmsim/pois.py`). With `demand.to_pois` a commute goes to a `work` place, an errand to a
  `shop`, and so on; nearer and weightier places are likelier. An activity of a day plan can be put at a
  place (`at: {poi: shop}`: the one that adds the least detour) and then has its opening hours. A traveller
  is told the name and type of the place they are heading for; `pois.visits`, `pois.found_closed` and the
  map show what happened there.
- **Weather** slows every road and walking, and tells every agent so in its observation
  (`context.environment`); the built-in choice model weighs waiting and walking more heavily while it lasts.
- **A surge** adds trips that all start from (or head to) one place within a period: a match ending.
- **A road blocked or slowed**, as before. Who knows about it depends on the vehicles' and the
  orchestrator's agency.
- **A place closing early** ends what people planned to do there at that time; those who may adapt their
  day plan are told and review it.

The testbed has a road slowed by a crash; the others are added with `--set events=[...]`.

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

The testbed is a morning of single trips; day plans are switched on with `demand.plan`.

Not yet: editing the plan while travelling, forecasts of traffic (projections assume it stays as it is),
and MATSim's day-to-day loop itself as a reference.

### Experiments on the testbed

`experiments/suite.py` holds the experiments. Each is a set of arms that change one thing on the testbed
(`montreal`) and leave everything else as it is; all arms run on the same seeds, and differences from the
first arm are reported **paired**:

```bash
.venv/bin/python experiments/suite.py                         # 3 seeds while developing
.venv/bin/python experiments/suite.py vehicle_agency --seeds 10
```

`vehicle_agency` changes nothing but the level of the private cars, from L0 to L4; `vehicle_agency_alone`
does the same with every other agent passive. Results: `docs/EXPERIMENTS_zh.md` (in Chinese). An arm whose
label contains `{llm}` runs on a language model: `--llm qwen-local` (default), `--llm deepseek`, or both to
compare them as arms; `--answer choice explain reason` adds the answer formats as arms.

Earlier experiments were each run on a scenario of their own (an incident, a toll, a bottleneck with a bus
lane, scarce charging, traffic lights, on-demand cars, places and events, day plans, two bridges). Those
scenarios are no longer part of the package; they and their results are in the repository's history under
the tag `before-single-testbed`, and the write-up is `docs/archive/EXPERIMENTS_before_testbed_zh.md`.

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
New tasks subclass `agmsim.benchmark.Task` and register with `@register_task`.

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

- Traffic is a simple link-level speed–density model with stop lines at junctions that have traffic lights;
  no lanes, and queues do not spill back onto the road behind.
- Day plans (activity chains) are executed and edited within one period of a day; there is no day-to-day
  learning loop yet (the `feedback` hook exists), and plans cannot be changed while travelling.
- Bus lines are circular; park-and-ride estimates use timetable times.
- On-demand cars do not set their own price, claim requests themselves or charge; the earlier ride-sourcing
  code that did is parked in `archive/ridesourcing/` (not maintained).
- The built-in non-LLM policies are simple heuristics with assumed parameters, not calibrated models.
- Of the agency mechanisms, sight, memory for roads and car-to-car reports exist for cars only; reports
  are always truthful and taken at face value; there are no free-text messages between agents. Recalling
  its own decisions helps a car when a built-in rule uses them; the language models tried used them only
  when the memory said what came of a decision and routes could be told apart, and then only after being
  caught out once (`docs/archive/EXPERIMENTS_before_testbed_zh.md`, section 12).
- The operator's action set is small. No validation against real-world data has been done.

Tests: `.venv/bin/python -m pytest`. Experiments and measured results (in Chinese): `docs/EXPERIMENTS_zh.md`.

No licence has been chosen yet; until one is added, all rights are reserved.
