# Experiments

## How an experiment is written

An experiment is a set of **arms**. Each arm overrides a few settings of the testbed and leaves everything
else as it is. All arms run on the same seeds, that is on identical travellers, and results are reported as
**paired** differences from the first arm.

```bash
.venv/bin/python experiments/suite.py                         # every experiment, 3 seeds
.venv/bin/python experiments/suite.py vehicle_agency --seeds 10
```

Tables go to `experiments/results/<name>.md`, raw rows to `<name>.json`. An experiment in
`experiments/suite.py` looks like this:

```python
"vehicle_agency": {
    "question": "Only the level of the private cars changes, from L0 to L4. What does the system show?",
    "arms": {"cars at L0": {"agents.vehicle.agency": "L0"}, "cars at L1": {"agents.vehicle.agency": "L1"}},
    "metrics": {"outcomes.traffic_speed_mph": "speed (mph)", "outcomes.vht_hours": "VHT (h)"},
},
```

An arm whose label contains `{llm}` runs on a language model, once per `--llm` preset.

## Only the cars' level changes

Three seeds; differences from L0 in brackets, `*` where the 95% interval excludes zero.

**The testbed as it is** (the orchestrator publishes an advisory about the crash):

| Cars at | Speed (mph) | VHT (h) | VMT (miles) | Mean trip time (s) | Charging trips |
| - | - | - | - | - | - |
| L0 | 33.9 | 165 | 5581 | 991 | 18 |
| L1 | 34.0 (+0.1) | 163 (−1) | 5547 (−34) | 979 (−12) | 18 |
| L2 | 34.0 (+0.1) | 163 (−1) | 5551 (−30) | 981 (−10) | 18 |
| L3 | as L2 | | | | |
| L4 | 33.7 (−0.2) | 172 (+8)* | 5802 (+221)* | 1001 (+10) | 89 (+71)* |

**Every other agent passive** (no advisory; a fixed charge of 2 USD to enter the centre):

| Cars at | Speed (mph) | VHT (h) | VMT (miles) | Mean trip time (s) | Charges paid (USD) |
| - | - | - | - | - | - |
| L0 | 28.4 | 226 | 6413 | 1303 | 460 |
| L1 | 32.4 (+4.0)* | 206 (−21)* | 6663 (+249)* | 1193 (−110)* | 457 (−3) |
| L2 | 31.8 (+3.4) | 208 (−19) | 6601 (+188) | 1214 (−89) | 435 (−25)* |
| L3 | as L2 | | | | |
| L4 | 31.7 (+3.3)* | 216 (−11) | 6844 (+430)* | 1228 (−76) | 435 (−25)* |

What it shows:

- **Re-routing (L1) is the step that matters, where nobody else does that job.** With every other agent
  passive, speed rises from 28.4 to 32.4 mph and the mean trip is almost two minutes shorter. Once the
  orchestrator publishes an advisory, hardly anything is left for the cars to add.
- **L2 saves a little in charges**, not time. **L3 has no room yet**: transit is not quicker than driving by
  enough for a car to propose park-and-ride.
- **L4 adds traffic.** Cars below 20% charge drive themselves to a station after dropping their owner: about
  70 more charging trips per run and 4% to 7% more vehicle miles.

System performance does not simply rise with the level of agency: it depends on what the other agents do and
on what the setting leaves for a level to act on.
