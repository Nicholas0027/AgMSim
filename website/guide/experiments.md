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
| L0 | 36.5 | 306 | 11170 | 900 | 27 |
| L1 | 36.7 (+0.2) | 304 (−2) | 11180 (+8) | 896 (−4) | 27 |
| L2 | 36.6 (+0.1) | 305 (−2) | 11160 (−13) | 896 (−5) | 27 |
| L3 | as L2 | | | | |
| L4 | 36.5 (0.0) | 313 (+7)* | 11400 (+235)* | 901 (+1) | 134 (+107)* |

**Every other agent passive** (no advisory; a fixed charge of 2 USD to enter the centre):

| Cars at | Speed (mph) | VHT (h) | VMT (miles) | Mean trip time (s) | Charges paid (USD) |
| - | - | - | - | - | - |
| L0 | 31.4 | 408 | 12750 | 1160 | 705 |
| L1 | 34.9 (+3.5) | 367 (−41) | 12780 (+28) | 974 (−186) | 703 (−3) |
| L2 | 34.4 (+3.0) | 372 (−36) | 12770 (+19) | 997 (−163) | 681 (−25)* |
| L3 | as L2 | | | | |
| L4 | 34.2 (+2.9) | 380 (−28) | 12990 (+239) | 1001 (−159) | 689 (−16) |

What it shows:

- **Re-routing (L1) is the step that matters, where nobody else does that job.** With every other agent
  passive, speed rises from 31.4 to 34.9 mph and the mean trip is three minutes shorter (with three seeds the
  spread between seeds is still wide). Once the
  orchestrator publishes an advisory, hardly anything is left for the cars to add.
- **L2 saves a little in charges**, not time. **L3 has no room yet**: transit is not quicker than driving by
  enough for a car to propose park-and-ride.
- **L4 adds traffic.** Cars below 20% charge drive themselves to a station after dropping their owner: about
  110 more charging trips per run and 2% more vehicle miles.

System performance does not simply rise with the level of agency: it depends on what the other agents do and
on what the setting leaves for a level to act on.
