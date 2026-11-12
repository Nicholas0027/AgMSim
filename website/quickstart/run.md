# Run the testbed

```bash
.venv/bin/agmsim run montreal
```

prints the outcome measures and writes `runs/<name>/` with the configuration, the trace of every decision and
the metrics.

## Change a setting

`--set key=value` overrides any setting of the scenario; repeat it for several.

```bash
.venv/bin/agmsim run montreal --set agents.vehicle.agency=L3
.venv/bin/agmsim run montreal --set agents.vehicle.agency=L0 --set agents.operator.agency=L0
.venv/bin/agmsim run montreal --set simulation.seed=5 --json        # all metrics as JSON
```

## Compare settings over seeds

```bash
.venv/bin/agmsim matrix montreal --factor agents.vehicle.agency=L0,L1,L2,L3,L4 --seeds 3 --workers 5
```

runs every level on the same seeds, that is on identical travellers, and prints means with 95% confidence
half-widths. `--out DIR` writes `runs.csv`. For designed experiments with paired differences see
[](../guide/experiments.md).

## Look into a run

```bash
.venv/bin/agmsim explain runs/<name> V012 --observations     # what car V012 was asked, saw and chose
.venv/bin/agmsim replay runs/<name>                          # re-execute from the recorded decisions and verify
```

All commands: [](../reference/cli.md).
