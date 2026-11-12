# Command line

```text
agmsim run <scenario> [--set KEY=VALUE]... [--json] [--quiet]
agmsim serve [--host HOST] [--port PORT]
agmsim matrix <scenario> --factor KEY=V1,V2 [--factor ...] [--seeds N] [--workers N] [--metric M] [--out DIR]
agmsim replay <run directory>
agmsim explain <run directory> <agent> [--observations]
agmsim list
```

| Command | What it does |
| - | - |
| `run` | runs one simulation, prints the outcome measures, writes `runs/<name>/` |
| `serve` | starts the web interface on your machine |
| `matrix` | runs every combination of the factors' levels over common seeds; `--llm-2x2` adds the human and machine LLM switches as factors |
| `replay` | re-executes a recorded run from its decisions and verifies the state after every tick |
| `explain` | shows why an agent did what it did: every decision it was asked for, with `--observations` also what it saw |
| `list` | lists scenarios, kinds of agent, agency levels, decision models, presets and plug-ins |

`<scenario>` is `montreal` or the path of a YAML file.

Experiments with paired differences: `python experiments/suite.py [name] [--seeds N] [--llm PRESET]`, see
[](../guide/experiments.md).
