---
hide-toc: true
---

# Agentic Mobility

<p class="tagline"><strong>AgMSim</strong>, the Agentic Mobility Simulator, is a testbed for mobility systems in
which every participant can be an agent: travellers, their cars, on-demand cars, transit, charging stations,
traffic lights and the orchestrator of the network.</p>

<a class="start-button" href="simulator/index.html">Start the simulator</a>
<a class="start-button quiet" href="quickstart/installation.html">Install</a>
<a class="start-button quiet" href="https://github.com/Nicholas0027/AgMSim">Source</a>

## Start

The simulator runs in this page, in your browser: nothing to install, nothing sent to a server. It opens on
the testbed, a simplified Montreal on a weekday morning. Set who decides for each kind of agent and how much
it may decide, press **Run**, and click any decision to see what the agent saw. The page opens with a language
model deciding for most kinds, which needs your own API key; **built-in for all** runs at once without one.

<div class="sim-embed">
<iframe src="simulator/index.html" title="The AgMSim simulator" loading="lazy"></iframe>
</div>
<p class="sim-open"><a href="simulator/index.html">Open the simulator in a full page</a> · the first load fetches
a Python runtime and takes a few seconds.</p>

## What it is for

A mobility system is no longer one decision-maker and many followers. Cars re-route by themselves, weigh
tolls for their owners and propose where to park; traffic lights and charging stations react to what they
see; an orchestrator advises and prices. AgMSim lets you set, for each kind of participant and independently
of the others,

| Question | Setting | Examples |
| - | - | - |
| **How** does it decide? | decision model | a rule, a behavioural model, an optimiser, an LLM, a remote service |
| **What** may it decide? | agency level `L0` to `L4` | L0 follows defaults; L4 forms its own goals |
| **What** does it have to go on? | observation, memory, interaction | how far a car sees, what it remembers, what cars tell each other |

and then measures what that does to the system: waiting time, vehicle miles and hours travelled, traffic speed.

Agency follows the agentic-vehicle literature (Yu et al., *Agentic Vehicles for Human-Centered Mobility*,
[arXiv:2507.04996](https://arxiv.org/abs/2507.04996)): a continuum distinct from automation. AgMSim
operationalises it as **decision authority**, the set of actions the environment offers an agent. An agent
without authority at a decision point takes the default action and its decision model is not consulted, so
the same language model can be evaluated as a level-0 or a level-4 vehicle, and a rule can be compared with
a language model under identical authority.

## Features

- **Every kind of participant is an agent.** Seven kinds are built in: traveller, private car, on-demand car,
  transit vehicle, charging station, traffic light and orchestrator. A new kind is one class.
- **Agency is a setting, not a property of the model.** Levels L0 to L4, or any subset of permissions, per kind
  or for a share of a kind.
- **Any decision model behind any agent.** Rules, algorithms, language models and remote services receive the
  same observation and the same memory, so they can be compared like for like.
- **One testbed.** A simplified Montreal of 25 nodes and 51 roads, in the way transport research uses Sioux
  Falls. Experiments change settings of the testbed; they do not bring a network of their own.
- **Reproducible runs with language models.** Every decision is recorded with what the agent saw; a run is
  replayed from its decisions to the identical state, without the model.

## The testbed

```{raw} html
:file: _static/testbed.html
```

A weekday morning from 06:00 to 12:00 on the island of Montreal, its crossings, the South Shore and Laval:
900 travellers, their cars, 40 on-demand cars, four transit lines, traffic lights in the inner city, seven
charging stations, an orchestrator, and a crash on the Decarie expressway at 07:30. See [](testbed.md).

## Install

```bash
git clone https://github.com/Nicholas0027/AgMSim.git && cd AgMSim
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/agmsim run montreal
```

See [](quickstart/installation.md).

## Status

An early prototype: see [](status.md). Questions and problems:
[issues on GitHub](https://github.com/Nicholas0027/AgMSim/issues).

```{toctree}
:hidden:
:caption: Start

simulator
testbed
```

```{toctree}
:hidden:
:caption: Quick start

quickstart/installation
quickstart/run
quickstart/interface
quickstart/python
```

```{toctree}
:hidden:
:caption: Concepts

concepts/agency
concepts/mechanisms
concepts/engine
concepts/outcomes
```

```{toctree}
:hidden:
:caption: Guide

guide/experiments
guide/llm
guide/events
guide/extending
```

```{toctree}
:hidden:
:caption: Reference

configuration
reference/cli
status
```
