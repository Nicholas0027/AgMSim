# Using the web interface

```bash
.venv/bin/agmsim serve          # http://127.0.0.1:8765
```

opens the same page as the [online simulator](../simulator.md),
served from your machine.

## What the page shows

- **Assistant.** Say what you want to run ("vehicles at L3 with 30% staying at L0, slow the Champlain Bridge
  from 07:30 to 08:30, then run it") and a language model fills in the form. Its proposal is checked like any
  configuration before it is used. It needs a real language model: on your machine a hosted model whose key
  is in `.env` is picked automatically, online you type a key.
- **Agents.** For each kind: who decides (a built-in model or an LLM) and its agency level.
- **What they know.** For every kind, how many of its own decisions it recalls; for cars, how much of
  traffic they see, how long they keep a slow road in mind, and whether they tell each other.
- **Map.** Agents on the network; roads darken as traffic slows, incidents are marked, and the agent of the
  selected decision is ringed.
- **Over the run.** Three charts on one time axis: travellers by stage, traffic speed as a share of free-flow
  speed, and decisions asked per time slice by kind of agent. Hover to read every series at that moment,
  click to move the map there.
- **Results.** The outcome measures, how trips were made, and how often each kind of agent used its authority.
- **Decisions.** Every decision, filterable by kind, by agent and by whether authority was used. A selected
  decision shows why the agent was asked, what it could choose, what it chose and why, and the agent's whole
  morning on a timeline.

A kind of agent keeps one colour everywhere on the page; the palette was checked for colour-vision
deficiencies in light and dark mode.

## The online version

The online simulator is that page as a static site: Pyodide runs the `agmsim` package in the visitor's
browser. Any static host can serve it and nothing is sent to a server. A visitor who chooses an LLM types
their own key into the page; it is kept in that tab's memory only, never becomes part of a configuration or
trace, and the browser itself calls the provider (which must accept requests from web pages).
