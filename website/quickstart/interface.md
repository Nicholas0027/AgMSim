# Using the web interface

```bash
.venv/bin/agmsim serve          # http://127.0.0.1:8765
```

opens the same page as the [online simulator](../simulator.md),
served from your machine.

## What the page shows

- **Guide** (top right). Everything that is in the simulation, one entry each: the run, the trips, the outcome
  measures, the road network, places, events, transit lines, charging sites, junctions with lights, and every
  kind of agent. Click an entry to see what it is, what it may do (its functions, with those switched on at its
  level marked), what it is told (its goal, and the prompt a language model gets before every decision), all
  its settings, and what it is doing in the run: how often it was asked and why, what it chose, and its state
  at the moment shown on the map. A kind's name in the Agents table opens its entry.
- **Assistant.** Say what you want to run ("vehicles at L3 with 30% staying at L0, slow the Champlain Bridge
  from 07:30 to 08:30, then run it") and a language model fills in the form. Its proposal is checked like any
  configuration before it is used. It needs a real language model: on your machine a hosted model whose key
  is in `.env` is picked automatically, online you type a key.
- **Agents.** For each kind: who decides (a built-in model or an LLM) and its agency level. A kind decided by
  a language model has a link **goal and prompt**: it shows the prompt as it will be sent, and lets you change
  the kind's goal, add instructions to the prompt, or write the whole prompt yourself.
- **What they know.** For every kind, how many of its own decisions it recalls; for cars, how much of
  traffic they see, how long they keep a slow road in mind, and whether they tell each other.
- **Events.** The scenario's own events can be switched off. Weather is three numbers on sliders (how fast
  roads are, how fast walking is, how much longer waiting and walking feel), for a period of the run, with the
  word the agents are told; rain and snow are shortcuts.
- **Map.** Agents on the network; roads darken as traffic slows, incidents are marked, and the agent of the
  selected decision is ringed.
- **Over the run.** Three charts on one time axis: travellers by stage, traffic speed as a share of free-flow
  speed, and decisions asked per time slice by kind of agent. Hover to read every series at that moment,
  click to move the map there.
- **Results.** The outcome measures, how trips were made, and how often each kind of agent used its authority.
- **Decisions.** Every decision, filterable by kind, by agent and by whether authority was used. A selected
  decision shows why the agent was asked, what it could choose, what it chose and why, the agent's whole
  morning on a timeline and, when a language model decided, the text the model was sent, word for word.

A kind of agent keeps one colour everywhere on the page; the palette was checked for colour-vision
deficiencies in light and dark mode.

## The online version

The online simulator is that page as a static site: Pyodide runs the `agmsim` package in the visitor's
browser. Any static host can serve it and nothing is sent to a server. A visitor who chooses an LLM types
their own key into the page; it is kept in that tab's memory only, never becomes part of a configuration or
trace, and the browser itself calls the provider (which must accept requests from web pages).
