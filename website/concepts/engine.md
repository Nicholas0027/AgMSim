# The engine

## A controlled Markov process

A run is a controlled Markov process in discrete time. The state `S_t` is everything in the world: where
every person and vehicle is and what it is doing, queues, prices, road conditions, the clock, and what every
agent remembers. In each step

1. exogenous input arrives: trip needs and scheduled events, all fixed by the scenario and the seed;
2. every agent that has a decision point is shown an observation, a function of the state alone and the same
   state for all of them, and its decision model returns an action from those its agency permits; an agent
   without authority there takes the default action;
3. the state moves on, with actions applied in a fixed order.

Neither the transition nor the observation has randomness of its own, so the next state depends only on the
current state, the actions and the exogenous input, never on how the state was reached. A decision model
needs no state of its own. This is why a recorded run can be replayed decision by decision to the identical
state, and why the same model can be put behind any agent.

## Traffic

Roads congest: the speed on a road falls linearly with the number of vehicles on it. A junction with traffic
lights holds vehicles at the end of their road until their approach has green; they are let through in
arrival order at the saturation flow, and a change of green costs a few seconds. Each simulated car stands
for several real ones (`network.capacity_scale`).

## Reproducibility

- **Seeded streams.** Demand, personas and car ownership depend only on the seed, never on policies, so
  experimental arms that share a seed face identical travellers.
- **Trace.** Every decision is recorded with what the agent saw, what it was permitted to do and what it
  chose; every tick stores a fingerprint of the full state.
- **Replay.** `agmsim replay runs/<name>` re-executes a run from its recorded decisions and verifies the state
  after every tick. A run driven by a language model is thereby reproduced exactly, without the model.
- **Cassette.** `llm.cassette: file.jsonl` records model responses; re-running with it makes no model calls.

Sampling temperature is not a reliable route to reproducibility, which is why AgMSim relies on recorded
decisions.
