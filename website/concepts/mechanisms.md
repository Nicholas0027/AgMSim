# Observation, memory, interaction

The agency level says what a kind of agent may *do*. Three more settings of each kind say what it has to go
on when it decides:

```yaml
agents:
  vehicle:
    agency: L1
    observation: {traffic: local, radius_m: 900}       # traffic: none | local | global
    memory: {decisions: 4, reasons: false, traffic_seconds: 600}
    interaction: {share_traffic: true, range_m: 500}
```

## Observation

What it perceives now. `traffic` is how much of current road conditions a car sees for itself: nothing, the
roads within `radius_m`, or every road. A car that does not see every road is told so in its observation:
how far it sees, that roads beyond are reckoned as clear, and the reports its estimates rest on.

A route's number does not say which route it is (route 0 is always the route the car is on). Two optional
settings let a car recognise a route it remembers: `route_names` gives every route the short name of its
way, kept from one decision to the next; `route_roads` lists the junctions each route passes.

## Memory

What it carries from the past. `decisions` is how many of its own latest decisions and outcomes an agent is
shown at every decision point; with `reasons` each comes with the reason its decision model gave at the
time. These two apply to every kind. `traffic_seconds` is how long a slow road that a car saw or was told
about stays with it.

A remembered decision is put in words that still mean something later, and what came of it is added when it
shows: a car that finds the route it chose much slower than it reckoned is told so at its next look.

## Interaction

What it exchanges. With `share_traffic`, cars within `range_m` of each other pass on what they know about
slow roads, including that a road is clear again. Next to this there are the channels the levels open: a car
proposes to its owner, the orchestrator asks the lights, advisories, prices, ride proposals.

## Kept by the framework

All three are kept by the framework and not by a decision model. A rule, an algorithm, a language model and
a remote service behind the same kind are handed the same observation and the same memory, so they can be
compared like for like. Whether a decision model *uses* its memory is another matter.

The traffic settings belong to the kinds that find their own way (`vehicle`, `ondemand`); writing them for
another kind is an error.
