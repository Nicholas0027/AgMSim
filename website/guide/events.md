# Places and events

Two parts of the physical setting are given in a scenario next to the network: the **places** people go to,
and the **events** that happen during the run.

## Places

```yaml
pois:
  - {id: downtown_offices, type: work, node: 0, name: Downtown offices, weight: 8}
  - {id: mcgill, type: school, node: 1, name: McGill University, weight: 3}
  - {id: jean_talon_market, type: shop, node: 12, name: Jean-Talon Market, weight: 2, closes: "18:30"}
demand: {to_pois: true}
```

With `demand.to_pois` a commute goes to a `work` place, an errand to a `shop`, and so on; nearer and
weightier places are likelier. A traveller is told the name and type of the place they are heading for. A
place can have opening hours.

## Events

```yaml
events:
  - {type: link_condition, label: crash on the Decarie expressway, at_seconds: 1800, until_seconds: 4200,
     links: [[8, 7]], factor: 0.3}
  - {type: weather, label: snow, at_seconds: 0, until_seconds: 7200, speed_factor: 0.75, walk_factor: 0.85,
     discomfort: 1.8}
  - {type: demand_surge, poi: bell_centre, at_seconds: 3600, until_seconds: 4500, trips: 120}
  - {type: poi_closure, poi: jean_talon_market, at_seconds: 2400}
  - {type: set_agency, kind: vehicle, agency: L3, at_seconds: 3000}
```

| Event | What it does |
| - | - |
| `link_condition` | slows or blocks roads (`links` are pairs of node ids, `factor` the share of normal speed). Who knows about it depends on the cars' and the orchestrator's agency |
| `weather` | slows every road and walking, and tells every agent so in its observation |
| `demand_surge` | adds trips that all start from, or head to, one place within a period |
| `poi_closure` | closes a place early; those who may adapt their plans are told |
| `set_agency` | changes the agency of a kind during the run |

The testbed has one event, the crash on the Decarie expressway. Others are added with `--set events=[...]`
or in a copy of the scenario.
