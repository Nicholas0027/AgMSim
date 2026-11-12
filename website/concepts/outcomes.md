# Outcome variables

Every run reports four system-level measures under `outcomes`. They lead the results on the page and in
`agmsim run`.

| Variable | Key | Meaning |
| - | - | - |
| Average waiting time | `wait_time_s_mean` | time travellers wait for transport (at a stop, or for a car they called), per completed trip; trips without a wait count as 0 |
| VMT | `vmt_miles` (`vkt_km`) | vehicle miles travelled by private cars, on-demand cars and transit vehicles |
| VHT | `vht_hours` | vehicle hours travelled: time vehicles spend on the road |
| Average traffic speed | `traffic_speed_mph` (`traffic_speed_kmh`) | VMT ÷ VHT |

Further measures follow them, by kind of agent:

| Group | Examples |
| - | - |
| `travelers` | trips, completion rate, trip time (mean, 90th percentile), mode shares, generalised cost |
| `vehicles` | re-routes, tolls paid, proposals to owners, charging stops and queueing, empty kilometres |
| `ondemand` | requests served and turned down, empty kilometres, profit |
| `bus` | boardings, holds, stops skipped |
| `evcs` | sessions, revenue, utilisation, how evenly stations are used |
| `signals` | waiting at lights, switches |
| `operator` | advisories, charge, revenue |
| `decisions` | per kind: decision points, how often authority was used |
| `llm` | calls, tokens, fallbacks, latency, estimated cost |

In Python: `result.metrics["outcomes"]["traffic_speed_mph"]`; on the command line `agmsim run montreal --json`.
