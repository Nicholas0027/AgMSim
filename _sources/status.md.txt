# Status and limitations

AgMSim is an early prototype.

## Where each module stands

| Module | Part | Status |
| - | - | - |
| **Physical infrastructure** | road network | the Montreal testbed; also a grid, two banks with bridges, or from a file |
| | charging stations | done |
| | external events | a road blocked or slowed, weather, a surge of trips, a place closing early, a change of agency. Not yet: outages of stations or transit lines |
| | destinations as named places | done |
| **Entity agents** | traveller, private car, transit vehicle, charging station | done |
| | traffic lights | fixed-time, deciding by their own queues, or cooperating with the orchestrator |
| | orchestrator | advisories, transit priority, a cordon charge, requests to traffic lights. It does not address single cars |
| | on-demand car | call, dispatch to the nearest free car, re-routing, turning requests down, repositioning. Not yet: own prices, charging |
| **Agency mechanisms** | observation, memory, interaction | settings of each kind; sight, memory for roads and car-to-car reports for cars only. Not yet: free-text messages between agents |
| **Simulation engine** | a controlled Markov process | done |
| **Interface** | web page, online simulator, command line, Python, configuration by conversation | done |

## Limitations

- The testbed is an abstraction: positions, lengths, speeds and lanes are approximate, and nothing has been
  calibrated or validated against data from the city.
- Traffic is a simple speed-density model per road with stop lines at junctions that have traffic lights; no
  lanes, and queues do not spill back onto the road behind.
- Transit lines are loops; park-and-ride estimates use timetable times.
- The built-in decision models are simple heuristics with assumed parameters, not calibrated models.
- Day plans are executed and edited within one period of a day; there is no day-to-day learning loop.
- On the testbed, only some levels of agency have room to act so far: there is no charge by default, and
  transit is not quicker than driving by enough for a car to propose a transfer.
- Reports that cars pass on are always truthful and taken at face value.

No licence has been chosen yet; until one is added, all rights are reserved.
