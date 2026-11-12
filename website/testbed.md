# The testbed: a simplified Montreal

Experiments are run on one network, in the way much of transport research uses Sioux Falls (24 nodes): small
enough to be read off a map, fixed, and the same for everybody. The testbed is the scenario `montreal`.

```{raw} html
:file: _static/testbed.html
```

## The network

25 nodes and 51 two-way roads, about 21 km by 17 km (`agmsim/montreal.py`, `network: {type: montreal}`). It
keeps what shapes traffic in the city:

- **an island** reached over a handful of crossings: the Champlain, Victoria and Jacques-Cartier bridges and
  the Lafontaine tunnel to the South Shore, four bridges to Laval;
- **a centre on its south side**: Downtown, McGill, Old Montreal, the Quartier Latin;
- **a ring of expressways**: the A-40 Metropolitan, the A-15 Decarie, the A-20, the Ville-Marie and the A-25;
- **the arterials between them**: Sherbrooke, Saint-Denis, Papineau, Pie-IX, Henri-Bourassa and others.

| Class of road | Roads | Free-flow speed | Lanes each way |
| - | - | - | - |
| expressway | 15 | 90 km/h | 3 |
| bridge or tunnel | 8 | 60 to 70 km/h | 2 or 3 |
| arterial | 28 | 50 km/h | 2 |

| Id | Node | Id | Node | Id | Node |
| - | - | - | - | - | - |
| 0 | Downtown | 9 | Dorval (YUL) | 18 | Montreal-Nord |
| 1 | McGill | 10 | Saint-Laurent | 19 | Centre-Sud |
| 2 | Old Montreal | 11 | Ahuntsic | 20 | Longueuil |
| 3 | Quartier Latin | 12 | Villeray | 21 | Saint-Lambert |
| 4 | Plateau | 13 | Rosemont | 22 | Brossard |
| 5 | Cote-des-Neiges | 14 | Saint-Michel | 23 | Laval (Chomedey) |
| 6 | Verdun | 15 | Olympic Stadium | 24 | Laval (Pont-Viau) |
| 7 | Turcot | 16 | Anjou | | |
| 8 | Decarie / A-40 | 17 | Mercier-Est | | |

Positions are approximate (read off a map by district), lengths are straight-line distances times 1.2,
speeds and lanes are typical values for the class of road. **It is an abstraction to experiment on, not a
calibrated model of the city.**

## The morning

`agmsim run montreal` simulates a weekday from 06:00 to 12:00 with everything the simulator has:

| Part | In the testbed |
| - | - |
| Travellers | 900, living where people live (weights by district), 65% with a car; most set out around 08:00 |
| Places | 21 named destinations: downtown offices, the Technoparc, YUL airport, McGill, UdeM, UQAM, three hospitals, markets and malls, the Bell Centre, the Olympic Park |
| Private cars | one per car-owning traveller, electric, level L1 (re-route on traffic) |
| On-demand cars | a fleet of 40, level L1 |
| Transit | four lines on their own right of way, drawn as loops: Orange, Green, South, West |
| Traffic lights | at 10 junctions of the inner city, deciding by their own queues (L1) |
| Charging stations | 7 stations, 17 chargers, dynamic price (L1) |
| Orchestrator | publishes advisories (L1); a cordon around Downtown that can carry a charge |
| Event | a crash on the Decarie expressway, 07:30 to 08:10, traffic at 30% of its speed |

With these settings and seed 1 every trip is completed, the mean trip takes 15 minutes, 63% of trips are made
by car, 19% by transit and 18% by on-demand car; a run with the built-in decision models takes about 15 seconds.

## Changing a condition

An experiment changes settings of the testbed and leaves everything else as it is:

```bash
agmsim run montreal --set agents.vehicle.agency=L0          # cars that follow their first route
agmsim run montreal --set agents.operator.initial_toll=2    # a charge of 2 USD to enter the centre
agmsim run montreal --set events=[]                         # the same morning without the crash
agmsim run montreal --set agents.traveler.count=1500        # more people
```

Every setting is described in [](configuration.md).
