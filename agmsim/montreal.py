"""The testbed: a simplified road network of Montreal.

In the spirit of the Sioux Falls network (24 nodes): small enough to be read off a map, and keeping
what makes the city what it is for traffic -- an island reached over a handful of bridges and one
tunnel, a centre on its south side, a ring of expressways (the Metropolitan, Decarie, Ville-Marie,
A-25, A-20) and the radial arterials between them. 25 nodes (districts, interchanges and the far ends
of the crossings) and 51 two-way roads, about 20 km across.

Positions are approximate (read off a map by district), lengths are straight-line distances times
``CIRCUITY``, speeds and lanes are typical values for the class of road. It is an abstraction to
experiment on, not a calibrated model of the city.

``network: {type: montreal}`` builds it. ``NODE`` gives the id of a node by its short name, for
scenarios and experiments that want to refer to a place.
"""
from __future__ import annotations

import math

from .network import RoadNetwork, register_network

CIRCUITY = 1.2
ORIGIN = (45.52, -73.62)                    # latitude, longitude of (0, 0)
M_PER_DEG_LAT = 111_200.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(ORIGIN[0]))

#: (short name, full name, latitude, longitude); a node's id is its position in this list
NODES = [
    ("downtown", "Downtown", 45.4995, -73.5700),
    ("mcgill", "McGill", 45.5065, -73.5760),
    ("old_port", "Old Montreal", 45.5060, -73.5540),
    ("berri", "Quartier Latin", 45.5155, -73.5610),
    ("plateau", "Plateau", 45.5250, -73.5810),
    ("cdn", "Cote-des-Neiges", 45.5020, -73.6200),
    ("verdun", "Verdun", 45.4590, -73.5720),
    ("turcot", "Turcot", 45.4670, -73.5990),
    ("decarie", "Decarie / A-40", 45.5030, -73.6650),
    ("dorval", "Dorval (YUL)", 45.4500, -73.7300),
    ("st_laurent", "Saint-Laurent", 45.4900, -73.7370),
    ("ahuntsic", "Ahuntsic", 45.5556, -73.6680),
    ("villeray", "Villeray", 45.5430, -73.6250),
    ("rosemont", "Rosemont", 45.5440, -73.5850),
    ("st_michel", "Saint-Michel", 45.5650, -73.6050),
    ("hochelaga", "Olympic Stadium", 45.5580, -73.5515),
    ("anjou", "Anjou", 45.6000, -73.5650),
    ("mercier", "Mercier-Est", 45.5850, -73.5150),
    ("mtl_nord", "Montreal-Nord", 45.5950, -73.6400),
    ("centre_sud", "Centre-Sud", 45.5235, -73.5465),
    ("longueuil", "Longueuil", 45.5250, -73.5180),
    ("st_lambert", "Saint-Lambert", 45.4950, -73.5100),
    ("brossard", "Brossard", 45.4600, -73.4750),
    ("laval_w", "Laval (Chomedey)", 45.5600, -73.7450),
    ("laval_c", "Laval (Pont-Viau)", 45.5750, -73.6900),
]
NODE = {short: i for i, (short, *_rest) in enumerate(NODES)}

#: class of road -> (free-flow speed in m/s, lanes in each direction)
CLASSES = {"autoroute": (25.0, 3), "arterial": (13.9, 2), "bridge": (19.4, 3), "small_bridge": (16.7, 2)}

#: (from, to, class, name); every road is two-way
ROADS = [
    # the expressways
    ("st_laurent", "decarie", "autoroute", "A-40"),
    ("decarie", "villeray", "autoroute", "A-40 Metropolitan"),
    ("villeray", "st_michel", "autoroute", "A-40 Metropolitan"),
    ("st_michel", "anjou", "autoroute", "A-40 Metropolitan"),
    ("dorval", "turcot", "autoroute", "A-20"),
    ("dorval", "st_laurent", "autoroute", "A-13"),
    ("turcot", "decarie", "autoroute", "A-15 Decarie"),
    ("turcot", "verdun", "autoroute", "A-15"),
    ("turcot", "downtown", "autoroute", "R-136 Ville-Marie"),
    ("downtown", "centre_sud", "autoroute", "R-136 Ville-Marie"),
    ("downtown", "verdun", "autoroute", "A-10 Bonaventure"),
    ("mercier", "anjou", "autoroute", "A-25"),
    ("brossard", "st_lambert", "autoroute", "R-132"),
    ("st_lambert", "longueuil", "autoroute", "R-132"),
    ("laval_w", "laval_c", "autoroute", "A-440"),
    # the crossings
    ("verdun", "brossard", "bridge", "Champlain Bridge"),
    ("old_port", "st_lambert", "small_bridge", "Victoria Bridge"),
    ("centre_sud", "longueuil", "bridge", "Jacques-Cartier Bridge"),
    ("mercier", "longueuil", "bridge", "Lafontaine Tunnel"),
    ("st_laurent", "laval_w", "bridge", "A-13 bridge"),
    ("decarie", "laval_w", "bridge", "A-15 bridge"),
    ("ahuntsic", "laval_c", "small_bridge", "Viau Bridge"),
    ("mtl_nord", "laval_c", "small_bridge", "A-19 bridge"),
    # the arterials
    ("downtown", "mcgill", "arterial", "Peel"),
    ("downtown", "old_port", "arterial", "Rene-Levesque"),
    ("downtown", "cdn", "arterial", "Cote-des-Neiges"),
    ("mcgill", "berri", "arterial", "Sherbrooke"),
    ("mcgill", "plateau", "arterial", "Saint-Laurent"),
    ("old_port", "berri", "arterial", "Saint-Denis"),
    ("old_port", "centre_sud", "arterial", "Notre-Dame"),
    ("berri", "plateau", "arterial", "Saint-Denis"),
    ("berri", "centre_sud", "arterial", "Ontario"),
    ("plateau", "villeray", "arterial", "Saint-Denis"),
    ("plateau", "rosemont", "arterial", "Papineau"),
    ("plateau", "cdn", "arterial", "Cote-Sainte-Catherine"),
    ("cdn", "decarie", "arterial", "Jean-Talon W"),
    ("cdn", "turcot", "arterial", "Decarie Blvd"),
    ("dorval", "verdun", "arterial", "LaSalle Blvd"),
    ("decarie", "ahuntsic", "arterial", "Henri-Bourassa W"),
    ("st_laurent", "ahuntsic", "arterial", "Gouin"),
    ("villeray", "ahuntsic", "arterial", "Lajeunesse"),
    ("villeray", "rosemont", "arterial", "Beaubien"),
    ("ahuntsic", "mtl_nord", "arterial", "Henri-Bourassa E"),
    ("rosemont", "st_michel", "arterial", "Saint-Michel"),
    ("rosemont", "centre_sud", "arterial", "De Lorimier"),
    ("rosemont", "hochelaga", "arterial", "Rosemont Blvd"),
    ("st_michel", "hochelaga", "arterial", "Pie-IX"),
    ("st_michel", "mtl_nord", "arterial", "Pie-IX"),
    ("hochelaga", "centre_sud", "arterial", "Sainte-Catherine E"),
    ("hochelaga", "mercier", "arterial", "Sherbrooke E"),
    ("mtl_nord", "anjou", "arterial", "Henri-Bourassa E"),
]


def position(lat: float, lon: float) -> tuple:
    """Metres east and north of ``ORIGIN``."""
    return (lon - ORIGIN[1]) * M_PER_DEG_LON, (lat - ORIGIN[0]) * M_PER_DEG_LAT


@register_network("montreal")
def build_montreal(cfg) -> RoadNetwork:
    net = RoadNetwork()
    for node, (_short, name, lat, lon) in enumerate(NODES):
        net.add_node(node, *position(lat, lon))
        net.names[node] = name
    for a, b, kind, name in ROADS:
        u, v = NODE[a], NODE[b]
        speed, lanes = CLASSES[kind]
        net.add_road(u, v, CIRCUITY * net.euclid(u, v), speed, lanes)
        net.road_names[(u, v)] = net.road_names[(v, u)] = name
    net.check_connected()
    return net
