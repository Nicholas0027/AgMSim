"""Infrastructure layer: the road network, link conditions and routing.

The network is a directed graph. Each link has a length, a free-flow speed and
a dynamic *condition factor* (1.0 = nominal, 0.25 = incident slowing traffic to
a quarter of free-flow speed). Two travel-time views are offered everywhere:

* ``dynamic=False`` -- free-flow times (what a static navigation system knows);
* ``dynamic=True``  -- times under the current condition factors.

Which view a given agent may use is an *information* question decided by its
observation scope, not by the network.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .registry import Registry

NETWORKS = Registry("network builder")

#: Lowest condition factor. A "closed" link is modelled as a crawl so that
#: vehicles already on it can always leave and routing never meets infinity.
MIN_FACTOR = 0.02


def register_network(name: str):
    """Register a network builder ``(NetworkCfg) -> RoadNetwork``."""
    return NETWORKS.register(name)


@dataclass
class Link:
    id: int
    u: int
    v: int
    length: float           # metres
    speed: float            # free-flow speed, m/s
    lanes: int = 1
    factor: float = 1.0     # dynamic condition factor (incidents)
    flow: float = 1.0       # congestion factor from the vehicles currently on the link

    @property
    def free_time(self) -> float:
        return self.length / self.speed

    @property
    def cur_speed(self) -> float:
        return self.speed * max(self.factor, MIN_FACTOR) * self.flow

    @property
    def cur_time(self) -> float:
        return self.length / self.cur_speed


@dataclass(frozen=True)
class Route:
    time: float                 # seconds, under the weights used to compute it
    length: float               # metres
    links: tuple[int, ...]      # link ids in travel order


def free_flow_weight(link: Link) -> float:
    return link.free_time


def current_weight(link: Link) -> float:
    return link.cur_time


class RoadNetwork:
    def __init__(self) -> None:
        self.nodes: dict[int, tuple[float, float]] = {}
        self.links: list[Link] = []
        self.out: dict[int, list[int]] = {}
        self.version = 0                      # bumped whenever a condition factor changes
        self._static_trees: dict[int, tuple[dict, dict]] = {}
        self._dynamic_trees: dict[int, tuple[dict, dict]] = {}
        self._lengths: dict[tuple[int, int], float] = {}
        self._pairs: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ construction

    def add_node(self, node: int, x: float, y: float) -> None:
        self.nodes[int(node)] = (float(x), float(y))
        self.out.setdefault(int(node), [])

    def add_link(self, u: int, v: int, length: Optional[float] = None, speed: float = 10.0, lanes: int = 1) -> Link:
        if length is None:
            length = self.euclid(u, v)
        link = Link(len(self.links), int(u), int(v), float(length), float(speed), int(lanes))
        self.links.append(link)
        self.out[link.u].append(link.id)
        self._pairs[(link.u, link.v)] = link.id
        return link

    def add_road(self, u: int, v: int, length: Optional[float] = None, speed: float = 10.0, lanes: int = 1) -> None:
        """Add a two-way road as a pair of directed links."""
        self.add_link(u, v, length, speed, lanes)
        self.add_link(v, u, length, speed, lanes)

    def check_connected(self) -> None:
        """Every node must be reachable from every other node."""
        if not self.nodes:
            raise ValueError("network has no nodes")
        start = next(iter(self.nodes))
        reverse: dict[int, list[int]] = {n: [] for n in self.nodes}
        for link in self.links:
            reverse[link.v].append(link.u)

        def reach(neighbours: Callable[[int], list[int]]) -> set[int]:
            seen, stack = {start}, [start]
            while stack:
                for nxt in neighbours(stack.pop()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen

        forward = reach(lambda n: [self.links[i].v for i in self.out[n]])
        backward = reach(lambda n: reverse[n])
        if len(forward) != len(self.nodes) or len(backward) != len(self.nodes):
            raise ValueError("network is not strongly connected")

    # ------------------------------------------------------------------ geometry

    def xy(self, node: int) -> tuple[float, float]:
        return self.nodes[node]

    def euclid(self, a: int, b: int) -> float:
        (ax, ay), (bx, by) = self.nodes[a], self.nodes[b]
        return math.hypot(ax - bx, ay - by)

    def position(self, link_id: int, offset: float) -> tuple[float, float]:
        link = self.links[link_id]
        (ux, uy), (vx, vy) = self.nodes[link.u], self.nodes[link.v]
        frac = 0.0 if link.length <= 0 else min(1.0, max(0.0, offset / link.length))
        return ux + (vx - ux) * frac, uy + (vy - uy) * frac

    def link_between(self, u: int, v: int) -> Optional[Link]:
        lid = self._pairs.get((u, v))
        return None if lid is None else self.links[lid]

    # ------------------------------------------------------------------ conditions

    def set_factor(self, link_id: int, factor: float) -> bool:
        """Change a link's condition factor; returns True if it actually changed."""
        link = self.links[link_id]
        factor = max(float(factor), MIN_FACTOR)
        if abs(link.factor - factor) < 1e-12:
            return False
        link.factor = factor
        self.version += 1
        self._dynamic_trees.clear()
        return True

    def set_flows(self, counts: dict, spacing_m: float, scale: float, floor: float = 0.15) -> None:
        """Update congestion from the number of vehicles on each link (linear speed-density relation).

        A link holds ``length * lanes / spacing_m * scale`` vehicles at jam density; ``scale`` plays the
        role of a sample-size factor when each simulated car stands for several real ones.
        """
        changed = False
        for link in self.links:
            jam = max(1.0, link.length * link.lanes / spacing_m * scale)
            flow = max(floor, 1.0 - counts.get(link.id, 0) / jam)
            if abs(flow - link.flow) > 1e-9:
                link.flow, changed = flow, True
        if changed:
            self.version += 1
            self._dynamic_trees.clear()

    def disrupted(self) -> bool:
        return any(abs(link.factor - 1.0) > 1e-12 for link in self.links)

    # ------------------------------------------------------------------ routing

    def _dijkstra(
        self,
        src: int,
        weight: Callable[[Link], float],
        banned: frozenset = frozenset(),
        target: Optional[int] = None,
    ) -> tuple[dict[int, float], dict[int, int]]:
        """Shortest-path tree from ``src``. Ties are broken deterministically."""
        dist: dict[int, float] = {src: 0.0}
        pred: dict[int, int] = {}
        heap: list[tuple[float, int]] = [(0.0, src)]
        done: set[int] = set()
        while heap:
            d, node = heapq.heappop(heap)
            if node in done:
                continue
            done.add(node)
            if node == target:
                break
            for lid in self.out[node]:
                if lid in banned:
                    continue
                link = self.links[lid]
                nd = d + weight(link)
                if nd < dist.get(link.v, math.inf) - 1e-9:
                    dist[link.v] = nd
                    pred[link.v] = lid
                    heapq.heappush(heap, (nd, link.v))
        return dist, pred

    def _tree(self, src: int, dynamic: bool) -> tuple[dict[int, float], dict[int, int]]:
        cache = self._dynamic_trees if dynamic else self._static_trees
        tree = cache.get(src)
        if tree is None:
            tree = self._dijkstra(src, current_weight if dynamic else free_flow_weight)
            cache[src] = tree
        return tree

    def _unwind(self, pred: dict[int, int], src: int, dst: int) -> list[int]:
        links: list[int] = []
        node = dst
        while node != src:
            lid = pred[node]
            links.append(lid)
            node = self.links[lid].u
        links.reverse()
        return links

    def time(self, src: int, dst: int, dynamic: bool = False) -> float:
        """Travel time in seconds along the fastest route (inf if unreachable)."""
        if src == dst:
            return 0.0
        return self._tree(src, dynamic)[0].get(dst, math.inf)

    def route(self, src: int, dst: int, dynamic: bool = False) -> Route:
        if src == dst:
            return Route(0.0, 0.0, ())
        dist, pred = self._tree(src, dynamic)
        if dst not in dist:
            raise ValueError(f"no route from node {src} to node {dst}")
        links = self._unwind(pred, src, dst)
        return Route(dist[dst], self.route_length(links), tuple(links))

    def length(self, src: int, dst: int) -> float:
        """Network distance in metres along the free-flow fastest route."""
        key = (src, dst)
        value = self._lengths.get(key)
        if value is None:
            value = self.route(src, dst).length
            self._lengths[key] = value
        return value

    def route_length(self, links) -> float:
        return sum(self.links[i].length for i in links)

    def route_time(self, links, weight: Callable[[Link], float] = free_flow_weight) -> float:
        return sum(weight(self.links[i]) for i in links)

    def fastest(self, src: int, dst: int, weight: Callable[[Link], float] = free_flow_weight,
                banned: frozenset = frozenset()) -> Optional[Route]:
        """Fastest route under ``weight`` that avoids the ``banned`` links, or None."""
        if src == dst:
            return Route(0.0, 0.0, ())
        dist, pred = self._dijkstra(src, weight, banned=banned, target=dst)
        if dst not in dist:
            return None
        links = self._unwind(pred, src, dst)
        return Route(dist[dst], self.route_length(links), tuple(links))

    def alternatives(
        self,
        src: int,
        dst: int,
        k: int = 3,
        weight: Callable[[Link], float] = free_flow_weight,
        first: Optional[tuple[int, ...]] = None,
    ) -> list[Route]:
        """Up to ``k`` distinct routes.

        Candidates are the fastest route under ``weight`` plus, for each of its
        links, the fastest route that avoids that link. If ``first`` (a link
        sequence) is given it is always returned at index 0, so that callers can
        present "the default route" followed by alternatives.
        """
        if src == dst:
            return [Route(0.0, 0.0, ())]

        def run(banned: frozenset) -> Optional[Route]:
            dist, pred = self._dijkstra(src, weight, banned=banned, target=dst)
            if dst not in dist:
                return None
            links = self._unwind(pred, src, dst)
            return Route(dist[dst], self.route_length(links), tuple(links))

        best = run(frozenset())
        if best is None:
            return []
        found: dict[tuple[int, ...], Route] = {best.links: best}
        for lid in best.links:
            alt = run(frozenset([lid]))
            if alt is not None and alt.links not in found:
                found[alt.links] = alt
        if first is not None:
            first = tuple(first)
            head = Route(self.route_time(first, weight), self.route_length(first), first)
        else:
            head = best
        rest = sorted((r for r in found.values() if r.links != head.links), key=lambda r: (r.time, r.links))
        return [head] + rest[: max(0, k - 1)]

    # ------------------------------------------------------------------ export

    def to_dict(self) -> dict:
        return {
            "nodes": [[n, round(x, 2), round(y, 2)] for n, (x, y) in sorted(self.nodes.items())],
            "links": [[l.id, l.u, l.v, round(l.length, 2), round(l.speed, 3), l.lanes] for l in self.links],
        }


# --------------------------------------------------------------------------- builders


@register_network("grid")
def build_grid(cfg) -> RoadNetwork:
    """Rectangular grid of two-way streets.

    ``params``: ``arterial_every`` (int, every k-th row/column is an arterial),
    ``arterial_speed_mps`` (default 1.5x the street speed).
    """
    params = cfg.params or {}
    rows, cols, block, speed = int(cfg.rows), int(cfg.cols), float(cfg.block_m), float(cfg.speed_mps)
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError("grid network needs at least two nodes")
    every = int(params.get("arterial_every", 0))
    fast = float(params.get("arterial_speed_mps", speed * 1.5))
    net = RoadNetwork()
    for r in range(rows):
        for c in range(cols):
            net.add_node(r * cols + c, c * block, r * block)
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if c + 1 < cols:
                net.add_road(node, node + 1, block, fast if every and r % every == 0 else speed)
            if r + 1 < rows:
                net.add_road(node, node + cols, block, fast if every and c % every == 0 else speed)
    net.check_connected()
    return net


@register_network("bridge")
def build_bridge(cfg) -> RoadNetwork:
    """Two grids (west and east bank) joined by a few bridges.

    ``params``: ``gap_m`` (river width, default 3 blocks), ``bridge_rows``
    (rows that carry a bridge, default first and last), ``bridge_speed_mps``.
    """
    params = cfg.params or {}
    rows, cols, block, speed = int(cfg.rows), int(cfg.cols), float(cfg.block_m), float(cfg.speed_mps)
    gap = float(params.get("gap_m", 3 * block))
    bridge_rows = list(params.get("bridge_rows", sorted({0, rows - 1})))
    bridge_speed = float(params.get("bridge_speed_mps", speed))
    net = RoadNetwork()
    per_bank = rows * cols
    for bank in range(2):
        x0 = bank * ((cols - 1) * block + gap)
        for r in range(rows):
            for c in range(cols):
                net.add_node(bank * per_bank + r * cols + c, x0 + c * block, r * block)
        for r in range(rows):
            for c in range(cols):
                node = bank * per_bank + r * cols + c
                if c + 1 < cols:
                    net.add_road(node, node + 1, block, speed)
                if r + 1 < rows:
                    net.add_road(node, node + cols, block, speed)
    for r in bridge_rows:
        if not 0 <= int(r) < rows:
            raise ValueError(f"bridge row {r} is outside the grid")
        west = int(r) * cols + (cols - 1)
        east = per_bank + int(r) * cols
        net.add_road(west, east, gap, bridge_speed)
    net.check_connected()
    return net


@register_network("file")
def build_from_file(cfg) -> RoadNetwork:
    """Load a network from JSON.

    ``params.path`` points to ``{"nodes": [{"id", "x", "y"}], "links": [{"u",
    "v", "length"?, "speed"?, "lanes"?, "oneway"?}]}`` (metres, m/s).
    """
    params = cfg.params or {}
    if "path" not in params:
        raise ValueError("network.type=file requires network.params.path")
    path = Path(params["path"])
    if not path.is_absolute() and getattr(cfg, "base_dir", None):
        path = Path(cfg.base_dir) / path
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    net = RoadNetwork()
    for node in data["nodes"]:
        net.add_node(node["id"], node["x"], node["y"])
    for link in data["links"]:
        kwargs = dict(length=link.get("length"), speed=link.get("speed", cfg.speed_mps), lanes=link.get("lanes", 1))
        if link.get("oneway"):
            net.add_link(link["u"], link["v"], **kwargs)
        else:
            net.add_road(link["u"], link["v"], **kwargs)
    net.check_connected()
    return net


def build_network(cfg, base_dir: Optional[str] = None) -> RoadNetwork:
    """Build the network named by ``cfg.type``; ``base_dir`` anchors relative file paths."""
    cfg.base_dir = base_dir
    return NETWORKS.get(cfg.type)(cfg)
