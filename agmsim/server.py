"""Web interface: configure a run, watch it, inspect every decision.

``agmsim serve`` starts a small local HTTP server (standard library only).
Runs execute in a background thread; the page polls for new frames, so slow
LLM-driven runs can be watched while they progress.

The same page also works with no server at all: in the online demo AgMSim runs
inside the visitor's browser and :mod:`agmsim.browser` answers the page's
requests through :func:`route`, moving the run forward with :meth:`Run.advance`.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import kinds as _kinds, policies as _policies  # noqa: F401
from . import assistant
from .agency import LEVEL_NAMES
from .config import LLM_PRESETS, Config, list_scenarios
from .engine import Simulation
from .kinds.base import KINDS
from .llm import LLM_BACKENDS, load_env_file
from .policies.base import available_policies
from .recorder import Recorder

WEB = Path(__file__).parent / "web"
RUNS: dict[int, "Run"] = {}
_LOCK = threading.Lock()


class LiveRecorder(Recorder):
    """Keeps frames and decisions apart so the page can fetch them incrementally."""

    def __init__(self):
        super().__init__(None, keep=False)
        self.header, self.frames, self.decisions = None, [], []

    def write(self, record: dict) -> None:
        kind = record["type"]
        if kind == "frame":
            self.frames.append(record)
        elif kind == "decision":
            self.decisions.append(record)
        elif kind == "header":
            self.header = record


class Run:
    """One run started from the page: in a background thread, or advanced step by step by the caller."""

    def __init__(self, run_id: int, scenario: str, overrides: list, threaded: bool = True):
        self.id, self.status, self.error, self.metrics = run_id, "starting", None, None
        self.recorder = LiveRecorder()
        self.stop = False
        self.sim = None
        self._args = (scenario, overrides)
        self.thread = None
        if threaded:
            self.thread = threading.Thread(target=self._work, daemon=True)
            self.thread.start()

    def _work(self):
        while self.advance(3600.0):
            pass

    def advance(self, seconds: float) -> bool:
        """Work on the run for about ``seconds`` of wall-clock time; False once it is over."""
        if self.status in ("done", "stopped", "error"):
            return False
        try:
            if self.sim is None:
                scenario, overrides = self._args
                self.sim = Simulation(scenario, overrides=["record.dir=null", *overrides], recorder=self.recorder)
                self.status = "running"
                return True
            deadline = time.perf_counter() + seconds
            while not self.stop and self.sim.step():
                if time.perf_counter() >= deadline:
                    return True
            self.metrics = self.sim.finish().metrics
            self.status = "stopped" if self.stop else "done"
        except Exception as exc:        # shown to the user in the page
            self.error = f"{type(exc).__name__}: {exc.args[0] if exc.args else exc}"
            self.status = "error"
            traceback.print_exc()
        return False

    def view(self, frame_from: int, decision_from: int) -> dict:
        rec = self.recorder
        out = {"id": self.id, "status": self.status, "error": self.error, "metrics": self.metrics,
               "tick": self.sim.world.tick if self.sim else 0, "ticks": self.sim.n_ticks if self.sim else 0,
               "frames": rec.frames[frame_from:frame_from + 400],
               "decisions": [{k: v for k, v in d.items() if k != "obs"} | {"i": decision_from + i}
                             for i, d in enumerate(rec.decisions[decision_from:decision_from + 300])]}
        if frame_from == 0 and rec.header is not None:
            out["header"] = {k: rec.header[k] for k in ("network", "lines", "stations", "signals", "kinds", "config")}
        return out


def agent_story(run: "Run", agent_id: str) -> dict:
    """One agent's part in a run: every decision it took and, for a traveller, how its day went.

    ``timeline`` lists what the traveller did, in seconds since the start of the run: activities
    (``kind: activity``), trips (``travel``) and, for a single trip, the time before the need
    arose (``before``). Activities dropped from a day plan are listed under ``skipped``.
    """
    sim = run.sim
    if sim is None or agent_id not in sim.specs:
        return {}
    w, now = sim.world, sim.world.time_s
    out = {"id": agent_id, "kind": sim.specs[agent_id].kind, "level": sim.agency[agent_id].level,
           "policy": sim.specs[agent_id].policy, "now_s": now, "horizon_s": sim.n_ticks * sim.dt,
           "decisions": [{"i": i, "tick": d["tick"], "t": d["tick"] * sim.dt, "trigger": d["trigger"],
                          "action": d["decision"]["action"], "args": d["decision"].get("args") or {},
                          "source": d["source"], "exercised": bool(d.get("exercised"))}
                         for i, d in enumerate(run.recorder.decisions) if d["agent"] == agent_id]}
    tr = w.travelers.get(agent_id)
    if tr is None and agent_id in w.vehicles:
        out["owner"] = w.vehicles[agent_id].owner
    if agent_id in w.cabs:
        cab = w.cabs[agent_id]
        out["summary"] = {"rides": cab.trips, "earned_usd": round(cab.revenue, 2), "declined": cab.declined,
                          "km_with_passenger": round(cab.m_loaded / 1000.0, 1), "km_empty": round(cab.m_empty / 1000.0, 1)}
    if tr is None:
        return out
    out["car"] = tr.car
    items, skipped = [], []
    if tr.plan is not None:
        p, last = tr.plan, len(tr.plan.activities) - 1
        for k, a in enumerate(p.activities):
            if a.skipped:
                skipped.append(a.type)
            elif a.arrived_s is not None:
                end = a.left_s if a.left_s is not None else (out["horizon_s"] if k == last else now)
                item = {"kind": "activity", "label": a.type, "start": a.arrived_s, "end": end}
                if a.latest_start_s is not None:
                    item["must_start_by"] = a.latest_start_s
                if a.close_s is not None:
                    item["closes"] = a.close_s
                items.append(item)
        for leg in p.legs:
            items.append({"kind": "travel", "label": leg["mode"] or "travel", "start": leg["depart_s"], "end": leg["arrive_s"]})
        if p.leg is not None:
            items.append({"kind": "travel", "label": tr.mode or "travel", "start": p.leg["depart_s"], "end": now})
    else:
        need = tr.t_need * sim.dt
        items.append({"kind": "before", "label": "not travelling yet", "start": 0.0, "end": min(need, now)})
        if now > need:
            arrived = tr.t_arrival * sim.dt if tr.t_arrival is not None else None
            items.append({"kind": "travel", "label": tr.mode or tr.state.value, "start": need, "end": arrived or now})
            if arrived is not None:
                items.append({"kind": "activity", "label": "arrived", "start": arrived, "end": out["horizon_s"]})
    out["timeline"] = sorted(items, key=lambda item: item["start"])
    out["skipped"] = skipped
    return out


def meta(online: bool = False) -> dict:
    """What the page needs to build its form. ``online``: the browser demo, where a visitor types their own key."""
    kinds = {}
    for name in KINDS.names():
        kind = KINDS.get(name)
        kinds[name] = {"title": kind.title, "side": kind.side, "policies": available_policies(name),
                       "levels": {lvl: list(perms) for lvl, perms in sorted(kind.agency.levels.items())},
                       "permissions": kind.agency.permissions}
    scenarios = {}
    for name in list_scenarios():
        cfg = Config.load(name)
        scenarios[name] = {"description": cfg.description, "kinds": cfg.active_kinds(), "seed": cfg.simulation.seed,
                           "travelers": cfg.agents.traveler.count,
                           "agents": {k: {"use_llm": getattr(cfg.agents, k).use_llm,
                                          "policy": getattr(cfg.agents, k).non_llm_policy,
                                          "agency": getattr(cfg.agents, k).agency} for k in cfg.active_kinds()}}
    if not online:                      # so the page can tell which hosted models are ready to use on this machine
        load_env_file()
        load_env_file(Path(__file__).resolve().parent.parent)
    presets = {name: {"model": p.get("model", ""), "key_env": p.get("api_key_env"), "base_url": p.get("base_url", ""),
                      "has_key": bool(p.get("api_key_env") and os.environ.get(p["api_key_env"]))}
               for name, p in LLM_PRESETS.items() if name != "mock"}
    return {"scenarios": scenarios, "kinds": kinds, "level_names": LEVEL_NAMES, "llm_presets": presets,
            "llm_backends": LLM_BACKENDS.names(), "online": online}


def route(method: str, path: str, query: Optional[dict] = None, body: Optional[dict] = None, online: bool = False):
    """Answer one request of the page: ``(HTTP status, JSON-serialisable object)``.

    ``online`` is the browser demo: runs are not threaded (the caller advances them), model calls
    are made one at a time (a browser tab has no threads to spare) and only the latest run is kept.
    """
    parts = [p for p in path.split("/") if p]
    query = query or {}
    run = RUNS.get(int(parts[2])) if len(parts) >= 3 and parts[:2] == ["api", "run"] and parts[2].isdigit() else None
    if method == "GET":
        if parts == ["api", "meta"]:
            return 200, meta(online)
        if run is not None and len(parts) == 5 and parts[3] == "decision" and parts[4].isdigit():
            decisions, i = run.recorder.decisions, int(parts[4])
            return (200, decisions[i]) if i < len(decisions) else (404, {})
        if run is not None and len(parts) == 5 and parts[3] == "agent":
            story = agent_story(run, parts[4])
            return (200, story) if story else (404, {})
        if run is not None and len(parts) == 3:
            return 200, run.view(int(query.get("frames", ["0"])[0]), int(query.get("decisions", ["0"])[0]))
    elif parts == ["api", "run"]:
        body = body or {}
        with _LOCK:
            run_id = max(RUNS, default=0) + 1
            if online:
                RUNS.clear()
            overrides = list(body.get("overrides", [])) + (["llm.max_concurrency=1"] if online else [])
            RUNS[run_id] = Run(run_id, body.get("scenario", "demo_3x3"), overrides, threaded=not online)
        return 200, {"id": run_id}
    elif parts == ["api", "chat"]:               # the assistant: a request in words -> proposed settings
        return 200, assistant.configure(body or {})
    elif run is not None and len(parts) == 4 and parts[3] == "stop":
        run.stop = True
        return 200, {"ok": True}
    return 404, {"error": "not found"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, status=200, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.strip("/") == "":
            return self._send((WEB / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
        status, out = route("GET", url.path, parse_qs(url.query))
        self._send(out, status)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        status, out = route("POST", urlparse(self.path).path, body=json.loads(self.rfile.read(length) or b"{}"))
        self._send(out, status)

    def log_message(self, *args):
        pass


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"AgMSim interface: http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
