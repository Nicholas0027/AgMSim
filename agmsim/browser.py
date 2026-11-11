"""The web interface without a server: AgMSim running inside the visitor's browser.

In the online demo the simulator is loaded into the page with Pyodide (Python compiled to
WebAssembly). The page's requests go to :func:`request` instead of to ``agmsim serve``, and since a
browser tab has no threads to spare, the page's worker calls :func:`advance` again and again to move
the current run forward a little at a time, answering requests in between. Nothing leaves the
visitor's machine. No language model can be reached from there, so the demo offers the built-in
decision models and the offline stand-in only.
"""
from __future__ import annotations

import json
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .server import RUNS, route


def request(url: str, body: Optional[str] = None) -> str:
    """Answer one request of the page; ``body`` is the JSON text of a POST, None for a GET."""
    parsed = urlparse(url)
    _, out = route("GET" if body is None else "POST", parsed.path, parse_qs(parsed.query),
                   json.loads(body) if body else {}, online=True)
    return json.dumps(out, default=str)


def advance(seconds: float = 0.05) -> bool:
    """Work on the current run for about ``seconds``; True while there is more to do."""
    return any([run.advance(seconds) for run in list(RUNS.values())])
