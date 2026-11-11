"""The web interface without a server: AgMSim running inside the visitor's browser.

In the online demo the simulator is loaded into the page with Pyodide (Python compiled to
WebAssembly). The page's requests go to :func:`request` instead of to ``agmsim serve``, and since a
browser tab has no threads to spare, the page's worker calls :func:`advance` again and again to move
the current run forward a little at a time, answering requests in between.

A visitor who wants agents driven by a language model types their own API key into the page. The
key is handed to :func:`request` (``/api/key``) and kept in this tab's memory under the environment
variable the chosen provider reads; it is sent to that provider's address and nowhere else, and it
never becomes part of a configuration, a trace or anything shown on the page. Model calls are made
by the browser itself (see ``agmsim.llm._browser_post``), which works with providers that accept
requests from web pages.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .server import RUNS, route

GENERIC_KEY_ENV = "AGMSIM_API_KEY"       # for a server that is not one of the presets


def request(url: str, body: Optional[str] = None) -> str:
    """Answer one request of the page; ``body`` is the JSON text of a POST, None for a GET."""
    parsed = urlparse(url)
    data = json.loads(body) if body else {}
    if parsed.path == "/api/key" and body is not None:
        return json.dumps(_set_key(str(data.get("env") or GENERIC_KEY_ENV), str(data.get("key") or "")))
    _, out = route("GET" if body is None else "POST", parsed.path, parse_qs(parsed.query), data, online=True)
    return json.dumps(out, default=str)


def _set_key(env: str, key: str) -> dict:
    """Keep the visitor's key in memory for the provider that reads ``env``; an empty key forgets it."""
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", env):
        return {"ok": False, "error": "not a valid variable name"}
    if key.strip():
        os.environ[env] = key.strip()
    else:
        os.environ.pop(env, None)
    return {"ok": True}


def advance(seconds: float = 0.05) -> bool:
    """Work on the current run for about ``seconds``; True while there is more to do."""
    return any([run.advance(seconds) for run in list(RUNS.values())])
