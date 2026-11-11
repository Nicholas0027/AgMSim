"""Remote decision model: delegate each decision to an HTTP service.

This lets an agent built with any framework or language take part in a
simulation. The service receives ``{"observation": {...}}`` (the JSON form of
:class:`~agmsim.observation.Observation`) and replies with a decision object
``{"action": ..., "args": {...}, "rationale": "..."}``.

Config::

    machine_agents:
      non_llm_policy: remote
      policy_params: {url: "http://localhost:9000/decide", timeout_seconds: 30}
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import Policy, register_policy


@register_policy("remote", role="any")
class RemotePolicy(Policy):
    concurrent = True

    def __init__(self, url: str, timeout_seconds: float = 30.0, headers: dict | None = None, **params):
        super().__init__(**params)
        self.url = url
        self.timeout = float(timeout_seconds)
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def decide(self, obs):
        body = json.dumps({"observation": obs.to_dict()}).encode()
        request = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"remote policy at {self.url} failed for {obs.agent_id}: {exc}") from exc
