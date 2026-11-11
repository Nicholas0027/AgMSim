"""LLM backends.

A backend turns a (system, user) prompt pair into text. Everything around it
is shared by all backends: usage accounting (calls, tokens, latency, cost),
hard budgets, and a *cassette* that records every response so that a run can
later be re-executed deterministically without calling the model again.

Built-in backends:

* ``openai_compatible`` -- any server speaking the Chat Completions protocol
  (local Qwen through mlx-lm, Ollama or vLLM; hosted providers such as DeepSeek).
  Connections are kept open between calls, one per worker thread;
* ``mock`` -- an offline stand-in for tests and demos. It is **not** a language
  model: it answers with a heuristic policy's choice, formatted as model text,
  so that the whole LLM decision path can be exercised without a model.

Add a backend by subclassing :class:`LLMClient` and decorating it with
``@register_llm("name")``.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import Registry

LLM_BACKENDS = Registry("LLM backend")


def register_llm(name: str):
    return LLM_BACKENDS.register(name)


class LLMError(RuntimeError):
    """Misconfiguration or a non-retryable failure: the run should stop."""


class LLMTransientError(LLMError):
    """A failure that survived retries (overload, network): the decision falls back."""


class LLMBudgetExceeded(LLMError):
    """The configured call or spend limit is reached: the decision falls back."""


def load_env_file(start: "str | Path | None" = None) -> Optional[Path]:
    """Read ``KEY=VALUE`` lines from the nearest ``.env`` file into the environment.

    Looks in ``start`` (default: the working directory) and up to three parent folders. Variables
    that are already set are left alone, so an exported key always wins over the file.
    """
    here = Path(start or os.getcwd()).resolve()
    for folder in [here, *list(here.parents)[:3]]:
        path = folder / ".env"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
        return path
    return None


@dataclass
class LLMRequest:
    system: str
    user: str
    meta: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False              # served from the cassette, no model call
    cached_input_tokens: int = 0      # part of input_tokens the provider served from its prompt cache


class LLMClient:
    backend = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "replayed": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
                      "latency_ms": 0.0, "cost_usd": 0.0, "errors": 0}
        self.latencies: list[float] = []
        self._started = 0
        self._cassette: dict[str, dict] = {}
        self._cassette_path: Optional[Path] = Path(cfg.cassette) if cfg.cassette else None
        if self._cassette_path and self._cassette_path.exists() and cfg.cassette_mode != "record":
            with open(self._cassette_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self._cassette[rec["key"]] = rec

    # ------------------------------------------------------------------ public API

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = self._key(request)
        if self._cassette_path is not None and self.cfg.cassette_mode != "record":
            rec = self._cassette.get(key)
            if rec is not None:
                with self._lock:
                    self.usage["replayed"] += 1
                return LLMResponse(rec["text"], rec.get("model", ""), rec.get("input_tokens", 0),
                                   rec.get("output_tokens", 0), 0.0, cached=True)
            if self.cfg.cassette_mode == "replay":
                raise LLMError("cassette_mode is 'replay' but this prompt is not in the cassette "
                               "(the run has diverged from the recording)")
        with self._lock:
            if self.cfg.max_calls is not None and self._started >= self.cfg.max_calls:
                raise LLMBudgetExceeded(f"max_calls={self.cfg.max_calls} reached")
            if self.cfg.budget_usd is not None and self.usage["cost_usd"] >= self.cfg.budget_usd:
                raise LLMBudgetExceeded(f"budget_usd={self.cfg.budget_usd} reached")
            self._started += 1
        started = time.perf_counter()
        try:
            response = self._complete(request)
        except LLMError:
            with self._lock:
                self.usage["errors"] += 1
            raise
        response.latency_ms = (time.perf_counter() - started) * 1000.0
        cost = self.cost(response)
        with self._lock:
            u = self.usage
            u["calls"] += 1
            u["input_tokens"] += response.input_tokens
            u["cached_input_tokens"] += response.cached_input_tokens
            u["output_tokens"] += response.output_tokens
            u["latency_ms"] += response.latency_ms
            u["cost_usd"] += cost
            self.latencies.append(response.latency_ms)
            if self._cassette_path is not None:
                rec = {"key": key, "text": response.text, "model": response.model,
                       "input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
                self._cassette[key] = rec
                self._cassette_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cassette_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return response

    def cost(self, response: LLMResponse) -> float:
        """Estimated spend in USD from ``llm.pricing`` (per million tokens); 0 if no prices are configured."""
        pricing = self.cfg.pricing or {}
        fresh = float(pricing.get("input", 0.0))
        hit = min(response.cached_input_tokens, response.input_tokens)
        return ((response.input_tokens - hit) * fresh + hit * float(pricing.get("cached_input", fresh))
                + response.output_tokens * float(pricing.get("output", 0.0))) / 1e6

    def summary(self) -> dict:
        u = dict(self.usage)
        lat = sorted(self.latencies)
        u["backend"], u["model"] = self.backend, self.cfg.model or self.backend
        u["latency_ms_mean"] = round(u.pop("latency_ms") / max(1, u["calls"]), 1)
        u["latency_ms_p50"] = round(lat[len(lat) // 2], 1) if lat else 0.0
        u["latency_ms_p95"] = round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else 0.0
        u["cost_usd"] = round(u["cost_usd"], 6)
        return u

    def close(self) -> None:
        """Release what the calling thread holds open (connections); the client stays usable."""

    # ------------------------------------------------------------------ for subclasses

    def _complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def _key(self, request: LLMRequest) -> str:
        blob = json.dumps([self.backend, self.cfg.model, self.cfg.temperature, request.system, request.user],
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]


# --------------------------------------------------------------------------- OpenAI-compatible servers


@register_llm("openai_compatible")
class OpenAICompatibleClient(LLMClient):
    """Chat Completions over HTTP: a local Qwen served by ``mlx_lm.server``, Ollama, vLLM, or a hosted provider.

    ``llm.params`` may contain ``extra_body`` (merged into the request body), ``user_suffix`` (appended
    to every user message, e.g. ``/no_think``) and ``keep_alive: false`` (open a new connection per call).
    """

    backend = "openai_compatible"

    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.base_url:
            raise LLMError("llm.base_url is required for the openai_compatible backend, e.g. http://localhost:8080/v1")
        if not cfg.model:
            raise LLMError("llm.model is required for the openai_compatible backend")
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env)
            if not key:                      # not exported: look for a .env file here, then next to the package
                load_env_file()
                load_env_file(Path(__file__).resolve().parent.parent)
                key = os.environ.get(cfg.api_key_env)
            if not key:
                raise LLMError(f"the API key is missing: put a line {cfg.api_key_env}=<your key> in a file named "
                               f".env in the project folder, or export {cfg.api_key_env} (llm.api_key_env)")
            self.headers["Authorization"] = f"Bearer {key}"
        parts = urllib.parse.urlsplit(self.url)
        self._scheme, self._host, self._port = parts.scheme, parts.hostname or "", parts.port
        self._path = parts.path + (f"?{parts.query}" if parts.query else "")
        self._local = threading.local()      # one open connection per calling thread
        proxied = bool(urllib.request.getproxies().get(parts.scheme)) and not urllib.request.proxy_bypass(self._host)
        self._keep_alive = bool(cfg.params.get("keep_alive", True)) and not proxied

    def _post(self, data: bytes) -> tuple[int, bytes]:
        """POST the body and return (status, raw response)."""
        if not self._keep_alive:             # also the path that honours HTTP(S)_PROXY
            req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read()
        conn = getattr(self._local, "conn", None)
        reused = conn is not None
        while True:
            if conn is None:
                factory = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
                conn = factory(self._host, self._port, timeout=self.cfg.timeout_seconds)
            try:
                conn.request("POST", self._path, body=data, headers=self.headers)
                resp = conn.getresponse()
                raw = resp.read()
                self._local.conn = conn
                return resp.status, raw
            except (http.client.HTTPException, OSError) as exc:
                conn.close()
                self._local.conn = conn = None
                if reused and not isinstance(exc, TimeoutError):
                    reused = False           # the server had closed an idle connection: once more on a fresh one
                    continue
                raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _complete(self, request: LLMRequest) -> LLMResponse:
        cfg = self.cfg
        user = request.user + (cfg.params.get("user_suffix") or "")
        body = {"model": cfg.model,
                "messages": [{"role": "system", "content": request.system}, {"role": "user", "content": user}]}
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.max_tokens is not None:
            body["max_tokens"] = cfg.max_tokens
        if cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(cfg.params.get("extra_body") or {})
        data = json.dumps(body).encode()
        last = ""
        for attempt in range(3):
            try:
                status, raw = self._post(data)
            except (http.client.HTTPException, OSError) as exc:
                last = f"cannot reach {self.url}: {exc}"
            else:
                if status == 200:
                    try:
                        payload = json.loads(raw.decode())
                        message = payload["choices"][0]["message"]
                        usage = payload.get("usage") or {}
                        hit = usage.get("prompt_cache_hit_tokens")
                        if hit is None:
                            hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                        return LLMResponse(message.get("content") or "", payload.get("model", cfg.model),
                                           int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                                           cached_input_tokens=int(hit))
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise LLMError(f"unexpected response from LLM server: {exc}") from exc
                detail = raw.decode(errors="replace")[:300]
                if status in (408, 409, 429) or status >= 500:
                    last = f"HTTP {status}: {detail}"
                else:
                    raise LLMError(f"LLM server rejected the request (HTTP {status}): {detail}")
            time.sleep(1.5 * (attempt + 1))
        raise LLMTransientError(last)


# --------------------------------------------------------------------------- offline stand-in


@register_llm("mock")
class MockLLM(LLMClient):
    """Offline stand-in for a language model (for tests, CI and demos without a model).

    It reads the observation out of the prompt, lets the kind's heuristic
    policy choose, and returns that choice as JSON text. With probability
    ``params.explore`` it picks a random permitted action instead, and with
    probability ``params.error_rate`` it returns prose instead of JSON, which
    exercises the retry and fallback path. Results obtained with this backend
    say nothing about how a real LLM behaves.
    """

    backend = "mock"
    HEURISTICS = {"traveler": "utility"}

    def _complete(self, request: LLMRequest) -> LLMResponse:
        from .policies.llm_policy import observation_from_prompt   # late import: avoids a cycle
        from .policies.base import PolicyContext, create_policy
        from .actions import as_decision
        from .observation import OBJECTIVES

        params = self.cfg.params
        seed = hashlib.sha256(f"{params.get('seed', 0)}|{request.user}".encode()).digest()
        rng = random.Random(int.from_bytes(seed[:8], "big"))
        tokens_in = (len(request.system) + len(request.user)) // 4
        if rng.random() < float(params.get("error_rate", 0.0)):
            text = "I would rather think about this a little longer before choosing."
            return LLMResponse(text, "mock", tokens_in, len(text) // 4)
        obs = observation_from_prompt(request.user)
        if rng.random() < float(params.get("explore", 0.05)):
            spec = obs.actions[rng.randrange(len(obs.actions))]
            args = {p.name: p.choices[rng.randrange(len(p.choices))] for p in spec.params if p.choices and p.required}
            decision = {"action": spec.name, "args": args, "rationale": "mock: exploratory choice"}
        else:
            policy = create_policy(obs.kind, self.HEURISTICS.get(obs.kind, "algorithm"), {})
            text_objective = obs.state.get("objective", "")
            objective = next((k for k, v in OBJECTIVES.items() if v == text_objective), "balanced")
            policy.bind(PolicyContext(obs.agent_id, obs.kind, rng, objective))
            chosen = as_decision(policy.decide(obs))
            decision = {"action": chosen.action, "args": chosen.args,
                        "rationale": "mock: " + (chosen.rationale or "heuristic choice")}
        text = json.dumps(decision)
        return LLMResponse(text, "mock", tokens_in, len(text) // 4)


def build_llm_client(cfg) -> LLMClient:
    if hasattr(cfg, "resolved"):
        cfg = cfg.resolved()             # apply llm.preset
    if not cfg.backend:
        raise LLMError("an agent has use_llm: true but neither llm.preset nor llm.backend is set. "
                       f"Backends: {', '.join(LLM_BACKENDS.names())}")
    return LLM_BACKENDS.get(cfg.backend)(cfg)
