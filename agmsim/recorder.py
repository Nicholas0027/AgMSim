"""Run records.

Every run produces an append-only trace (``trace.jsonl``) with one JSON object
per line:

* ``header``   -- resolved config, code fingerprint, network, agent kinds, mechanism rules;
* ``decision`` -- who was asked what, what they saw, what they chose, whether it was valid;
* ``event``    -- what happened in the world;
* ``state``    -- fingerprint of the full state after a tick;
* ``frame``    -- positions for visualisation;
* ``summary``  -- metrics and the run's digests.

Together with the config this is enough to explain a run and to replay it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


def code_digest() -> str:
    """Fingerprint of the simulator's own source files."""
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def environment() -> dict:
    from . import __version__
    return {"agmsim": __version__, "python": sys.version.split()[0], "platform": platform.platform()}


class Recorder:
    """Collects trace records in memory and, if a directory is configured, on disk."""

    def __init__(self, cfg=None, directory: Optional[str] = None, keep: bool = True):
        self.keep = keep
        self.records: list[dict] = []
        self.run_dir: Optional[Path] = None
        self._fh = None
        self._cfg = cfg
        base = directory if directory is not None else (cfg.record.dir if cfg is not None else None)
        if base:
            name = (cfg.record.run_name if cfg is not None and cfg.record.run_name
                    else f"{cfg.name if cfg is not None else 'run'}-{time.strftime('%Y%m%d-%H%M%S')}")
            base_path = Path(base)
            self.run_dir = base_path / name
            suffix = 1
            while self.run_dir.exists():
                suffix += 1
                self.run_dir = base_path / f"{name}-{suffix}"
            self.run_dir.mkdir(parents=True)
            self._fh = open(self.run_dir / "trace.jsonl", "w", encoding="utf-8")
            if cfg is not None:
                with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as fh:
                    yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)

    @property
    def active(self) -> bool:
        """False when nothing consumes the records: not kept, not written, and ``write`` not overridden."""
        return self.keep or self._fh is not None or type(self).write is not Recorder.write

    def write(self, record: dict) -> None:
        if self.keep:
            self.records.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")

    def close(self, metrics: Optional[dict] = None, digests: Optional[dict] = None) -> Optional[str]:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.run_dir is None:
            return None
        if metrics is not None:
            with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as fh:
                json.dump({"metrics": metrics, **(digests or {}), "environment": environment(),
                           "code_digest": code_digest()}, fh, indent=2, ensure_ascii=False)
        return str(self.run_dir)


def read_trace(path: "str | Path") -> list[dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "trace.jsonl"
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
