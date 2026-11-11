"""Plugin registries.

Every swappable component of AgMSim (decision policies, LLM backends,
interaction mechanisms, matchers, network builders, demand generators,
movement models, metrics) is looked up by name in a :class:`Registry`.
Third-party code registers components with the decorators exported from
``agmsim`` and is loaded through ``plugins:`` in a config file (a module path
or a ``.py`` file) or through the ``agmsim.plugins`` entry-point group.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable


class Registry:
    """A name -> object table with helpful error messages."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str, obj: Any = None):
        """Register ``obj`` under ``name``; usable directly or as a decorator."""

        def deco(o: Any) -> Any:
            self._items[name] = o
            return o

        return deco(obj) if obj is not None else deco

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "(none)"
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {available}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


def load_plugins(specs: Iterable[str] | None, base_dir: str | Path | None = None) -> list[str]:
    """Import plugin modules so that their registrations take effect.

    Each spec is either a dotted module path (``my_pkg.my_agents``) or a path
    to a ``.py`` file (resolved against ``base_dir``, normally the directory of
    the config file).
    """
    loaded: list[str] = []
    for spec in specs or []:
        looks_like_file = spec.endswith(".py") or "/" in spec or "\\" in spec
        if looks_like_file:
            path = Path(spec)
            if not path.is_absolute():
                path = Path(base_dir or ".") / path
            path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"plugin file not found: {path}")
            modname = "agmsim_plugin_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
            module_spec = importlib.util.spec_from_file_location(modname, path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[modname] = module
            module_spec.loader.exec_module(module)  # re-executed on every load: registrations are idempotent
            loaded.append(str(path))
        else:
            importlib.import_module(spec)
            loaded.append(spec)
    return loaded


def load_entry_point_plugins() -> list[str]:
    """Load plugins advertised by installed packages (group ``agmsim.plugins``)."""
    from importlib.metadata import entry_points

    loaded = []
    for ep in entry_points(group="agmsim.plugins"):
        ep.load()
        loaded.append(ep.name)
    return loaded
