# source/core/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import threading
import logging

log = logging.getLogger(__name__)
# Do not configure global logging here; let main/app own it.
if not log.handlers:
    log.addHandler(logging.NullHandler())

Factory = Callable[..., Any]


@dataclass
class _Provider:
    kind: str
    name: str
    factory: Factory
    singleton: bool = True
    instance: Any = None


class ServiceRegistry:
    """
    Ultra-small DI registry keyed by (kind, name).

    - kind: category (e.g., "vector", "llm", "embeddings", "parser")
    - name: provider (e.g., "milvus", "weaviate", "azure", "nvidia")

    Features:
    - Idempotent registration: re-registering the same provider is a no-op (unless replace=True).
    - Thread-safe: internal RLock guards all mutations.
    - Singleton or factory mode per provider.
    - Defaults per kind for easy resolution by callers.
    """
    def __init__(self) -> None:
        self._providers: Dict[str, Dict[str, _Provider]] = {}
        self._defaults: Dict[str, str] = {}
        self._lock = threading.RLock()

    # ---- register / resolve ----

    def register(
        self,
        kind: str,
        name: str,
        factory: Factory,
        *,
        singleton: bool = True,
        replace: bool = False,
    ) -> None:
        """
        Register a provider. If the same (kind,name) already exists:
          - replace=False  -> no-op (quiet), logs at DEBUG
          - replace=True   -> replace factory and singleton flag; resets instance unless unchanged
        """
        with self._lock:
            bucket = self._providers.setdefault(kind, {})
            if name in bucket:
                if not replace:
                    log.debug("Provider already registered: %s:%s; skipping", kind, name)
                    return
                old = bucket[name]
                # If factory & singleton unchanged, keep existing instance; otherwise reset.
                keep_instance = (old.factory is factory) and (old.singleton == singleton)
                bucket[name] = _Provider(kind, name, factory, singleton, instance=(old.instance if keep_instance else None))
                log.info("Replaced provider %s:%s (singleton=%s, kept_instance=%s)", kind, name, singleton, keep_instance)
                return

            bucket[name] = _Provider(kind, name, factory, singleton)
            log.info("Registered provider %s:%s (singleton=%s)", kind, name, singleton)

    def set_default(self, kind: str, name: str) -> None:
        """
        Set the default provider name for a kind. Must exist.
        Logs at INFO only if it changes; otherwise DEBUG.
        """
        with self._lock:
            if kind not in self._providers or name not in self._providers[kind]:
                raise KeyError(f"No such provider to set default: {kind}:{name}")
            prev = self._defaults.get(kind)
            self._defaults[kind] = name
            if prev != name:
                log.info("Default for %s set to '%s'", kind, name)
            else:
                log.debug("Default for %s already '%s'; no change", kind, name)

    def get_default_name(self, kind: str) -> Optional[str]:
        return self._defaults.get(kind)

    def create(self, kind: str, name: Optional[str] = None, **kwargs) -> Any:
        """
        Create or get an instance. For singletons, caches the first created instance.
        Extra kwargs are only used at first creation for singletons.
        """
        with self._lock:
            prov = self._resolve_provider(kind, name)
            if prov.singleton:
                if prov.instance is None:
                    prov.instance = prov.factory(**kwargs)
                return prov.instance
            return prov.factory(**kwargs)

    def get(self, kind: str, name: Optional[str] = None) -> Any:
        return self.create(kind, name)

    # ---- utilities ----

    def has(self, kind: str, name: str) -> bool:
        return kind in self._providers and name in self._providers[kind]

    def list_kinds(self):
        return sorted(self._providers.keys())

    def list_names(self, kind: str):
        return sorted(self._providers.get(kind, {}).keys())

    def clear_instances(self, kind: Optional[str] = None) -> None:
        """
        Clears instantiated singleton objects (factories unaffected).
        Useful for tests or controlled reinit.
        """
        with self._lock:
            if kind:
                for p in self._providers.get(kind, {}).values():
                    p.instance = None
            else:
                for bucket in self._providers.values():
                    for p in bucket.values():
                        p.instance = None

    def clear_all(self) -> None:
        """Clears providers, defaults, and instances entirely."""
        with self._lock:
            self._providers.clear()
            self._defaults.clear()

    # ---- internal ----

    def _resolve_provider(self, kind: str, name: Optional[str]) -> _Provider:
        bucket = self._providers.get(kind, {})
        chosen = name or self._defaults.get(kind)
        if chosen and chosen in bucket:
            return bucket[chosen]
        if chosen is None and bucket:
            # If no default set, pick the first registered (stable across run)
            return next(iter(bucket.values()))
        raise KeyError(f"No provider registered for kind='{kind}' name='{name}'")


# Global singleton
registry = ServiceRegistry()

# Convenience functions
def register(kind: str, name: str, factory: Factory, *, singleton: bool = True, replace: bool = False) -> None:
    registry.register(kind, name, factory, singleton=singleton, replace=replace)

def set_default(kind: str, name: str) -> None:
    registry.set_default(kind, name)

def resolve(kind: str, name: Optional[str] = None) -> Any:
    return registry.get(kind, name)
