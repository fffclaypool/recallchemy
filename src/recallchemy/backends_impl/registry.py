from __future__ import annotations

from typing import Any

from .annoy import AnnoyBackend
from .base import VectorBackend
from .faiss_ivf import FaissIVFBackend
from .hnswlib import HnswlibBackend


BACKENDS: dict[str, type[VectorBackend]] = {
    HnswlibBackend.name: HnswlibBackend,
    AnnoyBackend.name: AnnoyBackend,
    FaissIVFBackend.name: FaissIVFBackend,
}


def resolve_backends(
    requested: list[str],
    backend_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[VectorBackend], dict[str, str]]:
    names = requested
    if "all" in names:
        names = list(BACKENDS.keys())

    backend_overrides = backend_overrides or {}
    selected: list[VectorBackend] = []
    skipped: dict[str, str] = {}
    for name in names:
        backend_cls = BACKENDS.get(name)
        if backend_cls is None:
            skipped[name] = "unknown backend name"
            continue
        ok, reason = backend_cls.availability()
        if not ok:
            skipped[name] = reason or "not available"
            continue
        selected.append(backend_cls(overrides=backend_overrides.get(name, {})))
    return selected, skipped


__all__ = ["BACKENDS", "resolve_backends"]
