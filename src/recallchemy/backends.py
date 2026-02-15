from __future__ import annotations

"""Public backend API.

This module keeps import compatibility while backend implementations live in
smaller, focused modules.
"""

from .backends_impl import (
    AnnoyBackend,
    BACKENDS,
    FaissIVFBackend,
    HnswlibBackend,
    VectorBackend,
    _neighbor_choices,
    _pq_m_candidates,
    resolve_backends,
)

__all__ = [
    "AnnoyBackend",
    "BACKENDS",
    "FaissIVFBackend",
    "HnswlibBackend",
    "VectorBackend",
    "_neighbor_choices",
    "_pq_m_candidates",
    "resolve_backends",
]
