"""Upstream repository handling: fetch, pin, attribute -- never vendor."""

from sdmbench.upstream.manager import UpstreamCheckout, UpstreamManager
from sdmbench.upstream.metadata import UPSTREAM_SOURCES, UpstreamSource, get_upstream

__all__ = [
    "UpstreamCheckout",
    "UpstreamManager",
    "UPSTREAM_SOURCES",
    "UpstreamSource",
    "get_upstream",
]
