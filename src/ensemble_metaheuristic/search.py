"""Backward-compatible re-export of weighted ensemble strategies."""
from __future__ import annotations

from .strategies.weighted_strategy import *  # noqa: F403
from .strategies.weighted_vns_strategy import run_vns_search  # noqa: F401
