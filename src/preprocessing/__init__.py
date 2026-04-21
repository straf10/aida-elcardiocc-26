"""Load raw ELCardioCC splits, clean text, and write ``data/processed/`` for all model tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import io_utils
from .pipeline import run_preprocessing

if TYPE_CHECKING:
    from .dataset import ELCardioDataset as ELCardioDataset

__all__ = ["io_utils", "ELCardioDataset", "run_preprocessing"]


def __getattr__(name: str):
    if name == "ELCardioDataset":
        from .dataset import ELCardioDataset

        return ELCardioDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
