"""Device and mixed-precision selection for the XLM-R train/predict entrypoints."""

from __future__ import annotations

import torch


def mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_built() and mps.is_available()


def get_device(explicit: str | None = None) -> torch.device:
    """Prefer CUDA, then Apple MPS (Metal), then CPU."""
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_cuda_amp_fp16(device: torch.device, config_fp16: bool) -> bool:
    """Whether to use CUDA GradScaler + the CUDA autocast training path."""
    return config_fp16 and device.type == "cuda"


def use_autocast_fp16(device: torch.device, config_fp16: bool) -> bool:
    """fp16 autocast on CUDA or MPS (Metal); disabled on CPU."""
    if not config_fp16:
        return False
    if device.type == "cuda":
        return True
    if device.type == "mps":
        return mps_available()
    return False


def autocast_device_type(device: torch.device) -> str:
    """Device type string for torch.amp.autocast."""
    if device.type in ("cuda", "mps"):
        return device.type
    return "cpu"


def seed_for_device(device: torch.device, seed: int) -> None:
    """Best-effort RNG seeding for the active device (CUDA / MPS)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps" and mps_available():
        mps_mod = getattr(torch, "mps", None)
        if mps_mod is not None and hasattr(mps_mod, "manual_seed"):
            mps_mod.manual_seed(seed)
