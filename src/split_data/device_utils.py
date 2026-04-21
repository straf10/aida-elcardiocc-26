import torch

def get_device(explicit: str | None = None) -> torch.device:
    """
    Determine the best available device or use the explicit one.
    """
    if explicit is not None:
        return torch.device(explicit)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
        
    return torch.device("cpu")

def use_amp_fp16(device: torch.device, config_fp16: bool) -> bool:
    """
    Return True if AMP fp16 should be used.
    Only enable on CUDA devices to avoid CPU/MPS warnings or issues.
    """
    return config_fp16 and device.type == "cuda"
