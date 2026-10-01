"""Hardware and Device Utilities."""

import torch


def get_optimal_device() -> torch.device:
    """Detect and return the fastest available PyTorch device.

    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs)
    3. CPU (Fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """Return a human-readable name for the device."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "Apple Silicon MPS (Metal Performance Shaders)"
    return "CPU"
