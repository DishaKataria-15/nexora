"""Detect usable compute. A GPU option is offered only when a device AND a capable backend exist."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field


class ComputeError(RuntimeError):
    """The requested compute configuration is not available."""


@dataclass
class ComputeInfo:
    cpu_count: int
    gpu_devices: list[str] = field(default_factory=list)
    xgboost_installed: bool = False
    note: str = ""

    @property
    def gpu_backend_ready(self) -> bool:
        return bool(self.gpu_devices) and self.xgboost_installed


def detect_compute(timeout: float = 3.0) -> ComputeInfo:
    names: list[str] = []
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=timeout, check=False)
            names = [ln.split(":", 1)[1].split("(")[0].strip() for ln in out.stdout.splitlines() if ln.startswith("GPU")]
        except (OSError, subprocess.SubprocessError):
            names = []
    xgb = importlib.util.find_spec("xgboost") is not None
    if names and not xgb:
        note = "A GPU was detected, but none of Nexora's models can use it without a GPU-capable library."
    elif not names:
        note = "No compatible GPU was detected. CPU is used; it is fast enough for most tabular data."
    else:
        note = "A GPU was detected. Nexora's built-in models run on CPU."
    return ComputeInfo(os.cpu_count() or 1, names, xgb, note)


def device_options(info: ComputeInfo, gpu_capable_models: bool) -> list[str]:
    opts = ["auto", "cpu"]
    if info.gpu_backend_ready and gpu_capable_models:
        opts.append("gpu")
    return opts


def resolve_device(requested: str, gpu_capable: bool, info: ComputeInfo) -> tuple[str, str]:
    """Return (device actually used, plain-language note). Raises ComputeError for an impossible request."""
    if requested == "gpu":
        if not info.gpu_devices:
            raise ComputeError("GPU was requested but no compatible GPU was detected. Choose Automatic or CPU.")
        if not gpu_capable or not info.xgboost_installed:
            raise ComputeError("GPU was requested, but this model has no GPU backend. Choose Automatic or CPU.")
        return "gpu", f"Using GPU ({info.gpu_devices[0]})."
    if requested == "cpu":
        return "cpu", "Using CPU."
    if gpu_capable and info.gpu_backend_ready:
        return "gpu", f"Automatic: using GPU ({info.gpu_devices[0]})."
    return "cpu", "Automatic: using CPU (this model runs on CPU)."
