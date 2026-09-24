"""Probe the runtime environment (Python / PyTorch / CUDA / device / AMP)."""

from __future__ import annotations

import platform
import sys
from typing import Any


def environment_info() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the current environment.

    Safe to call even if torch is missing or broken (fields fall back to None
    with an ``errors`` list explaining why).
    """
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "hostname": platform.node(),
        "torch_version": None,
        "torch_git_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "mps_available": False,
        "device_count": 0,
        "gpu_names": [],
        "bf16_supported": False,
        "errors": [],
    }

    try:
        import torch
    except Exception as exc:  # broken install, missing lib, etc.
        info["errors"].append(f"import torch failed: {exc}")
        return info

    info["torch_version"] = torch.__version__
    info["torch_git_version"] = getattr(torch.version, "git_version", None)

    try:
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            cudnn_v = torch.backends.cudnn.version()
            info["cudnn_version"] = str(cudnn_v) if cudnn_v else None
            try:
                info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                info["bf16_supported"] = False
    except Exception as exc:
        info["errors"].append(f"cuda probe failed: {exc}")

    try:
        if torch.backends.mps.is_available():
            info["mps_available"] = True
            if info["device_count"] == 0:
                info["device_count"] = 1
            if not info["gpu_names"]:
                info["gpu_names"] = [f"Apple MPS ({platform.processor() or 'arm64'})"]
    except Exception as exc:
        info["errors"].append(f"mps probe failed: {exc}")

    return info


if __name__ == "__main__":
    import json

    print(json.dumps(environment_info(), indent=2))
