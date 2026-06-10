"""Torch 设备与随机性工具。"""

import os
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def pick_dev(dev_cfg: Any, model_key: str | None = None) -> "torch.device":
    """选择运行设备。"""
    if torch is None:
        raise RuntimeError("?????? PyTorch????? Torch ??")

    name = str(getattr(dev_cfg, "device", "auto")).strip().lower()
    if name == "auto":
        cpu_mods = tuple(str(v).strip().lower() for v in getattr(dev_cfg, "cpu_mods", ()))
        key = "" if model_key is None else str(model_key).strip().lower()
        if key in cpu_mods:
            name = "cpu"
        else:
            name = "cuda" if torch.cuda.is_available() else "cpu"

    if name.startswith("cuda") and not torch.cuda.is_available():
        name = "cpu"

    return torch.device(name)


def seed_torch(seed: int, dev: "torch.device", det: bool = True) -> None:
    """设置 Torch 随机种子，并尽量启用确定性训练。"""
    if torch is None:
        raise RuntimeError("当前环境缺少 PyTorch，无法设置 Torch 随机性")

    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    if det:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False


def to_f32(x: Any, dev: "torch.device") -> "torch.Tensor":
    """转成目标设备上的 float32 Tensor。"""
    if torch is None:
        raise RuntimeError("?????? PyTorch????? Tensor")

    return torch.as_tensor(x, dtype=torch.float32, device=dev)
