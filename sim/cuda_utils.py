"""仿真阶段的 CUDA 工具。"""

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def pick_sim_dev(cfg: Any) -> "torch.device | None":
    """根据总配置选择仿真设备。"""
    if torch is None:
        return None

    name = str(getattr(getattr(cfg, "dev", None), "device", "auto")).strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    if not name.startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.device(name)


def delay_interp_c(sig: Any, tau: Any, dt: float, dev: "torch.device") -> "torch.Tensor":
    """对均匀采样复信号做批量可变时延插值。"""
    if torch is None:
        raise RuntimeError("当前环境缺少 PyTorch，无法执行 CUDA 插值")

    s = torch.as_tensor(sig, dtype=torch.complex64, device=dev)
    tau_t = torch.as_tensor(tau, dtype=torch.float32, device=dev)
    n_t = int(s.numel())
    base = torch.arange(tau_t.shape[-1], dtype=torch.float32, device=dev)
    idx_f = base.view((1,) * (tau_t.ndim - 1) + (tau_t.shape[-1],)) - tau_t / float(dt)

    valid = (idx_f >= 0.0) & (idx_f <= float(n_t - 1))
    idx0 = torch.floor(idx_f).to(torch.long)
    idx1 = idx0 + 1
    idx0 = torch.clamp(idx0, 0, n_t - 1)
    idx1 = torch.clamp(idx1, 0, n_t - 1)

    frac = torch.clamp(idx_f - idx0.to(idx_f.dtype), 0.0, 1.0)
    y0 = s[idx0]
    y1 = s[idx1]
    out = y0 * (1.0 - frac) + y1 * frac
    return torch.where(valid, out, torch.zeros_like(out))
