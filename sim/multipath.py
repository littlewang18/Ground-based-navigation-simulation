"""多径参数与多径信号生成。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import Cfg, MpCfg
from sim.cuda_utils import pick_sim_dev, delay_interp_c
from sim.geometry import GeoData
from sim.tx_signal import TxData

C0 = 299792458.0  # 光速 m/s


@dataclass
class MpData:
    """多径结果。"""

    t: np.ndarray  # 时间轴
    dly: np.ndarray  # 相对直达的多径时延 (n_bs, n_mp, n_t)
    amp: np.ndarray  # 多径幅度 (n_bs, n_mp, n_t)
    ph: np.ndarray  # 多径相位 (n_bs, n_mp, n_t)
    path: np.ndarray  # 分路径信号 (n_bs, n_mp, n_t)
    mix: np.ndarray  # 合成多径信号 (n_bs, n_t)


def _check_mp_cfg(cfg: MpCfg) -> None:
    """检查多径配置。"""
    if cfg.n <= 0:
        raise ValueError("mp.n 必须大于 0")
    if cfg.dly_min < 0.0:
        raise ValueError("mp.dly_min 不能小于 0")
    if cfg.dly_max <= cfg.dly_min:
        raise ValueError("mp.dly_max 必须大于 mp.dly_min")
    if cfg.k0 < 0.0:
        raise ValueError("mp.k0 不能小于 0")
    if cfg.decay < 0.0:
        raise ValueError("mp.decay 不能小于 0")
    if cfg.jit < 0.0:
        raise ValueError("mp.jit 不能小于 0")
    if cfg.fade < 0.0:
        raise ValueError("mp.fade 不能小于 0")
    if cfg.fade_f_min < 0.0 or cfg.fade_f_max < cfg.fade_f_min:
        raise ValueError("mp.fade_f_min / mp.fade_f_max 非法")
    if cfg.dly_wob < 0.0:
        raise ValueError("mp.dly_wob 不能小于 0")
    if cfg.ph_rate < 0.0:
        raise ValueError("mp.ph_rate 不能小于 0")
    if cfg.burst_n < 0:
        raise ValueError("mp.burst_n 不能小于 0")
    if cfg.burst_dur < 0.0:
        raise ValueError("mp.burst_dur 不能小于 0")
    if cfg.burst_gain < 1.0:
        raise ValueError("mp.burst_gain 不能小于 1")
    if cfg.ph_mode not in {"rand", "zero"}:
        raise ValueError("mp.ph_mode 仅支持 rand 或 zero")


def _delay_c(sig: np.ndarray, t: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """对复信号执行可变时延（线性插值）。"""
    q = t - tau
    re = np.interp(q, t, np.real(sig), left=0.0, right=0.0)
    im = np.interp(q, t, np.imag(sig), left=0.0, right=0.0)
    return re + 1j * im


def _gen_amp0(cfg: MpCfg, bs_n: int, rng: np.random.Generator) -> np.ndarray:
    """生成多径基准幅度矩阵。"""
    base = cfg.k0 * (cfg.decay ** np.arange(cfg.n, dtype=np.float64))
    amp = np.tile(base[None, :], (bs_n, 1))
    if cfg.jit > 0.0:
        jitter = rng.uniform(-cfg.jit, cfg.jit, size=(bs_n, cfg.n))
        amp = amp * (1.0 + jitter)
    return np.clip(amp, 0.0, None)


def _gen_ph0(cfg: MpCfg, bs_n: int, rng: np.random.Generator) -> np.ndarray:
    """生成多径基准相位矩阵。"""
    if cfg.ph_mode == "zero":
        return np.zeros((bs_n, cfg.n), dtype=np.float64)
    return rng.uniform(0.0, 2.0 * np.pi, size=(bs_n, cfg.n))


def _gen_time_field(cfg: MpCfg, t: np.ndarray, dly0: np.ndarray, amp0: np.ndarray, ph0: np.ndarray, rng: np.random.Generator):
    """生成时变多径参数场。"""
    bs_n, n_mp = dly0.shape
    n_t = t.size

    fade_f = rng.uniform(cfg.fade_f_min, cfg.fade_f_max, size=(bs_n, n_mp))
    fade_ph = rng.uniform(0.0, 2.0 * np.pi, size=(bs_n, n_mp))
    dly_f = rng.uniform(cfg.fade_f_min, cfg.fade_f_max, size=(bs_n, n_mp))
    dly_ph = rng.uniform(0.0, 2.0 * np.pi, size=(bs_n, n_mp))
    ph_rate = rng.uniform(-cfg.ph_rate, cfg.ph_rate, size=(bs_n, n_mp))

    tw = t[None, None, :]
    amp = amp0[:, :, None] * (1.0 + cfg.fade * np.sin(2.0 * np.pi * fade_f[:, :, None] * tw + fade_ph[:, :, None]))
    amp = np.clip(amp, 0.0, None)
    dly = dly0[:, :, None] + cfg.dly_wob * np.sin(2.0 * np.pi * dly_f[:, :, None] * tw + dly_ph[:, :, None])
    dly = np.clip(dly, cfg.dly_min, None)
    ph = ph0[:, :, None] + ph_rate[:, :, None] * tw

    if cfg.burst_n > 0 and cfg.burst_dur > 0.0:
        n_win = max(1, int(round(cfg.burst_dur / max(t[1] - t[0], 1e-12))))
        for _ in range(cfg.burst_n):
            b = int(rng.integers(0, bs_n))
            k = int(rng.integers(0, n_mp))
            s = int(rng.integers(0, max(1, n_t - n_win + 1)))
            e = min(n_t, s + n_win)
            amp[b, k, s:e] *= cfg.burst_gain
            dly[b, k, s:e] += 0.5 * cfg.dly_wob

    return dly, amp, ph


def _make_path_cpu(tx: TxData, tau0: np.ndarray, dly: np.ndarray, amp: np.ndarray, ph: np.ndarray, mp_n: int) -> tuple[np.ndarray, np.ndarray]:
    """CPU 路径下生成多径分路与合成信号。"""
    bs_n, n_t = tau0.shape
    path = np.zeros((bs_n, mp_n, n_t), dtype=np.complex128)
    for b in range(bs_n):
        for k in range(mp_n):
            tau = tau0[b] + dly[b, k]
            s = _delay_c(tx.tx, tx.t, tau)
            path[b, k] = amp[b, k] * np.exp(1j * ph[b, k]) * s
    mix = np.sum(path, axis=1)
    return path, mix


def _make_path_cuda(cfg: Cfg, tx: TxData, tau0: np.ndarray, dly: np.ndarray, amp: np.ndarray, ph: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """CUDA 路径下批量生成多径分路与合成信号。"""
    dev = pick_sim_dev(cfg)
    if dev is None:
        return None

    dt = float(tx.t[1] - tx.t[0]) if tx.t.size > 1 else 1.0
    tau = tau0[:, None, :] + dly
    s = delay_interp_c(tx.tx, tau, dt, dev)
    amp_t = np.asarray(amp, dtype=np.float32)
    ph_t = np.asarray(ph, dtype=np.float32)

    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    amp_c = torch.as_tensor(amp_t, dtype=torch.float32, device=dev)
    ph_c = torch.as_tensor(ph_t, dtype=torch.float32, device=dev)
    path = torch.polar(amp_c, ph_c) * s
    mix = torch.sum(path, dim=1)
    return path.cpu().numpy(), mix.cpu().numpy()


def gen_mp(cfg: Cfg, tx: TxData, geo: GeoData) -> MpData:
    """生成多径参数和多径信号。"""
    _check_mp_cfg(cfg.mp)
    if tx.t.size != geo.t.size:
        raise ValueError("tx.t 与 geo.t 长度不一致")
    if not np.allclose(tx.t, geo.t):
        raise ValueError("tx.t 与 geo.t 时间轴不一致")

    t = tx.t
    bs_n, n_t = geo.rng.shape
    rng = np.random.default_rng(cfg.mp.seed)

    # 每个基站每条多径的基准相对时延/幅度/相位
    dly0 = rng.uniform(cfg.mp.dly_min, cfg.mp.dly_max, size=(bs_n, cfg.mp.n))
    amp0 = _gen_amp0(cfg.mp, bs_n, rng)
    ph0 = _gen_ph0(cfg.mp, bs_n, rng)
    dly, amp, ph = _gen_time_field(cfg.mp, t, dly0, amp0, ph0, rng)

    # 直达时延 = 几何距离 / 光速
    tau0 = geo.rng / C0

    out = _make_path_cuda(cfg, tx, tau0, dly, amp, ph)
    if out is None:
        path, mix = _make_path_cpu(tx, tau0, dly, amp, ph, cfg.mp.n)
    else:
        path, mix = out
    return MpData(t=t, dly=dly, amp=amp, ph=ph, path=path, mix=mix)


def mp_stat(data: MpData) -> Dict[str, Any]:
    """输出多径摘要。"""
    return {
        "bs_n": int(data.mix.shape[0]),
        "mp_n": int(data.path.shape[1]),
        "t_n": int(data.t.size),
        "dly_ns_min": float(np.min(data.dly) * 1e9),
        "dly_ns_max": float(np.max(data.dly) * 1e9),
        "amp_mean": float(np.mean(data.amp)),
        "mix_pwr": float(np.mean(np.abs(data.mix) ** 2)),
    }


def save_mp(data: MpData, out_dir: Path) -> Path:
    """保存多径结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mp.npz"
    np.savez(
        out,
        t=data.t,
        dly=data.dly,
        amp=data.amp,
        ph=data.ph,
        path=data.path,
        mix=data.mix,
    )
    return out
