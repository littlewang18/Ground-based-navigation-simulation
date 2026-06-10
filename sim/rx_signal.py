"""接收信号生成。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import Cfg, RxCfg
from sim.cuda_utils import pick_sim_dev, delay_interp_c
from sim.geometry import GeoData
from sim.multipath import C0, MpData
from sim.tx_signal import TxData


@dataclass
class RxData:
    """接收信号结果。"""

    t: np.ndarray  # 时间轴
    los: np.ndarray  # 直达信号 (n_bs, n_t)
    mp: np.ndarray  # 合成多径信号 (n_bs, n_t)
    noise: np.ndarray  # 噪声 (n_bs, n_t)
    rx: np.ndarray  # 最终接收信号 (n_bs, n_t)


def _check_rx_cfg(cfg: RxCfg) -> None:
    """检查接收信号配置。"""
    if cfg.a0 < 0.0:
        raise ValueError("rx.a0 不能小于 0")
    if not np.isfinite(cfg.snr_db):
        raise ValueError("rx.snr_db 必须是有限值")
    if cfg.los_fade < 0.0:
        raise ValueError("rx.los_fade 不能小于 0")
    if cfg.los_fade_f < 0.0:
        raise ValueError("rx.los_fade_f 不能小于 0")
    if cfg.blk_n < 0 or cfg.imp_n < 0:
        raise ValueError("rx.blk_n / rx.imp_n 不能小于 0")
    if cfg.blk_dur < 0.0 or cfg.imp_dur < 0.0:
        raise ValueError("rx.blk_dur / rx.imp_dur 不能小于 0")
    if cfg.blk_gain <= 0.0 or cfg.blk_gain > 1.0:
        raise ValueError("rx.blk_gain 必须在 (0, 1] 内")
    if cfg.imp_gain < 1.0:
        raise ValueError("rx.imp_gain 不能小于 1")


def _delay_c(sig: np.ndarray, t: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """对复信号执行可变时延。"""
    q = t - tau
    re = np.interp(q, t, np.real(sig), left=0.0, right=0.0)
    im = np.interp(q, t, np.imag(sig), left=0.0, right=0.0)
    return re + 1j * im


def _gen_los(cfg: RxCfg, tx: TxData, geo: GeoData) -> np.ndarray:
    """生成每个基站的直达信号。"""
    bs_n = geo.bs.shape[0]
    los = np.zeros((bs_n, tx.t.size), dtype=np.complex128)
    tau0 = geo.rng / C0
    rng = np.random.default_rng(cfg.seed + 101)
    gain = np.ones((bs_n, tx.t.size), dtype=np.float64)

    if cfg.los_fade > 0.0 and cfg.los_fade_f > 0.0:
        ph = rng.uniform(0.0, 2.0 * np.pi, bs_n)
        gain = gain * (1.0 + cfg.los_fade * np.sin(2.0 * np.pi * cfg.los_fade_f * tx.t[None, :] + ph[:, None]))
        gain = np.clip(gain, 0.1, None)

    if cfg.blk_n > 0 and cfg.blk_dur > 0.0:
        n_win = max(1, int(round(cfg.blk_dur / max(tx.t[1] - tx.t[0], 1e-12))))
        for _ in range(cfg.blk_n):
            b = int(rng.integers(0, bs_n))
            s = int(rng.integers(0, max(1, tx.t.size - n_win + 1)))
            e = min(tx.t.size, s + n_win)
            gain[b, s:e] *= cfg.blk_gain

    for b in range(bs_n):
        los[b] = cfg.a0 * gain[b] * _delay_c(tx.tx, tx.t, tau0[b])

    return los


def _gen_los_cuda(cfg_all: Cfg, tx: TxData, geo: GeoData) -> np.ndarray | None:
    """CUDA 路径下批量生成直达信号。"""
    dev = pick_sim_dev(cfg_all)
    if dev is None:
        return None

    cfg = cfg_all.rx
    bs_n = geo.bs.shape[0]
    tau0 = geo.rng / C0
    rng = np.random.default_rng(cfg.seed + 101)
    gain = np.ones((bs_n, tx.t.size), dtype=np.float32)

    if cfg.los_fade > 0.0 and cfg.los_fade_f > 0.0:
        ph = rng.uniform(0.0, 2.0 * np.pi, bs_n)
        gain = gain * (1.0 + cfg.los_fade * np.sin(2.0 * np.pi * cfg.los_fade_f * tx.t[None, :] + ph[:, None]))
        gain = np.clip(gain, 0.1, None)

    if cfg.blk_n > 0 and cfg.blk_dur > 0.0:
        dt = float(tx.t[1] - tx.t[0]) if tx.t.size > 1 else cfg.blk_dur
        n_win = max(1, int(round(cfg.blk_dur / max(dt, 1e-12))))
        for _ in range(cfg.blk_n):
            b = int(rng.integers(0, bs_n))
            s = int(rng.integers(0, max(1, tx.t.size - n_win + 1)))
            e = min(tx.t.size, s + n_win)
            gain[b, s:e] *= cfg.blk_gain

    dt = float(tx.t[1] - tx.t[0]) if tx.t.size > 1 else 1.0
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    s = delay_interp_c(tx.tx, tau0, dt, dev)
    gain_t = torch.as_tensor(gain, dtype=torch.float32, device=dev)
    los = float(cfg.a0) * gain_t * s
    return los.cpu().numpy()


def _gen_noise(cfg: RxCfg, t: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """按目标信噪比生成复高斯白噪声。"""
    rng = np.random.default_rng(cfg.seed)
    snr = 10.0 ** (cfg.snr_db / 10.0)
    p_sig = np.mean(np.abs(clean) ** 2, axis=1, keepdims=True)
    p_noise = p_sig / snr
    sigma = np.sqrt(p_noise / 2.0)

    scale = np.ones(clean.shape, dtype=np.float64)
    if cfg.imp_n > 0 and cfg.imp_dur > 0.0:
        dt = float(t[1] - t[0]) if t.size > 1 else cfg.imp_dur
        n_win = max(1, int(round(cfg.imp_dur / max(dt, 1e-12))))
        for _ in range(cfg.imp_n):
            b = int(rng.integers(0, clean.shape[0]))
            s = int(rng.integers(0, max(1, clean.shape[1] - n_win + 1)))
            e = min(clean.shape[1], s + n_win)
            scale[b, s:e] *= cfg.imp_gain

    n_re = rng.standard_normal(size=clean.shape)
    n_im = rng.standard_normal(size=clean.shape)
    return sigma * scale * (n_re + 1j * n_im)


def _gen_noise_cuda(cfg_all: Cfg, t: np.ndarray, clean: np.ndarray) -> np.ndarray | None:
    """CUDA 路径下生成复高斯白噪声。"""
    dev = pick_sim_dev(cfg_all)
    if dev is None:
        return None

    cfg = cfg_all.rx
    snr = 10.0 ** (cfg.snr_db / 10.0)
    p_sig = np.mean(np.abs(clean) ** 2, axis=1, keepdims=True)
    p_noise = p_sig / snr
    sigma = np.sqrt(p_noise / 2.0).astype(np.float32)

    rng = np.random.default_rng(cfg.seed)
    scale = np.ones(clean.shape, dtype=np.float32)
    if cfg.imp_n > 0 and cfg.imp_dur > 0.0:
        dt = float(t[1] - t[0]) if t.size > 1 else cfg.imp_dur
        n_win = max(1, int(round(cfg.imp_dur / max(dt, 1e-12))))
        for _ in range(cfg.imp_n):
            b = int(rng.integers(0, clean.shape[0]))
            s = int(rng.integers(0, max(1, clean.shape[1] - n_win + 1)))
            e = min(clean.shape[1], s + n_win)
            scale[b, s:e] *= cfg.imp_gain

    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(cfg.seed))
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=dev)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=dev)
    n_re = torch.randn(clean.shape, generator=gen, device=dev, dtype=torch.float32)
    n_im = torch.randn(clean.shape, generator=gen, device=dev, dtype=torch.float32)
    noise = sigma_t * scale_t * (n_re + 1j * n_im)
    return noise.cpu().numpy()


def gen_rx(cfg: Cfg, tx: TxData, geo: GeoData, mp: MpData) -> RxData:
    """生成接收端输入信号。"""
    _check_rx_cfg(cfg.rx)
    if tx.t.size != geo.t.size or tx.t.size != mp.t.size:
        raise ValueError("tx、geo、mp 的时间长度不一致")
    if not np.allclose(tx.t, geo.t) or not np.allclose(tx.t, mp.t):
        raise ValueError("tx、geo、mp 的时间轴不一致")
    if geo.bs.shape[0] != mp.mix.shape[0]:
        raise ValueError("基站数量与多径数据不一致")

    t = tx.t
    los = _gen_los_cuda(cfg, tx, geo)
    if los is None:
        los = _gen_los(cfg.rx, tx, geo)
    clean = los + mp.mix
    noise = _gen_noise_cuda(cfg, t, clean)
    if noise is None:
        noise = _gen_noise(cfg.rx, t, clean)
    rx = clean + noise

    return RxData(t=t, los=los, mp=mp.mix, noise=noise, rx=rx)


def rx_stat(data: RxData) -> Dict[str, Any]:
    """输出接收信号摘要。"""
    clean = data.los + data.mp
    p_clean = np.mean(np.abs(clean) ** 2)
    p_los = np.mean(np.abs(data.los) ** 2)
    p_mp = np.mean(np.abs(data.mp) ** 2)
    p_noise = np.mean(np.abs(data.noise) ** 2)
    p_rx = np.mean(np.abs(data.rx) ** 2)
    snr = 10.0 * np.log10(p_clean / p_noise) if p_noise > 0.0 else np.inf
    return {
        "bs_n": int(data.rx.shape[0]),
        "t_n": int(data.t.size),
        "clean_pwr": float(p_clean),
        "los_pwr": float(p_los),
        "mp_pwr": float(p_mp),
        "noise_pwr": float(p_noise),
        "rx_pwr": float(p_rx),
        "snr_db": float(snr),
    }


def save_rx(data: RxData, out_dir: Path) -> Path:
    """保存接收信号结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "rx.npz"
    np.savez(out, t=data.t, los=data.los, mp=data.mp, noise=data.noise, rx=data.rx)
    return out
