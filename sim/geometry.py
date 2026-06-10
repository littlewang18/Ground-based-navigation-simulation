"""几何场景生成。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import Cfg, GeoCfg
from sim.tx_signal import TxData


@dataclass
class GeoData:
    """几何场景结果。"""

    t: np.ndarray  # 时间轴
    bs: np.ndarray  # 基站坐标 (n_bs, 3)
    rx: np.ndarray  # 接收机坐标 (n_t, 3)
    rng: np.ndarray  # 基站到接收机距离 (n_bs, n_t)


def _check_geo_cfg(cfg: GeoCfg) -> None:
    """检查几何配置。"""
    if cfg.bs_n <= 0:
        raise ValueError("bs_n 必须大于 0")
    if cfg.bs_r <= 0:
        raise ValueError("bs_r 必须大于 0")
    if cfg.bs_z_jit < 0.0:
        raise ValueError("bs_z_jit 不能小于 0")
    if cfg.bs_mode not in {"circle", "random"}:
        raise ValueError("bs_mode 仅支持 circle 或 random")
    if cfg.rx_mode not in {"static", "line", "curve"}:
        raise ValueError("rx_mode 仅支持 static / line / curve")
    if cfg.rx_wob_f < 0.0:
        raise ValueError("rx_wob_f 不能小于 0")


def _gen_bs(cfg: GeoCfg) -> np.ndarray:
    """生成基站坐标。"""
    rng = np.random.default_rng(cfg.bs_seed)
    if cfg.bs_mode == "circle":
        ang = np.linspace(0.0, 2.0 * np.pi, cfg.bs_n, endpoint=False, dtype=np.float64)
        x = cfg.bs_cx + cfg.bs_r * np.cos(ang)
        y = cfg.bs_cy + cfg.bs_r * np.sin(ang)
    else:
        x = cfg.bs_cx + rng.uniform(-cfg.bs_r, cfg.bs_r, cfg.bs_n)
        y = cfg.bs_cy + rng.uniform(-cfg.bs_r, cfg.bs_r, cfg.bs_n)

    z = np.full(cfg.bs_n, cfg.bs_h, dtype=np.float64)
    if cfg.bs_z_jit > 0.0:
        z = z + rng.uniform(-cfg.bs_z_jit, cfg.bs_z_jit, cfg.bs_n)
    return np.column_stack((x, y, z))


def _gen_rx(cfg: GeoCfg, t: np.ndarray) -> np.ndarray:
    """生成接收机坐标序列。"""
    if cfg.rx_mode == "static":
        x = np.full(t.size, cfg.rx_x0, dtype=np.float64)
        y = np.full(t.size, cfg.rx_y0, dtype=np.float64)
        z = np.full(t.size, cfg.rx_z0, dtype=np.float64)
    elif cfg.rx_mode == "line":
        x = cfg.rx_x0 + cfg.rx_vx * t
        y = cfg.rx_y0 + cfg.rx_vy * t
        z = cfg.rx_z0 + cfg.rx_vz * t
    else:
        w = 2.0 * np.pi * cfg.rx_wob_f
        x = cfg.rx_x0 + cfg.rx_vx * t + 0.5 * cfg.rx_ax * t**2 + cfg.rx_wob_x * np.sin(w * t)
        y = cfg.rx_y0 + cfg.rx_vy * t + 0.5 * cfg.rx_ay * t**2 + cfg.rx_wob_y * np.cos(w * t)
        z = np.full(t.size, cfg.rx_z0, dtype=np.float64) + cfg.rx_vz * t

    return np.column_stack((x, y, z))


def gen_geo(cfg: Cfg, tx: TxData) -> GeoData:
    """生成基站位置和接收机位置。"""
    _check_geo_cfg(cfg.geo)

    t = tx.t
    bs = _gen_bs(cfg.geo)
    rx = _gen_rx(cfg.geo, t)

    # 广播计算每个基站到每个时刻接收机的几何距离
    d = rx[None, :, :] - bs[:, None, :]
    rng = np.linalg.norm(d, axis=2)

    return GeoData(t=t, bs=bs, rx=rx, rng=rng)


def geo_stat(data: GeoData) -> Dict[str, Any]:
    """输出几何场景摘要。"""
    return {
        "bs_n": int(data.bs.shape[0]),
        "t_n": int(data.t.size),
        "rx0": data.rx[0].tolist(),
        "rx1": data.rx[-1].tolist(),
        "rng_min": float(np.min(data.rng)),
        "rng_max": float(np.max(data.rng)),
    }


def save_geo(data: GeoData, out_dir: Path) -> Path:
    """保存几何场景结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "geo.npz"
    np.savez(out, t=data.t, bs=data.bs, rx=data.rx, rng=data.rng)
    return out
