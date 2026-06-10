"""发射信号仿真。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import SigCfg


@dataclass
class TxData:
    """发射信号结果。"""

    t: np.ndarray       # 采样时刻
    prn: np.ndarray     # 一个周期 PRN 码
    nav_bits: np.ndarray  # 导航比特
    code: np.ndarray  # 逐采样码序列
    nav: np.ndarray  # 逐采样导航符号
    bb: np.ndarray  # 复调制前基带
    car: np.ndarray  # 载波
    tx: np.ndarray  # 最终发射复信号


def _check_cfg(cfg: SigCfg) -> None:
    """检查配置合法性。"""
    if cfg.fs <= 0:
        raise ValueError("fs 必须大于 0")
    if cfg.code_rate <= 0:
        raise ValueError("code_rate 必须大于 0")
    if cfg.code_len <= 0:
        raise ValueError("code_len 必须大于 0")
    if cfg.nav_rate <= 0:
        raise ValueError("nav_rate 必须大于 0")
    if cfg.dur <= 0:
        raise ValueError("dur 必须大于 0")
    if cfg.amp <= 0:
        raise ValueError("amp 必须大于 0")


def gen_prn(cfg: SigCfg) -> np.ndarray:
    """生成双极性 PRN 码（+1/-1）。"""
    rng = np.random.default_rng(cfg.prn_seed)
    x = rng.integers(0, 2, size=cfg.code_len, dtype=np.int8)
    return np.where(x == 0, -1.0, 1.0).astype(np.float64)


def gen_nav(cfg: SigCfg) -> np.ndarray:
    """生成覆盖全时长的导航比特（+1/-1）。"""
    n = int(np.ceil(cfg.dur * cfg.nav_rate))
    rng = np.random.default_rng(cfg.nav_seed)
    x = rng.integers(0, 2, size=n, dtype=np.int8)
    return np.where(x == 0, -1.0, 1.0).astype(np.float64)


def gen_tx(cfg: SigCfg) -> TxData:
    """生成发射复信号。"""
    _check_cfg(cfg)

    # 采样点数量
    n = int(round(cfg.dur * cfg.fs))
    if n <= 0:
        raise ValueError("采样点数量必须大于 0")

    # 时间轴 + 基础码元
    t = np.arange(n, dtype=np.float64) / cfg.fs
    prn = gen_prn(cfg)
    nav_bits = gen_nav(cfg)

    # 将码片和导航比特映射到每个采样点
    code_idx = np.floor(t * cfg.code_rate).astype(np.int64) % cfg.code_len
    nav_idx = np.floor(t * cfg.nav_rate).astype(np.int64)
    nav_idx = np.minimum(nav_idx, nav_bits.size - 1)

    # 形成基带并上变频
    code = prn[code_idx]
    nav = nav_bits[nav_idx]
    bb = cfg.amp * code * nav
    car = np.exp(1j * 2.0 * np.pi * cfg.fc * t)
    tx = bb * car

    return TxData(t=t, prn=prn, nav_bits=nav_bits, code=code, nav=nav, bb=bb, car=car, tx=tx)


def tx_stat(data: TxData) -> Dict[str, Any]:
    """统计发射信号摘要。"""
    return {
        "n": int(data.t.size),
        "dur": float(data.t[-1] - data.t[0]) if data.t.size > 1 else 0.0,
        "code_len": int(data.prn.size),
        "nav_n": int(data.nav_bits.size),
        "pwr": float(np.mean(np.abs(data.tx) ** 2)),
    }


def save_tx(data: TxData, out_dir: Path) -> Path:
    """保存发射信号到 npz 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "tx_signal.npz"
    np.savez(
        out,
        t=data.t,
        prn=data.prn,
        nav_bits=data.nav_bits,
        code=data.code,
        nav=data.nav,
        bb=data.bb,
        car=data.car,
        tx=data.tx,
    )
    return out
