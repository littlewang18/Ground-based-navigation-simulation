"""正式实验共用的经典场景与绘图配置。"""

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

from config import Cfg
import model.gru as gru_mod
import model.mstcn as mstcn_mod
import model.nav_tcn as ntcn_mod
import model.tcn as tcn_mod

_MODS = [tcn_mod, mstcn_mod, ntcn_mod, gru_mod]

_COLOR = {
    "baseline": "#4C78A8",
    "Hatch": "#BAB0AC",
    "Huber": "#54A24B",
    "GRU": "#FF9DA6",
    "TCN": "#F58518",
    "MSTCN": "#EECA3B",
    "NavTCN": "#9D755D",
}


def _legacy_q(cfg: Cfg, geo, rho_obs, rho_true, t_ep, scale: float):
    """兼容旧版残差标签口径。"""
    _ = (cfg, geo, t_ep)
    scale = max(float(scale), 1e-6)
    res = abs(rho_obs - rho_true)
    return 1.0 / (1.0 + (res / scale) ** 2)


@contextmanager
def legacy_label() -> Iterator[None]:
    """临时把深度模型标签切回旧版残差标签。"""
    bak = {m.__name__: getattr(m, "make_pos_q", None) for m in _MODS}
    try:
        for mod in _MODS:
            if getattr(mod, "make_pos_q", None) is not None:
                mod.make_pos_q = _legacy_q
        yield
    finally:
        for mod in _MODS:
            if bak[mod.__name__] is not None:
                mod.make_pos_q = bak[mod.__name__]


def make_cfg(cfg: Cfg) -> Cfg:
    """构造正式主表使用的经典多径场景。"""
    geo = replace(
        cfg.geo,
        bs_n=6,
        bs_z_jit=1.5,
        rx_mode="curve",
        rx_ax=1.0,
        rx_ay=-0.5,
        rx_wob_x=1.5,
        rx_wob_y=2.0,
        rx_wob_f=4.0,
    )
    mp = replace(
        cfg.mp,
        n=4,
        dly_min=50e-9,
        dly_max=600e-9,
        k0=0.45,
        decay=0.75,
        jit=0.13,
        fade=0.08,
        fade_f_min=6.0,
        fade_f_max=18.0,
        dly_wob=12e-9,
        ph_rate=60.0,
        burst_n=1,
        burst_dur=1.0e-3,
        burst_gain=1.4,
    )
    rx = replace(
        cfg.rx,
        snr_db=16.5,
        los_fade=0.05,
        los_fade_f=6.0,
        blk_n=1,
        blk_dur=1.0e-3,
        blk_gain=0.75,
        imp_n=1,
        imp_dur=0.5e-3,
        imp_gain=3.0,
    )
    run = replace(cfg.run, save=False)
    return replace(cfg, geo=geo, mp=mp, rx=rx, run=run)
