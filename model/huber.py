"""Huber 扩展对比模型。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import Cfg
from model.common_utils import calc_pos_err, code_postfit_res, interp_pos, interp_rng
from sim.geometry import GeoData
from sim.positioning import pick_dim, solve_epoch
from sim.tracking import ObsData


@dataclass(frozen=True)
class HuberCfg:
    """Huber 默认参数。"""

    c: float = 1.5
    irls_n: int = 4
    sig_floor: float = 0.5


@dataclass
class HuberData:
    """Huber 输出。"""

    t_ep: np.ndarray
    rho_true: np.ndarray
    mp_raw: np.ndarray
    rho_hat: np.ndarray
    mp_hat: np.ndarray
    w: np.ndarray
    pos: np.ndarray
    cb: np.ndarray
    res: np.ndarray
    dim: int
    it: np.ndarray
    true_pos: np.ndarray
    e: np.ndarray
    d3: np.ndarray
    dh: np.ndarray
    dz: np.ndarray


def _cfg() -> HuberCfg:
    return HuberCfg()


def _mad_scale(x: np.ndarray, floor: float) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1.4826 * mad, float(floor))


def _huber_w(res: np.ndarray, c: float, sig: float) -> np.ndarray:
    u = np.abs(res) / max(float(c) * sig, 1e-6)
    w = np.ones_like(u, dtype=np.float64)
    mask = u > 1.0
    w[mask] = 1.0 / u[mask]
    return w


def _solve_huber_epochs(cfg: Cfg, geo: GeoData, rho: np.ndarray, sub: HuberCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    bs = geo.bs
    dim = pick_dim(cfg.pos, bs)
    n_ep, n_bs = rho.shape
    pos = np.zeros((n_ep, 3), dtype=np.float64)
    cb = np.zeros(n_ep, dtype=np.float64)
    res = np.zeros((n_ep, n_bs), dtype=np.float64)
    w_out = np.ones((n_ep, n_bs), dtype=np.float64)
    it = np.zeros(n_ep, dtype=np.int64)

    x0 = np.mean(bs, axis=0)
    x0[2] = float(cfg.pos.z_init)
    z_fix = float(cfg.pos.z_fix)
    for e in range(n_ep):
        pe, cbe, re, ite = solve_epoch(bs, rho[e], dim, z_fix, x0)
        w = np.ones(n_bs, dtype=np.float64)
        for _ in range(sub.irls_n):
            sig = _mad_scale(re, sub.sig_floor)
            w = _huber_w(re, sub.c, sig)
            pe, cbe, re, ite = solve_epoch(bs, rho[e], dim, z_fix, pe, w=w)
        pos[e] = pe
        cb[e] = cbe
        res[e] = re
        w_out[e] = w
        it[e] = ite
        x0 = pe.copy()
    return pos, cb, res, w_out, dim, it


def run_huber(cfg: Cfg, obs: ObsData, geo: GeoData) -> HuberData:
    """运行 Huber。"""
    sub = _cfg()
    rho_true = interp_rng(geo, obs.t_ep)
    mp_raw = obs.rho_code - rho_true
    rho_hat = obs.rho_code.copy()
    mp_hat = code_postfit_res(cfg, geo, obs)
    pos, cb, res, w, dim, it = _solve_huber_epochs(cfg, geo, rho_hat, sub)
    true_pos = interp_pos(geo, obs.t_ep)
    e, dh, dz, d3 = calc_pos_err(pos, true_pos)
    return HuberData(
        t_ep=obs.t_ep,
        rho_true=rho_true,
        mp_raw=mp_raw,
        rho_hat=rho_hat,
        mp_hat=mp_hat,
        w=w,
        pos=pos,
        cb=cb,
        res=res,
        dim=dim,
        it=it,
        true_pos=true_pos,
        e=e,
        d3=d3,
        dh=dh,
        dz=dz,
    )


def huber_stat(data: HuberData) -> Dict[str, Any]:
    """输出摘要。"""
    return {
        "dim": int(data.dim),
        "n_ep": int(data.t_ep.size),
        "mp_raw_rms_m": float(np.sqrt(np.mean(data.mp_raw**2))),
        "mp_hat_rms_m": float(np.sqrt(np.mean(data.mp_hat**2))),
        "w_min": float(np.min(data.w)),
        "w_max": float(np.max(data.w)),
        "d3_rms_m": float(np.sqrt(np.mean(data.d3**2))),
        "dh_rms_m": float(np.sqrt(np.mean(data.dh**2))),
    }


def save_huber(data: HuberData, out_dir: Path) -> Path:
    """保存结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "huber.npz"
    np.savez(
        out,
        t_ep=data.t_ep,
        rho_true=data.rho_true,
        mp_raw=data.mp_raw,
        rho_hat=data.rho_hat,
        mp_hat=data.mp_hat,
        w=data.w,
        pos=data.pos,
        cb=data.cb,
        res=data.res,
        dim=np.array([data.dim], dtype=np.int64),
        it=data.it,
        true_pos=data.true_pos,
        e=data.e,
        d3=data.d3,
        dh=data.dh,
        dz=data.dz,
    )
    return out
