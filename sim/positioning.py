"""定位解算。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from config import Cfg, PosCfg
from sim.geometry import GeoData
from sim.tracking import ObsData


@dataclass
class PosData:
    """时序定位结果。"""

    t_ep: np.ndarray  # 历元时刻 (n_ep,)
    pos_code: np.ndarray  # 码伪距定位坐标 (n_ep, 3)
    cb_code: np.ndarray  # 码伪距钟差 m (n_ep,)
    res_code: np.ndarray  # 码伪距残差 (n_ep, n_bs)
    pos_car: np.ndarray  # 载波伪距定位坐标 (n_ep, 3)
    cb_car: np.ndarray  # 载波伪距钟差 m (n_ep,)
    res_car: np.ndarray  # 载波伪距残差 (n_ep, n_bs)
    dim: int  # 解算维数 2 / 3
    it_code: np.ndarray  # 码伪距迭代次数 (n_ep,)
    it_car: np.ndarray  # 载波伪距迭代次数 (n_ep,)


def solve_epoch(
    bs: np.ndarray,
    rho: np.ndarray,
    dim: int,
    z_fix: float,
    x0: np.ndarray,
    w: np.ndarray | None = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """单历元最小二乘/加权最小二乘定位。"""
    if dim == 3:
        x = np.array([x0[0], x0[1], x0[2], 0.0], dtype=np.float64)
    else:
        x = np.array([x0[0], x0[1], 0.0], dtype=np.float64)

    sw = None
    if w is not None:
        sw = np.sqrt(np.clip(np.asarray(w, dtype=np.float64), 1e-8, None))

    it_used = 0
    for it in range(max_iter):
        it_used = it + 1
        if dim == 3:
            p = x[:3]
            d = p[None, :] - bs
            r = np.linalg.norm(d, axis=1)
            r = np.maximum(r, 1e-9)
            h = r + x[3]
            v = rho - h
            H = np.column_stack((d[:, 0] / r, d[:, 1] / r, d[:, 2] / r, np.ones(bs.shape[0])))
        else:
            p = np.array([x[0], x[1], z_fix], dtype=np.float64)
            d = p[None, :] - bs
            r = np.linalg.norm(d, axis=1)
            r = np.maximum(r, 1e-9)
            h = r + x[2]
            v = rho - h
            H = np.column_stack((d[:, 0] / r, d[:, 1] / r, np.ones(bs.shape[0])))

        if sw is None:
            a = H
            b = v
        else:
            a = H * sw[:, None]
            b = v * sw

        dx, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break

    if dim == 3:
        p_est = x[:3]
        cb = float(x[3])
    else:
        p_est = np.array([x[0], x[1], z_fix], dtype=np.float64)
        cb = float(x[2])

    d = p_est[None, :] - bs
    r = np.linalg.norm(d, axis=1)
    res = rho - (r + cb)
    return p_est, cb, res, it_used


def pick_dim(cfg: PosCfg, bs: np.ndarray) -> int:
    """选择 2D/3D 解算模式。"""
    mode = cfg.mode.lower()
    n_bs = int(bs.shape[0])

    if mode == "2d":
        if n_bs < 3:
            raise ValueError("2D 解算至少需要 3 个基站")
        return 2
    if mode == "3d":
        if n_bs < 4:
            raise ValueError("3D 解算至少需要 4 个基站")
        return 3
    if mode != "auto":
        raise ValueError("pos.mode 仅支持 auto / 2d / 3d")

    if n_bs < 3:
        raise ValueError("基站数量不足，至少需要 3 个基站")
    if n_bs < 4:
        return 2

    z_span = float(np.max(bs[:, 2]) - np.min(bs[:, 2]))
    return 3 if z_span >= cfg.min_bs_z_span else 2


def solve_pos(cfg: Cfg, obs: ObsData, geo: GeoData) -> PosData:
    """根据时序伪距观测解算位置。"""
    bs = geo.bs
    n_bs = bs.shape[0]
    if obs.rho_code.ndim != 2 or obs.rho_car.ndim != 2:
        raise ValueError("obs.rho_code / obs.rho_car 必须是二维矩阵")
    if obs.rho_code.shape[1] != n_bs or obs.rho_car.shape[1] != n_bs:
        raise ValueError("观测数量与基站数量不一致")
    if obs.rho_code.shape[0] != obs.t_ep.size:
        raise ValueError("obs.t_ep 与观测历元数不一致")

    dim = pick_dim(cfg.pos, bs)
    n_ep = obs.t_ep.size

    pos_code = np.zeros((n_ep, 3), dtype=np.float64)
    pos_car = np.zeros((n_ep, 3), dtype=np.float64)
    cb_code = np.zeros(n_ep, dtype=np.float64)
    cb_car = np.zeros(n_ep, dtype=np.float64)
    res_code = np.zeros((n_ep, n_bs), dtype=np.float64)
    res_car = np.zeros((n_ep, n_bs), dtype=np.float64)
    it_code = np.zeros(n_ep, dtype=np.int64)
    it_car = np.zeros(n_ep, dtype=np.int64)

    x0_code = np.mean(bs, axis=0)
    x0_code[2] = float(cfg.pos.z_init)
    x0_car = x0_code.copy()
    z_fix = float(cfg.pos.z_fix)

    for e in range(n_ep):
        pc, cbc, rc, itc = solve_epoch(bs, obs.rho_code[e], dim, z_fix, x0_code)
        pr, cbr, rr, itr = solve_epoch(bs, obs.rho_car[e], dim, z_fix, x0_car)

        pos_code[e] = pc
        cb_code[e] = cbc
        res_code[e] = rc
        it_code[e] = itc

        pos_car[e] = pr
        cb_car[e] = cbr
        res_car[e] = rr
        it_car[e] = itr

        # 用上一历元结果作为下一历元初值，提升收敛稳定性。
        x0_code = pc.copy()
        x0_car = pr.copy()

    return PosData(
        t_ep=obs.t_ep,
        pos_code=pos_code,
        cb_code=cb_code,
        res_code=res_code,
        pos_car=pos_car,
        cb_car=cb_car,
        res_car=res_car,
        dim=dim,
        it_code=it_code,
        it_car=it_car,
    )


def pos_stat(data: PosData) -> Dict[str, Any]:
    """输出定位摘要。"""
    rms_code = np.sqrt(np.mean(data.res_code**2, axis=1))
    rms_car = np.sqrt(np.mean(data.res_car**2, axis=1))
    return {
        "dim": int(data.dim),
        "n_ep": int(data.t_ep.size),
        "t0_s": float(data.t_ep[0]),
        "t1_s": float(data.t_ep[-1]),
        "pos_code_0": data.pos_code[0].tolist(),
        "pos_code_1": data.pos_code[-1].tolist(),
        "pos_car_0": data.pos_car[0].tolist(),
        "pos_car_1": data.pos_car[-1].tolist(),
        "rms_code_mean_m": float(np.mean(rms_code)),
        "rms_car_mean_m": float(np.mean(rms_car)),
    }


def save_pos(data: PosData, out_dir: Path) -> Path:
    """保存定位结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pos.npz"
    np.savez(
        out,
        t_ep=data.t_ep,
        pos_code=data.pos_code,
        cb_code=data.cb_code,
        res_code=data.res_code,
        pos_car=data.pos_car,
        cb_car=data.cb_car,
        res_car=data.res_car,
        dim=np.array([data.dim], dtype=np.int64),
        it_code=data.it_code,
        it_car=data.it_car,
    )
    return out
