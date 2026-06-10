"""定位贡献标签工具。"""

import numpy as np

from config import Cfg
from sim.geometry import GeoData
from sim.positioning import pick_dim, solve_epoch


def interp_pos(geo: GeoData, t_ep: np.ndarray) -> np.ndarray:
    """插值得到历元真值坐标。"""
    return np.column_stack(
        (
            np.interp(t_ep, geo.t, geo.rx[:, 0]),
            np.interp(t_ep, geo.t, geo.rx[:, 1]),
            np.interp(t_ep, geo.t, geo.rx[:, 2]),
        )
    )


def make_pos_q(
    cfg: Cfg,
    geo: GeoData,
    rho_obs: np.ndarray,
    rho_true: np.ndarray,
    t_ep: np.ndarray,
    scale: float,
) -> np.ndarray:
    """按单观测对定位误差的改善量构造质量标签。"""
    bs = geo.bs
    dim = pick_dim(cfg.pos, bs)
    z_fix = float(cfg.pos.z_fix)
    q_scale = max(float(scale), 1e-6)

    true_pos = interp_pos(geo, t_ep)
    n_ep, n_bs = rho_obs.shape
    harm = np.zeros((n_ep, n_bs), dtype=np.float64)

    x0 = np.mean(bs, axis=0)
    x0[2] = float(cfg.pos.z_init)

    for e in range(n_ep):
        p_full, _, _, _ = solve_epoch(bs, rho_obs[e], dim, z_fix, x0)
        err_full = float(np.linalg.norm(p_full - true_pos[e]))

        for b in range(n_bs):
            rho_fix = rho_obs[e].copy()
            rho_fix[b] = rho_true[e, b]
            p_fix, _, _, _ = solve_epoch(bs, rho_fix, dim, z_fix, p_full)
            err_fix = float(np.linalg.norm(p_fix - true_pos[e]))
            harm[e, b] = max(err_full - err_fix, 0.0)

        x0 = p_full.copy()

    return 1.0 / (1.0 + (harm / q_scale) ** 2)
