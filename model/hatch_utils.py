"""Hatch 系列模型工具函数。"""

import numpy as np

from model.common_utils import (
    build_mc_train,
    build_seq4,
    calc_pos_err,
    interp_pos,
    interp_rng,
    local_rms,
    local_sig,
    make_mc_cfg,
    make_q_w,
    make_var_w,
    sim_obs_geo,
    solve_w_epochs,
)


def hatch_smooth(code: np.ndarray, car: np.ndarray, n_win: int, name: str) -> np.ndarray:
    """执行 Hatch 码载平滑。"""
    if n_win <= 0:
        raise ValueError(f"{name} 必须大于 0")

    n_ep, n_bs = code.shape
    out = np.zeros((n_ep, n_bs), dtype=np.float64)
    out[0] = code[0]
    for b in range(n_bs):
        for e in range(1, n_ep):
            m = min(e + 1, n_win)
            a = 1.0 / float(m)
            d_car = car[e, b] - car[e - 1, b]
            out[e, b] = a * code[e, b] + (1.0 - a) * (out[e - 1, b] + d_car)
    return out


__all__ = [
    "build_mc_train",
    "build_seq4",
    "calc_pos_err",
    "hatch_smooth",
    "interp_pos",
    "interp_rng",
    "local_rms",
    "local_sig",
    "make_mc_cfg",
    "make_q_w",
    "make_var_w",
    "sim_obs_geo",
    "solve_w_epochs",
]
