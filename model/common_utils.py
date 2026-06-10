"""模型通用工具函数。"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from config import Cfg
from sim.geometry import GeoData, gen_geo
from sim.multipath import gen_mp
from sim.positioning import pick_dim, solve_epoch
from sim.rx_signal import gen_rx
from sim.tracking import ObsData, track
from sim.tx_signal import gen_tx


def interp_rng(geo: GeoData, t_ep: np.ndarray) -> np.ndarray:
    """插值得到每个历元的真值几何距离。"""
    n_bs = geo.bs.shape[0]
    out = np.zeros((t_ep.size, n_bs), dtype=np.float64)
    for b in range(n_bs):
        out[:, b] = np.interp(t_ep, geo.t, geo.rng[b])
    return out


def interp_pos(geo: GeoData, t_ep: np.ndarray) -> np.ndarray:
    """插值得到每个历元的接收机真值位置。"""
    return np.column_stack(
        (
            np.interp(t_ep, geo.t, geo.rx[:, 0]),
            np.interp(t_ep, geo.t, geo.rx[:, 1]),
            np.interp(t_ep, geo.t, geo.rx[:, 2]),
        )
    )


def local_rms(x: np.ndarray, back_n: int = 3) -> np.ndarray:
    """计算因果局部 RMS。"""
    out = np.zeros_like(x)
    for i in range(x.size):
        s = max(0, i - back_n + 1)
        out[i] = float(np.sqrt(np.mean(x[s : i + 1] ** 2)))
    return out


def build_seq4(res: np.ndarray, seq_n: int, name: str) -> np.ndarray:
    """基于残差构造四通道因果时序特征。"""
    if seq_n <= 0:
        raise ValueError(f"{name} 必须大于 0")

    n_ep, n_bs = res.shape
    feat_n = 4
    seq = np.zeros((n_ep, n_bs, seq_n, feat_n), dtype=np.float64)

    for b in range(n_bs):
        for e in range(n_ep):
            s = max(0, e - seq_n + 1)
            seg = res[s : e + 1, b]
            pad_n = seq_n - seg.size
            if pad_n > 0:
                seg = np.concatenate((np.full(pad_n, seg[0]), seg))
            d1 = np.diff(seg, prepend=seg[0])
            mag = np.abs(seg)
            rms = local_rms(seg)
            seq[e, b, :, 0] = seg
            seq[e, b, :, 1] = d1
            seq[e, b, :, 2] = mag
            seq[e, b, :, 3] = rms

    return seq


def code_postfit_res(cfg: Cfg, geo: GeoData, obs: ObsData) -> np.ndarray:
    """用粗定位后验残差构造模型可观测输入。"""
    w = np.ones_like(obs.rho_code, dtype=np.float64)
    _, _, res, _, _ = solve_w_epochs(cfg, geo, obs.rho_code, w)
    return res


def sim_obs_geo(cfg: Cfg) -> tuple[ObsData, GeoData]:
    """运行 1 至 5 阶段，返回跟踪观测和几何真值。"""
    tx = gen_tx(cfg.sig)
    geo = gen_geo(cfg, tx)
    mp = gen_mp(cfg, tx, geo)
    rx = gen_rx(cfg, tx, geo, mp)
    obs = track(cfg, rx)
    return obs, geo


def make_mc_cfg(cfg: Cfg, subcfg: Any, idx: int) -> Cfg:
    """按子模型配置生成第 idx 个 Monte Carlo 场景。"""
    seed = int(subcfg.seed) + (idx + 1) * int(subcfg.mc_seed_step)
    rng = np.random.default_rng(seed)

    sig = replace(
        cfg.sig,
        prn_seed=cfg.sig.prn_seed + (idx + 1) * 11,
        nav_seed=cfg.sig.nav_seed + (idx + 1) * 11,
    )
    geo = replace(
        cfg.geo,
        bs_seed=cfg.geo.bs_seed + (idx + 1) * 13,
        bs_cx=cfg.geo.bs_cx + float(rng.normal(0.0, subcfg.mc_bs_xy_jit)),
        bs_cy=cfg.geo.bs_cy + float(rng.normal(0.0, subcfg.mc_bs_xy_jit)),
        rx_x0=cfg.geo.rx_x0 + float(rng.normal(0.0, subcfg.mc_rx_xy_jit)),
        rx_y0=cfg.geo.rx_y0 + float(rng.normal(0.0, subcfg.mc_rx_xy_jit)),
        rx_vx=cfg.geo.rx_vx + float(rng.normal(0.0, subcfg.mc_rx_v_jit)),
        rx_vy=cfg.geo.rx_vy + float(rng.normal(0.0, subcfg.mc_rx_v_jit)),
    )
    mp = replace(cfg.mp, seed=cfg.mp.seed + (idx + 1) * 17)
    rx = replace(
        cfg.rx,
        seed=cfg.rx.seed + (idx + 1) * 19,
        snr_db=float(np.clip(cfg.rx.snr_db + rng.normal(0.0, subcfg.mc_snr_jit), 5.0, 40.0)),
    )
    trk = replace(cfg.trk, amb_seed=cfg.trk.amb_seed + (idx + 1) * 23)
    return replace(cfg, sig=sig, geo=geo, mp=mp, rx=rx, trk=trk)


def build_mc_train(cfg: Cfg, subcfg: Any, prep_fn: Callable[[ObsData, GeoData, Cfg], Any]) -> tuple[np.ndarray, np.ndarray]:
    """构造质量评分类模型使用的 Monte Carlo 训练样本。"""
    if subcfg.mc_n <= 0:
        raise ValueError("mc_n 必须大于 0")

    seq_lst = []
    q_lst = []
    seq_n = int(subcfg.seq_n)
    for i in range(subcfg.mc_n):
        cfg_i = make_mc_cfg(cfg, subcfg, i)
        obs_i, geo_i = sim_obs_geo(cfg_i)
        prep_i = prep_fn(obs_i, geo_i, cfg_i)
        seq_lst.append(prep_i.seq.reshape(-1, seq_n, prep_i.seq.shape[-1]))
        q_lst.append(prep_i.q_tar.reshape(-1))

    seq = np.concatenate(seq_lst, axis=0)
    q = np.concatenate(q_lst, axis=0)
    return seq, q


def build_q_val_case(cfg: Cfg, subcfg: Any, prep_fn: Callable[[ObsData, GeoData, Cfg], Any], idx_add: int = 17) -> SimpleNamespace:
    """为质量评分模型构造固定验证场景。"""
    idx = int(subcfg.mc_n) + int(idx_add)
    cfg_i = make_mc_cfg(cfg, subcfg, idx)
    obs_i, geo_i = sim_obs_geo(cfg_i)
    prep_i = prep_fn(obs_i, geo_i, cfg_i)
    n_ep, n_bs = prep_i.rho_hat.shape
    seq_eval = prep_i.seq.reshape(-1, int(subcfg.seq_n), prep_i.seq.shape[-1])
    true_pos = interp_pos(geo_i, obs_i.t_ep)
    return SimpleNamespace(
        cfg=cfg_i,
        obs=obs_i,
        geo=geo_i,
        prep=prep_i,
        seq_eval=seq_eval,
        n_ep=n_ep,
        n_bs=n_bs,
        true_pos=true_pos,
    )


def build_q_eval_case(cfg: Cfg, subcfg: Any, obs: ObsData, geo: GeoData, prep: Any) -> SimpleNamespace:
    """为质量评分模型构造当前正式场景的按 epoch 测试包。"""
    n_ep, n_bs = prep.rho_hat.shape
    seq_eval = prep.seq.reshape(-1, int(subcfg.seq_n), prep.seq.shape[-1])
    true_pos = interp_pos(geo, obs.t_ep)
    return SimpleNamespace(
        cfg=cfg,
        obs=obs,
        geo=geo,
        prep=prep,
        seq_eval=seq_eval,
        n_ep=n_ep,
        n_bs=n_bs,
        true_pos=true_pos,
    )


def fill_val_curve(val_hist: np.ndarray) -> np.ndarray:
    """把稀疏记录的验证指标补成完整曲线。"""
    out = np.asarray(val_hist, dtype=np.float64).copy()
    if out.size == 0:
        return out
    finite = np.isfinite(out)
    if not np.any(finite):
        return np.zeros_like(out)
    first = int(np.argmax(finite))
    out[:first] = out[first]
    for i in range(first + 1, out.size):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out


def make_q_w(q_pred: np.ndarray, w_min: float, w_max: float, n_ep: int, n_bs: int) -> np.ndarray:
    """把观测质量评分映射为定位权重。"""
    w = float(w_min) + q_pred * (float(w_max) - float(w_min))
    w = w.reshape(n_ep, n_bs)
    w = np.clip(w, float(w_min), float(w_max))
    w = w / np.mean(w, axis=1, keepdims=True)
    return w


def solve_w_epochs(cfg: Cfg, geo: GeoData, rho: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """逐历元加权最小二乘定位。"""
    bs = geo.bs
    dim = pick_dim(cfg.pos, bs)
    n_ep, n_bs = rho.shape
    pos = np.zeros((n_ep, 3), dtype=np.float64)
    cb = np.zeros(n_ep, dtype=np.float64)
    res = np.zeros((n_ep, n_bs), dtype=np.float64)
    it = np.zeros(n_ep, dtype=np.int64)

    x0 = np.mean(bs, axis=0)
    x0[2] = float(cfg.pos.z_init)
    for e in range(n_ep):
        pe, cbe, re, ite = solve_epoch(
            bs=bs,
            rho=rho[e],
            dim=dim,
            z_fix=float(cfg.pos.z_fix),
            x0=x0,
            w=w[e],
        )
        pos[e] = pe
        cb[e] = cbe
        res[e] = re
        it[e] = ite
        x0 = pe.copy()

    return pos, cb, res, dim, it


def calc_pos_err(pos: np.ndarray, true_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算三维、水平和高程定位误差。"""
    e = pos - true_pos
    dh = np.linalg.norm(e[:, :2], axis=1)
    dz = e[:, 2]
    d3 = np.linalg.norm(e, axis=1)
    return e, dh, dz, d3


def local_sig(x: np.ndarray, n_win: int, sig_floor: float, name: str) -> np.ndarray:
    """计算因果局部尺度估计。"""
    if n_win <= 0:
        raise ValueError(f"{name} 必须大于 0")
    out = np.zeros_like(x)
    floor = max(float(sig_floor), 1e-6)
    for i in range(x.size):
        s = max(0, i - n_win + 1)
        seg = x[s : i + 1]
        out[i] = max(float(np.sqrt(np.mean(seg**2))), floor)
    return out


def make_var_w(
    inn: np.ndarray,
    n_win: int,
    sig_floor: float,
    w_pow: float,
    w_min: float,
    w_max: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """基于局部尺度估计构造方差型权重。"""
    n_ep, n_bs = inn.shape
    sig = np.zeros((n_ep, n_bs), dtype=np.float64)
    for b in range(n_bs):
        sig[:, b] = local_sig(inn[:, b], n_win, sig_floor, name)

    p = max(float(w_pow), 1e-6)
    w = 1.0 / np.maximum(sig, 1e-8) ** p
    w = w / np.mean(w, axis=1, keepdims=True)
    w = np.clip(w, float(w_min), float(w_max))
    w = w / np.mean(w, axis=1, keepdims=True)
    return w, sig
