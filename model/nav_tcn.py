"""慢变-稀疏 NavTCN。"""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from config import Cfg, NTcnCfg
from model.common_utils import (
    build_seq4,
    calc_pos_err,
    fill_val_curve,
    interp_pos,
    interp_rng,
    local_sig,
    make_mc_cfg,
    sim_obs_geo,
    solve_w_epochs,
)
from model.label_utils import make_pos_q
from model.torch_utils import pick_dev, seed_torch, to_f32
from sim.geometry import GeoData
from sim.positioning import pick_dim
from sim.tracking import ObsData

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


_RAW_LAST_I = 0
_CMC_LAST_I = 4
_VAL_GAP_N = 10


@dataclass
class NTcnPrepData:
    """慢变-稀疏 NavTCN 预处理结果。"""

    rho_true: np.ndarray
    mp_raw: np.ndarray
    rho_hat: np.ndarray
    mp_hat: np.ndarray
    cmc: np.ndarray
    geo_sens: np.ndarray
    seq: np.ndarray
    drho_tar: np.ndarray
    drho_slow_tar: np.ndarray
    sig_tar: np.ndarray
    pos_harm_tar: np.ndarray
    w_tar: np.ndarray


@dataclass
class NTcnTrainData:
    """慢变-稀疏 NavTCN 训练集。"""

    seq: np.ndarray
    corr_tar: np.ndarray
    slow_tar: np.ndarray
    sig_tar: np.ndarray
    pos_harm_tar: np.ndarray
    geo_sens: np.ndarray
    w_tar: np.ndarray
    rho_hat: np.ndarray
    bs: np.ndarray
    true_pos: np.ndarray
    dim: int


@dataclass
class NTcnData:
    """慢变-稀疏 NavTCN 推理输出。"""

    t_ep: np.ndarray
    rho_true: np.ndarray
    mp_raw: np.ndarray
    rho_hat: np.ndarray
    mp_hat: np.ndarray
    rho_fix: np.ndarray
    mp_fix: np.ndarray
    cmc: np.ndarray
    geo_sens: np.ndarray
    seq: np.ndarray
    drho_tar: np.ndarray
    drho_slow_tar: np.ndarray
    sig_tar: np.ndarray
    pos_harm_tar: np.ndarray
    w_tar: np.ndarray
    drho_pred: np.ndarray
    drho_slow_pred: np.ndarray
    slow_gain_pred: np.ndarray
    sig_pred: np.ndarray
    w_pow_ep: np.ndarray
    w_mix_ep: np.ndarray
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
    loss: np.ndarray
    val_d3: np.ndarray
    mc_n: int
    train_sample_n: int


def _interp_rng(geo: GeoData, t_ep: np.ndarray) -> np.ndarray:
    """插值得到历元真值几何距离。"""
    return interp_rng(geo, t_ep)


def _interp_pos(geo: GeoData, t_ep: np.ndarray) -> np.ndarray:
    """插值得到历元真值位置。"""
    return interp_pos(geo, t_ep)


def _norm_cmc_score(cmc: np.ndarray, clip_v: float) -> np.ndarray:
    """把 CMC 归一化为异常强度。"""
    med = np.median(cmc, axis=0, keepdims=True)
    mad = np.median(np.abs(cmc - med), axis=0, keepdims=True)
    scale = 1.4826 * mad + 1e-6
    z = np.clip((cmc - med) / scale, 0.0, float(clip_v))
    return z / max(float(clip_v), 1e-6)


def _build_seq(res: np.ndarray, seq_n: int) -> np.ndarray:
    """构造基础时序特征。"""
    return build_seq4(res, seq_n, "ntcn.seq_n")


def _build_aux_seq(x: np.ndarray, seq_n: int) -> np.ndarray:
    """将辅助标量展开为时序通道。"""
    if seq_n <= 0:
        raise ValueError("ntcn.seq_n 必须大于 0")

    n_ep, n_bs = x.shape
    out = np.zeros((n_ep, n_bs, seq_n, 1), dtype=np.float64)
    for b in range(n_bs):
        for e in range(n_ep):
            s = max(0, e - seq_n + 1)
            seg = x[s : e + 1, b]
            pad_n = seq_n - seg.size
            if pad_n > 0:
                seg = np.concatenate((np.full(pad_n, seg[0]), seg))
            out[e, b, :, 0] = seg
    return out


def _build_aux_seq4(x: np.ndarray, seq_n: int) -> np.ndarray:
    """将多通道辅助特征展开为时序通道。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("辅助特征必须是三维数组")
    out_lst = [_build_aux_seq(x[:, :, i], seq_n) for i in range(x.shape[2])]
    return np.concatenate(out_lst, axis=-1)


def _norm_cmc(cmc: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """把 CMC 归一化为稳定辅助特征。"""
    return _norm_cmc_score(cmc, cfg.cmc_clip)


def _make_feat_seq(res: np.ndarray, cmc: np.ndarray, geo_feat: np.ndarray, seq_n: int, cfg: NTcnCfg) -> np.ndarray:
    """组合残差、CMC 与几何辅助特征。"""
    seq_raw = _build_seq(res, seq_n)
    cmc_seq = _build_aux_seq(_norm_cmc(cmc, cfg), seq_n)
    geo_seq = _build_aux_seq4(geo_feat, seq_n)
    return np.concatenate((seq_raw, cmc_seq, geo_seq), axis=-1)


def _safe_dt(t_ep: np.ndarray) -> np.ndarray:
    """构造稳定历元间隔，避免除零。"""
    dt = np.diff(t_ep, prepend=t_ep[:1]).astype(np.float64)
    pos = dt[dt > 0.0]
    fill = float(np.median(pos)) if pos.size > 0 else 1.0
    dt[dt <= 0.0] = fill
    return dt


def _make_cmc(obs: ObsData) -> np.ndarray:
    """构造码载一致性指标。"""
    dt = _safe_dt(obs.t_ep)[:, None]
    d_code = np.diff(obs.rho_code, axis=0, prepend=obs.rho_code[:1]) / dt
    d_car = np.diff(obs.rho_car, axis=0, prepend=obs.rho_car[:1]) / dt
    return np.abs(d_code - d_car)


def _clip_corr(x: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """修正量限幅。"""
    lim = max(float(cfg.corr_lim), 1e-3)
    return np.clip(np.asarray(x, dtype=np.float64), -lim, lim)


def _clip_corr_torch(x: torch.Tensor, cfg: NTcnCfg) -> torch.Tensor:
    """Torch 修正量限幅。"""
    lim = max(float(cfg.corr_lim), 1e-3)
    return torch.clamp(x, -lim, lim)


def _demean_ep(x: np.ndarray) -> np.ndarray:
    """去掉每历元公共项，避免把公共钟差当成修正量。"""
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x, axis=1, keepdims=True)


def _causal_mean(x: np.ndarray, win_n: int) -> np.ndarray:
    """因果滑窗均值，用于慢变分量标签。"""
    x = np.asarray(x, dtype=np.float64)
    n_ep, n_bs = x.shape
    win_n = max(int(win_n), 1)
    out = np.zeros_like(x)
    for b in range(n_bs):
        cs = np.concatenate(([0.0], np.cumsum(x[:, b], dtype=np.float64)))
        for e in range(n_ep):
            s = max(0, e - win_n + 1)
            out[e, b] = (cs[e + 1] - cs[s]) / float(e - s + 1)
    return out


def _make_corr_tars(corr: np.ndarray, cfg: NTcnCfg) -> tuple[np.ndarray, np.ndarray]:
    """把总修正量分解为总量与慢变主分量。"""
    corr = _clip_corr(corr, cfg)
    slow = _causal_mean(corr, cfg.slow_win_n)
    return corr, slow


def _make_geo_feat(pos: np.ndarray, bs: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """构造当前粗定位对应的几何辅助特征与几何敏感度。"""
    d = pos[:, None, :] - bs[None, :, :]
    r = np.linalg.norm(d, axis=2, keepdims=True)
    r = np.clip(r, 1e-6, None)
    u = d / r
    r_med = np.median(r, axis=1, keepdims=True)
    r_rel = np.clip(r / np.clip(r_med, 1e-6, None), 0.0, 3.0)

    pos_dim = 3 if int(dim) == 3 else 2
    eye = np.eye(pos_dim, dtype=np.float64)
    lev = np.zeros((pos.shape[0], bs.shape[0]), dtype=np.float64)
    pinv_n = np.zeros_like(lev)
    for e in range(pos.shape[0]):
        h = d[e, :, :pos_dim] / r[e]
        a = h.T @ h + 1e-4 * eye
        a_inv = np.linalg.inv(a)
        g = a_inv @ h.T
        lev[e] = np.einsum("bi,ij,bj->b", h, a_inv, h)
        pinv_n[e] = np.linalg.norm(g.T, axis=1)

    lev_rel = np.clip(lev / np.clip(np.mean(lev, axis=1, keepdims=True), 1e-6, None), 0.0, 3.0)
    pinv_rel = np.clip(pinv_n / np.clip(np.mean(pinv_n, axis=1, keepdims=True), 1e-6, None), 0.0, 3.0)
    uz = np.abs(u[:, :, 2])
    geo_sens = np.clip(0.5 * (lev_rel + pinv_rel), 0.0, 3.0)
    feat = np.concatenate((u, r_rel, lev_rel[:, :, None], uz[:, :, None], pinv_rel[:, :, None]), axis=2)
    return feat, geo_sens


def _project_pos_corr(mp_raw: np.ndarray, bs: np.ndarray, pos0: np.ndarray, dim: int) -> np.ndarray:
    """提取会直接投影到定位偏差上的几何敏感分量。"""
    mp_use = _demean_ep(mp_raw)
    out = np.zeros_like(mp_use)
    pos_dim = 3 if int(dim) == 3 else 2
    eye = np.eye(pos_dim, dtype=np.float64)
    for e in range(mp_use.shape[0]):
        d = pos0[e] - bs
        r = np.linalg.norm(d, axis=1, keepdims=True)
        r = np.clip(r, 1e-6, None)
        h_pos = d[:, :pos_dim] / r
        a = h_pos.T @ h_pos + 1e-4 * eye
        b = h_pos.T @ mp_use[e]
        coef = np.linalg.solve(a, b)
        out[e] = h_pos @ coef
    return out


def _make_corr_tar(mp_raw: np.ndarray, mp_hat: np.ndarray, bs: np.ndarray, pos0: np.ndarray, dim: int, cfg: NTcnCfg) -> np.ndarray:
    """构造更贴近定位修正任务的监督目标。"""
    raw_nc = _demean_ep(mp_raw)
    geom = _project_pos_corr(raw_nc, bs, pos0, dim)
    obs = np.where(raw_nc * mp_hat >= 0.0, np.sign(raw_nc) * np.minimum(np.abs(raw_nc), np.abs(mp_hat)), 0.0)
    tar = float(cfg.tar_geom_use) * geom + float(cfg.tar_res_use) * obs
    return _clip_corr(_demean_ep(tar), cfg)


def _corr_dir_proxy(mp_hat: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """用后验残差的慢变趋势给出修正方向代理。"""
    slow = _causal_mean(mp_hat, cfg.slow_win_n)
    scale = np.median(np.abs(mp_hat), axis=1, keepdims=True)
    ref = np.where(np.abs(slow) > 0.15 * np.clip(scale, 1e-6, None), slow, mp_hat)
    sgn = -np.sign(ref)
    return np.where(sgn == 0.0, -np.sign(mp_hat), sgn)


def _safe_corr(corr: np.ndarray, mp_hat: np.ndarray, cmc: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """对预测修正量做导航约束，避免把基线结果拉坏。"""
    corr_mag = np.abs(_clip_corr(_demean_ep(corr), cfg))
    cmc_s = _norm_cmc(cmc, cfg)
    corr = _corr_dir_proxy(mp_hat, cfg) * corr_mag
    cap_k = np.clip(float(cfg.safe_res_use) + float(cfg.safe_cmc_use) * cmc_s, 0.0, 1.25)
    base = np.abs(mp_hat) + float(cfg.safe_med_use) * np.median(np.abs(mp_hat), axis=1, keepdims=True)
    corr = np.sign(corr) * np.minimum(np.abs(corr), cap_k * base)
    return _clip_corr(_demean_ep(corr), cfg)


def _make_sig_tar(mp_raw: np.ndarray, corr_tar: np.ndarray, geo_sens: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """构造修正后剩余不确定度标签。"""
    rem = np.abs(_demean_ep(mp_raw) - corr_tar)
    geo_k = max(float(cfg.geo_sig_use), 0.0)
    if geo_k > 0.0:
        rem = rem * (1.0 + geo_k * np.clip(geo_sens - 1.0, 0.0, None))
    return np.clip(rem, float(cfg.sig_min), float(cfg.sig_max))


def _make_pos_harm_tar(cfg: Cfg, geo: GeoData, rho_hat: np.ndarray, rho_true: np.ndarray, t_ep: np.ndarray) -> np.ndarray:
    """构造逐观测定位危害标签。"""
    q = make_pos_q(cfg, geo, rho_hat, rho_true, t_ep, cfg.ntcn.q_scale)
    harm = 1.0 - np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
    return np.clip(harm, 0.0, 1.0)


def _eff_sig_np(sig: np.ndarray, geo_sens: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """把几何敏感度显式折算到 sigma。"""
    sig = np.clip(np.asarray(sig, dtype=np.float64), float(cfg.sig_min), float(cfg.sig_max))
    geo_sens = np.asarray(geo_sens, dtype=np.float64)
    geo_pen = 1.0 + max(float(cfg.geo_w_use), 0.0) * np.clip(geo_sens - 1.0, 0.0, None)
    return np.clip(sig * geo_pen, float(cfg.sig_min), float(cfg.sig_max))


def _make_w_tar(sig_tar: np.ndarray, pos_harm_tar: np.ndarray, geo_sens: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """构造更贴近定位危害的权重标靶。"""
    harm_pen = 1.0 + max(float(cfg.harm_w_use), 0.0) * np.asarray(pos_harm_tar, dtype=np.float64)
    sig_eff = _eff_sig_np(np.asarray(sig_tar, dtype=np.float64) * harm_pen, geo_sens, cfg)
    w_raw = 1.0 / np.maximum(sig_eff, 1e-6) ** 2
    w_raw = w_raw / np.mean(w_raw, axis=1, keepdims=True)
    mix = _w_mix_np(sig_eff, cfg)
    w = (1.0 - mix) + mix * w_raw
    w = np.clip(w, float(cfg.w_min), float(cfg.w_max))
    return w / np.mean(w, axis=1, keepdims=True)


def _w_mix_np(sig: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """按历元质量自适应融合等权与预测权重。"""
    sig = np.clip(np.asarray(sig, dtype=np.float64), float(cfg.sig_min), float(cfg.sig_max))
    span = max(float(cfg.sig_max) - float(cfg.sig_min), 1e-6)
    sig_mean = np.mean(sig, axis=1, keepdims=True)
    risk = np.clip((sig_mean - float(cfg.sig_min)) / (1.2 * span), 0.0, 1.0)
    mix = float(cfg.w_mix_max) - (float(cfg.w_mix_max) - float(cfg.w_mix_min)) * risk
    return np.clip(mix, float(cfg.w_mix_min), float(cfg.w_mix_max))


def _sig_to_w_np(sig: np.ndarray, geo_sens: np.ndarray, cfg: NTcnCfg) -> tuple[np.ndarray, np.ndarray]:
    """把 sigma 预测映射为定位权重。"""
    sig_eff = _eff_sig_np(sig, geo_sens, cfg)
    sig_rel = sig_eff / np.clip(np.mean(sig_eff, axis=1, keepdims=True), 1e-6, None)
    p = max(float(cfg.w_pow) / max(float(cfg.w_temp), 1e-3), 1e-3)
    w = 1.0 / np.maximum(sig_rel, 1e-6) ** p
    w = w / np.mean(w, axis=1, keepdims=True)
    mix = _w_mix_np(sig_eff, cfg)
    w = (1.0 - mix) + mix * w
    w = np.clip(w, float(cfg.w_min), float(cfg.w_max))
    return w / np.mean(w, axis=1, keepdims=True), mix


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """稳定的 Sigmoid。"""
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _make_post_w(
    sig_pred: np.ndarray,
    res_eq: np.ndarray,
    cmc: np.ndarray,
    geo_sens: np.ndarray,
    cfg: NTcnCfg,
) -> tuple[np.ndarray, np.ndarray]:
    """按固定比例融合 sigma 与后验残差尺度构造权重。"""
    sig_pred = np.asarray(sig_pred, dtype=np.float64)
    res_eq = np.asarray(res_eq, dtype=np.float64)
    n_ep, n_bs = sig_pred.shape
    sig_res = np.zeros_like(sig_pred)
    for b in range(n_bs):
        sig_res[:, b] = local_sig(res_eq[:, b], int(cfg.post_sig_win_n), float(cfg.post_sig_floor), "ntcn.post_sig_win_n")

    _ = (cmc, geo_sens)
    pred_use = max(float(cfg.post_sig_pred_use), 0.0)
    res_use = max(float(cfg.post_sig_res_use), 0.0)
    if pred_use <= 1e-8 and res_use <= 1e-8:
        pred_use = 1.0
    route_res = np.full((n_ep, 1), res_use / max(pred_use + res_use, 1e-8), dtype=np.float64)
    sig = (1.0 - route_res) * sig_pred + route_res * sig_res
    sig = np.clip(sig, float(cfg.post_sig_floor), None)
    p = max(float(cfg.post_w_pow), 1e-6)
    w = 1.0 / sig**p
    w = w / np.mean(w, axis=1, keepdims=True)
    w = np.clip(w, float(cfg.post_w_min), float(cfg.post_w_max))
    return w / np.mean(w, axis=1, keepdims=True), route_res.reshape(n_ep)


def _batch_solve(cfg: Cfg, geo: GeoData, rho: np.ndarray, w: np.ndarray, pos0: np.ndarray, cb0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """带常速度平滑约束的批量定位解算。"""
    bs = np.asarray(geo.bs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n_ep, n_bs = rho.shape
    z_fix = float(cfg.pos.z_fix)
    lam_pos = max(float(cfg.ntcn.batch_lam_pos), 0.0)
    it_n = max(int(cfg.ntcn.batch_it_n), 1)

    x = np.column_stack((np.asarray(pos0[:, 0], dtype=np.float64), np.asarray(pos0[:, 1], dtype=np.float64), np.asarray(cb0, dtype=np.float64)))
    eye = np.eye(3 * n_ep, dtype=np.float64)

    for _ in range(it_n):
        rows = []
        rhs = []
        for e in range(n_ep):
            px, py, cb = x[e]
            p = np.array([px, py, z_fix], dtype=np.float64)
            d = p[None, :] - bs
            r = np.linalg.norm(d, axis=1)
            r = np.clip(r, 1e-6, None)
            v = rho[e] - (r + cb)
            for b in range(n_bs):
                sw = np.sqrt(max(w[e, b], 1e-6))
                h = np.zeros((1, 3 * n_ep), dtype=np.float64)
                h[0, 3 * e + 0] = (d[b, 0] / r[b]) * sw
                h[0, 3 * e + 1] = (d[b, 1] / r[b]) * sw
                h[0, 3 * e + 2] = sw
                rows.append(h)
                rhs.append(v[b] * sw)

        if lam_pos > 0.0:
            s = np.sqrt(lam_pos)
            for e in range(1, n_ep - 1):
                for k in (0, 1):
                    h = np.zeros((1, 3 * n_ep), dtype=np.float64)
                    h[0, 3 * (e - 1) + k] = s
                    h[0, 3 * e + k] = -2.0 * s
                    h[0, 3 * (e + 1) + k] = s
                    g = x[e - 1, k] - 2.0 * x[e, k] + x[e + 1, k]
                    rows.append(h)
                    rhs.append((-g) * s)

        H = np.concatenate(rows, axis=0)
        y = np.asarray(rhs, dtype=np.float64)
        dx = np.linalg.solve(H.T @ H + 1e-6 * eye, H.T @ y).reshape(n_ep, 3)
        x = x + dx
        if np.max(np.abs(dx)) < 1e-4:
            break

    pos = np.column_stack((x[:, 0], x[:, 1], np.full(n_ep, z_fix, dtype=np.float64)))
    cb = x[:, 2]
    d = pos[:, None, :] - bs[None, :, :]
    r = np.linalg.norm(d, axis=2)
    res = rho - (r + cb[:, None])
    it = np.full(n_ep, it_n, dtype=np.int64)
    return pos, cb, res, pick_dim(cfg.pos, bs), it


def _w_mix_torch(sig: torch.Tensor, cfg: NTcnCfg) -> torch.Tensor:
    """Torch 版自适应权重融合系数。"""
    sig = torch.clamp(sig, float(cfg.sig_min), float(cfg.sig_max))
    span = max(float(cfg.sig_max) - float(cfg.sig_min), 1e-6)
    sig_mean = torch.mean(sig, dim=2, keepdim=True)
    risk = torch.clamp((sig_mean - float(cfg.sig_min)) / (1.2 * span), 0.0, 1.0)
    return torch.clamp(float(cfg.w_mix_max) - (float(cfg.w_mix_max) - float(cfg.w_mix_min)) * risk, float(cfg.w_mix_min), float(cfg.w_mix_max))


def _eff_sig_torch(sig: torch.Tensor, geo_sens: torch.Tensor, cfg: NTcnCfg) -> torch.Tensor:
    """Torch 版几何敏感 sigma。"""
    sig = torch.clamp(sig, float(cfg.sig_min), float(cfg.sig_max))
    geo_pen = 1.0 + max(float(cfg.geo_w_use), 0.0) * torch.clamp(geo_sens - 1.0, min=0.0)
    return torch.clamp(sig * geo_pen, float(cfg.sig_min), float(cfg.sig_max))


def _sig_to_w_torch(sig: torch.Tensor, geo_sens: torch.Tensor, cfg: NTcnCfg) -> torch.Tensor:
    """Torch 版 sigma 到权重映射。"""
    sig_eff = _eff_sig_torch(sig, geo_sens, cfg)
    sig_rel = sig_eff / torch.clamp(torch.mean(sig_eff, dim=2, keepdim=True), min=1e-6)
    p = max(float(cfg.w_pow) / max(float(cfg.w_temp), 1e-3), 1e-3)
    w = 1.0 / torch.clamp(sig_rel, min=1e-6) ** p
    w = w / torch.clamp(torch.mean(w, dim=2, keepdim=True), min=1e-6)
    mix = _w_mix_torch(sig_eff, cfg)
    w = (1.0 - mix) + mix * w
    w = torch.clamp(w, float(cfg.w_min), float(cfg.w_max))
    return w / torch.clamp(torch.mean(w, dim=2, keepdim=True), min=1e-6)


def _prep_data(obs: ObsData, geo: GeoData, cfg: Cfg) -> NTcnPrepData:
    """构造慢变-稀疏 NavTCN 输入与监督。"""
    rho_true = _interp_rng(geo, obs.t_ep)
    mp_raw = obs.rho_code - rho_true
    cmc = _make_cmc(obs)
    rho_hat = obs.rho_code.copy()
    pos0, _, mp_hat, dim0, _ = _solve_eq(cfg, geo, rho_hat)
    geo_feat, geo_sens = _make_geo_feat(pos0, geo.bs, dim0)
    seq = _make_feat_seq(mp_hat, cmc, geo_feat, cfg.ntcn.seq_n, cfg.ntcn)
    corr_tar = _make_corr_tar(mp_raw, mp_hat, geo.bs, pos0, dim0, cfg.ntcn)
    pos_harm_tar = _make_pos_harm_tar(cfg, geo, rho_hat, rho_true, obs.t_ep)
    drho_tar, drho_slow_tar = _make_corr_tars(corr_tar, cfg.ntcn)
    sig_tar = _make_sig_tar(mp_raw, drho_tar, geo_sens, cfg.ntcn)
    if cfg.ntcn.pos_harm_use:
        sig_tar = np.clip(sig_tar * (1.0 + float(cfg.ntcn.pos_harm_sig_k) * pos_harm_tar), float(cfg.ntcn.sig_min), float(cfg.ntcn.sig_max))
    w_tar = _make_w_tar(sig_tar, pos_harm_tar, geo_sens, cfg.ntcn)
    return NTcnPrepData(
        rho_true=rho_true,
        mp_raw=mp_raw,
        rho_hat=rho_hat,
        mp_hat=mp_hat,
        cmc=cmc,
        geo_sens=geo_sens,
        seq=seq,
        drho_tar=drho_tar,
        drho_slow_tar=drho_slow_tar,
        sig_tar=sig_tar,
        pos_harm_tar=pos_harm_tar,
        w_tar=w_tar,
    )


def _make_ntcn_mc_cfg(cfg: Cfg, idx: int) -> Cfg:
    """构造更贴近主场景、并含少量难例的 NavTCN 训练场景。"""
    cfg_i = make_mc_cfg(cfg, cfg.ntcn, idx)
    if idx >= int(cfg.ntcn.mc_n) and int(cfg.ntcn.stealth_add_n) > 0:
        rng = np.random.default_rng(int(cfg.ntcn.seed) + (idx + 1) * 6151)
        dx = float(cfg_i.geo.rx_x0 - cfg_i.geo.bs_cx)
        dy = float(cfg_i.geo.rx_y0 - cfg_i.geo.bs_cy)
        scale = max(float(cfg.ntcn.stealth_rx_scale), 1.0)
        geo = replace(
            cfg_i.geo,
            rx_x0=float(cfg_i.geo.bs_cx + dx * scale + rng.normal(0.0, 0.35 * float(cfg.ntcn.mc_rx_xy_jit))),
            rx_y0=float(cfg_i.geo.bs_cy + dy * scale + rng.normal(0.0, 0.35 * float(cfg.ntcn.mc_rx_xy_jit))),
            rx_vx=float(cfg_i.geo.rx_vx) * float(cfg.ntcn.stealth_v_use),
            rx_vy=float(cfg_i.geo.rx_vy) * float(cfg.ntcn.stealth_v_use),
        )
        mp = replace(
            cfg_i.mp,
            k0=float(np.clip(float(cfg_i.mp.k0) * float(cfg.ntcn.stealth_k0_use), 0.05, 0.98)),
        )
        cfg_i = replace(cfg_i, geo=geo, mp=mp)
    if not bool(cfg.ntcn.hard_mc_use):
        return cfg_i

    frac = float(np.clip(cfg.ntcn.hard_frac, 0.0, 1.0))
    if frac <= 0.0:
        return cfg_i

    rng = np.random.default_rng(int(cfg.ntcn.seed) + (idx + 1) * 4099)
    if float(rng.random()) > frac:
        return cfg_i

    mp = replace(
        cfg_i.mp,
        n=max(1, int(cfg_i.mp.n) + int(cfg.ntcn.hard_mp_add)),
        k0=min(float(cfg_i.mp.k0) + float(cfg.ntcn.hard_k0_add), 0.98),
        jit=float(cfg_i.mp.jit) * (1.0 + float(cfg.ntcn.hard_jit_mul)),
        fade=min(float(cfg_i.mp.fade) * (1.0 + float(cfg.ntcn.hard_fade_mul)), 0.98),
        burst_n=max(0, int(cfg_i.mp.burst_n) + int(cfg.ntcn.hard_burst_add)),
    )
    rx = replace(
        cfg_i.rx,
        snr_db=float(np.clip(float(cfg_i.rx.snr_db) - float(cfg.ntcn.hard_snr_drop), 5.0, 40.0)),
        blk_n=max(0, int(cfg_i.rx.blk_n) + int(cfg.ntcn.hard_blk_add)),
        imp_n=max(0, int(cfg_i.rx.imp_n) + int(cfg.ntcn.hard_imp_add)),
    )
    return replace(cfg_i, mp=mp, rx=rx)


def _stealth_score(mp_hat: np.ndarray, pos_harm_tar: np.ndarray, geo_sens: np.ndarray, cfg: NTcnCfg) -> float:
    """给低残差但高定位危害场景打分。"""
    harm = np.asarray(pos_harm_tar, dtype=np.float64)
    geo = np.asarray(geo_sens, dtype=np.float64)
    res = np.asarray(mp_hat, dtype=np.float64)
    geo_k = max(float(cfg.stealth_geo_k), 0.0)
    harm_eff = harm * (1.0 + geo_k * np.clip(geo - 1.0, 0.0, None))
    harm_mean = float(np.mean(harm_eff))
    res_rms = float(np.sqrt(np.mean(res**2)))
    return harm_mean / max(res_rms, 1e-6)


def _build_mc_train(cfg: Cfg) -> NTcnTrainData:
    """构造带几何信息的 Monte Carlo 训练集。"""
    if cfg.ntcn.mc_n <= 0:
        raise ValueError("ntcn.mc_n 必须大于 0")

    seq_lst = []
    corr_lst = []
    slow_lst = []
    sig_lst = []
    harm_lst = []
    geo_sens_lst = []
    w_tar_lst = []
    rho_lst = []
    bs_lst = []
    pos_lst = []
    score_lst = []
    n_ep_ref = None
    bs_n_ref = None
    dim_ref = None

    pool_n = int(cfg.ntcn.mc_n) + max(int(cfg.ntcn.stealth_add_n), 0)
    for i in range(pool_n):
        cfg_i = _make_ntcn_mc_cfg(cfg, i)
        obs_i, geo_i = sim_obs_geo(cfg_i)
        prep_i = _prep_data(obs_i, geo_i, cfg_i)
        n_ep_i, bs_n_i = prep_i.mp_hat.shape
        dim_i = pick_dim(cfg_i.pos, geo_i.bs)

        if n_ep_ref is None:
            n_ep_ref = n_ep_i
            bs_n_ref = bs_n_i
            dim_ref = dim_i
        if n_ep_i != n_ep_ref or bs_n_i != bs_n_ref:
            raise ValueError("Monte Carlo 训练集中历元数或基站数不一致")
        if dim_i != dim_ref:
            raise ValueError("Monte Carlo 训练集中定位维数不一致")

        seq_lst.append(prep_i.seq)
        corr_lst.append(prep_i.drho_tar)
        slow_lst.append(prep_i.drho_slow_tar)
        sig_lst.append(prep_i.sig_tar)
        harm_lst.append(prep_i.pos_harm_tar)
        geo_sens_lst.append(prep_i.geo_sens)
        w_tar_lst.append(prep_i.w_tar)
        rho_lst.append(prep_i.rho_hat)
        bs_lst.append(geo_i.bs)
        pos_lst.append(_interp_pos(geo_i, obs_i.t_ep))
        score_lst.append(_stealth_score(prep_i.mp_hat, prep_i.pos_harm_tar, prep_i.geo_sens, cfg.ntcn))

    if pool_n > int(cfg.ntcn.mc_n) and int(cfg.ntcn.stealth_keep_n) > 0:
        base_n = int(cfg.ntcn.mc_n)
        extra_idx = np.arange(base_n, pool_n, dtype=np.int64)
        take_n = min(int(cfg.ntcn.stealth_keep_n), extra_idx.size)
        score = np.asarray(score_lst, dtype=np.float64)
        add_idx = extra_idx[np.argsort(score[extra_idx])[-take_n:]] if take_n > 0 else np.empty(0, dtype=np.int64)
        keep_idx = np.concatenate((np.arange(base_n, dtype=np.int64), add_idx), axis=0)
    else:
        keep_idx = np.arange(int(cfg.ntcn.mc_n), dtype=np.int64)

    seq_lst = [seq_lst[i] for i in keep_idx]
    corr_lst = [corr_lst[i] for i in keep_idx]
    slow_lst = [slow_lst[i] for i in keep_idx]
    sig_lst = [sig_lst[i] for i in keep_idx]
    harm_lst = [harm_lst[i] for i in keep_idx]
    geo_sens_lst = [geo_sens_lst[i] for i in keep_idx]
    w_tar_lst = [w_tar_lst[i] for i in keep_idx]
    rho_lst = [rho_lst[i] for i in keep_idx]
    bs_lst = [bs_lst[i] for i in keep_idx]
    pos_lst = [pos_lst[i] for i in keep_idx]

    return NTcnTrainData(
        seq=np.stack(seq_lst, axis=0).astype(np.float64),
        corr_tar=np.stack(corr_lst, axis=0).astype(np.float64),
        slow_tar=np.stack(slow_lst, axis=0).astype(np.float64),
        sig_tar=np.stack(sig_lst, axis=0).astype(np.float64),
        pos_harm_tar=np.stack(harm_lst, axis=0).astype(np.float64),
        geo_sens=np.stack(geo_sens_lst, axis=0).astype(np.float64),
        w_tar=np.stack(w_tar_lst, axis=0).astype(np.float64),
        rho_hat=np.stack(rho_lst, axis=0).astype(np.float64),
        bs=np.stack(bs_lst, axis=0).astype(np.float64),
        true_pos=np.stack(pos_lst, axis=0).astype(np.float64),
        dim=int(dim_ref),
    )


def _build_val_case(cfg: Cfg, idx_add: int = 17) -> SimpleNamespace:
    """为 NavTCN 构造固定验证场景。"""
    idx = int(cfg.ntcn.mc_n) + int(idx_add)
    cfg_i = _make_ntcn_mc_cfg(cfg, idx)
    obs_i, geo_i = sim_obs_geo(cfg_i)
    prep_i = _prep_data(obs_i, geo_i, cfg_i)
    true_pos = _interp_pos(geo_i, obs_i.t_ep)
    return SimpleNamespace(cfg=cfg_i, obs=obs_i, geo=geo_i, prep=prep_i, true_pos=true_pos)


def _build_eval_case(cfg: Cfg, obs: ObsData, geo: GeoData, prep: NTcnPrepData) -> SimpleNamespace:
    """为 NavTCN 构造当前正式场景的按 epoch 测试包。"""
    true_pos = _interp_pos(geo, obs.t_ep)
    return SimpleNamespace(cfg=cfg, obs=obs, geo=geo, prep=prep, true_pos=true_pos)


class Chomp1d(nn.Module):
    """裁剪卷积右侧填充，保持因果性。"""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = int(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n <= 0:
            return x
        return x[:, :, :-self.n]


class MsBlock(nn.Module):
    """多尺度残差块。"""

    def __init__(self, in_n: int, out_n: int, ker_lst: tuple[int, ...], dil: int) -> None:
        super().__init__()
        if len(ker_lst) == 0:
            raise ValueError("ntcn.ker_lst 不能为空")

        self.branches = nn.ModuleList()
        for ker in ker_lst:
            pad = (int(ker) - 1) * int(dil)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_n, out_n, int(ker), padding=pad, dilation=int(dil)),
                    Chomp1d(pad),
                    nn.ReLU(),
                    nn.Conv1d(out_n, out_n, int(ker), padding=pad, dilation=int(dil)),
                    Chomp1d(pad),
                    nn.ReLU(),
                )
            )
        self.fuse = nn.Conv1d(out_n * len(self.branches), out_n, 1)
        self.proj = nn.Conv1d(in_n, out_n, 1) if in_n != out_n else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([m(x) for m in self.branches], dim=1)
        y = self.fuse(y)
        z = self.proj(x)
        return F.relu(y + z)


class NTcnNet(nn.Module):
    """慢变-稀疏 NavTCN 主干。"""

    def __init__(
        self,
        feat_n: int,
        hid_n: int,
        layer_n: int,
        ker_lst: tuple[int, ...],
        corr_lim: float,
        gain_min: float,
        gain_max: float,
        sig_min: float,
        sig_max: float,
        use_log_sig: bool,
        spa_use: bool,
        spa_gain: float,
        slow_gain_cmc_k: float,
        slow_gain_geo_k: float,
        slow_gain_scene_k: float,
    ) -> None:
        super().__init__()
        mods = []
        c_in = int(feat_n)
        for i in range(int(layer_n)):
            mods.append(MsBlock(c_in, int(hid_n), ker_lst, dil=2**i))
            c_in = int(hid_n)
        self.net = nn.Sequential(*mods)
        self.corr_proj = nn.Sequential(nn.Linear(int(hid_n), int(hid_n)), nn.ReLU())
        self.sig_proj = nn.Sequential(
            nn.Linear(int(hid_n) + int(feat_n), int(hid_n)),
            nn.ReLU(),
            nn.Linear(int(hid_n), int(hid_n)),
            nn.ReLU(),
        )
        self.slow_head = nn.Linear(int(hid_n), 1)
        self.slow_gain_head = nn.Linear(int(hid_n), 1)
        self.sigma_head = nn.Linear(int(hid_n), 1)
        self.spa_use = bool(spa_use)
        self.spa_gain = float(spa_gain)
        if self.spa_use:
            attn_n = max(int(hid_n) // 2, 4)
            self.scene_fc1 = nn.Linear(int(hid_n), attn_n)
            self.scene_fc2 = nn.Linear(attn_n, int(hid_n))
            self.bs_gate = nn.Linear(int(hid_n) * 2, 1)
        self.corr_lim = float(corr_lim)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.sig_min = float(sig_min)
        self.sig_max = float(sig_max)
        self.use_log_sig = bool(use_log_sig)
        self.slow_gain_cmc_k = float(slow_gain_cmc_k)
        self.slow_gain_geo_k = float(slow_gain_geo_k)
        self.slow_gain_scene_k = float(slow_gain_scene_k)
        self.log_sig_min = float(np.log(max(sig_min, 1e-6)))
        self.log_sig_max = float(np.log(max(sig_max, 1e-6)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, bs_n, t_n, feat_n = x.shape
        y = x.reshape(b * bs_n, t_n, feat_n).transpose(1, 2)
        y = self.net(y)
        y = y[:, :, -1].reshape(b, bs_n, -1)
        if self.spa_use:
            scene = torch.mean(y, dim=1, keepdim=True)
            scene_w = torch.sigmoid(self.scene_fc2(F.relu(self.scene_fc1(scene))))
            bs_w = torch.sigmoid(self.bs_gate(torch.cat((y, scene.expand(-1, bs_n, -1)), dim=2)))
            y = y * (1.0 + self.spa_gain * scene_w) * (1.0 - 0.5 * self.spa_gain + self.spa_gain * bs_w)
        corr_y = self.corr_proj(y)
        sig_y = self.sig_proj(torch.cat((y.detach(), x[:, :, -1, :]), dim=2))
        slow = self.corr_lim * torch.tanh(self.slow_head(corr_y).squeeze(-1))
        gain_span = self.gain_max - self.gain_min
        gain_logit = self.slow_gain_head(corr_y).squeeze(-1)
        cmc_last = torch.clamp(x[:, :, -1, _CMC_LAST_I], 0.0, 1.0)
        if feat_n > (_CMC_LAST_I + 1):
            geo_last = torch.mean(torch.clamp(x[:, :, -1, _CMC_LAST_I + 1 :] - 1.0, min=0.0), dim=2)
        else:
            geo_last = torch.zeros_like(cmc_last)
        scene_risk = torch.mean(cmc_last + geo_last, dim=1, keepdim=True)
        gain_logit = (
            gain_logit
            - self.slow_gain_cmc_k * cmc_last
            - self.slow_gain_geo_k * geo_last
            - self.slow_gain_scene_k * scene_risk
        )
        slow_gain = self.gain_min + gain_span * torch.sigmoid(gain_logit)
        if self.use_log_sig:
            log_sig = self.log_sig_min + (self.log_sig_max - self.log_sig_min) * torch.sigmoid(self.sigma_head(sig_y).squeeze(-1))
            sigma = torch.exp(log_sig)
        else:
            sigma = self.sig_min + (self.sig_max - self.sig_min) * torch.sigmoid(self.sigma_head(sig_y).squeeze(-1))
        return slow, slow_gain, sigma


def _init_model(model: "nn.Module") -> None:
    """更稳的网络初始化。"""
    for mod in model.modules():
        if isinstance(mod, nn.Conv1d):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    if hasattr(model, "slow_head"):
        nn.init.normal_(model.slow_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(model.slow_head.bias)
    if hasattr(model, "slow_gain_head"):
        nn.init.normal_(model.slow_gain_head.weight, mean=0.0, std=5e-4)
        nn.init.zeros_(model.slow_gain_head.bias)
    if hasattr(model, "sigma_head"):
        nn.init.normal_(model.sigma_head.weight, mean=0.0, std=5e-4)
        nn.init.constant_(model.sigma_head.bias, -1.0)
def _build_model(seq: np.ndarray, cfg: NTcnCfg) -> "nn.Module":
    """构造慢变-稀疏 NavTCN 模型。"""
    feat_n = int(seq.shape[-1])
    model = NTcnNet(
        feat_n=feat_n,
        hid_n=cfg.hid_n,
        layer_n=cfg.layer_n,
        ker_lst=cfg.ker_lst,
        corr_lim=cfg.corr_lim,
        gain_min=cfg.gain_min,
        gain_max=cfg.gain_max,
        sig_min=cfg.sig_min,
        sig_max=cfg.sig_max,
        use_log_sig=cfg.sig_nll_use,
        spa_use=cfg.spa_use,
        spa_gain=cfg.spa_gain,
        slow_gain_cmc_k=cfg.slow_gain_cmc_k,
        slow_gain_geo_k=cfg.slow_gain_geo_k,
        slow_gain_scene_k=cfg.slow_gain_scene_k,
    )
    _init_model(model)
    return model


def _corr_total_loss(
    corr_hat: torch.Tensor,
    corr_tar: torch.Tensor,
    pos_harm_tar: torch.Tensor,
    cfg: NTcnCfg,
) -> torch.Tensor:
    """总修正损失，突出异常历元。"""
    lim = max(float(cfg.corr_lim), 1e-3)
    wei = 1.0 + torch.clamp(torch.abs(corr_tar) / lim, 0.0, 1.0)
    if cfg.pos_harm_use:
        wei = wei + float(cfg.pos_harm_corr_k) * pos_harm_tar
    loss = F.smooth_l1_loss(corr_hat, corr_tar, reduction="none")
    return torch.mean(wei * loss)


def _sign_loss(corr_hat: torch.Tensor, corr_tar: torch.Tensor, cfg: NTcnCfg) -> torch.Tensor:
    """约束总修正量方向与目标一致。"""
    lim = max(float(cfg.corr_lim), 1e-3)
    y = torch.sign(corr_tar)
    y = torch.where(y == 0.0, torch.ones_like(y), y)
    mask = (torch.abs(corr_tar) > 0.05 * lim).to(corr_hat.dtype)
    pred = torch.tanh(corr_hat / lim)
    loss = 0.5 * (1.0 - pred * y)
    den = torch.clamp(torch.sum(mask), min=1.0)
    return torch.sum(mask * loss) / den


def _slow_loss(slow_hat: torch.Tensor, slow_tar: torch.Tensor) -> torch.Tensor:
    """慢变分量损失。"""
    return F.smooth_l1_loss(slow_hat, slow_tar)

def _mix_slow_torch(slow_raw: torch.Tensor, slow_gain: torch.Tensor, cfg: NTcnCfg) -> torch.Tensor:
    """生成自适应慢变修正分量。"""
    return _clip_corr_torch(float(cfg.slow_use) * slow_gain * slow_raw, cfg)

def _mix_slow_np(slow_raw: np.ndarray, slow_gain: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """Numpy 版自适应慢变修正分量。"""
    return _clip_corr(float(cfg.slow_use) * np.asarray(slow_gain, dtype=np.float64) * np.asarray(slow_raw, dtype=np.float64), cfg)


def _rule_slow_gain(seq: np.ndarray, cfg: NTcnCfg) -> np.ndarray:
    """无 Torch 时按场景风险构造慢变分支自适应增益。"""
    cmc_last = np.clip(np.asarray(seq[:, :, -1, _CMC_LAST_I], dtype=np.float64), 0.0, 1.0)
    feat_n = int(seq.shape[-1])
    if feat_n > (_CMC_LAST_I + 1):
        geo_last = np.mean(np.clip(np.asarray(seq[:, :, -1, _CMC_LAST_I + 1 :], dtype=np.float64) - 1.0, 0.0, None), axis=2)
    else:
        geo_last = np.zeros_like(cmc_last)
    scene_risk = np.mean(cmc_last + geo_last, axis=1, keepdims=True)
    gain_score = -float(cfg.slow_gain_cmc_k) * cmc_last - float(cfg.slow_gain_geo_k) * geo_last - float(cfg.slow_gain_scene_k) * scene_risk
    gain_span = float(cfg.gain_max) - float(cfg.gain_min)
    return float(cfg.gain_min) + gain_span * _sigmoid_np(gain_score)


def _sig_nll_loss(corr_hat: torch.Tensor, corr_tar: torch.Tensor, sig_hat: torch.Tensor, cfg: NTcnCfg) -> torch.Tensor:
    """用高斯 NLL 口径约束 sigma 校准。"""
    lim = max(float(cfg.corr_lim), 1e-3)
    wei = 1.0 + torch.clamp(torch.abs(corr_tar) / lim, 0.0, 1.0)
    sig = torch.clamp(sig_hat, min=float(cfg.sig_min), max=float(cfg.sig_max))
    err = (corr_tar - corr_hat).detach()
    nll = 0.5 * (err / sig) ** 2 + torch.log(sig)
    return torch.mean(wei * nll)


def _sig_regress_loss(sig_hat: torch.Tensor, sig_tar: torch.Tensor) -> torch.Tensor:
    """旧版 sigma 监督损失。"""
    return F.smooth_l1_loss(sig_hat, sig_tar)


def _sig_reg_loss(sig_hat: torch.Tensor) -> torch.Tensor:
    """限制 sigma 过大或过小。"""
    return torch.mean(torch.abs(torch.log(torch.clamp(sig_hat, min=1e-4))))


def _w_cal_loss(w_hat: torch.Tensor, w_tar: torch.Tensor) -> torch.Tensor:
    """约束 sigma 到权重的映射保持可校准。"""
    return F.smooth_l1_loss(torch.log(torch.clamp(w_hat, min=1e-4)), torch.log(torch.clamp(w_tar, min=1e-4)))


def _time_loss(slow_hat: torch.Tensor, slow_tar: torch.Tensor) -> torch.Tensor:
    """慢变分量时间一致性损失。"""
    if slow_hat.shape[1] <= 1:
        return slow_hat.new_tensor(0.0)
    d_hat = slow_hat[:, 1:, :] - slow_hat[:, :-1, :]
    d_tar = slow_tar[:, 1:, :] - slow_tar[:, :-1, :]
    return F.smooth_l1_loss(d_hat, d_tar)


def _pos_scale(ep_idx: int, cfg: NTcnCfg) -> float:
    """定位一致性两阶段开关。"""
    stage1_n = max(int(cfg.pos_stage1_n), 0)
    ramp_n = max(int(cfg.pos_ramp_n), 0)
    if ep_idx < stage1_n:
        return 0.0
    if ramp_n <= 0:
        return 1.0
    frac = min((ep_idx - stage1_n) / float(ramp_n), 1.0)
    return max(frac, 0.0)


def _joint_ep_idx(ep_idx: int, cfg: NTcnCfg) -> int:
    """联合微调阶段内的相对 epoch。"""
    return max(int(ep_idx) - max(int(cfg.pretrain_n), 0), 0)


def _solve_eq_torch(
    bs: torch.Tensor,
    rho: torch.Tensor,
    dim: int,
    z_fix: float,
    z_init: float,
    it_n: int,
    w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """可微等权定位解算。"""
    b_n = int(rho.shape[0])
    dev = rho.device
    dtype = rho.dtype
    if w is None:
        w = torch.ones_like(rho)
    w = torch.clamp(w, min=1e-4)

    if dim == 3:
        x = torch.zeros((b_n, 4), dtype=dtype, device=dev)
        x[:, :3] = torch.mean(bs, dim=1)
        x[:, 2] = float(z_init)
        eye = torch.eye(4, dtype=dtype, device=dev).unsqueeze(0)
        for _ in range(int(it_n)):
            p = x[:, :3]
            d = p[:, None, :] - bs
            r = torch.linalg.norm(d, dim=2).clamp_min(1e-6)
            v = rho - (r + x[:, 3:4])
            h = torch.cat((d / r.unsqueeze(-1), torch.ones((b_n, bs.shape[1], 1), dtype=dtype, device=dev)), dim=2)
            hw = h * w.unsqueeze(-1)
            a = torch.matmul(h.transpose(1, 2), hw) + 1e-4 * eye
            b = torch.matmul(h.transpose(1, 2), (w * v).unsqueeze(-1))
            dx = torch.linalg.solve(a, b).squeeze(-1)
            x = x + dx
        pos = x[:, :3]
        cb = x[:, 3]
    else:
        x = torch.zeros((b_n, 3), dtype=dtype, device=dev)
        x[:, :2] = torch.mean(bs[:, :, :2], dim=1)
        eye = torch.eye(3, dtype=dtype, device=dev).unsqueeze(0)
        z_col = torch.full((b_n,), float(z_fix), dtype=dtype, device=dev)
        for _ in range(int(it_n)):
            p = torch.stack((x[:, 0], x[:, 1], z_col), dim=1)
            d = p[:, None, :] - bs
            r = torch.linalg.norm(d, dim=2).clamp_min(1e-6)
            v = rho - (r + x[:, 2:3])
            h = torch.stack((d[:, :, 0] / r, d[:, :, 1] / r, torch.ones_like(r)), dim=2)
            hw = h * w.unsqueeze(-1)
            a = torch.matmul(h.transpose(1, 2), hw) + 1e-4 * eye
            b = torch.matmul(h.transpose(1, 2), (w * v).unsqueeze(-1))
            dx = torch.linalg.solve(a, b).squeeze(-1)
            x = x + dx
        pos = torch.stack((x[:, 0], x[:, 1], z_col), dim=1)
        cb = x[:, 2]

    d = pos[:, None, :] - bs
    r = torch.linalg.norm(d, dim=2).clamp_min(1e-6)
    res = rho - (r + cb[:, None])
    return pos, cb, res


def _geom_res_loss(train: NTcnTrainData, corr_hat: torch.Tensor, w_hat: torch.Tensor, cfg: Cfg, dev: "torch.device") -> torch.Tensor:
    """几何残差一致性损失。"""
    mc_n, n_ep, n_bs = corr_hat.shape
    rho_hat = to_f32(train.rho_hat, dev)
    bs = to_f32(train.bs, dev)

    rho_fix = rho_hat - corr_hat
    bs_ep = bs[:, None, :, :].expand(-1, n_ep, -1, -1).reshape(mc_n * n_ep, n_bs, 3)
    rho_ep = rho_fix.reshape(mc_n * n_ep, n_bs)
    w_ep = w_hat.reshape(mc_n * n_ep, n_bs)
    pos_hat, _, res_hat = _solve_eq_torch(bs_ep, rho_ep, train.dim, cfg.pos.z_fix, cfg.pos.z_init, cfg.ntcn.gn_iter_n, w_ep)
    _ = pos_hat
    return F.smooth_l1_loss(torch.sqrt(torch.clamp(w_ep, min=1e-4)) * res_hat, torch.zeros_like(res_hat))


def _run_case(
    cfg: Cfg,
    geo: GeoData,
    prep: NTcnPrepData,
    corr_pred: np.ndarray,
    slow_pred: np.ndarray,
    sig_pred: np.ndarray,
) -> SimpleNamespace:
    """把预测修正量送入定位链路。"""
    n_ep, n_bs = prep.mp_hat.shape
    corr_pred = _safe_corr(corr_pred, prep.mp_hat, prep.cmc, cfg.ntcn)
    corr_pred = _clip_corr(float(cfg.ntcn.corr_gain_use) * corr_pred, cfg.ntcn)
    slow_pred = _clip_corr(slow_pred, cfg.ntcn)
    sig_pred = np.clip(sig_pred, float(cfg.ntcn.sig_min), float(cfg.ntcn.sig_max)).reshape(n_ep, n_bs)
    rho_fix = prep.rho_hat - corr_pred
    mp_fix = rho_fix - prep.rho_true
    pos_eq, cb_eq, res_eq, dim, it_eq = solve_w_epochs(cfg, geo, rho_fix, np.ones_like(rho_fix, dtype=np.float64))
    _ = pos_eq, cb_eq, it_eq
    w, w_mix_ep = _make_post_w(sig_pred, res_eq, prep.cmc, prep.geo_sens, cfg.ntcn)
    if bool(cfg.ntcn.batch_use):
        pos0, cb0, _, _, _ = solve_w_epochs(cfg, geo, rho_fix, w)
        pos, cb, res, dim, it = _batch_solve(cfg, geo, rho_fix, w, pos0, cb0)
    else:
        pos, cb, res, dim, it = solve_w_epochs(cfg, geo, rho_fix, w)
    return SimpleNamespace(
        corr_pred=corr_pred,
        slow_pred=slow_pred,
        sig_pred=sig_pred,
        rho_fix=rho_fix,
        mp_fix=mp_fix,
        w=w,
        w_mix_ep=w_mix_ep,
        pos=pos,
        cb=cb,
        res=res,
        dim=dim,
        it=it,
    )


def _eval_val_d3(model: nn.Module, eval_pack: Any, cfg: Cfg, dev_cfg: Any) -> float:
    """计算当前正式场景上的按 epoch 测试误差。"""
    corr_pred, slow_pred, _, sig_pred = _pred_torch(model, eval_pack.prep.seq, cfg.ntcn, dev_cfg)
    out = _run_case(eval_pack.cfg, eval_pack.geo, eval_pack.prep, corr_pred, slow_pred, sig_pred)
    _, _, _, d3 = calc_pos_err(out.pos, eval_pack.true_pos)
    return float(np.sqrt(np.mean(d3**2)))


def _train_torch(
    train: NTcnTrainData,
    cfg: NTcnCfg,
    dev_cfg: Any,
    pos_cfg: Any,
    full_cfg: Cfg,
    val_pack: Any | None = None,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """训练慢变-稀疏 NavTCN。"""
    if torch is None or nn is None or F is None:
        raise RuntimeError("当前环境缺少 PyTorch，无法训练 NavTCN")

    dev = pick_dev(dev_cfg, "ntcn")
    seed_torch(cfg.seed, dev)
    model = _build_model(train.seq[0], cfg).to(dev)

    mc_n, n_ep, n_bs, seq_n, feat_n = train.seq.shape
    x_all = to_f32(train.seq, dev)
    y_corr = to_f32(train.corr_tar, dev)
    y_slow = to_f32(train.slow_tar, dev)
    y_sig = to_f32(train.sig_tar, dev)
    y_harm = to_f32(train.pos_harm_tar, dev)
    y_w = to_f32(train.w_tar, dev)
    y_geo = to_f32(train.geo_sens, dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    ema_lst: list[torch.Tensor] | None = None
    if bool(cfg.ema_use):
        ema_lst = [p.detach().clone() for p in model.parameters()]

    val_n = 0
    if bool(cfg.best_use) and mc_n >= 4 and float(cfg.val_frac) > 0.0:
        val_n = min(max(int(round(mc_n * float(cfg.val_frac))), 1), mc_n - 1)
    tr_n = mc_n - val_n

    loss_hist = []
    model.train()
    wrapper = type("_TrainCfg", (), {"pos": pos_cfg, "ntcn": cfg})
    pre_n = max(int(cfg.pretrain_n), 0)
    best_state: dict[str, torch.Tensor] | None = None
    best_score: float | None = None
    val_hist = np.full(int(cfg.train_n), np.nan, dtype=np.float64)

    def _loss_core(
        x_mc: torch.Tensor,
        y_corr_mc: torch.Tensor,
        y_slow_mc: torch.Tensor,
        y_sig_mc: torch.Tensor,
        y_harm_mc: torch.Tensor,
        y_w_mc: torch.Tensor,
        y_geo_mc: torch.Tensor,
        rho_hat_mc: np.ndarray,
        bs_mc: np.ndarray,
        ep_idx: int,
    ) -> torch.Tensor:
        mc_b = int(x_mc.shape[0])
        x = x_mc.reshape(mc_b * n_ep, n_bs, seq_n, feat_n)
        slow_raw_f, slow_gain_f, sig_f = model(x)
        slow_raw = slow_raw_f.reshape(mc_b, n_ep, n_bs)
        slow_gain = slow_gain_f.reshape(mc_b, n_ep, n_bs)
        sig_hat = sig_f.reshape(mc_b, n_ep, n_bs)
        slow_hat = _mix_slow_torch(slow_raw, slow_gain, cfg)

        loss_slow = _slow_loss(slow_hat, y_slow_mc)
        loss_sig = _sig_nll_loss(slow_hat, y_corr_mc, sig_hat, cfg) if bool(cfg.sig_nll_use) else _sig_regress_loss(sig_hat, y_sig_mc)
        loss_sig_reg = _sig_reg_loss(sig_hat)
        w_hat = _sig_to_w_torch(sig_hat, y_geo_mc, cfg)
        loss_w_cal = _w_cal_loss(w_hat, y_w_mc)

        if ep_idx < pre_n:
            corr_hat = _clip_corr_torch(slow_hat, cfg)
            loss_corr = _corr_total_loss(corr_hat, y_corr_mc, y_harm_mc, cfg)
            loss_sign = _sign_loss(corr_hat, y_corr_mc, cfg)
            return (
                float(cfg.corr_lam) * loss_corr
                + float(cfg.sign_lam) * loss_sign
                + float(cfg.slow_lam) * loss_slow
                + float(cfg.sig_lam) * loss_sig
                + float(cfg.sig_reg_lam) * loss_sig_reg
                + float(cfg.w_cal_lam) * loss_w_cal
            )

        joint_idx = _joint_ep_idx(ep_idx, cfg)
        corr_hat = _clip_corr_torch(slow_hat, cfg)
        loss_corr = _corr_total_loss(corr_hat, y_corr_mc, y_harm_mc, cfg)
        loss_sign = _sign_loss(corr_hat, y_corr_mc, cfg)
        loss_time = _time_loss(slow_hat, y_slow_mc)
        batch_train = SimpleNamespace(rho_hat=rho_hat_mc, bs=bs_mc, dim=train.dim)
        loss_res = _geom_res_loss(batch_train, corr_hat, w_hat, wrapper, dev)
        pos_scale = _pos_scale(joint_idx, cfg)
        return (
            float(cfg.corr_lam) * loss_corr
            + float(cfg.sign_lam) * loss_sign
            + float(cfg.slow_lam) * loss_slow
            + float(cfg.sig_lam) * loss_sig
            + float(cfg.sig_reg_lam) * loss_sig_reg
            + float(cfg.w_cal_lam) * loss_w_cal
            + float(cfg.time_lam) * loss_time
            + pos_scale * float(cfg.res_lam) * loss_res
        )

    for ep_idx in range(int(cfg.train_n)):
        opt.zero_grad()
        loss = _loss_core(
            x_all[:tr_n],
            y_corr[:tr_n],
            y_slow[:tr_n],
            y_sig[:tr_n],
            y_harm[:tr_n],
            y_w[:tr_n],
            y_geo[:tr_n],
            train.rho_hat[:tr_n],
            train.bs[:tr_n],
            ep_idx,
        )
        loss.backward()
        clip = max(float(cfg.grad_clip), 0.0)
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if ema_lst is not None and ep_idx >= max(int(cfg.ema_start_n), 0):
            beta = float(np.clip(cfg.ema_beta, 0.0, 0.99999))
            with torch.no_grad():
                for ema_p, mod_p in zip(ema_lst, model.parameters()):
                    ema_p.mul_(beta).add_(mod_p.detach(), alpha=1.0 - beta)
        loss_hist.append(float(loss.detach().cpu()))
        if bool(cfg.best_use):
            with torch.no_grad():
                if val_n > 0:
                    score_t = _loss_core(
                        x_all[tr_n:],
                        y_corr[tr_n:],
                        y_slow[tr_n:],
                        y_sig[tr_n:],
                        y_harm[tr_n:],
                        y_w[tr_n:],
                        y_geo[tr_n:],
                        train.rho_hat[tr_n:],
                        train.bs[tr_n:],
                        ep_idx,
                    )
                else:
                    score_t = loss.detach()
                score = float(score_t.cpu())
                if best_score is None or score < best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if val_pack is not None and (ep_idx == 0 or (ep_idx + 1) % _VAL_GAP_N == 0 or ep_idx == cfg.train_n - 1):
            val_hist[ep_idx] = _eval_val_d3(model, val_pack, full_cfg, dev_cfg)

    if ema_lst is not None:
        with torch.no_grad():
            for mod_p, ema_p in zip(model.parameters(), ema_lst):
                mod_p.copy_(ema_p)
    if bool(cfg.best_use) and best_state is not None:
        model.load_state_dict(best_state)

    return model, np.asarray(loss_hist, dtype=np.float64), fill_val_curve(val_hist)


def _pred_torch(model: nn.Module, seq: np.ndarray, cfg: NTcnCfg, dev_cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Torch 推理。"""
    dev = pick_dev(dev_cfg, "ntcn")
    x = to_f32(seq, dev)
    model.eval()
    with torch.no_grad():
        slow_raw, slow_gain, sig_pred = model(x)
        slow = _mix_slow_torch(slow_raw, slow_gain, cfg)
        corr = torch.clamp(slow, -float(model.corr_lim), float(model.corr_lim))
    return (
        corr.cpu().numpy(),
        slow.cpu().numpy(),
        slow_gain.cpu().numpy(),
        sig_pred.cpu().numpy(),
    )


def _pred_rule(seq: np.ndarray, cfg: NTcnCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """无 Torch 时的规则回退。"""
    raw_last = seq[:, :, -1, _RAW_LAST_I]
    raw_mean = np.mean(seq[:, :, :, _RAW_LAST_I], axis=2)
    slow_gain = _rule_slow_gain(seq, cfg)
    slow = _mix_slow_np(raw_mean, slow_gain, cfg)
    corr = _clip_corr(slow, cfg)
    sig = np.clip(np.abs(raw_last - corr) + float(cfg.sig_min), float(cfg.sig_min), float(cfg.sig_max))
    return (
        corr.astype(np.float64),
        slow.astype(np.float64),
        slow_gain.astype(np.float64),
        sig.astype(np.float64),
    )


def _fit_predict(
    train: NTcnTrainData,
    seq_eval: np.ndarray,
    cfg: NTcnCfg,
    dev_cfg: Any,
    pos_cfg: Any,
    full_cfg: Cfg,
    eval_pack: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """先训练，再对当前历元序列推理。"""
    if cfg.use_torch and torch is not None:
        model, loss, val_d3 = _train_torch(train, cfg, dev_cfg, pos_cfg, full_cfg, eval_pack)
        corr_pred, slow_pred, slow_gain_pred, sig_pred = _pred_torch(model, seq_eval, cfg, dev_cfg)
        return corr_pred, slow_pred, slow_gain_pred, sig_pred, loss, val_d3

    corr_pred, slow_pred, slow_gain_pred, sig_pred = _pred_rule(seq_eval, cfg)
    return corr_pred, slow_pred, slow_gain_pred, sig_pred, np.zeros(1, dtype=np.float64), np.full(1, np.nan, dtype=np.float64)


def _solve_eq(cfg: Cfg, geo: GeoData, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """等权定位解算。"""
    w = np.ones_like(rho, dtype=np.float64)
    return solve_w_epochs(cfg, geo, rho, w)


def run_ntcn(cfg: Cfg, obs: ObsData, geo: GeoData) -> NTcnData:
    """运行慢变-稀疏 NavTCN。"""
    if obs.rho_code.ndim != 2 or obs.rho_car.ndim != 2:
        raise ValueError("obs.rho_code / obs.rho_car 必须是二维矩阵")

    prep = _prep_data(obs, geo, cfg)
    train = _build_mc_train(cfg)
    eval_pack = _build_eval_case(cfg, obs, geo, prep)

    n_ep, n_bs = prep.mp_hat.shape
    corr_pred, slow_pred, slow_gain_pred, sig_pred, loss, val_d3 = _fit_predict(
        train, prep.seq, cfg.ntcn, cfg.dev, cfg.pos, cfg, eval_pack
    )
    out = _run_case(cfg, geo, prep, corr_pred, slow_pred, sig_pred)
    true_pos = _interp_pos(geo, obs.t_ep)
    e, dh, dz, d3 = calc_pos_err(out.pos, true_pos)

    return NTcnData(
        t_ep=obs.t_ep,
        rho_true=prep.rho_true,
        mp_raw=prep.mp_raw,
        rho_hat=prep.rho_hat,
        mp_hat=prep.mp_hat,
        rho_fix=out.rho_fix,
        mp_fix=out.mp_fix,
        cmc=prep.cmc,
        geo_sens=prep.geo_sens,
        seq=prep.seq,
        drho_tar=prep.drho_tar,
        drho_slow_tar=prep.drho_slow_tar,
        sig_tar=prep.sig_tar,
        pos_harm_tar=prep.pos_harm_tar,
        w_tar=prep.w_tar,
        drho_pred=out.corr_pred.reshape(n_ep, n_bs),
        drho_slow_pred=out.slow_pred.reshape(n_ep, n_bs),
        slow_gain_pred=slow_gain_pred.reshape(n_ep, n_bs),
        sig_pred=out.sig_pred,
        w_pow_ep=np.full(n_ep, float(cfg.ntcn.w_pow), dtype=np.float64),
        w_mix_ep=out.w_mix_ep.reshape(n_ep),
        w=out.w,
        pos=out.pos,
        cb=out.cb,
        res=out.res,
        dim=out.dim,
        it=out.it,
        true_pos=true_pos,
        e=e,
        d3=d3,
        dh=dh,
        dz=dz,
        loss=loss,
        val_d3=val_d3,
        mc_n=int(cfg.ntcn.mc_n),
        train_sample_n=int(train.seq.shape[0] * train.seq.shape[1]),
    )


def ntcn_stat(data: NTcnData) -> Dict[str, Any]:
    """输出慢变-稀疏 NavTCN 摘要。"""
    return {
        "dim": int(data.dim),
        "n_ep": int(data.t_ep.size),
        "mc_n": int(data.mc_n),
        "train_sample_n": int(data.train_sample_n),
        "mp_raw_rms_m": float(np.sqrt(np.mean(data.mp_raw**2))),
        "mp_hat_rms_m": float(np.sqrt(np.mean(data.mp_hat**2))),
        "mp_fix_rms_m": float(np.sqrt(np.mean(data.mp_fix**2))),
        "corr_tar_rms_m": float(np.sqrt(np.mean(data.drho_tar**2))),
        "corr_pred_rms_m": float(np.sqrt(np.mean(data.drho_pred**2))),
        "slow_tar_rms_m": float(np.sqrt(np.mean(data.drho_slow_tar**2))),
        "slow_pred_rms_m": float(np.sqrt(np.mean(data.drho_slow_pred**2))),
        "sig_tar_mean_m": float(np.mean(data.sig_tar)),
        "pos_harm_mean": float(np.mean(data.pos_harm_tar)),
        "geo_sens_mean": float(np.mean(data.geo_sens)),
        "slow_gain_mean": float(np.mean(data.slow_gain_pred)),
        "sig_mean_m": float(np.mean(data.sig_pred)),
        "w_tar_mean": float(np.mean(data.w_tar)),
        "w_pow_mean": float(np.mean(data.w_pow_ep)),
        "w_mix_mean": float(np.mean(data.w_mix_ep)),
        "post_res_mix_mean": float(np.mean(data.w_mix_ep)),
        "w_min": float(np.min(data.w)),
        "w_max": float(np.max(data.w)),
        "cmc_mean": float(np.mean(data.cmc)),
        "d3_rms_m": float(np.sqrt(np.mean(data.d3**2))),
        "dh_rms_m": float(np.sqrt(np.mean(data.dh**2))),
        "loss_last": float(data.loss[-1]),
    }


def save_ntcn(data: NTcnData, out_dir: Path) -> Path:
    """保存慢变-稀疏 NavTCN 结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "nav_tcn.npz"
    np.savez(
        out,
        t_ep=data.t_ep,
        rho_true=data.rho_true,
        mp_raw=data.mp_raw,
        rho_hat=data.rho_hat,
        mp_hat=data.mp_hat,
        rho_fix=data.rho_fix,
        mp_fix=data.mp_fix,
        cmc=data.cmc,
        geo_sens=data.geo_sens,
        seq=data.seq,
        drho_tar=data.drho_tar,
        drho_slow_tar=data.drho_slow_tar,
        sig_tar=data.sig_tar,
        pos_harm_tar=data.pos_harm_tar,
        w_tar=data.w_tar,
        drho_pred=data.drho_pred,
        drho_slow_pred=data.drho_slow_pred,
        slow_gain_pred=data.slow_gain_pred,
        sig_pred=data.sig_pred,
        w_pow_ep=data.w_pow_ep,
        w_mix_ep=data.w_mix_ep,
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
        loss=data.loss,
        val_d3=data.val_d3,
        mc_n=np.array([data.mc_n], dtype=np.int64),
        train_sample_n=np.array([data.train_sample_n], dtype=np.int64),
    )
    return out

