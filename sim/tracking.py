"""码跟踪与载波跟踪。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from config import Cfg, TrkCfg
from sim.rx_signal import RxData
from sim.tx_signal import gen_tx

C0 = 299792458.0  # 光速 m/s


@dataclass
class ObsData:
    """多历元伪距观测结果。"""

    t: np.ndarray  # 全采样时间轴
    t_ep: np.ndarray  # 历元观测时刻 (n_ep,)
    lag: np.ndarray  # 延迟估计(采样点) (n_ep, n_bs)
    tau: np.ndarray  # 延迟估计(s) (n_ep, n_bs)
    rho_code: np.ndarray  # 码伪距(m) (n_ep, n_bs)
    rho_car: np.ndarray  # 载波伪距(m) (n_ep, n_bs)
    phi: np.ndarray  # 载波相位(rad) (n_ep, n_bs)
    frac: np.ndarray  # 载波相位小数周 (n_ep, n_bs)
    amb_n: np.ndarray  # 整周模糊度整数 N (n_bs,)
    amb_b: np.ndarray  # 载波偏置 b (m) (n_bs,)
    peak: np.ndarray  # 相关峰值 (n_ep, n_bs)


def _check_trk_cfg(cfg: TrkCfg, fs: float, n_t: int) -> Tuple[int, int, int, int]:
    """检查跟踪配置并返回关键采样参数。"""
    if cfg.epoch_s <= 0.0:
        raise ValueError("trk.epoch_s 必须大于 0")
    if cfg.step_s <= 0.0:
        raise ValueError("trk.step_s 必须大于 0")
    if cfg.lag_min_s < 0.0:
        raise ValueError("trk.lag_min_s 不能小于 0")
    if cfg.lag_max_s <= cfg.lag_min_s:
        raise ValueError("trk.lag_max_s 必须大于 trk.lag_min_s")
    if cfg.amb_mode not in {"code", "zero", "rand"}:
        raise ValueError("trk.amb_mode 仅支持 code / zero / rand")
    if cfg.amb_n_max < cfg.amb_n_min:
        raise ValueError("trk.amb_n_max 不能小于 trk.amb_n_min")

    n_ep = int(round(cfg.epoch_s * fs))
    n_st = int(round(cfg.step_s * fs))
    lag_min = int(np.floor(cfg.lag_min_s * fs))
    lag_max = int(np.ceil(cfg.lag_max_s * fs))

    if n_ep < 8:
        raise ValueError("trk.epoch_s 太小，单历元采样点过少")
    if n_st < 1:
        raise ValueError("trk.step_s 太小，历元步长过小")
    if n_ep > n_t:
        raise ValueError("trk.epoch_s 太大，超过接收信号长度")
    if lag_max >= n_ep:
        raise ValueError("trk.lag_max_s 过大，超过历元长度")

    return n_ep, n_st, lag_min, lag_max


def _xcorr_fft(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FFT 计算线性互相关。"""
    n = a.size + b.size - 1
    n_fft = 1 << (n - 1).bit_length()
    c = np.fft.ifft(np.fft.fft(a, n_fft) * np.conj(np.fft.fft(b, n_fft)), n_fft)[:n]
    c_lin = np.concatenate((c[-(b.size - 1) :], c[: a.size]))
    lags = np.arange(-(b.size - 1), a.size, dtype=np.int64)
    return c_lin, lags


def _frac_lag(mag: np.ndarray, idx: int) -> float:
    """抛物线细化峰值位置。"""
    if idx <= 0 or idx >= mag.size - 1:
        return 0.0
    y1, y2, y3 = mag[idx - 1], mag[idx], mag[idx + 1]
    d = y1 - 2.0 * y2 + y3
    if d == 0.0:
        return 0.0
    delta = 0.5 * (y1 - y3) / d
    if np.abs(delta) > 1.0:
        return 0.0
    return float(delta)


def _delay_c(sig: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    """对复信号做分数时延。"""
    q = t - tau
    re = np.interp(q, t, np.real(sig), left=0.0, right=0.0)
    im = np.interp(q, t, np.imag(sig), left=0.0, right=0.0)
    return re + 1j * im


def _epoch_starts(n_t: int, n_ep: int, n_st: int) -> np.ndarray:
    """生成历元起始索引。"""
    return np.arange(0, n_t - n_ep + 1, n_st, dtype=np.int64)


def _init_amb(
    cfg: TrkCfg, rho_code: np.ndarray, frac: np.ndarray, lam: float, bs_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """初始化整周模糊度 N 和载波偏置 b。"""
    if cfg.amb_mode == "zero":
        n = np.zeros(bs_n, dtype=np.int64)
        b = np.zeros(bs_n, dtype=np.float64)
        return n, b

    if cfg.amb_mode == "rand":
        rng = np.random.default_rng(cfg.amb_seed)
        n = rng.integers(cfg.amb_n_min, cfg.amb_n_max + 1, size=bs_n, dtype=np.int64)
        b = np.zeros(bs_n, dtype=np.float64)
        return n, b

    # code 辅助初始化：多历元平均后取整，提升稳健性
    n_float = np.mean(rho_code / lam - frac, axis=0)
    n = np.round(n_float).astype(np.int64)
    b = np.mean(rho_code - (n[None, :] + frac) * lam, axis=0)
    return n, b


def track(cfg: Cfg, rx: RxData) -> ObsData:
    """输出多历元码伪距和载波伪距。"""
    if rx.rx.ndim != 2:
        raise ValueError("rx.rx 必须是二维矩阵 (n_bs, n_t)")

    bs_n, n_t = rx.rx.shape
    n_ep, n_st, lag_min, lag_max = _check_trk_cfg(cfg.trk, cfg.sig.fs, n_t)

    # 本地副本信号：与发射配置一致
    tx_ref = gen_tx(cfg.sig)
    if tx_ref.t.size != n_t:
        raise ValueError("本地副本长度与接收信号长度不一致")
    if not np.allclose(tx_ref.t, rx.t):
        raise ValueError("本地副本时间轴与接收信号不一致")

    starts = _epoch_starts(n_t, n_ep, n_st)
    n_obs = starts.size
    if n_obs <= 0:
        raise ValueError("没有可用历元，请检查 trk.epoch_s / trk.step_s")

    lag = np.zeros((n_obs, bs_n), dtype=np.float64)
    tau = np.zeros((n_obs, bs_n), dtype=np.float64)
    rho_code = np.zeros((n_obs, bs_n), dtype=np.float64)
    phi = np.zeros((n_obs, bs_n), dtype=np.float64)
    frac = np.zeros((n_obs, bs_n), dtype=np.float64)
    peak = np.zeros((n_obs, bs_n), dtype=np.float64)

    lam = C0 / cfg.sig.fc

    for e, s in enumerate(starts):
        r = s + n_ep
        rx_seg = rx.rx[:, s:r]
        ref_seg = tx_ref.tx[s:r]
        t_seg = rx.t[s:r]

        for b in range(bs_n):
            c, lags = _xcorr_fft(rx_seg[b], ref_seg)
            m = np.abs(c)
            mask = (lags >= lag_min) & (lags <= lag_max)
            if not np.any(mask):
                raise ValueError("相关搜索窗口为空，请调整 trk.lag_min_s / trk.lag_max_s")

            idx_win = np.flatnonzero(mask)
            idx0 = idx_win[int(np.argmax(m[idx_win]))]
            d = _frac_lag(m, idx0) if cfg.trk.frac else 0.0
            lag_b = float(lags[idx0]) + d
            tau_b = lag_b / cfg.sig.fs
            rho_code_b = tau_b * C0

            # 估计复增益相位，得到小数周
            rep = _delay_c(ref_seg, t_seg, tau_b)
            den = np.vdot(rep, rep) + 1e-12
            h = np.vdot(rep, rx_seg[b]) / den
            phi_b = float(np.angle(h))
            frac_b = float(np.mod(phi_b / (2.0 * np.pi), 1.0))

            lag[e, b] = lag_b
            tau[e, b] = tau_b
            rho_code[e, b] = rho_code_b
            phi[e, b] = phi_b
            frac[e, b] = frac_b
            peak[e, b] = float(m[idx0])

    amb_n, amb_b = _init_amb(cfg.trk, rho_code, frac, lam, bs_n)
    rho_car = (amb_n[None, :].astype(np.float64) + frac) * lam + amb_b[None, :]
    t_ep = rx.t[starts + (n_ep // 2)]

    return ObsData(
        t=rx.t,
        t_ep=t_ep,
        lag=lag,
        tau=tau,
        rho_code=rho_code,
        rho_car=rho_car,
        phi=phi,
        frac=frac,
        amb_n=amb_n,
        amb_b=amb_b,
        peak=peak,
    )


def track_stat(data: ObsData) -> Dict[str, Any]:
    """输出跟踪摘要。"""
    return {
        "bs_n": int(data.rho_code.shape[1]),
        "n_ep": int(data.rho_code.shape[0]),
        "t0_s": float(data.t_ep[0]),
        "t1_s": float(data.t_ep[-1]),
        "code_min_m": float(np.min(data.rho_code)),
        "code_max_m": float(np.max(data.rho_code)),
        "car_min_m": float(np.min(data.rho_car)),
        "car_max_m": float(np.max(data.rho_car)),
        "amb_n_min": int(np.min(data.amb_n)),
        "amb_n_max": int(np.max(data.amb_n)),
        "peak_mean": float(np.mean(data.peak)),
    }


def save_track(data: ObsData, out_dir: Path) -> Path:
    """保存跟踪结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "obs.npz"
    np.savez(
        out,
        t=data.t,
        t_ep=data.t_ep,
        lag=data.lag,
        tau=data.tau,
        rho_code=data.rho_code,
        rho_car=data.rho_car,
        phi=data.phi,
        frac=data.frac,
        amb_n=data.amb_n,
        amb_b=data.amb_b,
        peak=data.peak,
    )
    return out
