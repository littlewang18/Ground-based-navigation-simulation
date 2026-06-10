"""经典多径场景下的总体模型对比实验。"""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation.common import _COLOR, legacy_label, make_cfg
from config import Cfg, load_cfg
import model.gru as gru_mod
import model.huber as huber_mod
import model.mstcn as mstcn_mod
import model.nav_tcn as ntcn_mod
import model.tcn as tcn_mod
from model.common_utils import calc_pos_err, interp_pos, interp_rng, make_var_w, solve_w_epochs
from model.hatch_utils import hatch_smooth
from sim.error_eval import ErrData, calc_err
from sim.geometry import gen_geo
from sim.multipath import gen_mp
from sim.positioning import solve_pos
from sim.rx_signal import gen_rx
from sim.tracking import track
from sim.tx_signal import gen_tx


MODEL_ORDER = ["baseline", "Hatch", "Huber", "GRU", "TCN", "MSTCN", "NavTCN"]
TRAD_MODEL_ORDER = ["baseline", "Hatch", "Huber"]
DEEP_MODEL_ORDER = ["GRU", "TCN", "MSTCN", "NavTCN"]
LOSS_MODEL_ORDER = DEEP_MODEL_ORDER

_SEED_KEYS = {
    "sig_prn_seed": ("sig", "prn_seed"),
    "sig_nav_seed": ("sig", "nav_seed"),
    "geo_bs_seed": ("geo", "bs_seed"),
    "mp_seed": ("mp", "seed"),
    "rx_seed": ("rx", "seed"),
    "trk_amb_seed": ("trk", "amb_seed"),
    "tcn_seed": ("tcn", "seed"),
    "mstcn_seed": ("mstcn", "seed"),
    "ntcn_seed": ("ntcn", "seed"),
}


@dataclass
class TotalCmpData:
    """单次总体对比实验输出。"""

    cfg: Cfg
    out: dict[str, Any]
    df_all: pd.DataFrame
    curve_df: pd.DataFrame
    loss_df: pd.DataFrame
    val_df: pd.DataFrame


def _rep_idx_lst(cfg: Cfg) -> list[int]:
    """返回本次总对比实际使用的重复场景索引。"""
    rep_idx_lst = [int(x) for x in getattr(cfg.total_cmp, "rep_idx_lst", ())]
    if len(rep_idx_lst) > 0:
        return rep_idx_lst
    return list(range(int(cfg.total_cmp.rep_n)))


def _key(model: str) -> str:
    return model.lower().replace("+", "_").replace("-", "_").replace(" ", "")


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x**2)))


def _pad_series(x: np.ndarray, n: int) -> np.ndarray:
    y = np.full(n, np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    y[: x.size] = x
    return y


def _make_cmp_cfg(cfg: Cfg) -> Cfg:
    """构造经典场景下的统一训练配置。"""
    cfg = make_cfg(cfg)
    tc = cfg.total_cmp
    tcn = replace(cfg.tcn, train_n=tc.train_n, lr=tc.lr, wd=tc.wd, q_scale=tc.q_scale, w_min=tc.w_min, w_max=tc.w_max, mc_n=tc.mc_n)
    mstcn = replace(cfg.mstcn, train_n=tc.train_n, lr=tc.lr, wd=tc.wd, q_scale=tc.q_scale, w_min=tc.w_min, w_max=tc.w_max, mc_n=tc.mc_n)
    ntcn = replace(cfg.ntcn, train_n=tc.train_n, lr=tc.lr, wd=tc.wd, q_scale=tc.q_scale, mc_n=tc.mc_n)
    return replace(cfg, tcn=tcn, mstcn=mstcn, ntcn=ntcn)


def _make_rep_cfg(cfg: Cfg, rep_idx: int) -> Cfg:
    """构造某次重复的确定性配置。

    为保证总体对比的公平性，重复实验只改变场景随机种子，
    不改变各模型的训练初始化种子，避免把“场景波动”和“优化波动”混在一起。
    """
    off = int(rep_idx) * int(cfg.total_cmp.seed_step)
    if off == 0:
        return cfg
    return replace(
        cfg,
        sig=replace(cfg.sig, prn_seed=cfg.sig.prn_seed + off, nav_seed=cfg.sig.nav_seed + off),
        geo=replace(cfg.geo, bs_seed=cfg.geo.bs_seed + off),
        mp=replace(cfg.mp, seed=cfg.mp.seed + off),
        rx=replace(cfg.rx, seed=cfg.rx.seed + off),
        trk=replace(cfg.trk, amb_seed=cfg.trk.amb_seed + off),
    )


def _pack_baseline(err: ErrData) -> SimpleNamespace:
    return SimpleNamespace(d3=err.d3_code, dh=err.dh_code, dz=err.dz_code, loss=np.asarray([], dtype=np.float64))


def _run_hatch(cfg: Cfg, obs, geo) -> SimpleNamespace:
    """运行 Hatch 平滑 + 后验残差加权最小二乘定位。"""
    rho_true = interp_rng(geo, obs.t_ep)
    rho_hat = hatch_smooth(obs.rho_code, obs.rho_car, cfg.hatch.n, "hatch.n")
    w0 = np.ones_like(rho_hat, dtype=np.float64)
    _, _, res0, _, _ = solve_w_epochs(cfg, geo, rho_hat, w0)
    w, sig = make_var_w(res0, 5, 0.5, 1.0, 0.2, 4.0, "hatch_wls.sig_win_n")
    pos, cb, res, dim, it = solve_w_epochs(cfg, geo, rho_hat, w)
    true_pos = interp_pos(geo, obs.t_ep)
    e, dh, dz, d3 = calc_pos_err(pos, true_pos)
    return SimpleNamespace(
        t_ep=obs.t_ep,
        rho_true=rho_true,
        rho_hat=rho_hat,
        sig=sig,
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
        loss=np.asarray([], dtype=np.float64),
    )


def _make_curve_df(err: ErrData, out: dict[str, Any]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {"t_ep_s": err.t_ep}
    for model in MODEL_ORDER:
        data[f"{_key(model)}_d3_m"] = np.asarray(out[model].d3, dtype=np.float64)
        data[f"{_key(model)}_dh_m"] = np.asarray(out[model].dh, dtype=np.float64)
        data[f"{_key(model)}_dz_m"] = np.asarray(out[model].dz, dtype=np.float64)
    return pd.DataFrame(data)


def _make_loss_df(out: dict[str, Any]) -> pd.DataFrame:
    loss_map = {f"{_key(model)}_loss": getattr(out[model], "loss", np.asarray([], dtype=np.float64)) for model in LOSS_MODEL_ORDER}
    n = max([np.asarray(v).size for v in loss_map.values()] + [1])
    data: dict[str, np.ndarray] = {"epoch": np.arange(1, n + 1, dtype=np.int64)}
    for key, val in loss_map.items():
        data[key] = _pad_series(val, n)
    return pd.DataFrame(data)


def _make_val_df(out: dict[str, Any]) -> pd.DataFrame:
    val_map = {f"{_key(model)}_val_d3": getattr(out[model], "val_d3", np.asarray([], dtype=np.float64)) for model in LOSS_MODEL_ORDER}
    n = max([np.asarray(v).size for v in val_map.values()] + [1])
    data: dict[str, np.ndarray] = {"epoch": np.arange(1, n + 1, dtype=np.int64)}
    for key, val in val_map.items():
        data[key] = _pad_series(val, n)
    return pd.DataFrame(data)


def _run_once(cfg: Cfg) -> TotalCmpData:
    tx = gen_tx(cfg.sig)
    geo = gen_geo(cfg, tx)
    mp = gen_mp(cfg, tx, geo)
    rx = gen_rx(cfg, tx, geo, mp)
    obs = track(cfg, rx)

    pos = solve_pos(cfg, obs, geo)
    err = calc_err(cfg, geo, pos)

    out: dict[str, Any] = {
        "baseline": _pack_baseline(err),
        "Hatch": _run_hatch(cfg, obs, geo),
        "Huber": huber_mod.run_huber(cfg, obs, geo),
    }

    ctx = legacy_label() if bool(cfg.total_cmp.legacy_label) else nullcontext()
    with ctx:
        out["GRU"] = gru_mod.run_gru(cfg, obs, geo)
        out["TCN"] = tcn_mod.run_tcn(cfg, obs, geo)
        out["MSTCN"] = mstcn_mod.run_mstcn(cfg, obs, geo)
    out["NavTCN"] = ntcn_mod.run_ntcn(cfg, obs, geo)

    rows = [
        {"model": model, "d3_rms_m": _rms(out[model].d3), "group": "classic" if model in TRAD_MODEL_ORDER else "deep"}
        for model in MODEL_ORDER
    ]
    df_all = pd.DataFrame(rows).sort_values("d3_rms_m").reset_index(drop=True)
    df_all.insert(0, "rank", np.arange(1, df_all.shape[0] + 1, dtype=np.int64))

    loss_df = _make_loss_df(out)
    val_df = _make_val_df(out)
    return TotalCmpData(cfg=cfg, out=out, df_all=df_all, curve_df=_make_curve_df(err, out), loss_df=loss_df, val_df=val_df)


def _make_run_df(data_lst: list[TotalCmpData], rep_idx_lst: list[int]) -> pd.DataFrame:
    rows = []
    for rep_ord, (rep_idx, data) in enumerate(zip(rep_idx_lst, data_lst), start=1):
        df = data.df_all.copy()
        base = float(df.loc[df["model"] == "baseline", "d3_rms_m"].iloc[0])
        df["rep"] = rep_ord
        df["rep_idx"] = int(rep_idx)
        df["gain_vs_baseline_pct"] = (base - df["d3_rms_m"]) / base * 100.0
        rows.append(df)
    return pd.concat(rows, axis=0, ignore_index=True)


def _make_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        run_df.groupby(["model", "group"], as_index=False)
        .agg(
            d3_rms_mean_m=("d3_rms_m", "mean"),
            d3_rms_std_m=("d3_rms_m", "std"),
            d3_rms_min_m=("d3_rms_m", "min"),
            d3_rms_max_m=("d3_rms_m", "max"),
            gain_mean_pct=("gain_vs_baseline_pct", "mean"),
            gain_std_pct=("gain_vs_baseline_pct", "std"),
            rep_n=("rep", "count"),
        )
        .sort_values("d3_rms_mean_m")
        .reset_index(drop=True)
    )
    base_mean = float(grp.loc[grp["model"] == "baseline", "d3_rms_mean_m"].iloc[0])
    grp["gain_mean_pct"] = (base_mean - grp["d3_rms_mean_m"]) / base_mean * 100.0
    grp["d3_rms_std_m"] = grp["d3_rms_std_m"].fillna(0.0)
    grp["gain_std_pct"] = grp["gain_std_pct"].fillna(0.0)
    grp.insert(0, "rank", np.arange(1, grp.shape[0] + 1, dtype=np.int64))
    return grp


def _stack_col(data_lst: list[TotalCmpData], df_name: str, col: str) -> np.ndarray:
    arr_lst = [getattr(data, df_name)[col].to_numpy(dtype=np.float64) for data in data_lst]
    return np.stack(arr_lst, axis=0)


def _make_curve_stat(data_lst: list[TotalCmpData]) -> pd.DataFrame:
    out = {"t_ep_s": data_lst[0].curve_df["t_ep_s"].to_numpy(dtype=np.float64)}
    for model in MODEL_ORDER:
        arr = _stack_col(data_lst, "curve_df", f"{_key(model)}_d3_m")
        out[f"{_key(model)}_mean_m"] = np.mean(arr, axis=0)
        out[f"{_key(model)}_std_m"] = np.std(arr, axis=0)
    return pd.DataFrame(out)


def _make_stat_df(data_lst: list[TotalCmpData], df_name: str, col_map: dict[str, str], tail: str) -> pd.DataFrame:
    ref = getattr(data_lst[0], df_name)
    out = {"epoch": ref["epoch"].to_numpy(dtype=np.int64)}
    for model, col in col_map.items():
        arr = _stack_col(data_lst, df_name, col)
        mask = ~np.isnan(arr)
        cnt = np.sum(mask, axis=0)
        s1 = np.nansum(arr, axis=0)
        mean = np.divide(s1, cnt, out=np.full(cnt.shape, np.nan, dtype=np.float64), where=cnt > 0)
        diff = np.where(mask, arr - mean[None, :], 0.0)
        s2 = np.sum(diff**2, axis=0)
        std = np.sqrt(np.divide(s2, cnt, out=np.full(cnt.shape, np.nan, dtype=np.float64), where=cnt > 0))
        out[f"{_key(model)}_{tail}_mean"] = mean
        out[f"{_key(model)}_{tail}_std"] = std
    return pd.DataFrame(out)


def _make_norm_stat_df(data_lst: list[TotalCmpData], df_name: str, col_map: dict[str, str], tail: str) -> pd.DataFrame:
    """对每次重复先按初始值归一化，再统计均值与方差。"""
    ref = getattr(data_lst[0], df_name)
    out = {"epoch": ref["epoch"].to_numpy(dtype=np.int64)}
    for model, col in col_map.items():
        arr = _stack_col(data_lst, df_name, col).astype(np.float64)
        for i in range(arr.shape[0]):
            row = arr[i]
            finite = np.isfinite(row)
            if not np.any(finite):
                continue
            first = int(np.argmax(finite))
            den = float(abs(row[first]))
            if den <= 1e-12:
                den = 1.0
            arr[i] = row / den
        mask = ~np.isnan(arr)
        cnt = np.sum(mask, axis=0)
        s1 = np.nansum(arr, axis=0)
        mean = np.divide(s1, cnt, out=np.full(cnt.shape, np.nan, dtype=np.float64), where=cnt > 0)
        diff = np.where(mask, arr - mean[None, :], 0.0)
        s2 = np.sum(diff**2, axis=0)
        std = np.sqrt(np.divide(s2, cnt, out=np.full(cnt.shape, np.nan, dtype=np.float64), where=cnt > 0))
        out[f"{_key(model)}_{tail}_mean"] = mean
        out[f"{_key(model)}_{tail}_std"] = std
    return pd.DataFrame(out)


def _make_seed_df(cfg: Cfg) -> pd.DataFrame:
    rows = []
    for rep_ord, rep_idx in enumerate(_rep_idx_lst(cfg), start=1):
        cfg_i = _make_rep_cfg(cfg, rep_idx)
        row = {"rep": rep_ord, "rep_idx": int(rep_idx)}
        for key, (obj_name, attr_name) in _SEED_KEYS.items():
            row[key] = int(getattr(getattr(cfg_i, obj_name), attr_name))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_model_df(cfg: Cfg) -> pd.DataFrame:
    gru = gru_mod._cfg(cfg)
    return pd.DataFrame(
        [
            {"model": "baseline", "seq_n": "", "hid_n": "", "layer_n": "", "kernel": "", "train_n": 0, "mc_n": 0, "extra": "code positioning baseline"},
            {"model": "Hatch", "seq_n": "", "hid_n": "", "layer_n": "", "kernel": "", "train_n": 0, "mc_n": 0, "extra": "Hatch smoothing + post-fit residual WLS"},
            {"model": "Huber", "seq_n": "", "hid_n": "", "layer_n": "", "kernel": "", "train_n": 0, "mc_n": 0, "extra": "raw code Huber IRLS"},
            {"model": "GRU", "seq_n": gru.seq_n, "hid_n": gru.hid_n, "layer_n": gru.layer_n, "kernel": "", "train_n": gru.train_n, "mc_n": gru.mc_n, "extra": "quality score + weighted positioning"},
            {"model": "TCN", "seq_n": cfg.tcn.seq_n, "hid_n": cfg.tcn.hid_n, "layer_n": cfg.tcn.layer_n, "kernel": cfg.tcn.ker_n, "train_n": cfg.tcn.train_n, "mc_n": cfg.tcn.mc_n, "extra": "quality score + weighted positioning"},
            {"model": "MSTCN", "seq_n": cfg.mstcn.seq_n, "hid_n": cfg.mstcn.hid_n, "layer_n": cfg.mstcn.layer_n, "kernel": str(cfg.mstcn.ker_lst), "train_n": cfg.mstcn.train_n, "mc_n": cfg.mstcn.mc_n, "extra": "multi-scale quality score + weighted positioning"},
            {"model": "NavTCN", "seq_n": cfg.ntcn.seq_n, "hid_n": cfg.ntcn.hid_n, "layer_n": cfg.ntcn.layer_n, "kernel": str(cfg.ntcn.ker_lst), "train_n": cfg.ntcn.train_n, "mc_n": cfg.ntcn.mc_n, "extra": "post-fit residual + CMC + decoupled corr/sigma branch + geometry-calibrated sigma weighting"},
        ]
    )


def _plot_rank(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "total_cmp_rank.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [_COLOR[m] for m in df["model"]]
    bars = ax.bar(df["model"], df["d3_rms_mean_m"], yerr=df["d3_rms_std_m"], capsize=4, color=colors)
    ax.set_title("Model Comparison in Classical Multipath Scene")
    ax.set_ylabel("3D RMS Error (m)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    for bar, mean_v, std_v in zip(bars, df["d3_rms_mean_m"], df["d3_rms_std_m"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean_v, f"{mean_v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=700)
    plt.close(fig)
    return out


def _plot_err(curve_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "total_cmp_err.png"
    t_ms = curve_df["t_ep_s"].to_numpy(dtype=np.float64) * 1e3
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model in summary_df["model"]:
        y = curve_df[f"{_key(model)}_mean_m"].to_numpy(dtype=np.float64)
        s = curve_df[f"{_key(model)}_std_m"].to_numpy(dtype=np.float64)
        ax.plot(t_ms, y, label=model, lw=1.6, color=_COLOR[model])
        ax.fill_between(t_ms, y - s, y + s, color=_COLOR[model], alpha=0.10)
    ax.set_title("3D Error Curve")
    ax.set_xlabel("Epoch Time (ms)")
    ax.set_ylabel("3D Error (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=700)
    plt.close(fig)
    return out


def _plot_fit(loss_norm_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "total_cmp_fit.png"
    ep = loss_norm_df["epoch"].to_numpy(dtype=np.int64)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for model in LOSS_MODEL_ORDER:
        y = loss_norm_df[f"{_key(model)}_loss_mean"].to_numpy(dtype=np.float64)
        s = loss_norm_df[f"{_key(model)}_loss_std"].to_numpy(dtype=np.float64)
        ax[0].plot(ep, y, label=model, lw=1.6, color=_COLOR[model])
        ax[0].fill_between(ep, y - s, y + s, color=_COLOR[model], alpha=0.10)
    ax[0].set_title("Normalized Training Objective")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss / Loss@Epoch1")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    for model in LOSS_MODEL_ORDER:
        y = test_df[f"{_key(model)}_val_d3_mean"].to_numpy(dtype=np.float64)
        s = test_df[f"{_key(model)}_val_d3_std"].to_numpy(dtype=np.float64)
        ax[1].plot(ep, y, label=model, lw=1.6, color=_COLOR[model])
        ax[1].fill_between(ep, y - s, y + s, color=_COLOR[model], alpha=0.10)
    ax[1].set_title("Epoch Test 3D RMS on Formal Comparison Scenes")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("3D RMS Error (m)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=700)
    plt.close(fig)
    return out


def _save(
    cfg: Cfg,
    summary_df: pd.DataFrame,
    run_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    loss_raw_df: pd.DataFrame,
    loss_norm_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / "total_cmp.xlsx"
    csv = out_dir / "total_cmp.csv"
    scene_df = pd.DataFrame(
        [
            {
                "scene_name": "classical_total_cmp",
                "cmp_models": ",".join(MODEL_ORDER),
                "rx_mode": cfg.geo.rx_mode,
                "bs_z_jit": cfg.geo.bs_z_jit,
                "mp_n": cfg.mp.n,
                "dly_min_ns": cfg.mp.dly_min * 1e9,
                "dly_max_ns": cfg.mp.dly_max * 1e9,
                "k0": cfg.mp.k0,
                "decay": cfg.mp.decay,
                "jit": cfg.mp.jit,
                "fade": cfg.mp.fade,
                "burst_n": cfg.mp.burst_n,
                "snr_db": cfg.rx.snr_db,
                "legacy_label": int(bool(cfg.total_cmp.legacy_label)),
                "rep_n": int(len(_rep_idx_lst(cfg))),
                "rep_idx_lst": ",".join(str(i) for i in _rep_idx_lst(cfg)),
                "seed_step": int(cfg.total_cmp.seed_step),
                "device": cfg.dev.device,
            }
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "train_n": int(cfg.total_cmp.train_n),
                "lr": float(cfg.total_cmp.lr),
                "wd": float(cfg.total_cmp.wd),
                "q_scale": float(cfg.total_cmp.q_scale),
                "w_min": float(cfg.total_cmp.w_min),
                "w_max": float(cfg.total_cmp.w_max),
                "mc_n": int(cfg.total_cmp.mc_n),
                "deep_model_note": "Deep models share train_n/lr/wd/q_scale/mc_n; quality-weight models also share w_min/w_max.",
            }
        ]
    )
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        scene_df.to_excel(writer, sheet_name="scene", index=False)
        train_df.to_excel(writer, sheet_name="train_cfg", index=False)
        _make_model_df(cfg).to_excel(writer, sheet_name="model_cfg", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        run_df.to_excel(writer, sheet_name="runs", index=False)
        curve_df.to_excel(writer, sheet_name="err_curve", index=False)
        loss_raw_df.to_excel(writer, sheet_name="loss_raw", index=False)
        loss_norm_df.to_excel(writer, sheet_name="loss_norm", index=False)
        test_df.to_excel(writer, sheet_name="epoch_test_d3", index=False)
        seed_df.to_excel(writer, sheet_name="seed_plan", index=False)
    summary_df.to_csv(csv, index=False, encoding="utf-8-sig")
    return xlsx, csv


def run_total_cmp(cfg: Cfg | None = None) -> tuple[pd.DataFrame, Path, Path, Path, Path, Path]:
    cfg = _make_cmp_cfg(load_cfg() if cfg is None else cfg)
    rep_idx_lst = _rep_idx_lst(cfg)
    data_lst = [_run_once(_make_rep_cfg(cfg, i)) for i in rep_idx_lst]
    run_df = _make_run_df(data_lst, rep_idx_lst)
    summary_df = _make_summary(run_df)
    curve_df = _make_curve_stat(data_lst)
    loss_map = {model: f"{_key(model)}_loss" for model in LOSS_MODEL_ORDER}
    val_map = {model: f"{_key(model)}_val_d3" for model in LOSS_MODEL_ORDER}
    loss_raw_df = _make_stat_df(data_lst, "loss_df", loss_map, "loss")
    loss_norm_df = _make_norm_stat_df(data_lst, "loss_df", loss_map, "loss")
    test_df = _make_stat_df(data_lst, "val_df", val_map, "val_d3")
    seed_df = _make_seed_df(cfg)
    rank_png = _plot_rank(summary_df, cfg.run.out_dir)
    err_png = _plot_err(curve_df, summary_df, cfg.run.out_dir)
    fit_png = _plot_fit(loss_norm_df, test_df, cfg.run.out_dir)
    xlsx, csv = _save(cfg, summary_df, run_df, curve_df, loss_raw_df, loss_norm_df, test_df, seed_df, cfg.run.out_dir)
    return summary_df, rank_png, err_png, fit_png, xlsx, csv


if __name__ == "__main__":
    df, rank_png, err_png, fit_png, xlsx, csv = run_total_cmp()
    print("TOTAL_MODEL_SET")
    print(df.to_string(index=False))
    print("SAVE_RANK_PNG", rank_png)
    print("SAVE_ERR_PNG", err_png)
    print("SAVE_FIT_PNG", fit_png)
    print("SAVE_XLSX", xlsx)
    print("SAVE_CSV", csv)
