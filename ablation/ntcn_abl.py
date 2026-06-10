"""当前 NavTCN 主线模块消融实验。"""

from dataclasses import replace
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

from ablation.total_cmp import _COLOR, _make_cmp_cfg, _make_rep_cfg, _rep_idx_lst
from config import Cfg, load_cfg
import model.nav_tcn as ntcn_mod
from sim.error_eval import calc_err
from sim.geometry import gen_geo
from sim.multipath import gen_mp
from sim.positioning import solve_pos
from sim.rx_signal import gen_rx
from sim.tracking import track
from sim.tx_signal import gen_tx


_MODEL_ORDER = ["baseline", "full", "no_corr", "no_slow", "no_gain", "no_sig", "no_post_res", "no_batch"]
_MODEL_LABEL = {
    "baseline": "baseline",
    "full": "full",
    "no_corr": "no_corr",
    "no_slow": "no_slow",
    "no_gain": "no_gain",
    "no_sig": "no_sigma",
    "no_post_res": "no_post_res",
    "no_batch": "no_batch",
}
_MODEL_COLOR = {
    "baseline": _COLOR["baseline"],
    "full": _COLOR["NavTCN"],
    "no_corr": "#E45756",
    "no_slow": "#54A24B",
    "no_gain": "#4C78A8",
    "no_sig": "#B279A2",
    "no_post_res": "#72B7B2",
    "no_batch": "#F58518",
}


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x**2)))


def _pad_series(x: np.ndarray, n: int) -> np.ndarray:
    y = np.full(n, np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    y[: x.size] = x
    return y


def _nan_mean_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(arr)
    cnt = np.sum(mask, axis=0).astype(np.float64)
    val = np.where(mask, arr, 0.0)
    mean = np.divide(np.sum(val, axis=0), cnt, out=np.full(arr.shape[1], np.nan, dtype=np.float64), where=cnt > 0)
    diff = np.where(mask, arr - mean[None, :], 0.0)
    var = np.divide(np.sum(diff * diff, axis=0), cnt, out=np.full(arr.shape[1], np.nan, dtype=np.float64), where=cnt > 0)
    return mean, np.sqrt(var)


def _pack_baseline(err) -> SimpleNamespace:
    return SimpleNamespace(d3=err.d3_code, loss=np.asarray([], dtype=np.float64), val_d3=np.full(1, np.nan, dtype=np.float64))


def _build_variants(cfg: Cfg) -> dict[str, Cfg]:
    out: dict[str, Cfg] = {"full": cfg}
    abl = cfg.abl_ntcn
    if abl.use_no_corr:
        out["no_corr"] = replace(
            cfg,
            ntcn=replace(
                cfg.ntcn,
                corr_gain_use=0.0,
                corr_lam=0.0,
                sign_lam=0.0,
                slow_lam=0.0,
                time_lam=0.0,
                res_lam=0.0,
            ),
        )
    if abl.use_no_slow:
        out["no_slow"] = replace(cfg, ntcn=replace(cfg.ntcn, slow_use=0.0, slow_lam=0.0, time_lam=0.0))
    if abl.use_no_gain:
        out["no_gain"] = replace(
            cfg,
            ntcn=replace(
                cfg.ntcn,
                gain_min=1.0,
                gain_max=1.0,
                slow_gain_cmc_k=0.0,
                slow_gain_geo_k=0.0,
                slow_gain_scene_k=0.0,
            ),
        )
    if abl.use_no_sig:
        out["no_sig"] = replace(cfg, ntcn=replace(cfg.ntcn, post_sig_pred_use=0.0, post_sig_res_use=1.0))
    if abl.use_no_post_res:
        out["no_post_res"] = replace(cfg, ntcn=replace(cfg.ntcn, post_sig_pred_use=1.0, post_sig_res_use=0.0))
    if abl.use_no_batch:
        out["no_batch"] = replace(cfg, ntcn=replace(cfg.ntcn, batch_use=False))
    return out


def run_one(cfg: Cfg) -> dict[str, Any]:
    """运行一次当前 NavTCN 消融实验。"""
    tx = gen_tx(cfg.sig)
    geo = gen_geo(cfg, tx)
    mp = gen_mp(cfg, tx, geo)
    rx = gen_rx(cfg, tx, geo, mp)
    obs = track(cfg, rx)

    pos = solve_pos(cfg, obs, geo)
    err = calc_err(cfg, geo, pos)
    out: dict[str, Any] = {"baseline": _pack_baseline(err)}
    for name, cfg_i in _build_variants(cfg).items():
        out[name] = ntcn_mod.run_ntcn(cfg_i, obs, geo)
    return out


def _make_run_df(run_lst: list[tuple[int, Cfg, dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rep_ord, (rep_idx, _, out) in enumerate(run_lst, start=1):
        base = _rms(out["baseline"].d3)
        full = _rms(out["full"].d3)
        for model in _MODEL_ORDER:
            if model not in out:
                continue
            d3 = _rms(out[model].d3)
            rows.append(
                {
                    "rep": rep_ord,
                    "rep_idx": int(rep_idx),
                    "model": model,
                    "d3_rms_m": d3,
                    "gain_vs_baseline_pct": (base - d3) / base * 100.0,
                    "delta_vs_full_m": d3 - full,
                }
            )
    return pd.DataFrame(rows)


def _make_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    df = (
        run_df.groupby("model", as_index=False)
        .agg(
            d3_rms_mean_m=("d3_rms_m", "mean"),
            d3_rms_std_m=("d3_rms_m", "std"),
            d3_rms_min_m=("d3_rms_m", "min"),
            d3_rms_max_m=("d3_rms_m", "max"),
            gain_mean_pct=("gain_vs_baseline_pct", "mean"),
            gain_std_pct=("gain_vs_baseline_pct", "std"),
            delta_full_mean_m=("delta_vs_full_m", "mean"),
            delta_full_std_m=("delta_vs_full_m", "std"),
            rep_n=("rep", "count"),
        )
    )
    base_mean = float(df.loc[df["model"] == "baseline", "d3_rms_mean_m"].iloc[0])
    df["gain_mean_pct"] = (base_mean - df["d3_rms_mean_m"]) / base_mean * 100.0
    df["d3_rms_std_m"] = df["d3_rms_std_m"].fillna(0.0)
    df["gain_std_pct"] = df["gain_std_pct"].fillna(0.0)
    df["delta_full_std_m"] = df["delta_full_std_m"].fillna(0.0)
    df["order"] = df["model"].map({m: i for i, m in enumerate(_MODEL_ORDER)})
    return df.sort_values("order").drop(columns="order").reset_index(drop=True)


def _stack_curve(arr_lst: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.stack(arr_lst, axis=0).astype(np.float64)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def _make_curve_df(run_lst: list[tuple[int, Cfg, dict[str, Any]]]) -> pd.DataFrame:
    baseline_d3 = run_lst[0][2]["baseline"].d3
    n_ep = int(np.asarray(baseline_d3).size)
    out: dict[str, Any] = {"epoch": np.arange(1, n_ep + 1, dtype=np.int64)}

    for model in _MODEL_ORDER:
        arr_lst = []
        corr_lst = []
        for _, _, data in run_lst:
            if model not in data:
                continue
            arr_lst.append(np.asarray(data[model].d3, dtype=np.float64))
            if hasattr(data[model], "drho_pred"):
                corr_lst.append(np.mean(np.abs(np.asarray(data[model].drho_pred, dtype=np.float64)), axis=1))
        if not arr_lst:
            continue
        mean, std = _stack_curve(arr_lst)
        out[f"{model}_d3_mean_m"] = mean
        out[f"{model}_d3_std_m"] = std
        if corr_lst:
            mean, std = _stack_curve(corr_lst)
            out[f"{model}_corr_mean_m"] = mean
            out[f"{model}_corr_std_m"] = std

    return pd.DataFrame(out)


def _make_loss_df(run_lst: list[tuple[int, Cfg, dict[str, Any]]]) -> pd.DataFrame:
    loss_map: dict[str, list[np.ndarray]] = {}
    val_map: dict[str, list[np.ndarray]] = {}
    for model in _MODEL_ORDER:
        loss_lst = []
        val_lst = []
        for _, _, data in run_lst:
            if model in data and hasattr(data[model], "loss"):
                loss = np.asarray(data[model].loss, dtype=np.float64)
                if loss.size > 0:
                    loss_lst.append(loss)
            if model in data and hasattr(data[model], "val_d3"):
                val = np.asarray(data[model].val_d3, dtype=np.float64)
                if val.size > 0:
                    val_lst.append(val)
        if loss_lst:
            loss_map[model] = loss_lst
        if val_lst:
            val_map[model] = val_lst

    if not loss_map and not val_map:
        return pd.DataFrame({"epoch": np.arange(1, 2, dtype=np.int64)})

    size_lst = [np.asarray(v).size for arr_lst in loss_map.values() for v in arr_lst]
    size_lst += [np.asarray(v).size for arr_lst in val_map.values() for v in arr_lst]
    n = max(size_lst)
    out: dict[str, Any] = {"epoch": np.arange(1, n + 1, dtype=np.int64)}
    for key, arr_lst in loss_map.items():
        arr = np.stack([_pad_series(v, n) for v in arr_lst], axis=0)
        mean, std = _nan_mean_std(arr)
        out[f"{key}_loss_mean"] = mean
        out[f"{key}_loss_std"] = std
    for key, arr_lst in val_map.items():
        arr = np.stack([_pad_series(v, n) for v in arr_lst], axis=0)
        mean, std = _nan_mean_std(arr)
        out[f"{key}_val_d3_mean"] = mean
        out[f"{key}_val_d3_std"] = std
    return pd.DataFrame(out)


def _seed_df(cfg: Cfg) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for rep_ord, rep_idx in enumerate(_rep_idx_lst(cfg), start=1):
        cfg_i = _make_rep_cfg(cfg, rep_idx)
        rows.append(
            {
                "rep": rep_ord,
                "rep_idx": int(rep_idx),
                "sig_prn_seed": int(cfg_i.sig.prn_seed),
                "sig_nav_seed": int(cfg_i.sig.nav_seed),
                "geo_bs_seed": int(cfg_i.geo.bs_seed),
                "mp_seed": int(cfg_i.mp.seed),
                "rx_seed": int(cfg_i.rx.seed),
                "trk_amb_seed": int(cfg_i.trk.amb_seed),
                "ntcn_seed": int(cfg_i.ntcn.seed),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary_df: pd.DataFrame, curve_df: pd.DataFrame, loss_df: pd.DataFrame, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.png"

    show_models = [m for m in _MODEL_ORDER if m in set(summary_df["model"])]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    x = np.arange(len(show_models), dtype=np.int64)
    show_df = summary_df.set_index("model").loc[show_models].reset_index()
    colors = [_MODEL_COLOR[m] for m in show_models]
    ax[0, 0].bar(x, show_df["d3_rms_mean_m"], yerr=show_df["d3_rms_std_m"], capsize=4, color=colors)
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([_MODEL_LABEL[m] for m in show_models], rotation=20)
    ax[0, 0].set_ylabel("3D RMS Error (m)")
    ax[0, 0].set_title("Current NavTCN Module Ablation")
    ax[0, 0].grid(axis="y", alpha=0.3)

    ep = curve_df["epoch"].to_numpy(dtype=np.int64)
    for model in show_models:
        ax[0, 1].plot(ep, curve_df[f"{model}_d3_mean_m"], label=_MODEL_LABEL[model], color=_MODEL_COLOR[model], lw=1.7)
        ax[0, 1].fill_between(
            ep,
            curve_df[f"{model}_d3_mean_m"] - curve_df[f"{model}_d3_std_m"],
            curve_df[f"{model}_d3_mean_m"] + curve_df[f"{model}_d3_std_m"],
            color=_MODEL_COLOR[model],
            alpha=0.10,
        )
    ax[0, 1].set_title("3D Error Curve (Mean +/- Std)")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("3D Error (m)")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend(fontsize=8, ncol=2)

    for model in [m for m in show_models if m != "baseline" and f"{m}_corr_mean_m" in curve_df.columns]:
        ax[1, 0].plot(ep, curve_df[f"{model}_corr_mean_m"], label=_MODEL_LABEL[model], color=_MODEL_COLOR[model], lw=1.7)
        ax[1, 0].fill_between(
            ep,
            curve_df[f"{model}_corr_mean_m"] - curve_df[f"{model}_corr_std_m"],
            curve_df[f"{model}_corr_mean_m"] + curve_df[f"{model}_corr_std_m"],
            color=_MODEL_COLOR[model],
            alpha=0.10,
        )
    ax[1, 0].set_title("Mean Correction Magnitude (Mean +/- Std)")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("|corr| Mean (m)")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8, ncol=2)

    ep_loss = loss_df["epoch"].to_numpy(dtype=np.int64)
    for model in [m for m in show_models if f"{m}_val_d3_mean" in loss_df.columns]:
        ax[1, 1].plot(ep_loss, loss_df[f"{model}_val_d3_mean"], label=_MODEL_LABEL[model], color=_MODEL_COLOR[model], lw=1.7)
        ax[1, 1].fill_between(
            ep_loss,
            loss_df[f"{model}_val_d3_mean"] - loss_df[f"{model}_val_d3_std"],
            loss_df[f"{model}_val_d3_mean"] + loss_df[f"{model}_val_d3_std"],
            color=_MODEL_COLOR[model],
            alpha=0.10,
        )
    ax[1, 1].set_title("Validation 3D RMS (Mean +/- Std)")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("3D RMS Error (m)")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=700)
    plt.close(fig)
    return out


def main() -> None:
    cfg = _make_cmp_cfg(load_cfg())
    out_dir = Path(cfg.run.out_dir)

    run_lst: list[tuple[int, Cfg, dict[str, Any]]] = []
    for rep_idx in _rep_idx_lst(cfg):
        cfg_i = _make_rep_cfg(cfg, rep_idx)
        out = run_one(cfg_i)
        run_lst.append((int(rep_idx), cfg_i, out))

    run_df = _make_run_df(run_lst)
    summary_df = _make_summary(run_df)
    curve_df = _make_curve_df(run_lst)
    loss_df = _make_loss_df(run_lst)
    seed_df = _seed_df(cfg)

    tag = cfg.abl_ntcn.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / f"{tag}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / f"{tag}.xlsx") as xw:
        summary_df.to_excel(xw, sheet_name="summary", index=False)
        run_df.to_excel(xw, sheet_name="runs", index=False)
        curve_df.to_excel(xw, sheet_name="curve", index=False)
        loss_df.to_excel(xw, sheet_name="loss_val", index=False)
        seed_df.to_excel(xw, sheet_name="seed_plan", index=False)
    png = _plot(summary_df, curve_df, loss_df, out_dir, tag)

    print(summary_df.to_string(index=False))
    print(f"\npng: {png}")
    print(f"csv: {out_dir / f'{tag}.csv'}")
    print(f"xlsx: {out_dir / f'{tag}.xlsx'}")


if __name__ == "__main__":
    main()
