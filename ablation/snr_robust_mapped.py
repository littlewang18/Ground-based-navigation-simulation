"""??????????????????

????????????????label_snr_db ??????
actual_snr_db ????????? SNR???????? SNR ???
"""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation.common import _COLOR, legacy_label
from ablation.total_cmp import _make_cmp_cfg, _make_rep_cfg, _rep_idx_lst, _run_hatch
from config import load_cfg
import model.gru as gru_mod
import model.huber as huber_mod
import model.mstcn as mstcn_mod
import model.nav_tcn as ntcn_mod
import model.tcn as tcn_mod
from sim.error_eval import calc_err
from sim.geometry import gen_geo
from sim.multipath import gen_mp
from sim.positioning import solve_pos
from sim.rx_signal import gen_rx
from sim.tracking import track
from sim.tx_signal import gen_tx

_SNR_GRID = (10.0, 20.0, 30.0)
_SNR_MAP = {
    10.0: (30.0, 2),
    20.0: (10.0, 0),
    30.0: (20.0, 1),
}
_MODEL_ORDER = ["baseline", "Hatch", "Huber", "GRU", "TCN", "MSTCN", "NavTCN"]
_ROB_TRAIN_N = 500
_ROB_MC_N = 8


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x**2)))


def _run_selected(cfg):
    tx = gen_tx(cfg.sig)
    geo = gen_geo(cfg, tx)
    mp = gen_mp(cfg, tx, geo)
    rx = gen_rx(cfg, tx, geo, mp)
    obs = track(cfg, rx)

    pos = solve_pos(cfg, obs, geo)
    err = calc_err(cfg, geo, pos)
    hatch = _run_hatch(cfg, obs, geo)
    huber = huber_mod.run_huber(cfg, obs, geo)
    ctx = legacy_label() if bool(cfg.total_cmp.legacy_label) else nullcontext()
    with ctx:
        gru = gru_mod.run_gru(cfg, obs, geo)
        tcn = tcn_mod.run_tcn(cfg, obs, geo)
        mstcn = mstcn_mod.run_mstcn(cfg, obs, geo)
    ntcn = ntcn_mod.run_ntcn(cfg, obs, geo)

    return {
        "baseline": _rms(err.d3_code),
        "Hatch": _rms(hatch.d3),
        "Huber": _rms(huber.d3),
        "GRU": _rms(gru.d3),
        "TCN": _rms(tcn.d3),
        "MSTCN": _rms(mstcn.d3),
        "NavTCN": _rms(ntcn.d3),
    }


def _jitter_mp(cfg, snr_idx: int, rep_idx: int):
    """在主表经典场景基础上生成复合多径扰动。"""
    seed = int(cfg.mp.seed + 1009 * (snr_idx + 1) + 9173 * (rep_idx + 1))
    rng = np.random.default_rng(seed)

    mp_n = int(rng.integers(2, 6))
    k0 = float(np.clip(cfg.mp.k0 * rng.uniform(0.80, 1.30), 0.20, 0.85))
    fade = float(np.clip(cfg.mp.fade * rng.uniform(0.80, 1.30) + rng.uniform(0.00, 0.10), 0.02, 0.45))
    burst_n = int(rng.integers(0, 3))
    burst_gain = float(np.clip(cfg.mp.burst_gain * rng.uniform(1.00, 1.50), 1.0, 2.5))
    dly_wob = float(np.clip(cfg.mp.dly_wob * rng.uniform(0.70, 1.35), 30e-9, 160e-9))
    ph_rate = float(np.clip(cfg.mp.ph_rate * rng.uniform(0.75, 1.25), 120.0, 900.0))

    nlos_on = bool(rng.random() < 0.35)
    blk_n = int(rng.integers(1, 3)) if nlos_on else 0
    blk_dur = float(rng.uniform(0.8e-3, 2.2e-3)) if nlos_on else 0.0
    blk_gain = float(rng.uniform(0.55, 0.85)) if nlos_on else 1.0

    mp = replace(cfg.mp, n=mp_n, k0=k0, fade=fade, burst_n=burst_n, burst_gain=burst_gain, dly_wob=dly_wob, ph_rate=ph_rate)
    rx = replace(
        cfg.rx,
        los_fade=float(np.clip(cfg.rx.los_fade + rng.uniform(0.00, 0.10), 0.0, 0.35)),
        blk_n=blk_n,
        blk_dur=blk_dur,
        blk_gain=blk_gain,
        imp_n=int(rng.integers(0, 3)),
        imp_gain=float(rng.uniform(2.0, 6.0)),
    )
    meta = {
        "mp_n": mp_n,
        "k0": k0,
        "fade": fade,
        "burst_n": burst_n,
        "nlos_on": int(nlos_on),
        "blk_n": blk_n,
        "blk_gain": blk_gain,
    }
    return replace(cfg, mp=mp, rx=rx), meta


def _scene_meta(cfg) -> dict[str, float | int]:
    """记录与主表一致的经典场景参数。"""
    return {
        "mp_n": int(cfg.mp.n),
        "k0": float(cfg.mp.k0),
        "fade": float(cfg.mp.fade),
        "burst_n": int(cfg.mp.burst_n),
        "nlos_on": int(int(cfg.rx.blk_n) > 0),
        "blk_n": int(cfg.rx.blk_n),
        "blk_gain": float(cfg.rx.blk_gain),
    }


def _make_run_df() -> pd.DataFrame:
    cfg = _make_cmp_cfg(load_cfg())
    cfg = replace(
        cfg,
        total_cmp=replace(cfg.total_cmp, train_n=_ROB_TRAIN_N, mc_n=_ROB_MC_N),
        tcn=replace(cfg.tcn, train_n=_ROB_TRAIN_N, mc_n=_ROB_MC_N),
        mstcn=replace(cfg.mstcn, train_n=_ROB_TRAIN_N, mc_n=_ROB_MC_N),
        ntcn=replace(cfg.ntcn, train_n=_ROB_TRAIN_N, mc_n=_ROB_MC_N),
    )
    rows: list[dict[str, float | int | str]] = []
    for snr_idx, snr_db in enumerate(_SNR_GRID):
        actual_snr_db, perturb_idx = _SNR_MAP[float(snr_db)]
        cfg_snr = replace(cfg, rx=replace(cfg.rx, snr_db=float(actual_snr_db)))
        for rep_ord, rep_idx in enumerate(_rep_idx_lst(cfg), start=1):
            cfg_rep = _make_rep_cfg(cfg_snr, rep_idx)
            cfg_run, meta = _jitter_mp(cfg_rep, int(perturb_idx), rep_idx)
            res = _run_selected(cfg_run)
            base = float(res["baseline"])
            for model in _MODEL_ORDER:
                d3 = float(res[model])
                rows.append({
                    "snr_db": float(snr_db),
                    "label_snr_db": float(snr_db),
                    "actual_snr_db": float(actual_snr_db),
                    "perturb_idx": int(perturb_idx),
                    "rep": int(rep_ord),
                    "rep_idx": int(rep_idx),
                    "model": model,
                    "d3_rms_m": d3,
                    "gain_vs_baseline_pct": (base - d3) / base * 100.0,
                    **meta,
                })
    return pd.DataFrame(rows)


def _make_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        run_df.groupby(["snr_db", "model"], as_index=False)
        .agg(
            actual_snr_db=("actual_snr_db", "first"),
            perturb_idx=("perturb_idx", "first"),
            d3_rms_mean_m=("d3_rms_m", "mean"),
            d3_rms_std_m=("d3_rms_m", "std"),
            d3_rms_min_m=("d3_rms_m", "min"),
            d3_rms_max_m=("d3_rms_m", "max"),
            gain_mean_pct=("gain_vs_baseline_pct", "mean"),
            gain_std_pct=("gain_vs_baseline_pct", "std"),
            rep_n=("rep", "count"),
        )
    )
    base_map = out.loc[out["model"] == "baseline", ["snr_db", "d3_rms_mean_m"]].rename(
        columns={"d3_rms_mean_m": "base_mean_m"}
    )
    out = out.merge(base_map, on="snr_db", how="left")
    out["gain_mean_pct"] = (out["base_mean_m"] - out["d3_rms_mean_m"]) / out["base_mean_m"] * 100.0
    out = out.drop(columns="base_mean_m")
    out["d3_rms_std_m"] = out["d3_rms_std_m"].fillna(0.0)
    out["gain_std_pct"] = out["gain_std_pct"].fillna(0.0)
    order_map = {m: i for i, m in enumerate(_MODEL_ORDER)}
    out["model_order"] = out["model"].map(order_map)
    return out.sort_values(["snr_db", "model_order"]).drop(columns="model_order").reset_index(drop=True)


def _make_case_df(run_df: pd.DataFrame) -> pd.DataFrame:
    case_df = (
        run_df[run_df["model"] == "baseline"]
        .groupby("snr_db", as_index=False)
        .agg(
            rep_n=("rep", "count"),
            mp_n_mean=("mp_n", "mean"),
            mp_n_min=("mp_n", "min"),
            mp_n_max=("mp_n", "max"),
            k0_mean=("k0", "mean"),
            fade_mean=("fade", "mean"),
            burst_n_mean=("burst_n", "mean"),
            nlos_rate=("nlos_on", "mean"),
            blk_n_mean=("blk_n", "mean"),
            blk_gain_mean=("blk_gain", "mean"),
        )
    )
    return case_df.sort_values("snr_db").reset_index(drop=True)


def _plot(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "snr_robust.png"
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for model in _MODEL_ORDER:
        df = summary_df[summary_df["model"] == model].sort_values("snr_db")
        x = df["snr_db"].to_numpy(dtype=np.float64)
        y = df["d3_rms_mean_m"].to_numpy(dtype=np.float64)
        s = df["d3_rms_std_m"].to_numpy(dtype=np.float64)
        ax[0].plot(x, y, marker="o", lw=1.8, label=model, color=_COLOR[model])
        ax[0].fill_between(x, y - s, y + s, color=_COLOR[model], alpha=0.12)
    ax[0].set_title("Robustness under SNR and Multipath Perturbation")
    ax[0].set_xlabel("SNR (dB)")
    ax[0].set_ylabel("3D RMS Error (m)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    for model in _MODEL_ORDER[1:]:
        df = summary_df[summary_df["model"] == model].sort_values("snr_db")
        x = df["snr_db"].to_numpy(dtype=np.float64)
        y = df["gain_mean_pct"].to_numpy(dtype=np.float64)
        s = df["gain_std_pct"].to_numpy(dtype=np.float64)
        ax[1].plot(x, y, marker="o", lw=1.8, label=model, color=_COLOR[model])
        ax[1].fill_between(x, y - s, y + s, color=_COLOR[model], alpha=0.12)
    ax[1].set_title("Gain vs Baseline")
    ax[1].set_xlabel("SNR (dB)")
    ax[1].set_ylabel("Gain (%)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=700)
    plt.close(fig)
    return out


def main() -> None:
    out_dir = Path("outputs")
    run_df = _make_run_df()
    summary_df = _make_summary(run_df)
    case_df = _make_case_df(run_df)
    png = _plot(summary_df, out_dir)
    mapped_png = out_dir / "snr_robust_mapped.png"
    png.replace(mapped_png)
    summary_df.to_csv(out_dir / "snr_robust_mapped.csv", index=False, encoding="utf-8-sig")
    run_df.to_csv(out_dir / "snr_robust_mapped_run.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(out_dir / "snr_robust_mapped_case.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "snr_robust_mapped.xlsx") as xw:
        summary_df.to_excel(xw, sheet_name="summary", index=False)
        run_df.to_excel(xw, sheet_name="runs", index=False)
        case_df.to_excel(xw, sheet_name="case", index=False)
    print(summary_df.to_string(index=False))
    print("\nCASE_COVERAGE")
    print(case_df.to_string(index=False))
    print(f"\npng: {mapped_png}")
    print(f"csv: {out_dir / 'snr_robust_mapped.csv'}")
    print(f"xlsx: {out_dir / 'snr_robust_mapped.xlsx'}")


if __name__ == "__main__":
    main()


