"""NavTCN 复杂度与时间统计。"""

from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ablation.common import make_cfg
from config import Cfg, load_cfg
import model.nav_tcn as ntcn_mod
from model.torch_utils import pick_dev
from sim.geometry import gen_geo
from sim.multipath import gen_mp
from sim.rx_signal import gen_rx
from sim.tracking import track
from sim.tx_signal import gen_tx

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


_TRAIN_N = 500
_MC_N = 8
_LR = 5e-3
_WD = 1e-4
_INFER_REP_N = 50


def _sync(dev: object | None = None) -> None:
    """同步设备计时。"""
    if torch is None:
        return
    if dev is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if getattr(dev, "type", None) == "cuda":
        torch.cuda.synchronize(dev)


def _count_params(model: object) -> int:
    """统计可训练参数量。"""
    return int(sum(int(p.numel()) for p in model.parameters()))


def _timeit(fn, rep_n: int = _INFER_REP_N, dev: object | None = None) -> float:
    """统计平均单次推理时间，单位 ms。"""
    fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(rep_n):
        fn()
    _sync(dev)
    return float((time.perf_counter() - t0) / rep_n * 1e3)


def _build_scene(cfg: Cfg):
    """构造正式经典场景观测。"""
    tx = gen_tx(cfg.sig)
    geo = gen_geo(cfg, tx)
    mp = gen_mp(cfg, tx, geo)
    rx = gen_rx(cfg, tx, geo, mp)
    obs = track(cfg, rx)
    return obs, geo


def _make_cfg(cfg: Cfg) -> Cfg:
    """构造 NavTCN 复杂度统计配置。"""
    cfg = make_cfg(cfg)
    dev_name = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return replace(
        cfg,
        dev=replace(cfg.dev, device=dev_name, cpu_mods=()),
        ntcn=replace(cfg.ntcn, use_torch=True, train_n=_TRAIN_N, mc_n=_MC_N, lr=_LR, wd=_WD),
    )


def measure_ntcn_cost(cfg: Cfg | None = None) -> pd.DataFrame:
    """测量 NavTCN 参数量、训练时间和推理时间。"""
    cfg = _make_cfg(load_cfg() if cfg is None else cfg)
    obs, geo = _build_scene(cfg)
    dev = pick_dev(cfg.dev, "ntcn") if torch is not None else None

    prep = ntcn_mod._prep_data(obs, geo, cfg)
    train = ntcn_mod._build_mc_train(cfg)
    train_unit_n = int(train.seq.shape[0] * train.seq.shape[1])

    t0 = time.perf_counter()
    model, _, _ = ntcn_mod._train_torch(train, cfg.ntcn, cfg.dev, cfg.pos, cfg)
    _sync(dev)
    train_time_s = float(time.perf_counter() - t0)

    def _infer() -> None:
        corr_pred, slow_pred, _, sig_pred = ntcn_mod._pred_torch(model, prep.seq, cfg.ntcn, cfg.dev)
        ntcn_mod._run_case(cfg, geo, prep, corr_pred, slow_pred, sig_pred)

    infer_time_ms = _timeit(_infer, dev=dev)
    df = pd.DataFrame(
        [
            {
                "model": "NavTCN",
                "param_n": _count_params(model),
                "mc_scene_n": int(cfg.ntcn.mc_n),
                "mc_sample_n": train_unit_n,
                "train_iter_n": int(cfg.ntcn.train_n),
                "train_time_s": train_time_s,
                "infer_time_ms": infer_time_ms,
                "device": str(dev),
                "gpu_name": torch.cuda.get_device_name(dev) if torch is not None and getattr(dev, "type", None) == "cuda" else "",
                "optimizer": "Adam",
                "lr": float(cfg.ntcn.lr),
                "wd": float(cfg.ntcn.wd),
            }
        ]
    )
    return df


def main() -> None:
    df = measure_ntcn_cost()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "model_cost.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "model_cost.xlsx") as xw:
        df.to_excel(xw, sheet_name="NavTCN", index=False)
    print(df.to_string(index=False))
    print(f"\ncsv: {out_dir / 'model_cost.csv'}")
    print(f"xlsx: {out_dir / 'model_cost.xlsx'}")


if __name__ == "__main__":
    main()
