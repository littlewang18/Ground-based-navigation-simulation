"""项目全局配置。"""

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SigCfg:
    """发射信号参数。"""

    fs: float = 5e6
    code_rate: float = 1.023e6
    code_len: int = 1023
    nav_rate: float = 50.0
    fc: float = 1.25e6
    dur: float = 0.02
    amp: float = 1.0
    prn_seed: int = 20261304
    nav_seed: int = 20261305


@dataclass(frozen=True)
class RunCfg:
    """主流程控制参数。"""

    run_to: int = 10
    save: bool = True
    out_dir: Path = Path("outputs")


@dataclass(frozen=True)
class GeoCfg:
    """基站与接收机几何参数。"""

    bs_n: int = 6
    bs_mode: str = "circle"
    bs_r: float = 3000.0
    bs_h: float = 30.0
    bs_z_jit: float = 1.5
    bs_cx: float = 0.0
    bs_cy: float = 0.0
    bs_seed: int = 20261306
    rx_mode: str = "curve"
    rx_x0: float = 200.0
    rx_y0: float = -1200.0
    rx_z0: float = 2.0
    rx_vx: float = 15.0
    rx_vy: float = 4.0
    rx_vz: float = 0.0
    rx_ax: float = 1.0
    rx_ay: float = -0.5
    rx_wob_x: float = 1.5
    rx_wob_y: float = 2.0
    rx_wob_f: float = 4.0


@dataclass(frozen=True)
class MpCfg:
    """多径参数。"""

    n: int = 4
    dly_min: float = 50e-9
    dly_max: float = 600e-9
    k0: float = 0.45
    decay: float = 0.75
    jit: float = 0.13
    fade: float = 0.08
    fade_f_min: float = 6.0
    fade_f_max: float = 18.0
    dly_wob: float = 12e-9
    ph_rate: float = 60.0
    burst_n: int = 1
    burst_dur: float = 1.0e-3
    burst_gain: float = 1.4
    ph_mode: str = "rand"
    seed: int = 20261307


@dataclass(frozen=True)
class RxCfg:
    """接收信号参数。"""

    a0: float = 1.0
    snr_db: float = 16.5
    los_fade: float = 0.05
    los_fade_f: float = 6.0
    blk_n: int = 1
    blk_dur: float = 1.0e-3
    blk_gain: float = 0.75
    imp_n: int = 1
    imp_dur: float = 0.5e-3
    imp_gain: float = 3.0
    seed: int = 20261308


@dataclass(frozen=True)
class TrkCfg:
    """码跟踪与载波跟踪参数。"""

    epoch_s: float = 1e-3
    step_s: float = 1e-3
    lag_min_s: float = 0.0
    lag_max_s: float = 30e-6
    frac: bool = True
    amb_mode: str = "code"
    amb_seed: int = 20261309
    amb_n_min: int = -20
    amb_n_max: int = 20


@dataclass(frozen=True)
class PosCfg:
    """定位求解参数。"""

    mode: str = "auto"
    z_fix: float = 2.0
    z_init: float = 2.0
    min_bs_z_span: float = 30.0


@dataclass(frozen=True)
class DevCfg:
    """PyTorch 设备参数。"""

    device: str = "auto"
    cpu_mods: tuple[str, ...] = ("tcn", "mstcn", "ntcn", "gru", "ssm", "tfm")


@dataclass(frozen=True)
class TcnCfg:
    """TCN 质量评估模型参数。"""

    seq_n: int = 12
    hid_n: int = 16
    layer_n: int = 3
    ker_n: int = 3
    train_n: int = 900
    lr: float = 1e-2
    wd: float = 1e-4
    q_scale: float = 8.0
    w_min: float = 0.2
    w_max: float = 4.0
    seed: int = 20260312
    use_torch: bool = True
    mc_n: int = 8
    mc_seed_step: int = 97
    mc_rx_xy_jit: float = 60.0
    mc_rx_v_jit: float = 2.0
    mc_bs_xy_jit: float = 80.0
    mc_snr_jit: float = 3.0


@dataclass(frozen=True)
class MsTcnCfg:
    """MS-TCN 质量评估模型参数。"""

    seq_n: int = 10
    hid_n: int = 16
    layer_n: int = 1
    ker_lst: tuple[int, ...] = (1, 3, 5)
    train_n: int = 900
    lr: float = 1e-2
    wd: float = 1e-4
    q_scale: float = 6.0
    w_min: float = 0.1
    w_max: float = 5.0
    seed: int = 20260314
    use_torch: bool = True
    mc_n: int = 16
    mc_seed_step: int = 97
    mc_rx_xy_jit: float = 60.0
    mc_rx_v_jit: float = 2.0
    mc_bs_xy_jit: float = 80.0
    mc_snr_jit: float = 3.0


@dataclass(frozen=True)
class NTcnCfg:
    """慢变-稀疏 NavTCN 模型参数。"""

    seq_n: int = 10
    hid_n: int = 12
    layer_n: int = 1
    ker_lst: tuple[int, ...] = (1, 3, 5)
    slow_win_n: int = 5
    slow_use: float = 1.4
    cmc_clip: float = 8.0
    tar_geom_use: float = 0.80
    tar_res_use: float = 0.25
    pos_harm_use: bool = False
    pos_harm_corr_k: float = 1.00
    pos_harm_sig_k: float = 0.60
    spa_use: bool = False
    spa_gain: float = 0.50
    slow_gain_cmc_k: float = 0.80
    slow_gain_geo_k: float = 0.30
    slow_gain_scene_k: float = 0.80
    safe_res_use: float = 1.25
    safe_cmc_use: float = 1.10
    safe_med_use: float = 0.70
    gain_min: float = 0.40
    gain_max: float = 1.60
    sig_min: float = 0.50
    sig_max: float = 4.00
    geo_sig_use: float = 0.50
    geo_w_use: float = 0.30
    harm_w_use: float = 0.35
    w_temp: float = 0.90
    w_pow: float = 2.00
    sig_lam: float = 0.20
    sig_reg_lam: float = 0.02
    w_cal_lam: float = 0.10
    w_min: float = 0.50
    w_max: float = 2.50
    w_mix_min: float = 0.20
    w_mix_max: float = 0.65
    sig_nll_use: bool = False
    corr_lam: float = 1.00
    sign_lam: float = 0.20
    slow_lam: float = 0.80
    time_lam: float = 0.00
    pos_lam: float = 0.0
    res_lam: float = 0.05
    train_n: int = 500
    pretrain_n: int = 100
    lr: float = 5e-3
    wd: float = 1e-4
    q_scale: float = 6.0
    corr_lim: float = 30.0
    gn_iter_n: int = 6
    grad_clip: float = 1.0
    ema_use: bool = False
    ema_beta: float = 0.995
    ema_start_n: int = 20
    best_use: bool = False
    val_frac: float = 0.25
    corr_gain_use: float = 1.15
    post_sig_win_n: int = 1
    post_sig_floor: float = 1e-3
    post_sig_pred_use: float = 1.0
    post_sig_res_use: float = 0.3
    post_w_pow: float = 0.65
    post_w_min: float = 0.2
    post_w_max: float = 3.0
    batch_use: bool = True
    batch_lam_pos: float = 5000.0
    batch_it_n: int = 10
    pos_stage1_n: int = 10
    pos_ramp_n: int = 30
    seed: int = 20261315
    use_torch: bool = True
    mc_n: int = 8
    mc_seed_step: int = 97
    mc_rx_xy_jit: float = 12.0
    mc_rx_v_jit: float = 0.5
    mc_bs_xy_jit: float = 12.0
    mc_snr_jit: float = 0.8
    stealth_add_n: int = 0
    stealth_keep_n: int = 0
    stealth_geo_k: float = 1.0
    stealth_rx_scale: float = 1.55
    stealth_k0_use: float = 0.85
    stealth_v_use: float = 0.90
    hard_mc_use: bool = True
    hard_frac: float = 0.50
    hard_mp_add: int = 1
    hard_burst_add: int = 1
    hard_blk_add: int = 1
    hard_imp_add: int = 1
    hard_snr_drop: float = 1.5
    hard_jit_mul: float = 0.35
    hard_fade_mul: float = 0.35
    hard_k0_add: float = 0.16


@dataclass(frozen=True)
class AblNtcnCfg:
    """慢变-稀疏 NavTCN 消融实验配置。"""

    tag: str = "abl_03_ntcn"
    use_no_corr: bool = True
    use_no_slow: bool = True
    use_no_gain: bool = True
    use_no_sig: bool = True
    use_no_post_res: bool = True
    use_no_batch: bool = True


@dataclass(frozen=True)
class TotalCmpCfg:
    """模型对比实验统一配置。"""

    rep_n: int = 3
    rep_idx_lst: tuple[int, ...] = (0, 2, 7)
    seed_step: int = 1000
    legacy_label: bool = True
    train_n: int = 500
    lr: float = 5e-3
    wd: float = 1e-4
    q_scale: float = 6.0
    w_min: float = 0.10
    w_max: float = 5.00
    mc_n: int = 8


@dataclass(frozen=True)
class HatchCfg:
    """单独 Hatch 基线参数。"""

    n: int = 8


@dataclass(frozen=True)
class Cfg:
    """总配置。"""

    sig: SigCfg = field(default_factory=SigCfg)
    geo: GeoCfg = field(default_factory=GeoCfg)
    mp: MpCfg = field(default_factory=MpCfg)
    rx: RxCfg = field(default_factory=RxCfg)
    trk: TrkCfg = field(default_factory=TrkCfg)
    pos: PosCfg = field(default_factory=PosCfg)
    dev: DevCfg = field(default_factory=DevCfg)
    hatch: HatchCfg = field(default_factory=HatchCfg)
    tcn: TcnCfg = field(default_factory=TcnCfg)
    mstcn: MsTcnCfg = field(default_factory=MsTcnCfg)
    ntcn: NTcnCfg = field(default_factory=NTcnCfg)
    abl_ntcn: AblNtcnCfg = field(default_factory=AblNtcnCfg)
    total_cmp: TotalCmpCfg = field(default_factory=TotalCmpCfg)
    run: RunCfg = field(default_factory=RunCfg)


CFG_JSON_ENV = "LDNAV_CFG_JSON"


def _cfg_to_obj(x: Any) -> Any:
    """???????? JSON ?????????"""
    if is_dataclass(x):
        return {f.name: _cfg_to_obj(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, tuple):
        return [_cfg_to_obj(v) for v in x]
    return x


def cfg_to_dict(cfg: Cfg) -> dict[str, Any]:
    """??????????"""
    return _cfg_to_obj(cfg)


def _coerce_bool(x: Any) -> bool:
    """????????????"""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"???????: {x}")


def _coerce_seq(x: Any) -> list[Any]:
    """?????????????"""
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x).strip()
    if not s:
        return []
    return [v.strip() for v in s.split(",")]


def _coerce_like(raw: Any, ref: Any) -> Any:
    """?????????????????"""
    if is_dataclass(ref):
        if not isinstance(raw, dict):
            raise ValueError("?????????")
        return merge_cfg_dict(ref, raw)
    if isinstance(ref, bool):
        return _coerce_bool(raw)
    if isinstance(ref, int) and not isinstance(ref, bool):
        return int(raw)
    if isinstance(ref, float):
        return float(raw)
    if isinstance(ref, Path):
        return Path(raw)
    if isinstance(ref, tuple):
        seq = _coerce_seq(raw)
        if len(ref) == 0:
            return tuple(seq)
        item_ref = ref[0]
        return tuple(_coerce_like(v, item_ref) for v in seq)
    if isinstance(ref, str):
        return str(raw)
    return raw


def merge_cfg_dict(base_cfg: Any, patch: dict[str, Any]) -> Any:
    """?????????????"""
    if not is_dataclass(base_cfg):
        raise TypeError("base_cfg ??? dataclass ??")
    if not isinstance(patch, dict):
        raise TypeError("patch ?????")

    upd: dict[str, Any] = {}
    for f in fields(base_cfg):
        if f.name not in patch:
            continue
        cur = getattr(base_cfg, f.name)
        upd[f.name] = _coerce_like(patch[f.name], cur)
    return replace(base_cfg, **upd)


def save_cfg_json(cfg: Cfg, path: str | Path) -> Path:
    """????? JSON ???"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg_to_dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_cfg(path: str | Path | None = None) -> Cfg:
    """?????????? JSON ???"""
    cfg = Cfg()
    src = str(path).strip() if path is not None else str(os.environ.get(CFG_JSON_ENV, "")).strip()
    if not src:
        return cfg

    p = Path(src)
    data = json.loads(p.read_text(encoding="utf-8"))
    return merge_cfg_dict(cfg, data)
