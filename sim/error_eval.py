"""定位误差评估。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import Cfg
from sim.geometry import GeoData
from sim.positioning import PosData


@dataclass
class ErrData:
    """时序定位误差结果。"""

    t_ep: np.ndarray  # 历元时刻 (n_ep,)
    true_pos: np.ndarray  # 真值坐标 (n_ep, 3)
    e_code: np.ndarray  # 码伪距定位误差向量 (n_ep, 3)
    e_car: np.ndarray  # 载波伪距定位误差向量 (n_ep, 3)
    d3_code: np.ndarray  # 码伪距 3D 误差 (n_ep,)
    d3_car: np.ndarray  # 载波伪距 3D 误差 (n_ep,)
    dh_code: np.ndarray  # 码伪距水平误差 (n_ep,)
    dh_car: np.ndarray  # 载波伪距水平误差 (n_ep,)
    dz_code: np.ndarray  # 码伪距垂直误差 (n_ep,)
    dz_car: np.ndarray  # 载波伪距垂直误差 (n_ep,)


def calc_err(cfg: Cfg, geo: GeoData, pos: PosData) -> ErrData:
    """计算时序定位误差。"""
    _ = cfg  # 当前阶段无需额外参数，保留接口一致性
    if geo.rx.ndim != 2 or geo.rx.shape[1] != 3:
        raise ValueError("geo.rx 必须是 (n_t, 3) 的坐标矩阵")
    if geo.t.ndim != 1 or geo.t.size != geo.rx.shape[0]:
        raise ValueError("geo.t 与 geo.rx 长度不一致")
    if pos.pos_code.shape[0] != pos.t_ep.size or pos.pos_car.shape[0] != pos.t_ep.size:
        raise ValueError("定位结果与历元时刻长度不一致")

    t_ep = np.clip(pos.t_ep, geo.t[0], geo.t[-1]).astype(np.float64)
    true_pos = np.column_stack(
        (
            np.interp(t_ep, geo.t, geo.rx[:, 0]),
            np.interp(t_ep, geo.t, geo.rx[:, 1]),
            np.interp(t_ep, geo.t, geo.rx[:, 2]),
        )
    )

    e_code = pos.pos_code - true_pos
    e_car = pos.pos_car - true_pos

    dh_code = np.linalg.norm(e_code[:, :2], axis=1)
    dh_car = np.linalg.norm(e_car[:, :2], axis=1)
    dz_code = e_code[:, 2]
    dz_car = e_car[:, 2]
    d3_code = np.linalg.norm(e_code, axis=1)
    d3_car = np.linalg.norm(e_car, axis=1)

    return ErrData(
        t_ep=t_ep,
        true_pos=true_pos,
        e_code=e_code,
        e_car=e_car,
        d3_code=d3_code,
        d3_car=d3_car,
        dh_code=dh_code,
        dh_car=dh_car,
        dz_code=dz_code,
        dz_car=dz_car,
    )


def err_stat(data: ErrData) -> Dict[str, Any]:
    """输出误差摘要。"""
    return {
        "n_ep": int(data.t_ep.size),
        "t0_s": float(data.t_ep[0]),
        "t1_s": float(data.t_ep[-1]),
        "d3_code_rms_m": float(np.sqrt(np.mean(data.d3_code**2))),
        "d3_car_rms_m": float(np.sqrt(np.mean(data.d3_car**2))),
        "d3_code_max_m": float(np.max(data.d3_code)),
        "d3_car_max_m": float(np.max(data.d3_car)),
        "dh_code_rms_m": float(np.sqrt(np.mean(data.dh_code**2))),
        "dh_car_rms_m": float(np.sqrt(np.mean(data.dh_car**2))),
    }


def save_err(data: ErrData, out_dir: Path) -> Path:
    """保存误差结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "err.npz"
    np.savez(
        out,
        t_ep=data.t_ep,
        true_pos=data.true_pos,
        e_code=data.e_code,
        e_car=data.e_car,
        d3_code=data.d3_code,
        d3_car=data.d3_car,
        dh_code=data.dh_code,
        dh_car=data.dh_car,
        dz_code=data.dz_code,
        dz_car=data.dz_car,
    )
    return out
