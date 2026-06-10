"""TCN 权重模型。"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import Cfg, TcnCfg
from model.common_utils import (
    build_mc_train,
    build_q_eval_case,
    build_seq4,
    calc_pos_err,
    code_postfit_res,
    fill_val_curve,
    interp_pos,
    interp_rng,
    make_mc_cfg,
    make_q_w,
    sim_obs_geo,
    solve_w_epochs,
)
from model.label_utils import make_pos_q
from model.torch_utils import pick_dev, seed_torch, to_f32
from sim.geometry import GeoData
from sim.positioning import pick_dim
from sim.tracking import ObsData, track

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


_VAL_GAP_N = 10


@dataclass
class TcnPrepData:
    """TCN 预处理数据。"""

    rho_true: np.ndarray  # 真值几何距离 (n_ep, n_bs)
    mp_raw: np.ndarray  # 原始多径残差 (n_ep, n_bs)
    rho_hat: np.ndarray  # 模型输入伪距 (n_ep, n_bs)
    mp_hat: np.ndarray  # 模型输入残差 (n_ep, n_bs)
    seq: np.ndarray  # 时间序列特征 (n_ep, n_bs, seq_n, feat_n)
    q_tar: np.ndarray  # 监督质量标签 (n_ep, n_bs)


@dataclass
class TcnData:
    """TCN 模型输出。"""

    t_ep: np.ndarray  # 历元时刻 (n_ep,)
    rho_true: np.ndarray  # 真值几何距离 (n_ep, n_bs)
    mp_raw: np.ndarray  # 原始多径残差 (n_ep, n_bs)
    rho_hat: np.ndarray  # 模型输入伪距 (n_ep, n_bs)
    mp_hat: np.ndarray  # 模型输入残差 (n_ep, n_bs)
    seq: np.ndarray  # 时间序列特征 (n_ep, n_bs, seq_n, feat_n)
    q_tar: np.ndarray  # 目标质量 (n_ep, n_bs)
    q_pred: np.ndarray  # TCN 预测质量 (n_ep, n_bs)
    w: np.ndarray  # 动态权重 (n_ep, n_bs)
    pos: np.ndarray  # 加权定位坐标 (n_ep, 3)
    cb: np.ndarray  # 加权定位钟差 (n_ep,)
    res: np.ndarray  # 加权定位残差 (n_ep, n_bs)
    dim: int  # 解算维数 2 / 3
    it: np.ndarray  # 每历元迭代次数 (n_ep,)
    true_pos: np.ndarray  # 真值坐标 (n_ep, 3)
    e: np.ndarray  # 位置误差向量 (n_ep, 3)
    d3: np.ndarray  # 三维误差 (n_ep,)
    dh: np.ndarray  # 水平误差 (n_ep,)
    dz: np.ndarray  # 垂直误差 (n_ep,)
    loss: np.ndarray  # 训练损失曲线
    val_d3: np.ndarray  # 正式场景按 epoch 测试三维误差曲线
    mc_n: int  # Monte Carlo 训练次数
    train_sample_n: int  # 训练样本数


def _interp_rng(geo: GeoData, t_ep: np.ndarray) -> np.ndarray:
    """?????????????"""
    return interp_rng(geo, t_ep)


def _interp_pos(geo: GeoData, t_ep: np.ndarray) -> np.ndarray:
    """???????????"""
    return interp_pos(geo, t_ep)



def _build_seq(res: np.ndarray, seq_n: int) -> np.ndarray:
    """???????"""
    return build_seq4(res, seq_n, "tcn.seq_n")


def _prep_data(obs: ObsData, geo: GeoData, cfg: Cfg) -> TcnPrepData:
    """构造 TCN 的输入与监督。"""
    rho_true = _interp_rng(geo, obs.t_ep)
    mp_raw = obs.rho_code - rho_true
    rho_hat = obs.rho_code.copy()
    mp_hat = code_postfit_res(cfg, geo, obs)
    seq = _build_seq(mp_hat, cfg.tcn.seq_n)
    q_tar = make_pos_q(cfg, geo, rho_hat, rho_true, obs.t_ep, cfg.tcn.q_scale)
    return TcnPrepData(
        rho_true=rho_true,
        mp_raw=mp_raw,
        rho_hat=rho_hat,
        mp_hat=mp_hat,
        seq=seq,
        q_tar=q_tar,
    )


def _sim_obs_geo(cfg: Cfg) -> tuple[ObsData, GeoData]:
    """?? 1~5 ?????????????"""
    return sim_obs_geo(cfg)


def _make_mc_cfg(cfg: Cfg, idx: int) -> Cfg:
    """???? Monte Carlo ?????"""
    return make_mc_cfg(cfg, cfg.tcn, idx)


def _build_mc_train(cfg: Cfg) -> tuple[np.ndarray, np.ndarray]:
    """?? Monte Carlo ????"""
    return build_mc_train(cfg, cfg.tcn, _prep_data)


class Chomp1d(nn.Module):
    """裁掉卷积右侧填充，保持因果性。"""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = int(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算。"""
        if self.n <= 0:
            return x
        return x[:, :, :-self.n]


class TemporalBlock(nn.Module):
    """单个 TCN 膨胀卷积块。"""

    def __init__(self, in_n: int, out_n: int, ker_n: int, dil: int) -> None:
        super().__init__()
        pad = (ker_n - 1) * dil
        self.net = nn.Sequential(
            nn.Conv1d(in_n, out_n, ker_n, padding=pad, dilation=dil),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Conv1d(out_n, out_n, ker_n, padding=pad, dilation=dil),
            Chomp1d(pad),
            nn.ReLU(),
        )
        self.proj = nn.Conv1d(in_n, out_n, 1) if in_n != out_n else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算。"""
        y = self.net(x)
        z = self.proj(x)
        return F.relu(y + z)


class TcnNet(nn.Module):
    """轻量 TCN 质量预测网络。"""

    def __init__(self, in_n: int, hid_n: int, layer_n: int, ker_n: int) -> None:
        super().__init__()
        mods = []
        c_in = in_n
        for i in range(layer_n):
            mods.append(TemporalBlock(c_in, hid_n, ker_n, dil=2**i))
            c_in = hid_n
        self.net = nn.Sequential(*mods)
        self.out = nn.Linear(hid_n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算。"""
        y = x.transpose(1, 2)
        y = self.net(y)
        z = y[:, :, -1]
        return torch.sigmoid(self.out(z))


def _eval_val_d3(model: TcnNet, eval_pack: Any, cfg: TcnCfg, dev_cfg: Any) -> float:
    """计算当前正式场景上的按 epoch 测试误差。"""
    q_pred = _pred_torch(model, eval_pack.seq_eval, dev_cfg)
    q_pred = np.clip(q_pred, 0.0, 1.0)
    w = _make_w(q_pred, cfg, eval_pack.n_ep, eval_pack.n_bs)
    pos, _, _, _, _ = _solve_w(eval_pack.cfg, eval_pack.geo, eval_pack.prep.rho_hat, w)
    _, _, _, d3 = calc_pos_err(pos, eval_pack.true_pos)
    return float(np.sqrt(np.mean(d3**2)))


def _train_torch(seq: np.ndarray, q_tar: np.ndarray, cfg: TcnCfg, dev_cfg: Any, eval_pack: Any | None = None) -> tuple[TcnNet, np.ndarray, np.ndarray]:
    """训练 Torch 版 TCN。"""
    if torch is None or nn is None or F is None:
        raise RuntimeError("当前环境缺少 PyTorch，无法运行 Torch 版 TCN")

    dev = pick_dev(dev_cfg, "tcn")
    seed_torch(cfg.seed, dev)
    model = TcnNet(in_n=seq.shape[-1], hid_n=cfg.hid_n, layer_n=cfg.layer_n, ker_n=cfg.ker_n).to(dev)
    x = to_f32(seq, dev)
    y = to_f32(q_tar[:, None], dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    loss_hist = []
    val_hist = np.full(int(cfg.train_n), np.nan, dtype=np.float64)
    model.train()
    for ep_idx in range(cfg.train_n):
        opt.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.detach().cpu()))
        if eval_pack is not None and (ep_idx == 0 or (ep_idx + 1) % _VAL_GAP_N == 0 or ep_idx == cfg.train_n - 1):
            val_hist[ep_idx] = _eval_val_d3(model, eval_pack, cfg, dev_cfg)

    return model, np.asarray(loss_hist, dtype=np.float64), fill_val_curve(val_hist)


def _pred_torch(model: TcnNet, seq: np.ndarray, dev_cfg: Any) -> np.ndarray:
    """Torch 模型推理。"""
    dev = pick_dev(dev_cfg, "tcn")
    x = to_f32(seq, dev)
    model.eval()
    with torch.no_grad():
        return model(x).squeeze(-1).cpu().numpy()


def _pred_rule(seq: np.ndarray, cfg: TcnCfg) -> np.ndarray:
    """Torch 不可用时的规则回退。"""
    last = seq[:, -1, 0]
    rms = seq[:, -1, 3]
    return np.exp(-(np.abs(last) + rms) / max(float(cfg.q_scale), 1e-6)).astype(np.float64)


def _fit_predict(
    seq_train: np.ndarray,
    q_train: np.ndarray,
    seq_eval: np.ndarray,
    cfg: TcnCfg,
    dev_cfg: Any,
    eval_pack: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """先训练，再对当前序列推理。"""
    if cfg.use_torch and torch is not None:
        model, loss, val_d3 = _train_torch(seq_train, q_train, cfg, dev_cfg, eval_pack)
        q_pred = _pred_torch(model, seq_eval, dev_cfg)
        return q_pred, loss, val_d3

    q_pred = _pred_rule(seq_eval, cfg)
    return q_pred, np.zeros(1, dtype=np.float64), np.full(1, np.nan, dtype=np.float64)


def _make_w(q_pred: np.ndarray, cfg: TcnCfg, n_ep: int, n_bs: int) -> np.ndarray:
    """?????????????"""
    return make_q_w(q_pred, cfg.w_min, cfg.w_max, n_ep, n_bs)


def _solve_w(cfg: Cfg, geo: GeoData, rho: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """?????????"""
    return solve_w_epochs(cfg, geo, rho, w)


def run_tcn(cfg: Cfg, obs: ObsData, geo: GeoData) -> TcnData:
    """运行 TCN 模型并输出加权定位结果。"""
    if obs.rho_code.ndim != 2 or obs.rho_car.ndim != 2:
        raise ValueError("obs.rho_code / obs.rho_car 必须是二维矩阵")

    prep = _prep_data(obs, geo, cfg)
    seq_train, q_train = _build_mc_train(cfg)
    eval_pack = build_q_eval_case(cfg, cfg.tcn, obs, geo, prep)

    n_ep, n_bs = prep.mp_hat.shape
    seq_eval = prep.seq.reshape(n_ep * n_bs, cfg.tcn.seq_n, prep.seq.shape[-1])
    q_pred, loss, val_d3 = _fit_predict(seq_train, q_train, seq_eval, cfg.tcn, cfg.dev, eval_pack)
    q_pred = np.clip(q_pred, 0.0, 1.0)
    w = _make_w(q_pred, cfg.tcn, n_ep, n_bs)

    pos, cb, res, dim, it = _solve_w(cfg, geo, prep.rho_hat, w)
    true_pos = _interp_pos(geo, obs.t_ep)
    e, dh, dz, d3 = calc_pos_err(pos, true_pos)

    return TcnData(
        t_ep=obs.t_ep,
        rho_true=prep.rho_true,
        mp_raw=prep.mp_raw,
        rho_hat=prep.rho_hat,
        mp_hat=prep.mp_hat,
        seq=prep.seq,
        q_tar=prep.q_tar,
        q_pred=q_pred.reshape(n_ep, n_bs),
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
        loss=loss,
        val_d3=val_d3,
        mc_n=int(cfg.tcn.mc_n),
        train_sample_n=int(seq_train.shape[0]),
    )


def tcn_stat(data: TcnData) -> Dict[str, Any]:
    """输出 TCN 摘要。"""
    return {
        "dim": int(data.dim),
        "n_ep": int(data.t_ep.size),
        "mc_n": int(data.mc_n),
        "train_sample_n": int(data.train_sample_n),
        "mp_raw_rms_m": float(np.sqrt(np.mean(data.mp_raw**2))),
        "mp_hat_rms_m": float(np.sqrt(np.mean(data.mp_hat**2))),
        "q_pred_min": float(np.min(data.q_pred)),
        "q_pred_max": float(np.max(data.q_pred)),
        "w_min": float(np.min(data.w)),
        "w_max": float(np.max(data.w)),
        "d3_rms_m": float(np.sqrt(np.mean(data.d3**2))),
        "dh_rms_m": float(np.sqrt(np.mean(data.dh**2))),
        "loss_last": float(data.loss[-1]),
    }


def save_tcn(data: TcnData, out_dir: Path) -> Path:
    """保存 TCN 结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "tcn.npz"
    np.savez(
        out,
        t_ep=data.t_ep,
        rho_true=data.rho_true,
        mp_raw=data.mp_raw,
        rho_hat=data.rho_hat,
        mp_hat=data.mp_hat,
        seq=data.seq,
        q_tar=data.q_tar,
        q_pred=data.q_pred,
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
