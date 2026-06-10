"""GRU 扩展对比模型。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import Cfg
from model.common_utils import build_mc_train, build_q_eval_case, build_seq4, calc_pos_err, code_postfit_res, fill_val_curve, interp_pos, interp_rng, make_q_w, solve_w_epochs
from model.label_utils import make_pos_q
from model.torch_utils import pick_dev, seed_torch, to_f32
from sim.geometry import GeoData
from sim.tracking import ObsData

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


_VAL_GAP_N = 10


@dataclass(frozen=True)
class GruCfg:
    """GRU 默认参数。"""

    seq_n: int = 8
    hid_n: int = 16
    layer_n: int = 1
    train_n: int = 500
    lr: float = 5e-3
    wd: float = 1e-4
    q_scale: float = 6.0
    w_min: float = 0.1
    w_max: float = 5.0
    seed: int = 20260321
    use_torch: bool = True
    mc_n: int = 8
    mc_seed_step: int = 97
    mc_rx_xy_jit: float = 60.0
    mc_rx_v_jit: float = 2.0
    mc_bs_xy_jit: float = 80.0
    mc_snr_jit: float = 3.0


@dataclass
class GruPrepData:
    """GRU 预处理数据。"""

    rho_true: np.ndarray
    mp_raw: np.ndarray
    rho_hat: np.ndarray
    mp_hat: np.ndarray
    seq: np.ndarray
    q_tar: np.ndarray


@dataclass
class GruData:
    """GRU 输出。"""

    t_ep: np.ndarray
    rho_true: np.ndarray
    mp_raw: np.ndarray
    rho_hat: np.ndarray
    mp_hat: np.ndarray
    seq: np.ndarray
    q_tar: np.ndarray
    q_pred: np.ndarray
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


def _cfg(cfg: Cfg) -> GruCfg:
    base = getattr(cfg, "total_cmp", None)
    if base is None:
        return GruCfg()
    return GruCfg(train_n=int(base.train_n), lr=float(base.lr), wd=float(base.wd), q_scale=float(base.q_scale), w_min=float(base.w_min), w_max=float(base.w_max), mc_n=int(base.mc_n))


def _prep_data(obs: ObsData, geo: GeoData, cfg: Cfg) -> GruPrepData:
    sub = _cfg(cfg)
    rho_true = interp_rng(geo, obs.t_ep)
    mp_raw = obs.rho_code - rho_true
    rho_hat = obs.rho_code.copy()
    mp_hat = code_postfit_res(cfg, geo, obs)
    seq = build_seq4(mp_hat, sub.seq_n, "gru.seq_n")
    q_tar = make_pos_q(cfg, geo, rho_hat, rho_true, obs.t_ep, sub.q_scale)
    return GruPrepData(rho_true=rho_true, mp_raw=mp_raw, rho_hat=rho_hat, mp_hat=mp_hat, seq=seq, q_tar=q_tar)


class GruNet(nn.Module):
    """轻量 GRU 质量预测网络。"""

    def __init__(self, in_n: int, hid_n: int, layer_n: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=in_n, hidden_size=hid_n, num_layers=layer_n, batch_first=True)
        self.out = nn.Linear(hid_n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return torch.sigmoid(self.out(y[:, -1]))


def _eval_val_d3(model: GruNet, eval_pack: Any, sub: GruCfg, dev_cfg: Any) -> float:
    """计算当前正式场景上的按 epoch 测试误差。"""
    q_pred = _pred_torch(model, eval_pack.seq_eval, dev_cfg)
    q_pred = np.clip(q_pred, 0.0, 1.0)
    w = make_q_w(q_pred, sub.w_min, sub.w_max, eval_pack.n_ep, eval_pack.n_bs)
    pos, _, _, _, _ = solve_w_epochs(eval_pack.cfg, eval_pack.geo, eval_pack.prep.rho_hat, w)
    _, _, _, d3 = calc_pos_err(pos, eval_pack.true_pos)
    return float(np.sqrt(np.mean(d3**2)))


def _train_torch(seq: np.ndarray, q_tar: np.ndarray, sub: GruCfg, dev_cfg: Any, eval_pack: Any | None = None) -> tuple[GruNet, np.ndarray, np.ndarray]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("当前环境缺少 PyTorch，无法运行 GRU")
    dev = pick_dev(dev_cfg, "gru")
    seed_torch(sub.seed, dev)
    model = GruNet(seq.shape[-1], sub.hid_n, sub.layer_n).to(dev)
    x = to_f32(seq, dev)
    y = to_f32(q_tar[:, None], dev)
    opt = torch.optim.Adam(model.parameters(), lr=sub.lr, weight_decay=sub.wd)
    loss_hist: list[float] = []
    val_hist = np.full(int(sub.train_n), np.nan, dtype=np.float64)
    model.train()
    for ep_idx in range(sub.train_n):
        opt.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.detach().cpu()))
        if eval_pack is not None and (ep_idx == 0 or (ep_idx + 1) % _VAL_GAP_N == 0 or ep_idx == sub.train_n - 1):
            val_hist[ep_idx] = _eval_val_d3(model, eval_pack, sub, dev_cfg)
    return model, np.asarray(loss_hist, dtype=np.float64), fill_val_curve(val_hist)


def _pred_torch(model: GruNet, seq: np.ndarray, dev_cfg: Any) -> np.ndarray:
    dev = pick_dev(dev_cfg, "gru")
    x = to_f32(seq, dev)
    model.eval()
    with torch.no_grad():
        return model(x).squeeze(-1).cpu().numpy()


def _pred_rule(seq: np.ndarray, sub: GruCfg) -> np.ndarray:
    last = seq[:, -1, 0]
    rms = seq[:, -1, 3]
    return np.exp(-(np.abs(last) + rms) / max(sub.q_scale, 1e-6)).astype(np.float64)


def run_gru(cfg: Cfg, obs: ObsData, geo: GeoData) -> GruData:
    sub = _cfg(cfg)
    prep = _prep_data(obs, geo, cfg)
    seq_train, q_train = build_mc_train(cfg, sub, _prep_data)
    eval_pack = build_q_eval_case(cfg, sub, obs, geo, prep)
    n_ep, n_bs = prep.mp_hat.shape
    seq_eval = prep.seq.reshape(n_ep * n_bs, sub.seq_n, prep.seq.shape[-1])
    if sub.use_torch and torch is not None:
        model, loss, val_d3 = _train_torch(seq_train, q_train, sub, cfg.dev, eval_pack)
        q_pred = _pred_torch(model, seq_eval, cfg.dev)
    else:
        q_pred = _pred_rule(seq_eval, sub)
        loss = np.zeros(1, dtype=np.float64)
        val_d3 = np.full(1, np.nan, dtype=np.float64)
    q_pred = np.clip(q_pred, 0.0, 1.0)
    w = make_q_w(q_pred, sub.w_min, sub.w_max, n_ep, n_bs)
    pos, cb, res, dim, it = solve_w_epochs(cfg, geo, prep.rho_hat, w)
    true_pos = interp_pos(geo, obs.t_ep)
    e, dh, dz, d3 = calc_pos_err(pos, true_pos)
    return GruData(
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
        mc_n=int(sub.mc_n),
        train_sample_n=int(seq_train.shape[0]),
    )


def gru_stat(data: GruData) -> Dict[str, Any]:
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


def save_gru(data: GruData, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gru.npz"
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
