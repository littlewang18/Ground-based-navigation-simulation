"""项目主入口。"""

from config import load_cfg
from model.mstcn import mstcn_stat, run_mstcn, save_mstcn
from model.nav_tcn import ntcn_stat, run_ntcn, save_ntcn
from model.tcn import run_tcn, save_tcn, tcn_stat
from sim.error_eval import calc_err, err_stat, save_err
from sim.geometry import gen_geo, geo_stat, save_geo
from sim.multipath import gen_mp, mp_stat, save_mp
from sim.positioning import pos_stat, save_pos, solve_pos
from sim.rx_signal import gen_rx, rx_stat, save_rx
from sim.tracking import save_track, track, track_stat
from sim.tx_signal import gen_tx, save_tx, tx_stat


def _save_if(cfg, data, save_fn, stage: int) -> None:
    if not cfg.run.save:
        return
    out = save_fn(data, cfg.run.out_dir)
    print(f"[阶段{stage}] 输出文件: {out}")


def _stop_if(cfg, stage: int) -> bool:
    if cfg.run.run_to == stage:
        print(f"按配置停止在阶段 {stage}。")
        return True
    return False


def run() -> None:
    cfg = load_cfg()
    print("=== 陆基导航仿真 ===")
    print(f"运行到阶段 {cfg.run.run_to}")

    tx = gen_tx(cfg.sig)
    print(f"[阶段1] 发射信号完成: {tx_stat(tx)}")
    _save_if(cfg, tx, save_tx, 1)
    if _stop_if(cfg, 1):
        return

    geo = gen_geo(cfg, tx)
    print(f"[阶段2] 几何场景完成: {geo_stat(geo)}")
    _save_if(cfg, geo, save_geo, 2)
    if _stop_if(cfg, 2):
        return

    mp = gen_mp(cfg, tx, geo)
    print(f"[阶段3] 多径生成完成: {mp_stat(mp)}")
    _save_if(cfg, mp, save_mp, 3)
    if _stop_if(cfg, 3):
        return

    rx = gen_rx(cfg, tx, geo, mp)
    print(f"[阶段4] 接收信号完成: {rx_stat(rx)}")
    _save_if(cfg, rx, save_rx, 4)
    if _stop_if(cfg, 4):
        return

    obs = track(cfg, rx)
    print(f"[阶段5] 跟踪完成: {track_stat(obs)}")
    _save_if(cfg, obs, save_track, 5)
    if _stop_if(cfg, 5):
        return

    pos = solve_pos(cfg, obs, geo)
    print(f"[阶段6] 定位完成: {pos_stat(pos)}")
    _save_if(cfg, pos, save_pos, 6)
    if _stop_if(cfg, 6):
        return

    err = calc_err(cfg, geo, pos)
    print(f"[阶段7] 误差评估完成: {err_stat(err)}")
    _save_if(cfg, err, save_err, 7)
    if _stop_if(cfg, 7):
        return

    tcn = run_tcn(cfg, obs, geo)
    print(f"[阶段8] TCN 完成: {tcn_stat(tcn)}")
    _save_if(cfg, tcn, save_tcn, 8)
    if _stop_if(cfg, 8):
        return

    mstcn = run_mstcn(cfg, obs, geo)
    print(f"[阶段9] MS-TCN 完成: {mstcn_stat(mstcn)}")
    _save_if(cfg, mstcn, save_mstcn, 9)
    if _stop_if(cfg, 9):
        return

    ntcn = run_ntcn(cfg, obs, geo)
    print(f"[阶段10] NavTCN 完成: {ntcn_stat(ntcn)}")
    _save_if(cfg, ntcn, save_ntcn, 10)
    if _stop_if(cfg, 10):
        return

    print("主线流程完成。")


if __name__ == "__main__":
    run()
