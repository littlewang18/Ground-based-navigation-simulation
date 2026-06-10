"""按论文 Table VI 固定结果重新生成 Fig. 5。

该脚本不重新运行仿真实验，只读取已经与 Table VI 对齐的
``outputs/snr_robust_mapped.csv``，避免表格和曲线因随机实验重跑而漂移。
"""

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

from ablation.common import _COLOR

MODEL_ORDER = ["baseline", "Hatch", "Huber", "GRU", "TCN", "MSTCN", "NavTCN"]


def _load_table_vi(path: Path) -> pd.DataFrame:
    """读取与 Table VI 一致的汇总数据。"""
    df = pd.read_csv(path)
    need = {"snr_db", "model", "d3_rms_mean_m", "d3_rms_std_m"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"{path} 缺少列: {sorted(miss)}")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["model_order"] = df["model"].map(order_map)
    return df.sort_values(["snr_db", "model_order"]).reset_index(drop=True)


def _plot(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """生成高分辨率 PNG 和矢量 PDF。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig5_snr_robust_table_vi.png"
    pdf = out_dir / "fig5_snr_robust_table_vi.pdf"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("snr_db")
        x = sub["snr_db"].to_numpy(dtype=np.float64)
        y = sub["d3_rms_mean_m"].to_numpy(dtype=np.float64)
        s = sub["d3_rms_std_m"].to_numpy(dtype=np.float64)
        color = _COLOR.get(model, None)
        lw = 2.4 if model == "NavTCN" else 1.7
        zorder = 5 if model == "NavTCN" else 2
        ax.plot(x, y, marker="o", lw=lw, ms=5.2, label=model, color=color, zorder=zorder)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.12 if model == "NavTCN" else 0.07, zorder=1)

    ax.set_title("SNR Robustness under Compound Multipath Perturbations")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("3D RMS Error (m)")
    ax.set_xticks(sorted(df["snr_db"].unique()))
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(png, dpi=700, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    out_dir = ROOT / "outputs"
    src = out_dir / "snr_robust_mapped.csv"
    df = _load_table_vi(src)
    png, pdf = _plot(df, out_dir)
    # 论文正文仍引用旧文件名时，保持同步输出。
    df.to_csv(out_dir / "table_vi_for_fig5.csv", index=False, encoding="utf-8-sig")
    (out_dir / "snr_robust_mapped.png").write_bytes(png.read_bytes())
    print(f"source: {src}")
    print(f"png: {png}")
    print(f"pdf: {pdf}")
    print(f"synced: {out_dir / 'snr_robust_mapped.png'}")


if __name__ == "__main__":
    main()
