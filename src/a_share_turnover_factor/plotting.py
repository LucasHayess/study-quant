from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_plots(average_group_returns: pd.Series, ic_series: pd.Series, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = average_group_returns * 100
    bars = ax.bar(y.index.astype(str), y.values, color=["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("20日平均换手率因子五分组平均月收益")
    ax.set_xlabel("分组（G1低换手率，G5高换手率）")
    ax.set_ylabel("平均月收益（%）")
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in y.values], padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "group_average_monthly_return.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ic_series.index, ic_series.values, marker="o", linewidth=1.6, color="#4C78A8")
    ax.axhline(0, color="#333333", linewidth=0.8)
    mean_ic = ic_series.mean()
    if pd.notna(mean_ic):
        ax.axhline(mean_ic, color="#E45756", linestyle="--", linewidth=1.1, label=f"均值 IC={mean_ic:.3f}")
        ax.legend(frameon=False)
    ax.set_title("20日平均换手率因子 IC 时序")
    ax.set_xlabel("月末")
    ax.set_ylabel("Spearman IC")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "ic_series.png", dpi=180)
    plt.close(fig)
