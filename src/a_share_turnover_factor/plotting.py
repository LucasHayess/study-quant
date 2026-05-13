from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_plots(
    average_group_returns: pd.Series,
    ic_series: pd.Series,
    output_dir: Path,
    group_title: str = "20日平均换手率因子五分组平均月收益",
    ic_title: str = "20日平均换手率因子 IC 时序",
    period_label: str = "月",
    group_xlabel: str = "分组（G1低换手率，G5高换手率）",
    group_filename: str = "group_average_monthly_return.png",
    ic_filename: str = "ic_series.png",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = average_group_returns * 100
    bars = ax.bar(y.index.astype(str), y.values, color=["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(group_title)
    ax.set_xlabel(group_xlabel)
    ax.set_ylabel(f"平均{period_label}收益（%）")
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in y.values], padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / group_filename, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ic_series.index, ic_series.values, marker="o", linewidth=1.6, color="#4C78A8")
    ax.axhline(0, color="#333333", linewidth=0.8)
    mean_ic = ic_series.mean()
    if pd.notna(mean_ic):
        ax.axhline(mean_ic, color="#E45756", linestyle="--", linewidth=1.1, label=f"均值 IC={mean_ic:.3f}")
        ax.legend(frameon=False)
    ax.set_title(ic_title)
    ax.set_xlabel(f"{period_label}末")
    ax.set_ylabel("Spearman IC")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / ic_filename, dpi=180)
    plt.close(fig)


def save_nav_plot(
    nav: pd.DataFrame,
    output_dir: Path,
    title: str,
    filename: str = "strategy_vs_benchmarks_nav.png",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(nav.index, nav["strategy_nav"], linewidth=1.9, color="#4C78A8", label="综合评分Top20%策略")
    ax.plot(nav.index, nav["benchmark_nav"], linewidth=1.7, color="#E45756", label="沪深300")
    if "benchmark_500_nav" in nav:
        ax.plot(nav.index, nav["benchmark_500_nav"], linewidth=1.7, color="#72B7B2", label="中证500")
    ax.axhline(1.0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("月份")
    ax.set_ylabel("累计净值")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.7)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def save_risk_control_nav_plot(
    baseline_nav: pd.DataFrame,
    risk_nav: pd.DataFrame,
    output_dir: Path,
    title: str,
    filename: str,
    risk_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(baseline_nav.index, baseline_nav["strategy_nav"], linewidth=1.7, color="#4C78A8", label="原始策略")
    ax.plot(risk_nav.index, risk_nav["strategy_nav"], linewidth=1.9, color="#F58518", label=risk_label)
    ax.plot(baseline_nav.index, baseline_nav["benchmark_nav"], linewidth=1.6, color="#E45756", label="沪深300")
    ax.axhline(1.0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("月份")
    ax.set_ylabel("累计净值")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.7)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def save_monthly_return_heatmap(
    monthly_returns: pd.Series,
    output_dir: Path,
    title: str,
    filename: str = "monthly_return_heatmap.png",
) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei", "DejaVu Sans"]

    returns = monthly_returns.dropna().copy()
    returns.index = pd.to_datetime(returns.index)
    heatmap = returns.groupby([returns.index.year, returns.index.month]).last().unstack()
    heatmap = heatmap.reindex(columns=range(1, 13))
    values = heatmap * 100
    max_abs = float(np.nanmax(np.abs(values.to_numpy()))) if values.notna().any().any() else 1.0
    max_abs = max(max_abs, 1.0)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(values.to_numpy(), cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("月份")
    ax.set_ylabel("年份")
    ax.set_xticks(range(12), [str(month) for month in range(1, 13)])
    ax.set_yticks(range(len(values.index)), [str(year) for year in values.index])
    for row_idx, year in enumerate(values.index):
        for col_idx, month in enumerate(values.columns):
            value = values.loc[year, month]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=8, color="#1f1f1f")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("月收益（%）")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)
