# A股换手率因子检验

这个新项目用 Tushare Pro 获取中国 A 股日线数据，计算每只股票过去 20 个交易日换手率均值作为因子，在月末截面等分 5 组，统计下月平均收益，并输出分组收益柱状图与 IC 时序折线图。

## 口径

- 数据区间默认：`2022-01-01` 到 `2023-12-31`，滚动窗口预热从 `2021-11-01` 开始。
- 因子：`daily_basic.turnover_rate` 的 20 日滚动均值，要求窗口内有 20 个非空交易观测。
- 收益：月末复权收盘价到下一月末复权收盘价的收益，复权价用 `daily.close * adj_factor`。
- 分组：每个月末按因子值等分 5 组，`G1` 为低换手率，`G5` 为高换手率。
- IC：每个月末截面上因子值与下月收益的 Spearman 相关系数。
- 清洗：剔除 ST 股票、上市不足 180 个自然日的股票；停牌或缺数据的交易日通过完整交易日 x 股票面板补为 `NaN`。
- 默认只使用到 2023 年末的数据，因此 2023-12 月末因子没有 2024-01 的下月收益，不参与收益与 IC 统计。若要纳入 2023-12，运行时加 `--fetch-end-date 20240131`。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 配置 Tushare Token

```bash
export TUSHARE_TOKEN="你的token"
```

也可以在项目根目录创建本地 `.env` 文件：

```bash
TUSHARE_TOKEN=你的token
```

`.env` 已加入 `.gitignore`，不会被提交。

也可以运行时传入：

```bash
a-share-turnover-factor --token "你的token"
```

## 运行

```bash
a-share-turnover-factor
```

如果希望 2023-12 的分组收益也进入统计：

```bash
a-share-turnover-factor --fetch-end-date 20240131
```

常用参数：

```bash
a-share-turnover-factor \
  --start-date 20220101 \
  --end-date 20231231 \
  --warmup-start-date 20211101 \
  --groups 5 \
  --rolling-window 20 \
  --min-listed-days 180
```

## 对比实验

运行 3 个对比实验：

- 实验1：`rolling(5)`，月末换仓。
- 实验2：`rolling(60)`，月末换仓。
- 实验3：`rolling(20)`，周末换仓。

```bash
a-share-turnover-factor --run-experiments --output-dir outputs/experiments --pause 0.05
```

每个实验会输出一张分组收益柱状图和一张 IC 时序图，图表标题会注明参数。

## 输出

默认写入 `outputs/`：

- `group_average_monthly_return.png`：五分组平均月收益柱状图。
- `ic_series.png`：IC 时序折线图。
- `group_monthly_returns.csv`：每个月末每组下月平均收益。
- `group_average_monthly_returns.csv`：每组跨月平均收益。
- `ic_series.csv`：月度 IC。
- `monthly_sample_counts.csv`：每个月有效样本数。

Tushare 原始接口结果会缓存到 `data/cache/`，再次运行会优先使用缓存，避免重复请求。
