from __future__ import annotations

import unittest

import pandas as pd

from a_share_turnover_factor.backtest import (
    _analyst_eps_revision_panel,
    _latest_annual_net_profit_yoy_panel,
    _latest_annual_quality_panel,
    _latest_annual_roe_panel,
    run_top_quantile_portfolio_backtest,
    run_turnover_factor_backtest,
)


class TurnoverBacktestTest(unittest.TestCase):
    def test_backtest_filters_and_groups(self) -> None:
        codes = [f"00000{i}.SZ" for i in range(1, 7)]
        trade_dates = pd.bdate_range("2022-01-03", "2022-04-29")
        stock_basic = pd.DataFrame(
            {
                "ts_code": codes,
                "name": ["A", "B", "C", "D", "ST E", "F"],
                "list_date": ["20200101", "20200101", "20200101", "20200101", "20200101", "20220401"],
                "exchange": ["SZSE"] * 6,
                "list_status": ["L"] * 6,
            }
        )
        rows_daily = []
        rows_basic = []
        rows_adj = []
        for day_idx, day in enumerate(trade_dates):
            trade_date = day.strftime("%Y%m%d")
            for code_idx, code in enumerate(codes):
                if code == "000004.SZ" and day_idx % 13 == 0:
                    continue
                rows_daily.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "close": 10 + code_idx + day_idx * (0.01 + code_idx * 0.001),
                    }
                )
                rows_basic.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "turnover_rate": 1 + code_idx + day_idx * 0.01,
                    }
                )
                rows_adj.append({"ts_code": code, "trade_date": trade_date, "adj_factor": 1.0})

        result = run_turnover_factor_backtest(
            stock_basic=stock_basic,
            namechange=pd.DataFrame(),
            daily=pd.DataFrame(rows_daily),
            daily_basic=pd.DataFrame(rows_basic),
            adj_factor=pd.DataFrame(rows_adj),
            trade_dates=[d.strftime("%Y%m%d") for d in trade_dates],
            start_date="20220201",
            end_date="20220429",
            group_count=3,
            rolling_window=20,
            min_listed_days=180,
        )

        self.assertEqual(list(result.group_returns.columns), ["G1", "G2", "G3"])
        self.assertTrue(result.sample_counts.iloc[0] >= 3)
        self.assertTrue(result.average_group_returns.notna().any())
        self.assertEqual(result.ic_series.index.name, "rebalance_date")

    def test_momentum_backtest_runs(self) -> None:
        codes = [f"00000{i}.SZ" for i in range(1, 7)]
        trade_dates = pd.bdate_range("2022-01-03", "2022-04-29")
        stock_basic = pd.DataFrame(
            {
                "ts_code": codes,
                "name": ["A", "B", "C", "D", "E", "F"],
                "list_date": ["20200101"] * 6,
                "exchange": ["SZSE"] * 6,
                "list_status": ["L"] * 6,
            }
        )
        rows_daily = []
        rows_basic = []
        rows_adj = []
        for day_idx, day in enumerate(trade_dates):
            trade_date = day.strftime("%Y%m%d")
            for code_idx, code in enumerate(codes):
                rows_daily.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "close": 10 + code_idx + day_idx * (0.02 + code_idx * 0.002),
                    }
                )
                rows_basic.append({"ts_code": code, "trade_date": trade_date, "turnover_rate": 1.0})
                rows_adj.append({"ts_code": code, "trade_date": trade_date, "adj_factor": 1.0})

        result = run_turnover_factor_backtest(
            stock_basic=stock_basic,
            namechange=pd.DataFrame(),
            daily=pd.DataFrame(rows_daily),
            daily_basic=pd.DataFrame(rows_basic),
            adj_factor=pd.DataFrame(rows_adj),
            trade_dates=[d.strftime("%Y%m%d") for d in trade_dates],
            start_date="20220201",
            end_date="20220429",
            group_count=3,
            min_listed_days=180,
            factor_type="momentum",
            momentum_window=20,
            momentum_skip_days=1,
        )

        self.assertEqual(list(result.group_returns.columns), ["G1", "G2", "G3"])
        self.assertTrue(result.average_group_returns.notna().any())

    def test_composite_backtest_runs(self) -> None:
        codes = [f"00000{i}.SZ" for i in range(1, 7)]
        trade_dates = pd.bdate_range("2022-01-03", "2022-04-29")
        stock_basic = pd.DataFrame(
            {
                "ts_code": codes,
                "name": ["A", "B", "C", "D", "E", "F"],
                "list_date": ["20200101"] * 6,
                "exchange": ["SZSE"] * 6,
                "list_status": ["L"] * 6,
            }
        )
        rows_daily = []
        rows_basic = []
        rows_adj = []
        rows_amount = []
        for day_idx, day in enumerate(trade_dates):
            trade_date = day.strftime("%Y%m%d")
            for code_idx, code in enumerate(codes):
                rows_daily.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "close": 10 + code_idx + day_idx * (0.02 + code_idx * 0.002),
                    }
                )
                rows_basic.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "turnover_rate": 1 + code_idx + day_idx * 0.01,
                    }
                )
                rows_adj.append({"ts_code": code, "trade_date": trade_date, "adj_factor": 1.0})
                rows_amount.append({"ts_code": code, "trade_date": trade_date, "amount": 30000.0})

        result = run_turnover_factor_backtest(
            stock_basic=stock_basic,
            namechange=pd.DataFrame(),
            daily=pd.DataFrame(rows_daily),
            daily_basic=pd.DataFrame(rows_basic),
            adj_factor=pd.DataFrame(rows_adj),
            trade_dates=[d.strftime("%Y%m%d") for d in trade_dates],
            start_date="20220201",
            end_date="20220429",
            group_count=3,
            rolling_window=5,
            min_listed_days=180,
            daily_amount=pd.DataFrame(rows_amount),
            liquidity_min_avg_amount_yuan=20000000,
            factor_type="composite",
            momentum_window=20,
            momentum_skip_days=1,
        )

        self.assertEqual(list(result.group_returns.columns), ["G1", "G2", "G3"])
        self.assertTrue(result.average_group_returns.notna().any())

    def test_top_quantile_portfolio_backtest_runs(self) -> None:
        codes = [f"00000{i}.SZ" for i in range(1, 11)]
        trade_dates = pd.bdate_range("2021-11-01", "2022-04-29")
        stock_basic = pd.DataFrame(
            {
                "ts_code": codes,
                "name": [f"Alpha{i}" for i in range(10)],
                "list_date": ["20200101"] * 10,
                "exchange": ["SZSE"] * 10,
                "list_status": ["L"] * 10,
            }
        )
        rows_daily = []
        rows_basic = []
        rows_adj = []
        rows_amount = []
        benchmark_rows = []
        for day_idx, day in enumerate(trade_dates):
            trade_date = day.strftime("%Y%m%d")
            benchmark_rows.append({"ts_code": "000300.SH", "trade_date": trade_date, "close": 4000 + day_idx})
            for code_idx, code in enumerate(codes):
                rows_daily.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "close": 10 + code_idx + day_idx * (0.01 + code_idx * 0.002),
                    }
                )
                rows_basic.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "turnover_rate": 1 + code_idx + day_idx * 0.01,
                    }
                )
                rows_adj.append({"ts_code": code, "trade_date": trade_date, "adj_factor": 1.0})
                rows_amount.append({"ts_code": code, "trade_date": trade_date, "amount": 50000.0})

        result = run_top_quantile_portfolio_backtest(
            stock_basic=stock_basic,
            namechange=pd.DataFrame(),
            daily=pd.DataFrame(rows_daily),
            daily_basic=pd.DataFrame(rows_basic),
            adj_factor=pd.DataFrame(rows_adj),
            daily_amount=pd.DataFrame(rows_amount),
            benchmark_daily=pd.DataFrame(benchmark_rows),
            trade_dates=[d.strftime("%Y%m%d") for d in trade_dates],
            start_date="20220101",
            end_date="20220429",
            top_quantile=0.2,
            rolling_window=5,
            liquidity_min_avg_amount_yuan=20000000,
        )

        self.assertIn("strategy_nav", result.nav.columns)
        self.assertIn("benchmark_return", result.annual_returns.columns)
        self.assertIn("max_drawdown", result.metrics.index)
        self.assertTrue(result.period_returns["total_cost"].notna().all())

    def test_quality_roe_uses_ann_date_not_report_period(self) -> None:
        codes = [f"00000{i}.SZ" for i in range(1, 7)]
        trade_dates = pd.bdate_range("2021-11-01", "2022-04-29")
        stock_basic = pd.DataFrame(
            {
                "ts_code": codes,
                "name": [f"Alpha{i}" for i in range(6)],
                "list_date": ["20200101"] * 6,
                "exchange": ["SZSE"] * 6,
                "list_status": ["L"] * 6,
            }
        )
        rows_daily = []
        rows_basic = []
        rows_adj = []
        rows_amount = []
        for day_idx, day in enumerate(trade_dates):
            trade_date = day.strftime("%Y%m%d")
            for code_idx, code in enumerate(codes):
                rows_daily.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "close": 10 + code_idx + day_idx * (0.01 + code_idx * 0.001),
                    }
                )
                rows_basic.append({"ts_code": code, "trade_date": trade_date, "turnover_rate": 1.0})
                rows_adj.append({"ts_code": code, "trade_date": trade_date, "adj_factor": 1.0})
                rows_amount.append({"ts_code": code, "trade_date": trade_date, "amount": 50000.0})

        financial_rows = []
        for code_idx, code in enumerate(codes):
            financial_rows.append(
                {
                    "ts_code": code,
                    "ann_date": "20210430",
                    "end_date": "20201231",
                    "roe": 5 + code_idx,
                }
            )
            financial_rows.append(
                {
                    "ts_code": code,
                    "ann_date": "20220430",
                    "end_date": "20211231",
                    "roe": 100 - code_idx,
                }
            )

        result = run_turnover_factor_backtest(
            stock_basic=stock_basic,
            namechange=pd.DataFrame(),
            daily=pd.DataFrame(rows_daily),
            daily_basic=pd.DataFrame(rows_basic),
            adj_factor=pd.DataFrame(rows_adj),
            trade_dates=[d.strftime("%Y%m%d") for d in trade_dates],
            start_date="20220101",
            end_date="20220429",
            group_count=3,
            min_listed_days=180,
            daily_amount=pd.DataFrame(rows_amount),
            liquidity_min_avg_amount_yuan=20000000,
            factor_type="quality_roe",
            financial_indicator=pd.DataFrame(financial_rows),
        )

        self.assertEqual(int(result.sample_counts.loc[pd.Timestamp("2022-01-31")]), 6)
        roe_panel = _latest_annual_roe_panel(
            pd.DataFrame(financial_rows),
            pd.DatetimeIndex([pd.Timestamp("2022-01-31")]),
            codes,
        )
        self.assertEqual(float(roe_panel.loc[pd.Timestamp("2022-01-31"), "000001.SZ"]), 5.0)

    def test_quality_metrics_use_disclosed_annual_reports(self) -> None:
        codes = ["000001.SZ", "000002.SZ"]
        financial = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20200331",
                    "end_date": "20191231",
                    "grossprofit_margin": 30.0,
                    "ocf_to_profit": 0.8,
                },
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20210331",
                    "end_date": "20201231",
                    "grossprofit_margin": 35.0,
                    "ocf_to_profit": 1.1,
                },
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20220331",
                    "end_date": "20211231",
                    "grossprofit_margin": 99.0,
                    "ocf_to_profit": 9.9,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20200331",
                    "end_date": "20191231",
                    "grossprofit_margin": 40.0,
                    "ocf_to_profit": 0.6,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20210331",
                    "end_date": "20201231",
                    "grossprofit_margin": 37.0,
                    "ocf_to_profit": 1.4,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20220331",
                    "end_date": "20211231",
                    "grossprofit_margin": 10.0,
                    "ocf_to_profit": 0.1,
                },
            ]
        )
        rebalance_dates = pd.DatetimeIndex([pd.Timestamp("2022-01-31")])

        gpm_yoy = _latest_annual_quality_panel(
            financial,
            rebalance_dates,
            codes,
            "grossprofit_margin_yoy",
        )
        ocf_to_np = _latest_annual_quality_panel(
            financial,
            rebalance_dates,
            codes,
            "ocf_to_profit",
        )

        self.assertEqual(float(gpm_yoy.loc[pd.Timestamp("2022-01-31"), "000001.SZ"]), 5.0)
        self.assertEqual(float(gpm_yoy.loc[pd.Timestamp("2022-01-31"), "000002.SZ"]), -3.0)
        self.assertEqual(float(ocf_to_np.loc[pd.Timestamp("2022-01-31"), "000001.SZ"]), 1.1)
        self.assertEqual(float(ocf_to_np.loc[pd.Timestamp("2022-01-31"), "000002.SZ"]), 1.4)

    def test_analyst_eps_revision_uses_past_three_months(self) -> None:
        codes = ["000001.SZ", "000002.SZ"]
        reports = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20211015",
                    "quarter": "2022Q4",
                    "eps": 1.00,
                    "org_name": "AlphaSec",
                    "author_name": "Ann",
                },
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20220110",
                    "quarter": "2022Q4",
                    "eps": 1.20,
                    "org_name": "AlphaSec",
                    "author_name": "Ann",
                },
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20220210",
                    "quarter": "2022Q4",
                    "eps": 1.25,
                    "org_name": "AlphaSec",
                    "author_name": "Ann",
                },
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20220501",
                    "quarter": "2022Q4",
                    "eps": 2.00,
                    "org_name": "AlphaSec",
                    "author_name": "Ann",
                },
                {
                    "ts_code": "000002.SZ",
                    "report_date": "20211120",
                    "quarter": "2022Q4",
                    "eps": 0.90,
                    "org_name": "BetaSec",
                    "author_name": "Ben",
                },
                {
                    "ts_code": "000002.SZ",
                    "report_date": "20220120",
                    "quarter": "2022Q4",
                    "eps": 0.80,
                    "org_name": "BetaSec",
                    "author_name": "Ben",
                },
            ]
        )
        rebalance_dates = pd.DatetimeIndex([pd.Timestamp("2022-03-31")])

        count_panel = _analyst_eps_revision_panel(reports, rebalance_dates, codes, "count")
        magnitude_panel = _analyst_eps_revision_panel(reports, rebalance_dates, codes, "magnitude")

        self.assertEqual(float(count_panel.loc[pd.Timestamp("2022-03-31"), "000001.SZ"]), 2.0)
        self.assertEqual(float(count_panel.loc[pd.Timestamp("2022-03-31"), "000002.SZ"]), 0.0)
        self.assertAlmostEqual(
            float(magnitude_panel.loc[pd.Timestamp("2022-03-31"), "000001.SZ"]),
            0.25,
        )
        self.assertEqual(float(magnitude_panel.loc[pd.Timestamp("2022-03-31"), "000002.SZ"]), 0.0)

    def test_net_profit_yoy_excludes_negative_previous_profit(self) -> None:
        codes = ["000001.SZ", "000002.SZ"]
        income = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20210331",
                    "end_date": "20201231",
                    "n_income_attr_p": 100.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20220331",
                    "end_date": "20211231",
                    "n_income_attr_p": 150.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20210331",
                    "end_date": "20201231",
                    "n_income_attr_p": -20.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20220331",
                    "end_date": "20211231",
                    "n_income_attr_p": 40.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20230331",
                    "end_date": "20221231",
                    "n_income_attr_p": 300.0,
                },
            ]
        )
        rebalance_dates = pd.DatetimeIndex([pd.Timestamp("2022-04-29")])

        panel = _latest_annual_net_profit_yoy_panel(income, rebalance_dates, codes)

        self.assertEqual(float(panel.loc[pd.Timestamp("2022-04-29"), "000001.SZ"]), 0.5)
        self.assertTrue(pd.isna(panel.loc[pd.Timestamp("2022-04-29"), "000002.SZ"]))


if __name__ == "__main__":
    unittest.main()
