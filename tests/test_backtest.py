from __future__ import annotations

import unittest

import pandas as pd

from a_share_turnover_factor.backtest import run_turnover_factor_backtest


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


if __name__ == "__main__":
    unittest.main()
