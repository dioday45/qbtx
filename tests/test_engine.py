import pandas as pd
import pytest

from qbtx.engine.engine import AssetBacktest, Backtester, Portfolio, Strategy, TradeLog

# ---------- Fixtures ----------


@pytest.fixture
def sample_data() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=5)
    prices = pd.Series([100, 102, 101, 103, 99], index=index, name="price")
    df = pd.DataFrame({"price": prices})
    return df


@pytest.fixture
def sample_portfolio(sample_data: pd.DataFrame) -> Portfolio:
    return Portfolio({"A": sample_data, "B": sample_data.copy()})


class DummyStrategy(Strategy):
    def run(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for the given data.

        Args:
            data (pd.DataFrame): Input price data.

        Returns:
            pd.Series: Position signals (0 or 1) indexed by date.
        """
        return pd.Series([0, 1, 1, 0, 0], index=data.index)


# ---------- Portfolio Tests ----------


def test_portfolio_add_and_get_asset(sample_data: pd.DataFrame) -> None:
    pf = Portfolio({})
    pf.add_asset("TEST", sample_data)
    assert "TEST" in pf.get_all_tickers()
    pd.testing.assert_frame_equal(pf.get_asset_data("TEST"), sample_data)


def test_portfolio_remove_asset(sample_data: pd.DataFrame) -> None:
    pf = Portfolio({"TEST": sample_data})
    pf.remove_asset("TEST")
    assert "TEST" not in pf.get_all_tickers()


# ---------- TradeLog Tests ----------


def test_tradelog_no_trades(sample_data: pd.DataFrame) -> None:
    positions = pd.Series([0, 0, 0, 0, 0], index=sample_data.index)
    trades = TradeLog(sample_data["price"], positions).get_trades()
    assert trades.empty


def test_tradelog_long_trade(sample_data: pd.DataFrame) -> None:
    positions = pd.Series([0, 1, 1, 0, 0], index=sample_data.index)
    trades = TradeLog(sample_data["price"], positions).get_trades()
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["direction"] == "long"
    assert trade["entry_price"] == sample_data["price"].iloc[1]
    assert trade["exit_price"] == sample_data["price"].iloc[3]


def test_tradelog_short_trade(sample_data: pd.DataFrame) -> None:
    positions = pd.Series([0, -1, -1, 0, 0], index=sample_data.index)
    trades = TradeLog(sample_data["price"], positions).get_trades()
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["direction"] == "short"
    assert trade["entry_price"] == sample_data["price"].iloc[1]
    assert trade["exit_price"] == sample_data["price"].iloc[3]


def test_tradelog_with_nan_positions(sample_data: pd.DataFrame) -> None:
    positions = pd.Series([0, float("nan"), 1, 0, 0], index=sample_data.index)
    trades = TradeLog(sample_data["price"], positions).get_trades()
    assert isinstance(trades, pd.DataFrame)


def test_tradelog_with_nan_prices() -> None:
    index = pd.date_range("2023-01-01", periods=5)
    prices = pd.Series([100, None, 101, None, 99], index=index, name="price")
    positions = pd.Series([0, 1, 1, 0, 0], index=index)
    trades = TradeLog(prices, positions).get_trades()
    assert isinstance(trades, pd.DataFrame)


def test_tradelog_persistent_position(sample_data: pd.DataFrame) -> None:
    positions = pd.Series([1, 1, 1, 1, 1], index=sample_data.index)
    trades = TradeLog(sample_data["price"], positions).get_trades()
    assert trades.empty  # No closing position, no trade logged


def test_tradelog_long_to_short(sample_data: pd.DataFrame) -> None:
    positions = pd.Series([0, 1, -1, 0, 0], index=sample_data.index)
    trades = TradeLog(sample_data["price"], positions).get_trades()
    assert len(trades) == 2
    assert set(trades["direction"]) == {"long", "short"}


def test_tradelog_missing_price_index() -> None:
    index = pd.date_range("2023-01-01", periods=5)
    prices = pd.Series([100, 102, 101], index=index[:3], name="price")
    positions = pd.Series([0, 1, 1, 0, 0], index=index)
    trades = TradeLog(prices, positions).get_trades()
    assert isinstance(trades, pd.DataFrame)


def test_tradelog_realistic_scenario() -> None:
    index = pd.date_range("2023-01-01", periods=6)
    prices = pd.Series([100, 102, 104, 106, 108, 110], index=index, name="price")
    positions = pd.Series([0, 1, 1, 1, 0, 0], index=index)

    trades = TradeLog(prices, positions, fee=0.0, slippage=0.0).get_trades()
    assert len(trades) == 1

    trade = trades.iloc[0]
    assert trade["entry_time"] == index[1]
    assert trade["exit_time"] == index[4]
    assert trade["entry_price"] == 102
    assert trade["exit_price"] == 108
    assert trade["gross_return"] == round((108 / 102 - 1) * 100, 4)
    assert trade["net_return"] == trade["gross_return"]  # no fee/slippage


# ---------- AssetBacktest Tests ----------


def test_asset_backtest_returns(sample_data: pd.DataFrame) -> None:
    strat = DummyStrategy()
    ab = AssetBacktest("TEST", sample_data, strat, fee=0.0, slippage=0.0).run()
    assert isinstance(ab.returns, pd.Series)
    assert len(ab.returns) == len(sample_data)
    assert ab.positions.iloc[-1] == 0


# ---------- Backtester Tests ----------


def test_backtester_combined_results(sample_portfolio: Portfolio) -> None:
    bt = Backtester(sample_portfolio, DummyStrategy).run()
    results = bt.get_combined_results()
    assert "returns" in results
    assert "all_trades" in results
    assert isinstance(results["returns"], pd.Series)
    assert isinstance(results["all_trades"], pd.DataFrame)


def test_backtester_empty_portfolio() -> None:
    pf = Portfolio({})
    bt = Backtester(pf, DummyStrategy).run()
    combined = bt.get_combined_results()
    assert combined["returns"].empty
    assert combined["all_trades"].empty


# ---------- Additional Coverage Tests ----------


def test_asset_backtest_with_fee_and_slippage(sample_data: pd.DataFrame) -> None:
    strat = DummyStrategy()
    ab = AssetBacktest("TEST", sample_data, strat, fee=0.01, slippage=0.01).run()
    trades = ab.trade_log.get_trades()
    assert isinstance(ab.returns, pd.Series)
    assert len(trades) == 1
    assert trades.iloc[0]["total_cost"] > 0


def test_backtester_trade_stats(sample_portfolio: Portfolio) -> None:
    bt = Backtester(sample_portfolio, DummyStrategy).run()
    results = bt.get_combined_results()
    stats = results["trade_stats"]
    assert not stats.empty
    assert "Num Trades" in stats.index
    assert stats.loc["Num Trades"].values[0] > 0


# ---------- Multi-Asset Return Combination Tests ----------


def test_backtester_combines_asset_returns_correctly(sample_data: pd.DataFrame) -> None:
    portfolio = Portfolio({"X": sample_data.copy(), "Y": sample_data.copy()})
    bt = Backtester(portfolio, DummyStrategy).run()
    combined = bt.get_combined_results()
    returns_df = pd.DataFrame({k: v.returns for k, v in bt.asset_backtests.items()})
    manual_avg = returns_df.mean(axis=1)
    pd.testing.assert_series_equal(combined["returns"], manual_avg, check_names=False)


def test_combined_results_index_matches_assets(sample_data: pd.DataFrame) -> None:
    portfolio = Portfolio({"A": sample_data.copy(), "B": sample_data.copy()})
    bt = Backtester(portfolio, DummyStrategy).run()
    combined = bt.get_combined_results()
    for asset_bt in bt.asset_backtests.values():
        assert all(combined["returns"].index == asset_bt.returns.index)


def test_tradelog_multiple_trades(sample_data: pd.DataFrame) -> None:
    # open long -> flat -> short -> flat = 2 trades
    positions = pd.Series([0, 1, 0, -1, 0], index=sample_data.index)
    trades = TradeLog(sample_data["price"], positions).get_trades()
    assert len(trades) == 2
    assert set(trades["direction"]) == {"long", "short"}
    assert all(trades["exit_time"] > trades["entry_time"])


def test_tradelog_fee_slippage_impact(sample_data: pd.DataFrame) -> None:
    positions = pd.Series([0, 1, 1, 0, 0], index=sample_data.index)
    trade_no_fee = TradeLog(
        sample_data["price"], positions, fee=0.0, slippage=0.0
    ).get_trades()
    trade_with_fee = TradeLog(
        sample_data["price"], positions, fee=0.01, slippage=0.01
    ).get_trades()
    assert trade_no_fee["net_return"].iloc[0] > trade_with_fee["net_return"].iloc[0]
