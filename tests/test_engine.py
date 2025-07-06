import numpy as np
import pandas as pd
import pytest

from qbtx.engine.backtester import Backtester
from qbtx.engine.portfolio import Portfolio
from qbtx.engine.tradelog import TradeLog
from qbtx.strategy.simple_strats import ReversalStrategy

# ---------- Fixtures ----------


@pytest.fixture
def sample_prices():
    return pd.DataFrame(
        {"A": [100, 102, 101, 103, 99], "B": [200, 202, 204, 206, 208]},
        index=pd.date_range("2023-01-01", periods=5),
    )


# ---------- Portfolio Tests ----------


def test_portfolio_no_fees(sample_prices):
    signals = pd.DataFrame(
        {"A": [0, 1, 1, 0, 0], "B": [0, 0, 1, 1, 0]}, index=sample_prices.index
    )
    pf = Portfolio(sample_prices, signals, fee=0.0, slippage=0.0)
    expected_positions = signals.shift(1).fillna(0)
    pd.testing.assert_frame_equal(pf.positions, expected_positions)
    assert isinstance(pf.returns, pd.DataFrame)


def test_portfolio_with_fees(sample_prices):
    signals = pd.DataFrame(
        {"A": [0, 1, 1, 0, 0], "B": [0, 0, 1, 1, 0]}, index=sample_prices.index
    )
    pf = Portfolio(sample_prices, signals, fee=0.001, slippage=0.002)
    assert (
        (pf.returns <= pf.positions * sample_prices.pct_change().fillna(0)).all().all()
    )


# ---------- TradeLog Tests ----------


def test_tradelog_long_trade(sample_prices):
    prices = sample_prices["A"]
    positions = pd.Series([0, 1, 1, 0, 0], index=prices.index)
    trades = TradeLog(prices, positions, fee=0.001, slippage=0.001).get_trades()
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["direction"] == "long"
    assert trade["entry_price"] == prices.iloc[1]
    assert trade["exit_price"] == prices.iloc[3]


# ---------- Strategy Tests ----------


def test_reversal_strategy_signal_range(sample_prices):
    strat = ReversalStrategy(sample_prices)
    signals = strat.generate_signals()
    assert set(np.unique(signals.values)) <= {-1, 0, 1}


# ---------- Backtester Integration ----------


def test_backtester_run(sample_prices):
    bt = Backtester(sample_prices, ReversalStrategy).run()
    results = bt.results()
    assert "equity" in results
    assert "returns" in results
    assert "trades" in results
    assert isinstance(results["trades"], dict)
    for asset, trades in results["trades"].items():
        assert isinstance(trades, pd.DataFrame)
        assert set(trades.columns).issuperset({"entry_time", "exit_time", "net_return"})
