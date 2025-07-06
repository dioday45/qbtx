from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
from tqdm import tqdm


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, **kwargs):
        """Initialize strategy with parameters."""
        self.params = kwargs

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on the strategy logic.

        The strategy should handle look-ahead bias internally by ensuring signals
        are based only on information available at each point in time.

        Args:
            data: DataFrame containing price data for a single asset

        Returns:
            pd.Series: Series containing actionable position sizes (1, 0, -1 or custom sizing)
                      These signals represent positions that can be taken immediately
                      (i.e., look-ahead bias has already been handled)
        """
        pass


class Portfolio:
    """Portfolio class containing multiple assets and their data."""

    def __init__(self, assets: Dict[str, pd.DataFrame]):
        """Initialize portfolio with asset data.

        Args:
            assets: Dictionary mapping ticker symbols to their price DataFrames
        """
        self.assets = assets
        self.tickers = list(assets.keys())

    def add_asset(self, ticker: str, data: pd.DataFrame):
        """Add a new asset to the portfolio."""
        self.assets[ticker] = data
        if ticker not in self.tickers:
            self.tickers.append(ticker)

    def remove_asset(self, ticker: str):
        """Remove an asset from the portfolio."""
        if ticker in self.assets:
            del self.assets[ticker]
            self.tickers.remove(ticker)

    def get_asset_data(self, ticker: str) -> pd.DataFrame:
        """Get data for a specific asset."""
        return self.assets.get(ticker)

    def get_all_tickers(self) -> list:
        """Get list of all tickers in the portfolio."""
        return self.tickers.copy()


class TradeLog:
    """Class to generate and manage trade logs from positions and prices."""

    def __init__(
        self,
        prices: pd.Series,
        positions: pd.Series,
        fee: float = 0.0005,
        slippage: float = 0.001,
    ):
        self.prices = prices
        self.positions = positions
        self.fee = fee
        self.slippage = slippage
        self.trades = self._generate_trades()

    def _generate_trades(self) -> pd.DataFrame:
        """Generate individual trades from position series."""
        trades = []
        pos = self.positions.fillna(0)
        current_position = 0
        entry_time, entry_price = None, None

        for time, p in pos.items():
            if time not in self.prices.index:
                continue

            price = self.prices[time]

            if p != current_position:
                # Close existing position
                if current_position != 0:
                    exit_time = time
                    exit_price = price

                    # Calculate returns
                    if current_position > 0:  # Long position
                        gross_return = (exit_price / entry_price - 1) * abs(
                            current_position
                        )
                    else:  # Short position
                        gross_return = (entry_price / exit_price - 1) * abs(
                            current_position
                        )

                    # Calculate costs
                    notional_entry = abs(current_position) * entry_price
                    notional_exit = abs(current_position) * exit_price

                    fee_paid = self.fee * (notional_entry + notional_exit)
                    slippage_paid = self.slippage * (notional_entry + notional_exit)
                    total_cost = fee_paid + slippage_paid

                    net_return = gross_return - total_cost / notional_entry

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "position_size": current_position,
                            "gross_return": round(gross_return * 100, 4),
                            "net_return": round(net_return * 100, 4),
                            "fee_paid": fee_paid,
                            "slippage_paid": slippage_paid,
                            "total_cost": total_cost,
                            "duration": (exit_time - entry_time).days,
                            "direction": "long" if current_position > 0 else "short",
                        }
                    )

                # Open new position
                if p != 0:
                    entry_time = time
                    entry_price = price

                current_position = p

        return pd.DataFrame(trades)

    def get_trades(self) -> pd.DataFrame:
        """Return the trades DataFrame."""
        return self.trades


class AssetBacktest:
    """Handles backtesting for a single asset."""

    def __init__(
        self,
        ticker: str,
        data: pd.DataFrame,
        strategy: Strategy,
        slippage: float = 0.0,
        fee: float = 0.0,
    ):
        self.ticker = ticker
        self.data = data
        self.strategy = strategy
        self.slippage = slippage
        self.fee = fee
        self.signals = None
        self.positions = None
        self.returns = None
        self.trade_log = None

    def run(self):
        """Run the backtest for this asset."""
        # Generate signals (already lag-adjusted by strategy)
        self.signals = self.strategy.run(self.data.copy())

        # Extract price column (assuming price column is named 'price' or is the first column)
        price_col = "price" if "price" in self.data.columns else self.data.columns[0]

        # Use signals directly as positions (no additional shifting needed)
        self.positions = self.signals.copy()
        self.positions.iloc[-1] = 0  # Force close position on final day

        # Calculate returns
        self.returns = self._compute_returns(price_col)

        # Generate trade log
        self.trade_log = TradeLog(
            self.data[price_col], self.positions, fee=self.fee, slippage=self.slippage
        )

        return self

    def _compute_returns(self, price_col: str) -> pd.Series:
        """Compute returns for this asset."""
        price_returns = self.data[price_col].pct_change().fillna(0)

        # Gross returns
        gross_returns = self.positions * price_returns

        # Calculate transaction costs
        position_changes = abs(self.positions - self.positions.shift(1).fillna(0))
        transaction_costs = position_changes * (self.fee + self.slippage)

        # Net returns
        net_returns = gross_returns - transaction_costs

        return net_returns


class Backtester:
    """Main backtesting engine that handles multiple assets."""

    def __init__(
        self,
        portfolio: Portfolio,
        strategy: Strategy,
        strategy_kwargs: Optional[Dict] = None,
        slippage: float = 0.0,
        fee: float = 0.0,
    ):
        self.portfolio = portfolio
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs or {}
        self.slippage = slippage
        self.fee = fee
        self.asset_backtests = {}

    def run(self):
        """Run backtests for all assets in the portfolio."""
        for ticker in tqdm(self.portfolio.get_all_tickers(), desc="Running backtests"):
            data = self.portfolio.get_asset_data(ticker)
            strategy = self.strategy(**self.strategy_kwargs)

            asset_backtest = AssetBacktest(
                ticker=ticker,
                data=data,
                strategy=strategy,
                slippage=self.slippage,
                fee=self.fee,
            )

            self.asset_backtests[ticker] = asset_backtest.run()

        return self

    def get_results(self) -> Dict:
        """Get comprehensive results for all assets."""
        results = {}

        for ticker, backtest in self.asset_backtests.items():
            # Performance metrics
            performance = evaluate_returns(backtest.returns.to_frame(ticker))

            # Trade statistics
            trade_stats = evaluate_trade_log(backtest.trade_log.get_trades())

            results[ticker] = {
                "returns": backtest.returns,
                "trades": backtest.trade_log.get_trades(),
                "trade_stats": trade_stats,
                "signals": backtest.signals,
                "positions": backtest.positions,
            }

        return results

    def get_combined_results(self) -> Dict:
        """Get results for the combined portfolio."""
        # Combine all returns
        all_returns = pd.DataFrame(
            {
                ticker: backtest.returns
                for ticker, backtest in self.asset_backtests.items()
            }
        )

        # Equal weight portfolio returns
        portfolio_returns = all_returns.mean(axis=1)
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

        # All trades combined
        all_trades = pd.concat(
            [
                backtest.trade_log.get_trades().assign(ticker=ticker)
                for ticker, backtest in self.asset_backtests.items()
            ],
            ignore_index=True,
        )

        combined_trade_stats = (
            evaluate_trade_log(all_trades) if len(all_trades) > 0 else pd.DataFrame()
        )

        return {
            "returns": portfolio_returns,
            "all_trades": all_trades,
            "trade_stats": combined_trade_stats,
            "individual_results": self.get_results(),
        }


def evaluate_returns(
    returns: Union[pd.Series, pd.DataFrame], freq: int = 252
) -> pd.DataFrame:
    """Evaluate portfolio-level returns."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)
    sharpe = annual_return / volatility.replace(0, np.nan)

    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return pd.DataFrame(
        {
            "Total Return": total_return,
            "Annualized Return": annual_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
        }
    )


def evaluate_trade_log(trades: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a DataFrame of trades."""
    if len(trades) == 0:
        return pd.DataFrame()

    gross = trades["gross_return"] / 100  # Convert back to decimal
    net = trades["net_return"] / 100  # Convert back to decimal

    metrics = {
        "Num Trades": len(trades),
        "Win Rate [%]": f"{round((net > 0).mean() * 100, 2)}%",
        "Avg Gross Return [%]": f"{round(gross.mean() * 100, 2)}%",
        "Avg Net Return [%]": f"{round(net.mean() * 100, 2)}%",
        "Best Trade [%]": f"{round(net.max() * 100, 2)}%",
        "Worst Trade [%]": f"{round(net.min() * 100, 2)}%",
        "Avg Duration (days)": round(trades["duration"].mean(), 1),
        "Long Trades": (trades["direction"] == "long").sum(),
        "Short Trades": (trades["direction"] == "short").sum(),
    }

    return pd.DataFrame([metrics]).T
