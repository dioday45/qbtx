from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, **kwargs) -> None:
        """Initialize strategy with parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for strategy parameters.
        """
        self.params = kwargs

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on the strategy logic.

        The strategy should handle look-ahead bias internally by ensuring signals
        are based only on information available at each point in time.

        Args:
            data: DataFrame containing price data for a single asset.

        Returns:
            pd.Series: Series containing actionable position sizes (1, 0, -1 or custom sizing).
                       These signals represent positions that can be taken immediately
                       (i.e., look-ahead bias has already been handled).
        """
        pass


class Portfolio:
    """Portfolio class containing multiple assets and their data."""

    def __init__(self, assets: Dict[str, pd.DataFrame]) -> None:
        """Initialize portfolio with asset data.

        Args:
            assets: Dictionary mapping ticker symbols to their price DataFrames.
        """
        self.assets = assets
        self.tickers = list(assets.keys())

    def add_asset(self, ticker: str, data: pd.DataFrame) -> None:
        """Add a new asset to the portfolio.

        Args:
            ticker: Ticker symbol of the asset.
            data: Price DataFrame of the asset.
        """
        self.assets[ticker] = data
        if ticker not in self.tickers:
            self.tickers.append(ticker)

    def remove_asset(self, ticker: str) -> None:
        """Remove an asset from the portfolio.

        Args:
            ticker: Ticker symbol of the asset to remove.
        """
        if ticker in self.assets:
            del self.assets[ticker]
            self.tickers.remove(ticker)

    def get_asset_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get data for a specific asset.

        Args:
            ticker: Ticker symbol of the asset.

        Returns:
            pd.DataFrame or None: Price DataFrame of the asset if found, else None.
        """
        return self.assets.get(ticker)

    def get_all_tickers(self) -> List[str]:
        """Get list of all tickers in the portfolio.

        Returns:
            List[str]: List of ticker symbols.
        """
        return self.tickers.copy()


class TradeLog:
    """Class to generate and manage trade logs from positions and prices."""

    def __init__(
        self,
        prices: pd.Series,
        positions: pd.Series,
        fee: float = 0.0005,
        slippage: float = 0.001,
    ) -> None:
        """Initialize TradeLog with prices, positions, and cost parameters.

        Args:
            prices: Series of asset prices indexed by time.
            positions: Series of position sizes indexed by time.
            fee: Transaction fee rate.
            slippage: Slippage rate.
        """
        self.prices = prices
        self.positions = positions
        self.fee = fee
        self.slippage = slippage
        self.trades = self._generate_trades()

    def _generate_trades(self) -> pd.DataFrame:
        """Generate individual trades from position series.

        Identifies trade entry and exit points, calculates gross and net returns,
        and records trade details including fees, slippage, duration, and direction.

        Returns:
            pd.DataFrame: DataFrame with one row per completed trade containing columns:
                - entry_time: Timestamp of trade entry.
                - exit_time: Timestamp of trade exit.
                - entry_price: Price at entry.
                - exit_price: Price at exit.
                - position_size: Size of the position taken.
                - gross_return: Gross return of the trade in percentage.
                - net_return: Net return after costs in percentage.
                - fee_paid: Total fees paid for the trade.
                - slippage_paid: Total slippage cost for the trade.
                - total_cost: Sum of fees and slippage.
                - duration: Duration of the trade in days.
                - direction: 'long' or 'short' depending on position.
        """
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

                    # Skip trade if entry or exit price is NaN to avoid invalid calculations
                    if pd.isna(entry_price) or pd.isna(exit_price):
                        current_position = p
                        if p != 0:
                            entry_time = time
                            entry_price = price
                        else:
                            entry_time, entry_price = None, None
                        continue

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
                else:
                    entry_time, entry_price = None, None

                current_position = p

        return pd.DataFrame(trades)

    def get_trades(self) -> pd.DataFrame:
        """Return the DataFrame of completed trades.

        Returns:
            pd.DataFrame: DataFrame containing trade details.
        """
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
    ) -> None:
        """Initialize AssetBacktest with asset data and strategy.

        Args:
            ticker: Ticker symbol of the asset.
            data: Price DataFrame of the asset.
            strategy: Strategy instance to generate signals.
            slippage: Slippage rate.
            fee: Transaction fee rate.
        """
        self.ticker = ticker
        self.data = data
        self.strategy = strategy
        self.slippage = slippage
        self.fee = fee
        self.signals: Optional[pd.Series] = None
        self.positions: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.trade_log: Optional[TradeLog] = None

    def run(self) -> "AssetBacktest":
        """Run the backtest for this asset.

        Generates signals, computes positions, calculates returns, and creates trade log.

        Returns:
            AssetBacktest: Self for chaining.
        """
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
        """Compute returns for this asset.

        Args:
            price_col: Name of the price column in data.

        Returns:
            pd.Series: Series of net returns indexed by time.
        """
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
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        slippage: float = 0.0,
        fee: float = 0.0,
    ) -> None:
        """Initialize Backtester with portfolio, strategy, and cost parameters.

        Args:
            portfolio: Portfolio instance containing assets.
            strategy: Strategy class (not instance) to instantiate per asset.
            strategy_kwargs: Optional dict of keyword arguments for strategy initialization.
            slippage: Slippage rate.
            fee: Transaction fee rate.
        """
        self.portfolio = portfolio
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs or {}
        self.slippage = slippage
        self.fee = fee
        self.asset_backtests: Dict[str, AssetBacktest] = {}

    def run(self) -> "Backtester":
        """Run backtests for all assets in the portfolio.

        Returns:
            Backtester: Self for chaining.
        """
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

    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive results for all assets.

        Returns:
            Dict[str, Any]: Dictionary mapping ticker symbols to their backtest results,
                            including returns, trades, trade statistics, signals, and positions.
        """
        results = {}

        for ticker, backtest in self.asset_backtests.items():
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

    def get_combined_results(self) -> Dict[str, Any]:
        """Get results for the combined portfolio.

        Combines returns and trades across all assets and computes aggregate statistics.

        Returns:
            Dict[str, Any]: Dictionary containing combined portfolio returns,
                            all trades, combined trade statistics, and individual results.
        """
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
        trades_list = [
            backtest.trade_log.get_trades().assign(ticker=ticker)
            for ticker, backtest in self.asset_backtests.items()
            if not backtest.trade_log.get_trades().empty
        ]

        if trades_list:
            all_trades = pd.concat(trades_list, ignore_index=True)
            combined_trade_stats = evaluate_trade_log(all_trades)
        else:
            all_trades = pd.DataFrame()
            combined_trade_stats = pd.DataFrame()

        return {
            "returns": portfolio_returns,
            "all_trades": all_trades,
            "trade_stats": combined_trade_stats,
            "individual_results": self.get_results(),
        }


def evaluate_trade_log(trades: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a DataFrame of trades and compute summary statistics.

    Args:
        trades: DataFrame containing trade details with 'gross_return' and 'net_return' columns.

    Returns:
        pd.DataFrame: DataFrame with summary metrics as rows and their values.
    """
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
