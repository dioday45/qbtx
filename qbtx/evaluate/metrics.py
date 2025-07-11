import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series, ann_factor: float, risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Parameters:
    - returns: pd.Series of periodic returns in decimal (e.g., 0.001 = 0.1%)
    - ann_factor: Annualization factor (e.g., 365 for daily returns)
    - risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)

    Returns:
    - Annualized Sharpe Ratio (float)
    """
    if ann_factor <= 0:
        raise ValueError("ann_factor must be positive.")
    if returns.empty:
        return 0.0

    std = returns.std()
    if std == 0:
        return 0.0

    per_period_rf = risk_free_rate / ann_factor
    excess_return = returns.mean() - per_period_rf
    return (excess_return / std) * np.sqrt(ann_factor)


def sortino_ratio(
    returns: pd.Series, ann_factor: float, risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sortino Ratio.

    Parameters:
    - returns: pd.Series of periodic returns
    - ann_factor: Annualization factor
    - risk_free_rate: Annual risk-free rate

    Returns:
    - Annualized Sortino Ratio (float)
    """
    if returns.empty:
        return 0.0

    per_period_rf = risk_free_rate / ann_factor
    downside_returns = returns[returns < per_period_rf]
    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    excess_return = returns.mean() - per_period_rf
    return (excess_return / downside_std) * np.sqrt(ann_factor)


def profit_factor(returns: pd.Series) -> float:
    """
    Calculate the profit factor, defined as the ratio of gross profits to gross losses.

    Parameters:
    - returns: pd.Series of periodic returns

    Returns:
    - Profit Factor (float)
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def win_rate(returns: pd.Series) -> float:
    """
    Calculate the win rate: proportion of positive return periods.

    Parameters:
    - returns: pd.Series of periodic returns

    Returns:
    - Win rate as a float between 0 and 1
    """
    if returns.empty:
        return 0.0

    wins = (returns > 0).sum()
    total = returns.count()

    return wins / total
