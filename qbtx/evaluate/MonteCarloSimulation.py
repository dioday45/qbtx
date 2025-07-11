import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MonteCarloSimulator:
    def __init__(self, returns: pd.Series, num_simulations: int = 100):
        """
        Initializes the Monte Carlo simulator with returns data and number of simulations.

        :param returns: A pandas Series of historical returns.
        :param num_simulations: The number of simulations to run.
        :raises TypeError: If returns is not a pandas Series.
        """
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")
        self.returns = returns
        self.num_simulations = num_simulations

    def run_without_replacement(self) -> pd.DataFrame:
        """
        Simulates future returns by sampling without replacement.

        :return: A pandas DataFrame where each column represents a simulation of returns sampled without replacement.
        """
        return pd.DataFrame(
            {
                f"Simulation {i+1}": self.returns.sample(
                    n=len(self.returns), replace=False
                ).values
                for i in range(self.num_simulations)
            },
            index=self.returns.index,
        )

    def run_with_replacement(self) -> pd.DataFrame:
        """
        Simulates future returns by sampling with replacement.

        :return: A pandas DataFrame where each column represents a simulation of returns sampled with replacement.
        """
        return pd.DataFrame(
            {
                f"Simulation {i+1}": self.returns.sample(
                    n=len(self.returns), replace=True
                ).values
                for i in range(self.num_simulations)
            },
            index=self.returns.index,
        )

    @staticmethod
    def calculate_performance_table(
        simulated_returns: pd.DataFrame, original_returns: pd.Series = None
    ):
        """
        Calculates performance metrics for simulated returns and optionally for the original returns.

        :param simulated_returns: A non-empty pandas DataFrame of simulated returns.
        :param original_returns: An optional pandas Series of original returns to compare against simulations.
        :return: A pandas DataFrame summarizing performance metrics including annualized return, volatility, Sharpe ratio, drawdowns, and profit factor.
        :raises ValueError: If simulated_returns is not a non-empty DataFrame.
        """
        if not isinstance(simulated_returns, pd.DataFrame) or simulated_returns.empty:
            raise ValueError("simulated_returns must be a non-empty pandas DataFrame")

        def annualize_return(r):
            return (1 + r).prod() ** (252 / len(r)) - 1

        def annualize_volatility(r):
            return r.std() * (252**0.5)

        def sharpe_ratio(r):
            vol = annualize_volatility(r)
            return 0 if vol == 0 else annualize_return(r) / vol

        def max_drawdown(r):
            cumulative = (1 + r).cumprod()
            peak = cumulative.cummax()
            dd = (cumulative - peak) / peak
            return dd.min()

        def quantile_drawdown(r, q=0.95):
            cumulative = (1 + r).cumprod()
            peak = cumulative.cummax()
            dd = (cumulative - peak) / peak
            return dd.quantile(q)

        def profit_factor(r):
            profits = r[r > 0].sum()
            losses = -r[r < 0].sum()
            return float("inf") if losses == 0 else profits / losses

        percentiles = [95, 75, 50, 25, 10, 5, 1, 0]
        stats = {}

        metrics = simulated_returns.apply(
            lambda r: pd.Series(
                {
                    "Cumulative Return (p.a.)": annualize_return(r),
                    "Volatility (p.a.)": annualize_volatility(r),
                    "Sharpe Ratio": sharpe_ratio(r),
                    "Maximal Drawdown": max_drawdown(r),
                    "95% Drawdown": quantile_drawdown(r, 0.95),
                    "Profit Factor": profit_factor(r),
                }
            ),
            axis=0,
        ).T  # Now metrics is (simulations x metrics)

        if original_returns is not None:
            stats["Original Strategy"] = pd.Series(
                {
                    "Cumulative Return (p.a.)": annualize_return(original_returns),
                    "Volatility (p.a.)": annualize_volatility(original_returns),
                    "Sharpe Ratio": sharpe_ratio(original_returns),
                    "Maximal Drawdown": max_drawdown(original_returns),
                    "95% Drawdown": quantile_drawdown(original_returns, 0.95),
                    "Profit Factor": profit_factor(original_returns),
                }
            )

        for p in percentiles:
            stats[f"{p}th percentile" if p != 95 else "95th percentile"] = (
                metrics.quantile(p / 100)
            )

        order = ["Original Strategy"] + [
            f"{p}th percentile" if p != 95 else "95th percentile" for p in percentiles
        ]
        df = pd.DataFrame(stats).T.loc[
            order if original_returns is not None else order[1:]
        ]

        percent_format_cols = [
            "Cumulative Return (p.a.)",
            "Volatility (p.a.)",
            "Maximal Drawdown",
            "95% Drawdown",
        ]
        for col in percent_format_cols:
            if col in df.columns:
                df[col] = (df[col].astype(float) * 100).map("{:.2f}%".format)

        decimal_format_cols = ["Sharpe Ratio", "Profit Factor"]
        for col in decimal_format_cols:
            if col in df.columns:
                df[col] = df[col].astype(float).map("{:.2f}".format)

        return df

    @staticmethod
    def plot_simulations(
        simulated_returns: pd.DataFrame,
        fraction: float = 1.0,
        original_returns: pd.Series = None,
    ):
        """
        Plots the cumulative simulated returns using seaborn.

        :param simulated_returns: A pandas DataFrame of simulated returns.
        :param fraction: Fraction of simulations to plot (between 0 (exclusive) and 1 (inclusive)).
        :param original_returns: Optional pandas Series of original returns to overlay on the plot.
        :raises ValueError: If fraction is not between 0 (exclusive) and 1 (inclusive).
        """
        if not 0 < fraction <= 1.0:
            raise ValueError("fraction must be between 0 (exclusive) and 1 (inclusive)")

        plt.figure(figsize=(12, 6))
        sns.set_theme(style="white")
        # Sample simulations
        n_simulations = int(len(simulated_returns.columns) * fraction)
        sampled_columns = simulated_returns.columns.to_list()
        if 0 < fraction < 1.0:
            sampled_columns = (
                pd.Series(sampled_columns)
                .sample(n=n_simulations, random_state=42)
                .to_list()
            )

        cumulative_returns = (
            (1 + simulated_returns[sampled_columns]).cumprod() - 1
        ) * 100

        for column in cumulative_returns.columns:
            sns.lineplot(
                data=cumulative_returns[column], alpha=0.8, linewidth=1, legend=False
            )

        if original_returns is not None:
            original_cumulative = ((1 + original_returns).cumprod() - 1) * 100
            sns.lineplot(
                data=original_cumulative,
                color="black",
                label="Original Strategy",
                linewidth=1,
            )
        plt.legend(loc="upper left")
        plt.title("Monte Carlo Simulations of Cumulative Returns")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Return (%)")
        plt.show()
