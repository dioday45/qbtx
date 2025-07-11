from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from qbtx.evaluate.metrics import profit_factor, sharpe_ratio, sortino_ratio, win_rate


class PerformanceReport:
    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark: Union[str, pd.Series, None],
    ):
        self.strategy_returns = strategy_returns
        self.benchmark_input = benchmark
        self.benchmark_returns = None
        self.cumulative = None
        self.drawdown = None
        self.names = ["Strategy", "Benchmark"]

        self._validate_inputs()
        self._prepare_data()

    def _validate_inputs(self):
        """Validate input data"""
        if not isinstance(self.strategy_returns.index, pd.DatetimeIndex):
            raise ValueError("Strategy returns must have a DatetimeIndex.")

    def _prepare_data(self):
        """Prepare all data for analysis"""
        # Handle benchmark
        if isinstance(self.benchmark_input, str):
            self._fetch_benchmark_data()
        else:
            self.benchmark_returns = self.benchmark_input
            if not isinstance(self.benchmark_returns.index, pd.DatetimeIndex):
                raise ValueError("Benchmark series must have a DatetimeIndex.")
            if self.benchmark_returns.name is None:
                self.benchmark_returns.name = "Benchmark"

        self.strategy_returns.name = "Strategy"
        self.df = pd.merge(
            self.strategy_returns,
            self.benchmark_returns,
            how="outer",
            left_index=True,
            right_index=True,
        )
        self.df.columns = self.names
        self.df.fillna(0, inplace=True)

        self.df = self.df.sort_index()

        if self.df.empty:
            raise ValueError("No overlapping data between strategy and benchmark.")

        # Calculate cumulative returns and drawdowns
        self.cumulative = (1 + self.df.fillna(0)).cumprod()
        self.drawdown = self.cumulative.div(self.cumulative.cummax()) - 1

    def _fetch_benchmark_data(self):
        """Fetch benchmark data from Yahoo Finance"""
        ticker_map = {"SP500": "^GSPC", "BTC": "BTC-USD"}

        ticker = ticker_map[self.benchmark_input]

        try:
            # Download historical data
            hist = yf.download(
                ticker,
                start=self.strategy_returns.index[0],
                end=self.strategy_returns.index[-1],
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to fetch benchmark data for {ticker}: {e}")

        self.benchmark_returns = hist["Close"].pct_change().dropna()
        self.benchmark_returns.index.name = "timestamp"
        self.benchmark_returns.columns = [self.benchmark_input]

    def _get_annualization_factor(self):
        """Get annualization factor based on data frequency"""
        try:
            freq = pd.infer_freq(self.df.index)
            if freq and ("D" in freq or "B" in freq):
                return 365
            elif freq and "H" in freq:
                return 365 * 24
            elif freq and "W" in freq:
                return 52
            elif freq and "M" in freq:
                return 12
        except Exception:
            pass
        return 252  # Default to daily

    def compute_summary_metrics(self) -> pd.DataFrame:
        """Compute summary metrics using instance data"""
        ann_factor = self._get_annualization_factor()

        summary = {}

        for i, col in enumerate(
            self.df.columns
        ):  # Reverse order to have strategy first
            returns = self.df[col]
            cumulative = self.cumulative[col]
            drawdown = self.drawdown[col]

            # Basic metrics
            start_date = returns.index[0]
            end_date = returns.index[-1]
            duration_weeks = (end_date - start_date).days / 7

            # Return metrics
            annual_return = ((1 + returns).prod()) ** (ann_factor / len(returns)) - 1
            annual_volatility = returns.std() * np.sqrt(ann_factor)
            total_return = cumulative.iloc[-1] - 1

            # Risk metrics
            sharpe = sharpe_ratio(returns, ann_factor)
            sortino = sortino_ratio(returns, ann_factor)
            pf = profit_factor(returns)
            max_drawdown = drawdown.min()
            winrate = win_rate(returns)

            # Correlation (only meaningful for strategy vs benchmark)
            if col == self.names[0] and len(self.df.columns) > 1:
                other_col = [c for c in self.df.columns if c != col][0]
                correlation = self.df[col].corr(self.df[other_col])
                if pd.isna(correlation):
                    correlation = 0.0
            else:
                correlation = 1.0 if col != self.names[0] else 0.0

            summary[self.names[i]] = {
                "Start Date": start_date.strftime("%Y-%m-%d"),
                "End Date": end_date.strftime("%Y-%m-%d"),
                "Duration (weeks)": f"{duration_weeks:.1f}",
                "Total Return (%)": f"{total_return * 100:.2f}%",
                "Annualized Return (%)": f"{annual_return * 100:.2f}%",
                "Annualized Volatility (%)": f"{annual_volatility * 100:.2f}%",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Sortino Ratio": f"{sortino:.2f}",
                "Profit Factor": f"{pf:.2f}",
                "Win Rate (%)": f"{winrate * 100:.2f}%",
                "Correlation": f"{correlation:.2f}",
                "Max Drawdown (%)": f"{max_drawdown * 100:.2f}%",
            }

        return pd.DataFrame(summary)

    def plot(self) -> None:
        """Create performance plots using instance data"""
        # Colors
        strategy_color = "darkorange"
        benchmark_color = "indigo"
        colors = [benchmark_color, strategy_color]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.45, 0.3, 0.25],
            subplot_titles=[
                "Cumulative Return",
                "Drawdown",
                "Strategy Rolling Sharpe Ratio (6-Month)",
            ],
            x_title="Date",
        )

        # Cumulative returns
        for i, col in enumerate(self.df.columns):
            fig.add_trace(
                go.Scatter(
                    x=self.cumulative.index,
                    y=self.cumulative[col] - 1,
                    mode="lines",
                    name=self.names[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                ),
                row=1,
                col=1,
            )

            # final_value = self.cumulative[col].iloc[-1]
            # final_return = final_value - 1

            # fig.add_annotation(
            #     x=self.cumulative.index[-1],
            #     y=final_return,
            #     text=f"{final_return:+.1%}",
            #     showarrow=False,
            #     font=dict(color=colors[i % len(colors)], size=12),
            #     xanchor="left",
            #     yanchor="bottom",
            #     row=1,
            #     col=1,
            # )

        # Drawdown
        for i, col in enumerate(self.df.columns):
            fig.add_trace(
                go.Scatter(
                    x=self.drawdown.index,
                    y=self.drawdown[col],
                    mode="lines",
                    name=f"{col} Drawdown",
                    line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
                    fill="tozeroy",
                    opacity=0.2,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # 6-Month Rolling Sharpe Ratio
        rolling_window = 126  # 6 months of daily data
        if len(self.df[self.names[0]]) < rolling_window:
            raise ValueError(
                f"Not enough data for rolling window of {rolling_window} periods."
            )
        ann_factor = self._get_annualization_factor()
        rolling_sharpe = (
            self.df[self.names[0]]
            .rolling(window=rolling_window)
            .apply(lambda x: sharpe_ratio(x, ann_factor))
            .dropna()
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode="lines",
                name=f"{self.names[0]} 1y Sharpe",
                line=dict(color="grey", width=2),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        hline_value = sharpe_ratio(
            self.df[self.names[0]],
            ann_factor,
        )

        # Add horizontal line using add_hline, but include a dummy invisible scatter for hover
        fig.add_hline(
            y=hline_value,
            line=dict(color="red", width=1, dash="dash"),
            row=3,
            col=1,
        )

        # Layout
        fig.update_layout(
            height=1000,
            width=1000,
            hovermode="x unified",
            title_text=f"{self.names[0]} Performance Report",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color="black"),
            ),
        )

        fig.update_yaxes(
            title_text="Cumulative Return [%]", tickformat=".0%", row=1, col=1
        )
        fig.update_yaxes(
            range=[-1, 0], title_text="Drawdown", tickformat=".0%", row=2, col=1
        )
        fig.update_yaxes(
            range=[-5, 10],
            title_text="Yearly Sharpe Ratio",
            tickformat=".2f",
            row=3,
            col=1,
        )
        fig.show()

    def plot_monthly_returns(self) -> None:
        """
        Plot monthly returns as a confusion matrix (heatmap) with year in y and month in x.
        Add two columns: annual return and annual max drawdown.
        """
        # Compute monthly returns
        monthly = self.df.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly.index = pd.to_datetime(monthly.index)
        monthly["Year"] = monthly.index.year
        monthly["Month"] = monthly.index.month

        # Pivot to Year x Month
        pivot = monthly.pivot(index="Year", columns="Month", values=self.names[0])
        pivot = pivot.reindex(columns=range(1, 13), fill_value=np.nan)

        # Annual return
        annual_return = monthly.groupby("Year")[self.names[0]].apply(
            lambda x: (1 + x).prod() - 1
        )

        # Add columns
        pivot[""] = np.nan  # Placeholder for empty column
        pivot["Annual Return"] = annual_return
        pivot["Max Drawdown"] = monthly.groupby("Year")[self.names[0]].apply(
            lambda x: (1 + x).cumprod().div((1 + x).cumprod().cummax()).min() - 1
        )
        pivot.sort_index(ascending=False, inplace=True)

        # Month labels
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "",
            "Annual Return",
            "Max Drawdown",
        ]
        fig = px.imshow(
            pivot,
            labels=dict(x="Month", y="Year", color="Return"),
            x=month_labels,
            y=pivot.index.astype(str),
            color_continuous_scale=[
                "red",
                "lightgrey",
                "forestgreen",
            ],  # Use a continuous diverging color scale
            aspect="auto",
            title=f"{self.names[0]} Monthly Returns",
            text_auto=".1%",
            zmin=-0.2,  # adjust as needed for your data range
            zmax=0.2,  # adjust as needed for your data range
        )

        fig.update_layout(
            height=600,
            width=1000,
            title_x=0.5,
            font=dict(size=12),
            xaxis_title="Month",
            yaxis_title="Year",
            xaxis=dict(
                tickmode="array", tickvals=list(range(13)), ticktext=month_labels
            ),
            yaxis=dict(tickmode="linear", dtick=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.show()
