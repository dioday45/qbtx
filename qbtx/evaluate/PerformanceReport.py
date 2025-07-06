from typing import Union
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class PerformanceReport:
    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark: Union[str, pd.Series, None],
        strategy_name: str = "Strategy",
    ):
        self.strategy_name = strategy_name
        self.strategy_returns = strategy_returns
        self.benchmark_input = benchmark
        self.benchmark_returns = None
        self.cumulative = None
        self.drawdown = None

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

        # Set names
        self.strategy_returns.name = self.strategy_name

        self.df = pd.merge(
            self.strategy_returns,
            self.benchmark_returns,
            how="outer",
            left_index=True,
            right_index=True,
        )
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
                return 252
            elif freq and "H" in freq:
                return 252 * 24
            elif freq and "W" in freq:
                return 52
            elif freq and "M" in freq:
                return 12
        except:
            pass
        return 252  # Default to daily

    def _get_risk_free_rate(self):
        """Get risk-free rate from 10Y Treasury"""
        try:
            tnx = yf.download("^TNX", auto_adjust=True, progress=False)
            if not tnx.empty:
                return tnx["Close"].dropna().iloc[-1] / 100
        except:
            pass
        return 0.02  # Default 2%

    def compute_summary_metrics(self, starting_money: float = 10000) -> pd.DataFrame:
        """Compute summary metrics using instance data"""
        ann_factor = self._get_annualization_factor()
        risk_free_rate = self._get_risk_free_rate()

        summary = {}

        for col in self.df.columns:  # Reverse order to have strategy first
            returns = self.df[col]
            cumulative = self.cumulative[col]
            drawdown = self.drawdown[col]

            # Basic metrics
            start_date = returns.index[0]
            end_date = returns.index[-1]
            duration_weeks = (end_date - start_date).days / 7

            # Return metrics
            total_return = cumulative.iloc[-1] - 1
            annual_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
            annual_volatility = returns.std() * np.sqrt(ann_factor)

            # Risk metrics
            sharpe_ratio = (
                (annual_return - risk_free_rate) / annual_volatility
                if annual_volatility > 0
                else 0
            )
            max_drawdown = drawdown.min()

            # Value metrics
            start_value = starting_money
            final_value = starting_money * cumulative.iloc[-1]
            peak_value = starting_money * cumulative.max()

            # Correlation (only meaningful for strategy vs benchmark)
            if col == self.strategy_name and len(self.df.columns) > 1:
                other_col = [c for c in self.df.columns if c != col][0]
                correlation = self.df[col].corr(self.df[other_col])
                if pd.isna(correlation):
                    correlation = 0.0
            else:
                correlation = 1.0 if col != self.strategy_name else 0.0

            summary[col] = {
                "Start Date": start_date.strftime("%Y-%m-%d"),
                "End Date": end_date.strftime("%Y-%m-%d"),
                "Duration (weeks)": f"{duration_weeks:.1f}",
                "Exposure Time (%)": "100.00",
                "Start Value": f"${start_value:,.2f}",
                "Final Value": f"${final_value:,.2f}",
                "Peak Value": f"${peak_value:,.2f}",
                "Total Return (%)": f"{total_return * 100:.2f}%",
                "Annualized Return (%)": f"{annual_return * 100:.2f}%",
                "Annualized Volatility (%)": f"{annual_volatility * 100:.2f}%",
                "Sharpe Ratio": f"{sharpe_ratio.values[0]:.2f}",
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
            subplot_titles=["Cumulative Return", "Drawdown", "Yearly Sharpe Ratio"],
            x_title="Date",
        )

        # Cumulative returns
        for i, col in enumerate(self.df.columns):
            fig.add_trace(
                go.Scatter(
                    x=self.cumulative.index,
                    y=self.cumulative[col] - 1,
                    mode="lines",
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                ),
                row=1,
                col=1,
            )

            final_value = self.cumulative[col].iloc[-1]
            final_return = final_value - 1

            fig.add_annotation(
                x=self.cumulative.index[-1],
                y=final_return,
                text=f"{final_return:+.1%}",
                showarrow=False,
                font=dict(color=colors[i % len(colors)], size=12),
                xanchor="left",
                yanchor="bottom",
                row=1,
                col=1,
            )

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

        # Rolling Sharpe Ratio
        annual_returns = (self.df + 1).resample("YE").prod() - 1
        annual_std = self.df.resample("YE").std() * np.sqrt(252)  # Annualized std dev
        sharpe_ratio = (
            annual_returns - self._get_risk_free_rate().values[0]
        ) / annual_std

        fig.add_trace(
            go.Bar(
                x=sharpe_ratio.index.year,
                y=sharpe_ratio[self.strategy_name],
                name=f"{self.strategy_name} Sharpe",
                marker_color=benchmark_color,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Layout
        fig.update_layout(
            template="plotly_white",
            height=1000,
            width=1000,
            hovermode="x unified",
            title_text=f"{self.strategy_name} Performance Report",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color="black"),
            ),
        )

        fig.update_yaxes(title_text="Cumulative Return [%]", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        fig.update_yaxes(
            title_text="Yearly Sharpe Ratio", tickformat=".2f", row=3, col=1
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
        pivot = monthly.pivot(index="Year", columns="Month", values=self.strategy_name)
        pivot = pivot.reindex(columns=range(1, 13), fill_value=np.nan)

        # Annual return
        annual_return = monthly.groupby("Year")[self.strategy_name].apply(
            lambda x: (1 + x).prod() - 1
        )

        # Add columns
        pivot[""] = np.nan  # Placeholder for empty column
        pivot["Annual Return"] = annual_return
        pivot["Max Drawdown"] = monthly.groupby("Year")[self.strategy_name].apply(
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
            title=f"{self.strategy_name} Monthly Returns",
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
