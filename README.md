# qbtx

**qbtx** is a modular backtesting engine for quantitative trading strategies, built for flexibility and performance. It supports strategy definition, execution, and detailed performance evaluation through clean architecture and rich visualization.

It allows you to:

- Design and test custom trading strategies
- Evaluate performance metrics such as cumulative return, drawdown, and annual Sharpe ratio
- Visualize strategy performance using Plotly charts

---

## 📦 Setup with Poetry

Install dependencies and create the virtual environment:

```bash
poetry install
```

---

## 📂 Data Setup

Create a `data/` directory in the project root and place your historical data files there.

```bash
mkdir data
```

A good starting point is to use the [`binance_historical_data`](https://github.com/stas-prokopiev/binance_historical_data/tree/master) package:

> https://github.com/stas-prokopiev/binance_historical_data/tree/master

This dataset provides cleaned OHLCV historical data for Binance spot and futures markets.

---

## 🚀 Quick Start

Run the sample strategy notebook:

```bash
jupyter notebook notebooks/strategy/sample_strategy.ipynb
```

This shows how to:

- Define a strategy
- Run the backtest
- Generate interactive performance plots

---

## 🧱 Project Structure

```
qbtx/
├── dataloader/              # Loaders for market data (e.g., Binance, CoinGecko)
├── engine/                  # Core backtesting engine and components
├── evaluate/                # Performance reporting and plotting
├── strategy/                # Strategy definitions
notebooks/
└── strategy/
    └── sample_strategy.ipynb
data/                        # Place your price data files here (e.g., BTCUSDT.csv)
```

---

## 📊 Features

- 🧠 **Strategy Abstraction**  
  Build new strategies by subclassing a base strategy class.

- ⚡ **Vectorized Execution**  
  Efficient portfolio simulation using `pandas`.

- 📈 **Comprehensive Analytics**  
  Cumulative returns, drawdown, and annual Sharpe ratio plotted with Plotly.

- 🔌 **Modular Design**  
  Clear separation between engine, data, strategy, and reporting.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Open to pull requests and contributions!  
Feel free to suggest features, enhancements, or new strategy modules.