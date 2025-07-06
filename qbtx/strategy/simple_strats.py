from qbtx.strategy.base import Strategy
import numpy as np
import pandas as pd


class BuyAndHoldStrategy(Strategy):
    def generate_signals(self):
        signals = pd.DataFrame(1, index=self.prices.index, columns=self.prices.columns)
        # Force close position on final day
        signals.iloc[-1] = 0
        self.signals = signals
        return self.signals


class MACrossStrategy(Strategy):
    def __init__(self, prices, short_window=20, long_window=50):
        super().__init__(prices)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        short_ma = self.prices.rolling(self.short_window).mean()
        long_ma = self.prices.rolling(self.long_window).mean()
        self.signals = (short_ma > long_ma).astype(int)
        return self.signals


class ReversalStrategy(Strategy):
    def __init__(self, prices):
        super().__init__(prices)

    def generate_signals(self):
        ret = self.prices.pct_change().fillna(0)

        # Long if drop > 1%, short if jump > 1%
        signals = np.select([ret < -0.01, ret > 0.01], [1, -1], default=0)

        self.signals = pd.DataFrame(signals, index=ret.index, columns=ret.columns)
        return self.signals
