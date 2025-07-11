from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class BinanceDataLoader:
    """
    Binance market data loader for structured CSV files.

    Supports loading and formatting kline data by symbol and interval.
    """

    COLUMN_NAMES = [
        "open time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close time",
        "quote asset volume",
        "number of trades",
        "taker buy base asset volume",
        "taker buy quote asset volume",
        "ignore",
    ]

    KEEP_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote asset volume",
        "number of trades",
        "taker buy base asset volume",
        "taker buy quote asset volume",
    ]

    def __init__(
        self,
        root_dir: str = "data/market_data/binance/spot/monthly",
        data_type: str = "klines",
    ):
        self.base_path = Path(root_dir) / data_type
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base directory '{self.base_path}' not found.")

    def _safe_parse_datetime(self, val):
        try:
            val = int(val)

            if val > 1e17:
                unit = "ns"
            elif val > 1e14:
                unit = "us"  # <-- This is the correct one for your case
            elif val > 1e11:
                unit = "ms"
            else:
                unit = "s"

            ts = pd.to_datetime(val, unit=unit, errors="coerce")
            if ts is pd.NaT or not (2010 <= ts.year <= 2050):
                return pd.NaT
            return ts

        except Exception as e:
            print(f"[ERROR] Failed to parse timestamp: {val} → {e}")
            return pd.NaT

    def _load_csv(self, file: Path) -> pd.DataFrame:
        df = pd.read_csv(file, header=None, names=self.COLUMN_NAMES)

        # Safe timestamp conversion
        df["open time"] = df["open time"].apply(self._safe_parse_datetime)

        df.set_index(df["open time"].dt.date, inplace=True)
        df.index.name = "date"

        df = df[self.KEEP_COLUMNS].apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)

        return df

    def load(
        self,
        interval: str = "1d",
        symbols: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        min_years: int = 4,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for specified symbols or all symbols for a given interval.

        Args:
            interval (str): Kline interval, e.g., '1d', '4h', '1h', etc.
            symbols (Optional[List[str]]): List of symbols to load. If None, load all available symbols.

        Returns:
            Dict[str, pd.DataFrame]: Mapping from symbol to cleaned DataFrame.
        """
        results = {}

        if symbols:
            for symbol in symbols:
                interval_dir = self.base_path / symbol / interval
                if not interval_dir.exists():
                    print(f"[WARN] Skipping: {interval_dir} does not exist.")
                    continue
                csv_files = list(interval_dir.glob("*.csv"))
                if not csv_files:
                    print(f"[WARN] No CSV files found for {symbol} in {interval_dir}")
                    continue
                df = pd.concat(
                    [self._load_csv(f) for f in csv_files], ignore_index=False
                ).sort_index()
                results[symbol] = df[columns] if columns else df
        else:
            for sym_dir in self.base_path.iterdir():
                if sym_dir.is_dir():
                    interval_dir = sym_dir / interval
                    if interval_dir.exists():
                        csv_files = list(interval_dir.glob("*.csv"))
                        if csv_files:
                            df = pd.concat(
                                [self._load_csv(f) for f in csv_files],
                                ignore_index=False,
                            ).sort_index()
                            results[sym_dir.name] = df[columns] if columns else df

        if results:
            min_days = min_years * 365
            # Use dictionary comprehension for efficiency
            results = {
                symbol: df
                for symbol, df in results.items()
                if df.index.nunique() >= min_days
            }

        return results

    def get_columns(self) -> List[str]:
        """
        Get the list of columns used in the loaded DataFrames.

        Returns:
            List[str]: List of column names.
        """
        return self.KEEP_COLUMNS
