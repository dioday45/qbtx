from qbtx.dataloader.loader import Loader
import pandas as pd


class CoinGeckoLoader(Loader):
    """
    CoinGeckoLoader is a data loader class for retrieving and managing historical cryptocurrency market data
    from CoinGecko, stored in Parquet format. It inherits from the Loader base class and provides methods
    to load the data from a specified local path and access it as a pandas DataFrame.

    Attributes:
        data_path (str): The local file path to the CoinGecko market data in Parquet format.
        data (pd.DataFrame): The DataFrame containing the loaded CoinGecko market data.

    Methods:
        __init__():
            Initializes the CoinGeckoLoader with the predefined data path.

        load():
            Loads the CoinGecko market data from the specified Parquet file into a pandas DataFrame.

        get_data():
            Returns the loaded CoinGecko market data as a pandas DataFrame, loading it if not already loaded.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        assert data_path, "Data path must be provided"
        self.data = self._load() if data_path else None

    def _load(self):
        """
        Load the CoinGecko market data from the Parquet file into a pandas DataFrame.
        """
        try:
            df = pd.read_parquet(self.data_path, engine="pyarrow")
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found at {self.data_path}. Please check the path."
            )

    def get_data(self):
        """
        Get the loaded CoinGecko data.
        """
        return self.data

    def get_tickers(self) -> list:
        """
        Get the list of unique tickers in the dataset.

        Returns:
            list: A list of unique ticker symbols present in the data.
        """
        return (
            self.data["ticker"].unique().tolist()
            if "ticker" in self.data.columns
            else []
        )

    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Get the data for a specific ticker.

        Args:
            ticker (str): The ticker symbol to filter the data by.

        Returns:
            pd.DataFrame: A DataFrame containing only the data for the specified ticker.
        """
        if "ticker" in self.data.columns:
            return self.data[self.data["ticker"] == ticker].drop(columns=["ticker"])
        else:
            raise ValueError(f"Ticker '{ticker}' not found in the dataset.")
