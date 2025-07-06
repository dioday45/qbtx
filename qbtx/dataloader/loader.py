import pandas as pd
from abc import ABC, abstractmethod


class Loader(ABC):
    """
    Abstract base class for loading a Parquet dataset.
    """

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        """
        Load the Parquet file into a pandas DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
