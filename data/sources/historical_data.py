"""Historical data source for backtesting.

Provides access to historical market data from in-memory DataFrames or files.
"""

from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from strategies.sdk.events import BarEvent, MarketEvent


class HistoricalDataSource:
    """Data source that provides historical market data.

    Used for backtesting. Data is stored in memory for fast access.
    """

    def __init__(self, data: pd.DataFrame, timeframe: str = "1D"):
        """Initialize historical data source.

        Args:
            data: DataFrame with columns: timestamp, symbol, open, high, low, close, volume
            timeframe: Bar timeframe (e.g., "1D", "1H")
        """
        # Validate required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        # Ensure timestamp is datetime
        self._data = data.copy()
        self._data["timestamp"] = pd.to_datetime(self._data["timestamp"])

        # Sort by timestamp for efficient lookups
        self._data = self._data.sort_values("timestamp").reset_index(drop=True)

        self._timeframe = timeframe

        # Create index for fast lookups
        self._data_by_time = self._data.groupby("timestamp")

        # Track latest prices
        self._latest_prices: Dict[str, float] = {}

    @classmethod
    def from_csv(cls, filepath: str, timeframe: str = "1D") -> 'HistoricalDataSource':
        """Load data from CSV file.

        Args:
            filepath: Path to CSV file
            timeframe: Bar timeframe

        Returns:
            HistoricalDataSource instance
        """
        data = pd.read_csv(filepath)
        return cls(data, timeframe)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, pd.DataFrame], timeframe: str = "1D") -> 'HistoricalDataSource':
        """Create from dictionary of symbol -> DataFrame.

        Args:
            data_dict: Dict of symbol -> DataFrame (with timestamp, open, high, low, close, volume)
            timeframe: Bar timeframe

        Returns:
            HistoricalDataSource instance
        """
        dfs = []
        for symbol, df in data_dict.items():
            df = df.copy()
            df["symbol"] = symbol
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        return cls(combined, timeframe)

    def get_events(self, timestamp: datetime) -> List[MarketEvent]:
        """Get market events for a specific timestamp.

        Args:
            timestamp: Timestamp to get events for

        Returns:
            List of BarEvent objects
        """
        if timestamp not in self._data_by_time.groups:
            return []

        # Get all bars at this timestamp
        bars = self._data_by_time.get_group(timestamp)

        events = []
        for _, row in bars.iterrows():
            event = BarEvent(
                timestamp=timestamp,
                symbol=row["symbol"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                timeframe=self._timeframe,
                event_type=None  # Will be set by __post_init__
            )
            events.append(event)

            # Update latest price
            self._latest_prices[row["symbol"]] = row["close"]

        return events

    def get_bars(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get historical bars for symbols.

        Args:
            symbols: List of symbols
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            timeframe: Bar timeframe (must match data timeframe)
            limit: Maximum number of bars per symbol

        Returns:
            DataFrame with OHLCV data
        """
        if timeframe and timeframe != self._timeframe:
            raise ValueError(f"Requested timeframe {timeframe} doesn't match data timeframe {self._timeframe}")

        # Filter by symbols
        mask = self._data["symbol"].isin(symbols)

        # Filter by date range
        if start:
            mask &= self._data["timestamp"] >= start
        if end:
            mask &= self._data["timestamp"] <= end

        result = self._data[mask].copy()

        # Apply limit per symbol
        if limit:
            result = result.groupby("symbol").tail(limit).reset_index(drop=True)

        return result

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest known price for symbol.

        Args:
            symbol: Symbol to query

        Returns:
            Latest price or None
        """
        return self._latest_prices.get(symbol)

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict of symbol -> price
        """
        return {
            symbol: self._latest_prices.get(symbol, 0.0)
            for symbol in symbols
        }

    def get_symbols(self) -> List[str]:
        """Get list of all symbols in data.

        Returns:
            List of symbols
        """
        return self._data["symbol"].unique().tolist()

    def get_date_range(self) -> tuple:
        """Get date range of data.

        Returns:
            Tuple of (start_date, end_date)
        """
        return (self._data["timestamp"].min(), self._data["timestamp"].max())

    def __repr__(self) -> str:
        start, end = self.get_date_range()
        symbols = self.get_symbols()
        return (f"HistoricalDataSource(symbols={len(symbols)}, "
                f"timeframe={self._timeframe}, "
                f"range={start.date()} to {end.date()})")


def generate_sample_data(
    symbols: List[str],
    start: datetime,
    end: datetime,
    freq: str = "1D",
    initial_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Generate random walk sample data for testing.

    Args:
        symbols: List of symbols
        start: Start date
        end: End date
        freq: Frequency (pandas frequency string)
        initial_price: Starting price
        volatility: Daily volatility (std dev of returns)

    Returns:
        DataFrame with OHLCV data
    """
    timestamps = pd.date_range(start, end, freq=freq)

    dfs = []
    for symbol in symbols:
        # Generate random walk for close prices
        returns = np.random.normal(0, volatility, len(timestamps))
        close = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLCV from close
        high = close * (1 + np.abs(np.random.normal(0, volatility/2, len(timestamps))))
        low = close * (1 - np.abs(np.random.normal(0, volatility/2, len(timestamps))))
        open_ = close * (1 + np.random.normal(0, volatility/4, len(timestamps)))
        volume = np.random.uniform(1000000, 10000000, len(timestamps))

        df = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
