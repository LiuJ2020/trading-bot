"""Database-backed data source for backtesting.

Reads historical market data from SQLite + Parquet hybrid storage.
"""

from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import logging

from strategies.sdk.events import BarEvent, MarketEvent
from data.storage.market_data_store import MarketDataStore

logger = logging.getLogger(__name__)


class DatabaseDataSource:
    """Data source that reads from database storage.

    Used for backtesting with persisted historical data.
    Replaces in-memory HistoricalDataSource for production use.
    """

    def __init__(
        self,
        db_path: str,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1D"
    ):
        """Initialize database data source.

        Args:
            db_path: Path to database directory
            symbols: List of symbols to load (None = all available)
            timeframe: Bar timeframe (e.g., "1D", "1H")
        """
        self.store = MarketDataStore(db_path)
        self.timeframe = timeframe

        # Load symbols
        if symbols is None:
            self.symbols = self.store.list_symbols(active_only=True)
        else:
            self.symbols = symbols

        # Load all data into memory for fast backtesting
        self._data = self._load_all_data()

        # Group by timestamp for event delivery
        if not self._data.empty:
            self._data_by_time = self._data.groupby("timestamp")
        else:
            self._data_by_time = None

        # Track latest prices
        self._latest_prices: Dict[str, float] = {}

        logger.info(
            f"Loaded {len(self._data)} bars for {len(self.symbols)} symbols "
            f"({self.timeframe})"
        )

    def _load_all_data(self) -> pd.DataFrame:
        """Load all data for symbols into memory.

        Returns:
            Combined DataFrame with all symbols
        """
        dfs = []

        for symbol in self.symbols:
            try:
                df = self.store.get_bars(
                    symbol=symbol,
                    timeframe=self.timeframe
                )

                if not df.empty:
                    df['symbol'] = symbol
                    dfs.append(df)
                else:
                    logger.warning(f"No data for {symbol} ({self.timeframe})")

            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        if not dfs:
            logger.warning("No data loaded from database")
            return pd.DataFrame()

        # Combine and sort
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        return combined

    def get_events(self, timestamp: datetime) -> List[MarketEvent]:
        """Get market events for a specific timestamp.

        Args:
            timestamp: Timestamp to get events for

        Returns:
            List of BarEvent objects
        """
        if self._data_by_time is None or timestamp not in self._data_by_time.groups:
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
                timeframe=self.timeframe,
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
        if timeframe and timeframe != self.timeframe:
            raise ValueError(
                f"Requested timeframe {timeframe} doesn't match "
                f"data timeframe {self.timeframe}"
            )

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
        return self.symbols

    def get_date_range(self) -> tuple:
        """Get date range of data.

        Returns:
            Tuple of (start_date, end_date)
        """
        if self._data.empty:
            return (None, None)

        return (self._data["timestamp"].min(), self._data["timestamp"].max())

    def __repr__(self) -> str:
        start, end = self.get_date_range()
        if start and end:
            return (
                f"DatabaseDataSource(symbols={len(self.symbols)}, "
                f"timeframe={self.timeframe}, "
                f"range={start.date()} to {end.date()})"
            )
        else:
            return f"DatabaseDataSource(symbols={len(self.symbols)}, no data)"
