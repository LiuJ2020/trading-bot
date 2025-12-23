"""Basic price transforms and returns calculations.

These are fundamental features used as building blocks for more
complex indicators.
"""

import pandas as pd
import numpy as np

from features.store.base import Feature


class SimpleReturns(Feature):
    """Simple (arithmetic) returns.

    Calculation: (price[t] - price[t-1]) / price[t-1]
    """

    def __init__(self, column: str = "close", periods: int = 1):
        """Initialize simple returns feature.

        Args:
            column: Price column to use (default: 'close')
            periods: Number of periods for return calculation (default: 1)
        """
        super().__init__(
            name=f"returns_{column}_{periods}" if periods != 1 else f"returns_{column}",
            version="1.0.0",
            description=f"Simple returns on {column} over {periods} period(s)",
            dependencies=[column],
            window=periods + 1,
            parameters={"column": column, "periods": periods},
            tags={"returns", "transform"}
        )
        self.column = column
        self.periods = periods

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute simple returns."""
        self.validate_data(data)
        return data[self.column].pct_change(periods=self.periods)


class LogReturns(Feature):
    """Logarithmic returns.

    Calculation: ln(price[t] / price[t-1])

    Log returns are preferred for:
    - Multi-period aggregation (can sum)
    - Statistical analysis (more normal distribution)
    - Continuous compounding
    """

    def __init__(self, column: str = "close", periods: int = 1):
        """Initialize log returns feature.

        Args:
            column: Price column to use (default: 'close')
            periods: Number of periods for return calculation (default: 1)
        """
        super().__init__(
            name=f"log_returns_{column}_{periods}" if periods != 1 else f"log_returns_{column}",
            version="1.0.0",
            description=f"Log returns on {column} over {periods} period(s)",
            dependencies=[column],
            window=periods + 1,
            parameters={"column": column, "periods": periods},
            tags={"returns", "transform"}
        )
        self.column = column
        self.periods = periods

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute log returns."""
        self.validate_data(data)
        prices = data[self.column]
        return np.log(prices / prices.shift(periods=self.periods))


class PercentChange(Feature):
    """Percent change over N periods.

    Similar to SimpleReturns but expressed as percentage (0.05 -> 5.0).
    """

    def __init__(self, column: str = "close", periods: int = 1):
        """Initialize percent change feature.

        Args:
            column: Price column to use
            periods: Number of periods for calculation
        """
        super().__init__(
            name=f"pct_change_{column}_{periods}",
            version="1.0.0",
            description=f"Percent change on {column} over {periods} period(s)",
            dependencies=[column],
            window=periods + 1,
            parameters={"column": column, "periods": periods},
            tags={"returns", "transform"}
        )
        self.column = column
        self.periods = periods

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute percent change."""
        self.validate_data(data)
        return data[self.column].pct_change(periods=self.periods) * 100


class StandardizedPrice(Feature):
    """Z-score normalized price.

    Calculation: (price - mean) / std
    Useful for mean reversion strategies.
    """

    def __init__(self, column: str = "close", window: int = 20):
        """Initialize standardized price feature.

        Args:
            column: Price column to use
            window: Rolling window for mean/std calculation
        """
        super().__init__(
            name=f"zscore_{column}_{window}",
            version="1.0.0",
            description=f"Z-score of {column} over {window} periods",
            dependencies=[column],
            window=window,
            parameters={"column": column, "window": window},
            tags={"transform", "normalization"}
        )
        self.column = column
        self._window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute z-score."""
        self.validate_data(data)
        prices = data[self.column]
        rolling_mean = prices.rolling(window=self._window).mean()
        rolling_std = prices.rolling(window=self._window).std()

        # Avoid division by zero
        return (prices - rolling_mean) / rolling_std.replace(0, np.nan)


class PriceRank(Feature):
    """Percentile rank of current price within rolling window.

    Returns values between 0 and 1, where:
    - 1.0 means price is at highest point in window
    - 0.0 means price is at lowest point in window
    - 0.5 means price is at median
    """

    def __init__(self, column: str = "close", window: int = 20):
        """Initialize price rank feature.

        Args:
            column: Price column to use
            window: Rolling window for ranking
        """
        super().__init__(
            name=f"price_rank_{column}_{window}",
            version="1.0.0",
            description=f"Percentile rank of {column} over {window} periods",
            dependencies=[column],
            window=window,
            parameters={"column": column, "window": window},
            tags={"transform", "rank"}
        )
        self.column = column
        self._window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute price percentile rank."""
        self.validate_data(data)
        return data[self.column].rolling(window=self._window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )


class HighLowRange(Feature):
    """Range between high and low as percentage of close.

    Useful for volatility and breakout detection.
    """

    def __init__(self):
        """Initialize high-low range feature."""
        super().__init__(
            name="high_low_range",
            version="1.0.0",
            description="(High - Low) / Close as percentage",
            dependencies=["high", "low", "close"],
            window=1,
            parameters={},
            tags={"transform", "volatility"}
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute high-low range."""
        self.validate_data(data)
        return (data["high"] - data["low"]) / data["close"] * 100


class TypicalPrice(Feature):
    """Typical price: (High + Low + Close) / 3.

    Often used as a more stable price measure than close alone.
    """

    def __init__(self):
        """Initialize typical price feature."""
        super().__init__(
            name="typical_price",
            version="1.0.0",
            description="(High + Low + Close) / 3",
            dependencies=["high", "low", "close"],
            window=1,
            parameters={},
            tags={"transform", "price"}
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute typical price."""
        self.validate_data(data)
        return (data["high"] + data["low"] + data["close"]) / 3


class VWAP(Feature):
    """Volume Weighted Average Price over a window.

    Calculation: sum(price * volume) / sum(volume)
    """

    def __init__(self, window: int = 20):
        """Initialize VWAP feature.

        Args:
            window: Rolling window for VWAP calculation
        """
        super().__init__(
            name=f"vwap_{window}",
            version="1.0.0",
            description=f"Volume-weighted average price over {window} periods",
            dependencies=["close", "volume"],
            window=window,
            parameters={"window": window},
            tags={"transform", "volume", "price"}
        )
        self._window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute VWAP."""
        self.validate_data(data)

        price_volume = data["close"] * data["volume"]
        rolling_pv = price_volume.rolling(window=self._window).sum()
        rolling_vol = data["volume"].rolling(window=self._window).sum()

        # Avoid division by zero
        return rolling_pv / rolling_vol.replace(0, np.nan)


class VolumeRatio(Feature):
    """Current volume as ratio of average volume.

    Values > 1 indicate above-average volume.
    """

    def __init__(self, window: int = 20):
        """Initialize volume ratio feature.

        Args:
            window: Rolling window for average volume
        """
        super().__init__(
            name=f"volume_ratio_{window}",
            version="1.0.0",
            description=f"Volume / Average volume over {window} periods",
            dependencies=["volume"],
            window=window,
            parameters={"window": window},
            tags={"transform", "volume"}
        )
        self._window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute volume ratio."""
        self.validate_data(data)
        avg_volume = data["volume"].rolling(window=self._window).mean()
        return data["volume"] / avg_volume.replace(0, np.nan)
