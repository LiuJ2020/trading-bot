"""Common technical indicators.

Industry-standard technical analysis indicators with proper
edge case handling and vectorized implementations.
"""

import pandas as pd
import numpy as np

from features.store.base import Feature, CompositeFeature


class SMA(Feature):
    """Simple Moving Average.

    The arithmetic mean of prices over a specified period.
    """

    def __init__(self, period: int = 20, column: str = "close"):
        """Initialize SMA feature.

        Args:
            period: Number of periods for moving average
            column: Price column to use (default: 'close')
        """
        super().__init__(
            name=f"sma_{column}_{period}" if column != "close" else f"sma_{period}",
            version="1.0.0",
            description=f"Simple moving average of {column} over {period} periods",
            dependencies=[column],
            window=period,
            parameters={"period": period, "column": column},
            tags={"technical", "trend", "moving_average"}
        )
        self.period = period
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute SMA."""
        self.validate_data(data)
        return data[self.column].rolling(window=self.period).mean()


class EMA(Feature):
    """Exponential Moving Average.

    Gives more weight to recent prices, responds faster to price changes.
    """

    def __init__(self, period: int = 20, column: str = "close"):
        """Initialize EMA feature.

        Args:
            period: Number of periods for EMA calculation
            column: Price column to use (default: 'close')
        """
        super().__init__(
            name=f"ema_{column}_{period}" if column != "close" else f"ema_{period}",
            version="1.0.0",
            description=f"Exponential moving average of {column} over {period} periods",
            dependencies=[column],
            window=period,
            parameters={"period": period, "column": column},
            tags={"technical", "trend", "moving_average"}
        )
        self.period = period
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute EMA."""
        self.validate_data(data)
        return data[self.column].ewm(span=self.period, adjust=False).mean()


class RSI(Feature):
    """Relative Strength Index.

    Momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100:
    - Above 70: potentially overbought
    - Below 30: potentially oversold
    """

    def __init__(self, period: int = 14, column: str = "close"):
        """Initialize RSI feature.

        Args:
            period: Number of periods for RSI calculation (standard: 14)
            column: Price column to use (default: 'close')
        """
        super().__init__(
            name=f"rsi_{column}_{period}" if column != "close" else f"rsi_{period}",
            version="1.0.0",
            description=f"Relative Strength Index of {column} over {period} periods",
            dependencies=[column],
            window=period + 1,
            parameters={"period": period, "column": column},
            tags={"technical", "momentum", "oscillator"}
        )
        self.period = period
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute RSI using Wilder's smoothing method."""
        self.validate_data(data)

        # Calculate price changes
        delta = data[self.column].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate average gains and losses using EMA (Wilder's method)
        avg_gains = gains.ewm(span=self.period, adjust=False).mean()
        avg_losses = losses.ewm(span=self.period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi


class MACD(Feature):
    """Moving Average Convergence Divergence.

    Trend-following momentum indicator showing the relationship between
    two moving averages of a security's price.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        column: str = "close"
    ):
        """Initialize MACD feature.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            column: Price column to use (default: 'close')
        """
        super().__init__(
            name=f"macd_{fast_period}_{slow_period}",
            version="1.0.0",
            description=f"MACD line ({fast_period}, {slow_period})",
            dependencies=[column],
            window=slow_period,
            parameters={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "column": column
            },
            tags={"technical", "trend", "momentum"}
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute MACD line."""
        self.validate_data(data)

        fast_ema = data[self.column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.column].ewm(span=self.slow_period, adjust=False).mean()

        return fast_ema - slow_ema


class MACDSignal(CompositeFeature):
    """MACD Signal Line.

    9-period EMA of the MACD line.
    """

    def __init__(self, signal_period: int = 9):
        """Initialize MACD signal feature.

        Args:
            signal_period: Signal line EMA period (default: 9)
        """
        super().__init__(
            name=f"macd_signal_{signal_period}",
            version="1.0.0",
            description=f"MACD signal line ({signal_period}-period EMA of MACD)",
            feature_dependencies=["macd_12_26"],
            window=26 + signal_period,
            parameters={"signal_period": signal_period},
            tags={"technical", "trend", "momentum"}
        )
        self.signal_period = signal_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute MACD signal line."""
        self.validate_data(data)
        return data["macd_12_26"].ewm(span=self.signal_period, adjust=False).mean()


class MACDHistogram(CompositeFeature):
    """MACD Histogram.

    Difference between MACD line and signal line.
    """

    def __init__(self):
        """Initialize MACD histogram feature."""
        super().__init__(
            name="macd_histogram",
            version="1.0.0",
            description="MACD histogram (MACD - Signal)",
            feature_dependencies=["macd_12_26", "macd_signal_9"],
            window=26 + 9,
            parameters={},
            tags={"technical", "trend", "momentum"}
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute MACD histogram."""
        self.validate_data(data)
        return data["macd_12_26"] - data["macd_signal_9"]


class BollingerBands(Feature):
    """Bollinger Bands - all three bands in one.

    Returns a DataFrame with three columns: bb_upper, bb_middle, bb_lower.
    Note: This returns a DataFrame instead of Series.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0, column: str = "close"):
        """Initialize Bollinger Bands feature.

        Args:
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            column: Price column to use (default: 'close')
        """
        super().__init__(
            name=f"bb_{period}_{num_std}",
            version="1.0.0",
            description=f"Bollinger Bands ({period}, {num_std} std)",
            dependencies=[column],
            window=period,
            parameters={"period": period, "num_std": num_std, "column": column},
            tags={"technical", "volatility", "bands"}
        )
        self.period = period
        self.num_std = num_std
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Bollinger Bands middle band (SMA).

        Note: Use BollingerUpper/BollingerLower for individual bands.
        """
        self.validate_data(data)
        return data[self.column].rolling(window=self.period).mean()


class BollingerUpper(Feature):
    """Bollinger Bands Upper Band."""

    def __init__(self, period: int = 20, num_std: float = 2.0, column: str = "close"):
        """Initialize upper Bollinger Band.

        Args:
            period: Moving average period
            num_std: Number of standard deviations
            column: Price column to use
        """
        super().__init__(
            name=f"bb_upper_{period}_{num_std}",
            version="1.0.0",
            description=f"Bollinger upper band ({period}, {num_std} std)",
            dependencies=[column],
            window=period,
            parameters={"period": period, "num_std": num_std, "column": column},
            tags={"technical", "volatility", "bands"}
        )
        self.period = period
        self.num_std = num_std
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute upper Bollinger Band."""
        self.validate_data(data)
        rolling = data[self.column].rolling(window=self.period)
        sma = rolling.mean()
        std = rolling.std()
        return sma + (std * self.num_std)


class BollingerLower(Feature):
    """Bollinger Bands Lower Band."""

    def __init__(self, period: int = 20, num_std: float = 2.0, column: str = "close"):
        """Initialize lower Bollinger Band.

        Args:
            period: Moving average period
            num_std: Number of standard deviations
            column: Price column to use
        """
        super().__init__(
            name=f"bb_lower_{period}_{num_std}",
            version="1.0.0",
            description=f"Bollinger lower band ({period}, {num_std} std)",
            dependencies=[column],
            window=period,
            parameters={"period": period, "num_std": num_std, "column": column},
            tags={"technical", "volatility", "bands"}
        )
        self.period = period
        self.num_std = num_std
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute lower Bollinger Band."""
        self.validate_data(data)
        rolling = data[self.column].rolling(window=self.period)
        sma = rolling.mean()
        std = rolling.std()
        return sma - (std * self.num_std)


class ATR(Feature):
    """Average True Range.

    Volatility indicator measuring the degree of price movement.
    Higher ATR values indicate higher volatility.
    """

    def __init__(self, period: int = 14):
        """Initialize ATR feature.

        Args:
            period: Number of periods for ATR calculation (standard: 14)
        """
        super().__init__(
            name=f"atr_{period}",
            version="1.0.0",
            description=f"Average True Range over {period} periods",
            dependencies=["high", "low", "close"],
            window=period + 1,
            parameters={"period": period},
            tags={"technical", "volatility"}
        )
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute ATR."""
        self.validate_data(data)

        # Calculate True Range
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift())
        low_close = abs(data["low"] - data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR using EMA (Wilder's smoothing)
        atr = true_range.ewm(span=self.period, adjust=False).mean()

        return atr


class Stochastic(Feature):
    """Stochastic Oscillator %K.

    Momentum indicator comparing closing price to price range over a period.
    Values range from 0 to 100.
    """

    def __init__(self, period: int = 14):
        """Initialize Stochastic oscillator.

        Args:
            period: Lookback period for high/low range (standard: 14)
        """
        super().__init__(
            name=f"stoch_{period}",
            version="1.0.0",
            description=f"Stochastic oscillator %K over {period} periods",
            dependencies=["high", "low", "close"],
            window=period,
            parameters={"period": period},
            tags={"technical", "momentum", "oscillator"}
        )
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Stochastic %K."""
        self.validate_data(data)

        # Rolling highest high and lowest low
        rolling_high = data["high"].rolling(window=self.period).max()
        rolling_low = data["low"].rolling(window=self.period).min()

        # Stochastic calculation
        stoch = ((data["close"] - rolling_low) /
                 (rolling_high - rolling_low).replace(0, np.nan)) * 100

        return stoch


class StochasticD(CompositeFeature):
    """Stochastic %D - signal line.

    3-period SMA of %K.
    """

    def __init__(self, smooth_period: int = 3):
        """Initialize Stochastic %D.

        Args:
            smooth_period: Smoothing period for %D (standard: 3)
        """
        super().__init__(
            name=f"stoch_d_{smooth_period}",
            version="1.0.0",
            description=f"Stochastic %D ({smooth_period}-period SMA of %K)",
            feature_dependencies=["stoch_14"],
            window=14 + smooth_period,
            parameters={"smooth_period": smooth_period},
            tags={"technical", "momentum", "oscillator"}
        )
        self.smooth_period = smooth_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Stochastic %D."""
        self.validate_data(data)
        return data["stoch_14"].rolling(window=self.smooth_period).mean()


class OBV(Feature):
    """On-Balance Volume.

    Cumulative volume-based indicator measuring buying and selling pressure.
    """

    def __init__(self):
        """Initialize OBV feature."""
        super().__init__(
            name="obv",
            version="1.0.0",
            description="On-Balance Volume",
            dependencies=["close", "volume"],
            window=2,
            parameters={},
            tags={"technical", "volume"}
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute OBV."""
        self.validate_data(data)

        # Calculate price direction
        price_direction = data["close"].diff()

        # Assign volume based on price direction
        obv_values = np.where(
            price_direction > 0, data["volume"],
            np.where(price_direction < 0, -data["volume"], 0)
        )

        # Cumulative sum
        obv = pd.Series(obv_values, index=data.index).cumsum()

        return obv


class ADX(Feature):
    """Average Directional Index.

    Trend strength indicator (not direction).
    Values above 25 suggest a trending market.
    """

    def __init__(self, period: int = 14):
        """Initialize ADX feature.

        Args:
            period: Number of periods for ADX calculation (standard: 14)
        """
        super().__init__(
            name=f"adx_{period}",
            version="1.0.0",
            description=f"Average Directional Index over {period} periods",
            dependencies=["high", "low", "close"],
            window=period * 2,
            parameters={"period": period},
            tags={"technical", "trend", "strength"}
        )
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute ADX."""
        self.validate_data(data)

        # Calculate directional movement
        high_diff = data["high"].diff()
        low_diff = -data["low"].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Calculate True Range
        tr_high_low = data["high"] - data["low"]
        tr_high_close = abs(data["high"] - data["close"].shift())
        tr_low_close = abs(data["low"] - data["close"].shift())
        tr = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)

        # Smooth with EMA
        atr = tr.ewm(span=self.period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=data.index).ewm(
            span=self.period, adjust=False
        ).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=data.index).ewm(
            span=self.period, adjust=False
        ).mean() / atr

        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        return adx
