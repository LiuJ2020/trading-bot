"""Data loading utilities for research notebooks.

Provides easy access to market data for quick experimentation.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class DataLoader:
    """Unified interface for loading market data in research notebooks.

    Example:
        >>> loader = DataLoader()
        >>> data = loader.load_bars(['AAPL', 'MSFT'], '2023-01-01', '2024-01-01')
        >>> sample = loader.generate_sample_data(['SPY'], days=252)
    """

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize data loader.

        Args:
            data_dir: Optional directory containing data files
        """
        self.data_dir = Path(data_dir) if data_dir else None

    def load_bars(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Union[str, datetime],
        timeframe: str = '1D',
    ) -> pd.DataFrame:
        """Load historical bar data.

        Args:
            symbols: Symbol or list of symbols
            start: Start date (string or datetime)
            end: End date (string or datetime)
            timeframe: Bar timeframe ('1D', '1H', etc.)

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        # Try to load from data directory if specified
        if self.data_dir and self.data_dir.exists():
            return self._load_from_files(symbols, start, end, timeframe)

        # Fall back to generating sample data
        print(f"Warning: No data directory specified. Generating sample data for {symbols}")
        return generate_random_walk(symbols, start, end, timeframe=timeframe)

    def _load_from_files(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Load data from CSV/Parquet files.

        Args:
            symbols: List of symbols
            start: Start date
            end: End date
            timeframe: Timeframe

        Returns:
            Combined DataFrame
        """
        dfs = []
        for symbol in symbols:
            # Try different file formats
            for ext in ['.csv', '.parquet', '.pkl']:
                filepath = self.data_dir / f"{symbol}_{timeframe}{ext}"
                if filepath.exists():
                    df = self._read_file(filepath)
                    df['symbol'] = symbol
                    dfs.append(df)
                    break
            else:
                print(f"Warning: No data file found for {symbol}, using sample data")
                df = generate_random_walk([symbol], start, end, timeframe=timeframe)
                dfs.append(df)

        if not dfs:
            raise ValueError(f"No data found for symbols: {symbols}")

        combined = pd.concat(dfs, ignore_index=True)

        # Filter by date range
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        mask = (combined['timestamp'] >= start) & (combined['timestamp'] <= end)
        return combined[mask].reset_index(drop=True)

    @staticmethod
    def _read_file(filepath: Path) -> pd.DataFrame:
        """Read data file based on extension.

        Args:
            filepath: Path to file

        Returns:
            DataFrame
        """
        ext = filepath.suffix.lower()
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext == '.parquet':
            return pd.read_parquet(filepath)
        elif ext == '.pkl':
            return pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def generate_sample_data(
        self,
        symbols: Union[str, List[str]],
        days: int = 252,
        timeframe: str = '1D',
    ) -> pd.DataFrame:
        """Generate sample data for testing.

        Args:
            symbols: Symbol or list of symbols
            days: Number of days of data
            timeframe: Bar timeframe

        Returns:
            DataFrame with sample OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        end = datetime.now()
        start = end - timedelta(days=days)

        return generate_random_walk(symbols, start, end, timeframe=timeframe)


def load_bars(
    symbols: Union[str, List[str]],
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: str = '1D',
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function to load bar data.

    Args:
        symbols: Symbol or list of symbols
        start: Start date
        end: End date
        timeframe: Bar timeframe
        data_dir: Optional data directory

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> data = load_bars('AAPL', '2023-01-01', '2024-01-01')
        >>> data = load_bars(['AAPL', 'MSFT'], start='2023-01-01', end='2024-01-01')
    """
    loader = DataLoader(data_dir)
    return loader.load_bars(symbols, start, end, timeframe)


def load_sample_data(
    symbols: Union[str, List[str]],
    days: int = 252,
    timeframe: str = '1D',
) -> pd.DataFrame:
    """Generate sample data for quick testing.

    Args:
        symbols: Symbol or list of symbols
        days: Number of days of data
        timeframe: Bar timeframe

    Returns:
        DataFrame with sample OHLCV data

    Example:
        >>> data = load_sample_data('AAPL', days=100)
        >>> data = load_sample_data(['AAPL', 'MSFT'], days=252)
    """
    loader = DataLoader()
    return loader.generate_sample_data(symbols, days, timeframe)


def generate_random_walk(
    symbols: Union[str, List[str]],
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: str = '1D',
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005,
) -> pd.DataFrame:
    """Generate random walk price data for testing.

    Args:
        symbols: Symbol or list of symbols
        start: Start date
        end: End date
        timeframe: Bar timeframe
        initial_price: Starting price
        volatility: Daily volatility (std dev of returns)
        drift: Daily drift (mean return)

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> data = generate_random_walk('TEST', '2023-01-01', '2024-01-01')
        >>> data = generate_random_walk(['A', 'B'], '2023-01-01', '2023-12-31', volatility=0.03)
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)

    # Generate timestamps based on timeframe
    freq_map = {
        '1D': 'B',   # Business days
        '1H': 'H',
        '1T': 'T',
        '1min': 'T',
        '5min': '5T',
        '15min': '15T',
        '30min': '30T',
    }
    freq = freq_map.get(timeframe, 'B')
    timestamps = pd.date_range(start, end, freq=freq)

    dfs = []
    np.random.seed(42)  # For reproducibility in demos

    for i, symbol in enumerate(symbols):
        n = len(timestamps)

        # Generate correlated random walks for multiple symbols
        correlation = 0.3 if len(symbols) > 1 else 0
        common_factor = np.random.normal(drift, volatility, n)
        specific_factor = np.random.normal(0, volatility * np.sqrt(1 - correlation**2), n)
        returns = correlation * common_factor + specific_factor

        # Price path
        close = initial_price * (1 + i * 0.1) * np.exp(np.cumsum(returns))

        # Generate realistic OHLC from close
        high = close * (1 + np.abs(np.random.normal(0, volatility/3, n)))
        low = close * (1 - np.abs(np.random.normal(0, volatility/3, n)))
        open_price = np.roll(close, 1)
        open_price[0] = close[0]

        # Ensure OHLC consistency
        high = np.maximum.reduce([open_price, close, high])
        low = np.minimum.reduce([open_price, close, low])

        # Volume with some autocorrelation
        base_volume = 1_000_000
        volume_changes = np.random.normal(0, 0.2, n)
        volume = base_volume * np.exp(np.cumsum(volume_changes) * 0.1)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume.astype(int),
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def add_features(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add common technical features to price data.

    Args:
        data: DataFrame with OHLCV data
        features: List of features to add. If None, adds all common features.
                 Options: 'returns', 'sma', 'ema', 'rsi', 'bbands', 'volume_ma'

    Returns:
        DataFrame with added feature columns

    Example:
        >>> data = load_sample_data('AAPL')
        >>> data = add_features(data, ['returns', 'sma', 'rsi'])
    """
    df = data.copy()

    if features is None:
        features = ['returns', 'sma', 'ema', 'rsi', 'bbands', 'volume_ma']

    # Group by symbol for multi-symbol data
    if 'symbol' in df.columns:
        df = df.groupby('symbol', group_keys=False).apply(lambda x: _add_features_single(x, features))
        df = df.reset_index(drop=True)
    else:
        df = _add_features_single(df, features)

    return df


def _add_features_single(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Add features to single symbol data."""
    close = df['close']

    if 'returns' in features:
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))

    if 'sma' in features:
        df['sma_10'] = close.rolling(10).mean()
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()

    if 'ema' in features:
        df['ema_10'] = close.ewm(span=10).mean()
        df['ema_20'] = close.ewm(span=20).mean()
        df['ema_50'] = close.ewm(span=50).mean()

    if 'rsi' in features:
        df['rsi_14'] = _calculate_rsi(close, 14)

    if 'bbands' in features:
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    if 'volume_ma' in features:
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    return df


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        close: Close prices
        period: RSI period

    Returns:
        RSI values
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
