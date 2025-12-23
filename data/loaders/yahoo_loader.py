"""Yahoo Finance data loader.

Fetches historical OHLCV data from Yahoo Finance using yfinance library.
Handles rate limiting, error handling, and data validation.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class YahooFinanceLoader:
    """Load historical market data from Yahoo Finance.

    Features:
    - Batch symbol fetching with concurrency control
    - Automatic retry on failures
    - Rate limiting to avoid API throttling
    - Data validation and cleaning
    - Corporate action adjustment
    """

    def __init__(
        self,
        max_workers: int = 5,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.5
    ):
        """Initialize Yahoo Finance loader.

        Args:
            max_workers: Maximum concurrent downloads
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries (seconds)
            rate_limit_delay: Delay between requests (seconds)
        """
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay

    def fetch_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start: Start date
            end: End date
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, symbol
            Returns None if fetch fails
        """
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Fetching {symbol} from {start.date()} to {end.date()} (attempt {attempt + 1})")

                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,  # Adjust for splits and dividends
                    actions=False
                )

                # Check if data was returned
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None

                # Reset index to make timestamp a column
                df = df.reset_index()

                # Rename columns to lowercase
                df.columns = df.columns.str.lower()

                # Standardize column names (index might be called Date, Datetime, or datetime)
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'timestamp'})
                elif 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'timestamp'})

                # Check if timestamp column exists
                if 'timestamp' not in df.columns:
                    # If index name was capitalized or different, take first column as timestamp
                    first_col = df.columns[0]
                    logger.debug(f"Renaming first column '{first_col}' to 'timestamp'")
                    df = df.rename(columns={first_col: 'timestamp'})

                # Add symbol column
                df['symbol'] = symbol

                # Select and order columns
                df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

                # Convert timestamp to datetime and remove timezone info
                # (We store as UTC timestamps without timezone to avoid comparison issues)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

                # Basic validation
                if not self._validate_data(df, symbol):
                    logger.error(f"Data validation failed for {symbol}")
                    return None

                logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
                time.sleep(self.rate_limit_delay)  # Rate limiting
                return df

            except Exception as e:
                logger.warning(f"Error fetching {symbol} (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {symbol} after {self.retry_attempts} attempts")
                    return None

    def fetch_symbols(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date
            interval: Data interval

        Returns:
            Dict mapping symbol -> DataFrame (only successful fetches)
        """
        results = {}
        failed_symbols = []

        logger.info(f"Fetching {len(symbols)} symbols with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_symbol, symbol, start, end, interval): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[symbol] = df
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Exception fetching {symbol}: {e}")
                    failed_symbols.append(symbol)

        logger.info(f"Successfully fetched {len(results)}/{len(symbols)} symbols")
        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")

        return results

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get metadata for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with symbol metadata (name, exchange, sector, etc.)
            Returns None if fetch fails
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName') or info.get('shortName'),
                'exchange': info.get('exchange'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'asset_class': self._determine_asset_class(info)
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return None

    def _determine_asset_class(self, info: Dict) -> str:
        """Determine asset class from ticker info.

        Args:
            info: Ticker info dict

        Returns:
            Asset class string ('stock', 'etf', 'crypto', etc.)
        """
        quote_type = info.get('quoteType', '').lower()

        if quote_type == 'etf':
            return 'etf'
        elif quote_type == 'cryptocurrency':
            return 'crypto'
        elif quote_type in ['equity', 'mutualfund']:
            return 'stock'
        else:
            return 'unknown'

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate fetched data.

        Args:
            df: DataFrame to validate
            symbol: Symbol name (for logging)

        Returns:
            True if data is valid
        """
        if df.empty:
            logger.error(f"{symbol}: DataFrame is empty")
            return False

        # Check required columns
        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"{symbol}: Missing columns: {missing_cols}")
            return False

        # Check for null values in OHLCV
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        null_counts = df[ohlcv_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"{symbol}: Found null values: {null_counts[null_counts > 0].to_dict()}")

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        negative_prices = (df[price_cols] < 0).any()
        if negative_prices.any():
            logger.error(f"{symbol}: Found negative prices in columns: {negative_prices[negative_prices].index.tolist()}")
            return False

        # Check OHLC validity (high >= low, etc.)
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        )
        if invalid_ohlc.any():
            num_invalid = invalid_ohlc.sum()
            logger.warning(f"{symbol}: Found {num_invalid} bars with invalid OHLC relationships")

        return True

    def get_latest_available_date(self, symbol: str) -> Optional[datetime]:
        """Get the most recent date with available data.

        Args:
            symbol: Stock symbol

        Returns:
            Latest available date or None
        """
        try:
            # Fetch just the last few days
            end = datetime.now()
            start = end - timedelta(days=7)

            df = self.fetch_symbol(symbol, start, end)
            if df is not None and not df.empty:
                return df['timestamp'].max()
            return None
        except Exception as e:
            logger.error(f"Error getting latest date for {symbol}: {e}")
            return None


def interval_to_timeframe(interval: str) -> str:
    """Convert yfinance interval to timeframe string.

    Args:
        interval: yfinance interval ('1d', '1h', '5m', etc.)

    Returns:
        Timeframe string ('1D', '1H', '5T', etc.)
    """
    mapping = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '1d': '1D',
        '1wk': '1W',
        '1mo': '1M'
    }
    return mapping.get(interval, interval.upper())


def timeframe_to_interval(timeframe: str) -> str:
    """Convert timeframe string to yfinance interval.

    Args:
        timeframe: Timeframe string ('1D', '1H', '5T', etc.)

    Returns:
        yfinance interval ('1d', '1h', '5m', etc.)
    """
    mapping = {
        '1T': '1m',
        '5T': '5m',
        '15T': '15m',
        '30T': '30m',
        '1H': '1h',
        '1D': '1d',
        '1W': '1wk',
        '1M': '1mo'
    }
    return mapping.get(timeframe, timeframe.lower())
