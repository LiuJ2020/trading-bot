"""Unified market data store combining SQLite and Parquet.

This is the main interface for reading/writing market data.
Coordinates metadata (SQLite) with timeseries data (Parquet).
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, date
import pandas as pd
import logging
import time

from .database import MarketDatabase
from .parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class MarketDataStore:
    """Unified market data storage layer.

    Combines:
        - SQLite for metadata (symbols, availability, quality)
        - Parquet for timeseries data (OHLCV bars)

    Features:
        - Single interface: get_bars(symbol, start, end)
        - Automatic metadata tracking
        - Data quality validation
        - Efficient indexed queries
        - Schema versioning

    Storage structure:
    <base_path>/
        market_data.db           # SQLite database (metadata)
        parquet/
            AAPL/
                1D/data.parquet
                1H/data.parquet
            MSFT/
                1D/data.parquet
    """

    def __init__(
        self,
        base_path: str,
        compression: str = "snappy"
    ):
        """Initialize market data store.

        Args:
            base_path: Base directory for all data storage
            compression: Parquet compression codec
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / "market_data.db"
        parquet_path = base_path / "parquet"

        self.db = MarketDatabase(str(db_path))
        self.parquet = ParquetStore(str(parquet_path), compression=compression)

        logger.info(
            f"Initialized MarketDataStore: db={db_path}, parquet={parquet_path}"
        )

    # ===== Main Data Access Interface =====

    def get_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1D"
    ) -> pd.DataFrame:
        """Get OHLCV bars for a symbol.

        This is the primary data access method. Returns data from Parquet
        storage with automatic filtering.

        Args:
            symbol: Trading symbol
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            timeframe: Timeframe (e.g., '1D', '1H')

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Check if we have data
        availability = self.db.get_data_availability(symbol, timeframe)
        if not availability:
            logger.warning(f"No data available for {symbol} ({timeframe})")
            return pd.DataFrame()

        # Read from Parquet
        df = self.parquet.read_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end
        )

        return df

    def write_bars(
        self,
        symbol: str,
        data: pd.DataFrame,
        timeframe: str = "1D",
        source: str = "unknown",
        mode: str = "update"
    ) -> Dict:
        """Write OHLCV bars for a symbol.

        Writes to Parquet and updates SQLite metadata automatically.

        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
            timeframe: Timeframe
            source: Data source (for logging)
            mode: Write mode ('overwrite', 'append', 'update')

        Returns:
            Dict with operation results
        """
        start_time = time.time()

        # Ensure symbol exists in database
        if not self.db.get_symbol(symbol):
            self.db.add_symbol(symbol)

        # Write to Parquet
        try:
            parquet_meta = self.parquet.write_bars(
                symbol=symbol,
                timeframe=timeframe,
                data=data,
                mode=mode
            )

            # Register data file in database
            file_id = self.db.register_data_file(
                symbol=symbol,
                timeframe=timeframe,
                start_date=parquet_meta['start_date'].date(),
                end_date=parquet_meta['end_date'].date(),
                num_bars=parquet_meta['num_bars'],
                file_path=parquet_meta['file_path'],
                file_size_bytes=parquet_meta['file_size_bytes'],
                parquet_schema=parquet_meta['schema']
            )

            # Run quality check
            quality_issues = self._check_data_quality(data)

            # Record quality metrics
            self.db.record_quality_check(
                symbol=symbol,
                timeframe=timeframe,
                check_date=date.today(),
                total_bars=len(data),
                missing_bars=quality_issues.get('missing_bars', 0),
                duplicate_bars=quality_issues.get('duplicate_bars', 0),
                outlier_bars=quality_issues.get('outlier_bars', 0),
                zero_volume_bars=quality_issues.get('zero_volume_bars', 0),
                negative_prices=quality_issues.get('negative_prices', 0),
                invalid_ohlc=quality_issues.get('invalid_ohlc', 0)
            )

            # Log ingestion
            elapsed = time.time() - start_time
            self.db.log_ingestion(
                symbol=symbol,
                timeframe=timeframe,
                source=source,
                start_date=parquet_meta['start_date'].date(),
                end_date=parquet_meta['end_date'].date(),
                bars_ingested=len(data),
                status='success',
                ingestion_time_seconds=elapsed
            )

            result = {
                'status': 'success',
                'symbol': symbol,
                'timeframe': timeframe,
                'bars_written': len(data),
                'file_id': file_id,
                'quality_issues': quality_issues,
                'elapsed_seconds': elapsed
            }

            logger.info(
                f"Wrote {len(data)} bars for {symbol} ({timeframe}) "
                f"in {elapsed:.2f}s"
            )

            return result

        except Exception as e:
            # Log failure
            elapsed = time.time() - start_time
            self.db.log_ingestion(
                symbol=symbol,
                timeframe=timeframe,
                source=source,
                start_date=data['timestamp'].min().date(),
                end_date=data['timestamp'].max().date(),
                bars_ingested=0,
                status='failed',
                error_message=str(e),
                ingestion_time_seconds=elapsed
            )

            logger.error(f"Failed to write data for {symbol}: {e}")
            raise

    def delete_bars(
        self,
        symbol: str,
        timeframe: str
    ) -> bool:
        """Delete bars for symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            True if deleted, False if not found
        """
        # Delete from Parquet
        deleted = self.parquet.delete_bars(symbol, timeframe)

        if deleted:
            # Note: SQLite records remain for historical tracking
            logger.info(f"Deleted bars for {symbol} ({timeframe})")

        return deleted

    # ===== Symbol Management =====

    def add_symbol(
        self,
        symbol: str,
        **kwargs
    ) -> None:
        """Add symbol to database.

        Args:
            symbol: Trading symbol
            **kwargs: Additional symbol metadata (name, exchange, etc.)
        """
        self.db.add_symbol(symbol, **kwargs)

    def get_symbol(self, symbol: str) -> Optional[Dict]:
        """Get symbol metadata.

        Args:
            symbol: Trading symbol

        Returns:
            Symbol metadata dict or None
        """
        return self.db.get_symbol(symbol)

    def list_symbols(self, **kwargs) -> List[str]:
        """List available symbols.

        Args:
            **kwargs: Filter arguments (exchange, asset_class, etc.)

        Returns:
            List of symbol strings
        """
        symbols_data = self.db.list_symbols(**kwargs)
        return [s['symbol'] for s in symbols_data]

    # ===== Data Availability =====

    def get_date_range(
        self,
        symbol: str,
        timeframe: str = "1D"
    ) -> Optional[Tuple[datetime, datetime]]:
        """Get available date range for symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Tuple of (start_date, end_date) or None
        """
        # Try Parquet first (most accurate)
        date_range = self.parquet.get_date_range(symbol, timeframe)
        if date_range:
            return date_range

        # Fallback to database metadata
        db_range = self.db.get_date_range(symbol, timeframe)
        if db_range:
            # Convert dates to datetimes
            return (
                datetime.combine(db_range[0], datetime.min.time()),
                datetime.combine(db_range[1], datetime.max.time())
            )

        return None

    def get_bar_count(
        self,
        symbol: str,
        timeframe: str = "1D"
    ) -> int:
        """Get number of bars for symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Number of bars
        """
        return self.parquet.get_bar_count(symbol, timeframe)

    def has_data(
        self,
        symbol: str,
        timeframe: str = "1D"
    ) -> bool:
        """Check if data exists for symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            True if data exists
        """
        return self.get_bar_count(symbol, timeframe) > 0

    # ===== Quality & Monitoring =====

    def get_quality_metrics(
        self,
        symbol: str,
        timeframe: str = "1D"
    ) -> Optional[Dict]:
        """Get latest quality metrics for symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Quality metrics dict or None
        """
        return self.db.get_quality_metrics(symbol, timeframe)

    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of all available data.

        Returns:
            DataFrame with data availability summary
        """
        summary = self.db.get_data_summary()
        return pd.DataFrame(summary)

    def get_quality_summary(self) -> pd.DataFrame:
        """Get summary of data quality across all symbols.

        Returns:
            DataFrame with quality metrics
        """
        summary = self.db.get_quality_summary()
        return pd.DataFrame(summary)

    # ===== Corporate Actions =====

    def add_corporate_action(
        self,
        symbol: str,
        action_type: str,
        ex_date: date,
        **kwargs
    ) -> int:
        """Add corporate action.

        Args:
            symbol: Trading symbol
            action_type: Type ('split', 'dividend', etc.)
            ex_date: Ex-date
            **kwargs: Additional action details

        Returns:
            action_id
        """
        return self.db.add_corporate_action(
            symbol=symbol,
            action_type=action_type,
            ex_date=ex_date,
            **kwargs
        )

    def get_corporate_actions(
        self,
        symbol: str,
        **kwargs
    ) -> List[Dict]:
        """Get corporate actions for symbol.

        Args:
            symbol: Trading symbol
            **kwargs: Filter arguments

        Returns:
            List of corporate action dicts
        """
        return self.db.get_corporate_actions(symbol, **kwargs)

    # ===== Utilities =====

    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Check data quality and return issues.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with quality issue counts
        """
        issues = {}

        # Check for duplicates
        duplicates = df['timestamp'].duplicated().sum()
        issues['duplicate_bars'] = int(duplicates)

        # Check for zero volume
        zero_volume = (df['volume'] == 0).sum()
        issues['zero_volume_bars'] = int(zero_volume)

        # Check for negative prices
        negative = (
            (df['open'] < 0) |
            (df['high'] < 0) |
            (df['low'] < 0) |
            (df['close'] < 0)
        ).sum()
        issues['negative_prices'] = int(negative)

        # Check for invalid OHLC
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        ).sum()
        issues['invalid_ohlc'] = int(invalid_ohlc)

        # Check for outliers (simple z-score on returns)
        returns = df['close'].pct_change()
        z_scores = (returns - returns.mean()) / returns.std()
        outliers = (z_scores.abs() > 5).sum()  # 5 sigma threshold
        issues['outlier_bars'] = int(outliers)

        # Missing bars (assumes daily data, adjust for other timeframes)
        # This is a simplified check
        issues['missing_bars'] = 0

        return issues

    def bulk_write(
        self,
        data_dict: Dict[str, pd.DataFrame],
        timeframe: str = "1D",
        source: str = "bulk_import"
    ) -> Dict:
        """Write data for multiple symbols.

        Args:
            data_dict: Dict of symbol -> DataFrame
            timeframe: Timeframe
            source: Data source

        Returns:
            Dict with results for each symbol
        """
        results = {}

        for symbol, data in data_dict.items():
            try:
                result = self.write_bars(
                    symbol=symbol,
                    data=data,
                    timeframe=timeframe,
                    source=source
                )
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to write {symbol}: {e}")
                results[symbol] = {'status': 'failed', 'error': str(e)}

        return results

    def get_schema_version(self) -> int:
        """Get database schema version.

        Returns:
            Schema version number
        """
        return self.db.get_schema_version()

    def get_storage_stats(self) -> Dict:
        """Get storage statistics.

        Returns:
            Dict with storage stats
        """
        symbols = self.parquet.list_symbols()
        total_size = 0
        total_bars = 0

        for symbol in symbols:
            timeframes = self.parquet.list_timeframes(symbol)
            for tf in timeframes:
                info = self.parquet.get_file_info(symbol, tf)
                if info:
                    total_size += info['file_size_bytes']
                    total_bars += info['num_rows']

        return {
            'num_symbols': len(symbols),
            'total_bars': total_bars,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_bars_per_symbol': total_bars / len(symbols) if symbols else 0
        }

    def __repr__(self) -> str:
        stats = self.get_storage_stats()
        return (
            f"MarketDataStore(symbols={stats['num_symbols']}, "
            f"bars={stats['total_bars']:,}, "
            f"size={stats['total_size_mb']:.1f}MB)"
        )
