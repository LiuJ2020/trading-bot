"""Parquet storage backend for OHLCV timeseries data.

Provides efficient columnar storage and retrieval of market data.
Data is organized by symbol with automatic partitioning.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)


class ParquetStore:
    """Manages Parquet files for OHLCV data storage.

    Organization:
        {base_path}/{symbol}/{timeframe}/data.parquet

    Features:
        - Efficient columnar storage with compression
        - Indexed by timestamp for fast range queries
        - Automatic schema validation
        - Incremental append support
        - Metadata tracking
    """

    # Standard OHLCV schema
    OHLCV_SCHEMA = pa.schema([
        ('timestamp', pa.timestamp('ns')),
        ('open', pa.float64()),
        ('high', pa.float64()),
        ('low', pa.float64()),
        ('close', pa.float64()),
        ('volume', pa.float64()),
        ('vwap', pa.float64(), True),  # Optional
        ('trades', pa.int64(), True),  # Optional
    ])

    def __init__(
        self,
        base_path: str,
        compression: str = "snappy",
        create_dirs: bool = True
    ):
        """Initialize Parquet store.

        Args:
            base_path: Root directory for parquet files
            compression: Compression codec ('snappy', 'gzip', 'brotli', 'zstd')
            create_dirs: Create directories if they don't exist
        """
        self.base_path = Path(base_path)
        self.compression = compression

        if create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ParquetStore at {self.base_path}")

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        """Get file path for symbol/timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1D', '1H')

        Returns:
            Path to parquet file
        """
        return self.base_path / symbol / timeframe / "data.parquet"

    def write_bars(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        mode: str = "overwrite"
    ) -> Dict[str, any]:
        """Write OHLCV bars to parquet file.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1D', '1H')
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            mode: Write mode ('overwrite', 'append', 'update')

        Returns:
            Dict with metadata: {
                'file_path': str,
                'num_bars': int,
                'start_date': datetime,
                'end_date': datetime,
                'file_size_bytes': int
            }
        """
        # Validate input data
        self._validate_data(data)

        # Prepare data
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            missing = set(required) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        file_path = self._get_file_path(symbol, timeframe)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle different write modes
        if mode == "append" and file_path.exists():
            # Read existing data and combine
            existing = self.read_bars(symbol, timeframe)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif mode == "update" and file_path.exists():
            # Read existing, update overlapping, append new
            existing = self.read_bars(symbol, timeframe)
            # Remove duplicates, keeping new data
            combined = pd.concat([existing, df], ignore_index=True)
            df = combined.drop_duplicates(subset=['timestamp'], keep='last')
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Write to parquet
        table = pa.Table.from_pandas(df, schema=self._get_schema(df))
        pq.write_table(
            table,
            file_path,
            compression=self.compression,
            use_dictionary=True,
            write_statistics=True,
        )

        # Collect metadata
        file_size = file_path.stat().st_size
        metadata = {
            'file_path': str(file_path),
            'num_bars': len(df),
            'start_date': df['timestamp'].min(),
            'end_date': df['timestamp'].max(),
            'file_size_bytes': file_size,
            'schema': str(table.schema)
        }

        logger.info(
            f"Wrote {len(df)} bars for {symbol} ({timeframe}) "
            f"to {file_path} ({file_size / 1024:.2f} KB)"
        )

        return metadata

    def read_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Read OHLCV bars from parquet file.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1D', '1H')
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            columns: Specific columns to read (None = all)

        Returns:
            DataFrame with OHLCV data
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            logger.warning(f"No data file found for {symbol} ({timeframe})")
            return pd.DataFrame()

        # Build filters for efficient reading
        filters = []
        if start:
            filters.append(('timestamp', '>=', pd.Timestamp(start)))
        if end:
            filters.append(('timestamp', '<=', pd.Timestamp(end)))

        # Read parquet with filters
        try:
            if filters:
                # Use PyArrow for filtering
                table = pq.read_table(
                    file_path,
                    columns=columns,
                    filters=filters if filters else None
                )
                df = table.to_pandas()
            else:
                df = pd.read_parquet(file_path, columns=columns)

            logger.debug(
                f"Read {len(df)} bars for {symbol} ({timeframe}) "
                f"from {file_path}"
            )

            return df

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise

    def get_date_range(self, symbol: str, timeframe: str) -> Optional[tuple]:
        """Get date range for symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Tuple of (start_date, end_date) or None if no data
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            return None

        # Read metadata without loading full dataset
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata

        # Read just timestamp column
        df = pd.read_parquet(file_path, columns=['timestamp'])
        return (df['timestamp'].min(), df['timestamp'].max())

    def get_bar_count(self, symbol: str, timeframe: str) -> int:
        """Get number of bars for symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Number of bars
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            return 0

        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows

    def delete_bars(self, symbol: str, timeframe: str) -> bool:
        """Delete bars for symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            return False

        file_path.unlink()
        logger.info(f"Deleted data file: {file_path}")

        # Clean up empty directories
        try:
            file_path.parent.rmdir()  # Remove timeframe dir if empty
            file_path.parent.parent.rmdir()  # Remove symbol dir if empty
        except OSError:
            pass  # Directories not empty

        return True

    def list_symbols(self) -> List[str]:
        """List all symbols with data.

        Returns:
            List of symbols
        """
        if not self.base_path.exists():
            return []

        symbols = []
        for symbol_dir in self.base_path.iterdir():
            if symbol_dir.is_dir():
                symbols.append(symbol_dir.name)

        return sorted(symbols)

    def list_timeframes(self, symbol: str) -> List[str]:
        """List all timeframes for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of timeframes
        """
        symbol_dir = self.base_path / symbol

        if not symbol_dir.exists():
            return []

        timeframes = []
        for tf_dir in symbol_dir.iterdir():
            if tf_dir.is_dir() and (tf_dir / "data.parquet").exists():
                timeframes.append(tf_dir.name)

        return sorted(timeframes)

    def get_file_info(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get detailed file information.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with file info or None
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            return None

        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata

        return {
            'file_path': str(file_path),
            'file_size_bytes': file_path.stat().st_size,
            'num_rows': metadata.num_rows,
            'num_columns': metadata.num_columns,
            'num_row_groups': metadata.num_row_groups,
            'serialized_size': metadata.serialized_size,
            'schema': str(parquet_file.schema_arrow),
            'compression': self.compression,
        }

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate OHLCV data quality.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Cannot write empty DataFrame")

        # Check for required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for nulls in required columns
        null_counts = df[required].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        )
        if invalid_ohlc.any():
            num_invalid = invalid_ohlc.sum()
            raise ValueError(f"Found {num_invalid} bars with invalid OHLC relationships")

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        negative = (df[price_cols] < 0).any()
        if negative.any():
            raise ValueError(f"Negative prices found in columns: {negative[negative].index.tolist()}")

    def _get_schema(self, df: pd.DataFrame) -> pa.Schema:
        """Get PyArrow schema for DataFrame.

        Args:
            df: DataFrame

        Returns:
            PyArrow schema
        """
        # Build schema from DataFrame columns
        fields = [
            pa.field('timestamp', pa.timestamp('ns')),
            pa.field('open', pa.float64()),
            pa.field('high', pa.float64()),
            pa.field('low', pa.float64()),
            pa.field('close', pa.float64()),
            pa.field('volume', pa.float64()),
        ]

        # Add optional columns if present
        if 'vwap' in df.columns:
            fields.append(pa.field('vwap', pa.float64()))
        if 'trades' in df.columns:
            fields.append(pa.field('trades', pa.int64()))

        return pa.schema(fields)

    def __repr__(self) -> str:
        num_symbols = len(self.list_symbols())
        return (f"ParquetStore(base_path={self.base_path}, "
                f"compression={self.compression}, "
                f"symbols={num_symbols})")
