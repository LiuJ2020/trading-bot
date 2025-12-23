"""SQLite database layer for market data metadata.

Manages symbol information, data availability, and quality metrics.
Coordinates with ParquetStore for actual timeseries data.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, date
import json
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MarketDatabase:
    """SQLite database for market data metadata.

    Manages:
        - Symbol universe and metadata
        - Data file tracking and availability
        - Data quality metrics
        - Corporate actions
        - Ingestion history

    Works with ParquetStore to provide complete data layer.
    """

    def __init__(self, db_path: str):
        """Initialize market database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()
        logger.info(f"Initialized MarketDatabase at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic commit/rollback.

        Yields:
            sqlite3.Connection
        """
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        with self._get_connection() as conn:
            conn.executescript(schema_sql)

        logger.info("Database schema initialized")

    def get_schema_version(self) -> int:
        """Get current schema version.

        Returns:
            Schema version number
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(version) as version FROM schema_version"
            )
            row = cursor.fetchone()
            return row['version'] if row['version'] else 0

    # ===== Symbol Management =====

    def add_symbol(
        self,
        symbol: str,
        name: Optional[str] = None,
        exchange: Optional[str] = None,
        asset_class: str = "stock",
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        market_cap: Optional[float] = None,
        first_traded_date: Optional[date] = None
    ) -> None:
        """Add or update symbol metadata.

        Args:
            symbol: Trading symbol
            name: Company/asset name
            exchange: Exchange (e.g., 'NASDAQ', 'NYSE')
            asset_class: Asset class ('stock', 'etf', 'crypto')
            sector: Sector
            industry: Industry
            market_cap: Market capitalization
            first_traded_date: First trading date
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO symbols (
                    symbol, name, exchange, asset_class, sector, industry,
                    market_cap, first_traded_date, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    name = excluded.name,
                    exchange = excluded.exchange,
                    asset_class = excluded.asset_class,
                    sector = excluded.sector,
                    industry = excluded.industry,
                    market_cap = excluded.market_cap,
                    first_traded_date = excluded.first_traded_date,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, name, exchange, asset_class, sector, industry,
                  market_cap, first_traded_date))

        logger.info(f"Added/updated symbol: {symbol}")

    def get_symbol(self, symbol: str) -> Optional[Dict]:
        """Get symbol metadata.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with symbol info or None
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM symbols WHERE symbol = ?",
                (symbol,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def list_symbols(
        self,
        exchange: Optional[str] = None,
        asset_class: Optional[str] = None,
        sector: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict]:
        """List symbols with optional filtering.

        Args:
            exchange: Filter by exchange
            asset_class: Filter by asset class
            sector: Filter by sector
            active_only: Only return active symbols

        Returns:
            List of symbol dicts
        """
        query = "SELECT * FROM symbols WHERE 1=1"
        params = []

        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        if asset_class:
            query += " AND asset_class = ?"
            params.append(asset_class)
        if sector:
            query += " AND sector = ?"
            params.append(sector)
        if active_only:
            query += " AND is_active = 1"

        query += " ORDER BY symbol"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def deactivate_symbol(self, symbol: str) -> None:
        """Mark symbol as inactive.

        Args:
            symbol: Trading symbol
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE symbols SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE symbol = ?",
                (symbol,)
            )

        logger.info(f"Deactivated symbol: {symbol}")

    # ===== Data File Tracking =====

    def register_data_file(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        num_bars: int,
        file_path: str,
        file_size_bytes: int,
        parquet_schema: Optional[str] = None
    ) -> int:
        """Register a data file in the database.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1D')
            start_date: First bar date
            end_date: Last bar date
            num_bars: Number of bars in file
            file_path: Path to parquet file
            file_size_bytes: File size in bytes
            parquet_schema: JSON representation of schema

        Returns:
            file_id of inserted/updated record
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO data_files (
                    symbol, timeframe, start_date, end_date, num_bars,
                    file_path, file_size_bytes, parquet_schema, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol, timeframe, start_date, end_date) DO UPDATE SET
                    num_bars = excluded.num_bars,
                    file_path = excluded.file_path,
                    file_size_bytes = excluded.file_size_bytes,
                    parquet_schema = excluded.parquet_schema,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, timeframe, start_date, end_date, num_bars,
                  file_path, file_size_bytes, parquet_schema))

            # Get the file_id
            cursor = conn.execute("""
                SELECT file_id FROM data_files
                WHERE symbol = ? AND timeframe = ? AND start_date = ? AND end_date = ?
            """, (symbol, timeframe, start_date, end_date))

            row = cursor.fetchone()
            file_id = row['file_id']

        logger.debug(f"Registered data file {file_id} for {symbol} ({timeframe})")
        return file_id

    def get_data_availability(
        self,
        symbol: str,
        timeframe: Optional[str] = None
    ) -> List[Dict]:
        """Get data availability for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Optional timeframe filter

        Returns:
            List of data file records
        """
        query = "SELECT * FROM data_files WHERE symbol = ?"
        params = [symbol]

        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        query += " ORDER BY start_date"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_date_range(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Tuple[date, date]]:
        """Get overall date range for symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Tuple of (earliest_date, latest_date) or None
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT MIN(start_date) as earliest, MAX(end_date) as latest
                FROM data_files
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))

            row = cursor.fetchone()
            if row['earliest'] and row['latest']:
                return (row['earliest'], row['latest'])
            return None

    # ===== Data Quality =====

    def record_quality_check(
        self,
        symbol: str,
        timeframe: str,
        check_date: date,
        total_bars: int,
        missing_bars: int = 0,
        duplicate_bars: int = 0,
        outlier_bars: int = 0,
        zero_volume_bars: int = 0,
        negative_prices: int = 0,
        invalid_ohlc: int = 0,
        notes: Optional[str] = None
    ) -> None:
        """Record data quality check results.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            check_date: Date of quality check
            total_bars: Total number of bars checked
            missing_bars: Number of missing bars
            duplicate_bars: Number of duplicate timestamps
            outlier_bars: Number of outlier bars
            zero_volume_bars: Number of bars with zero volume
            negative_prices: Number of bars with negative prices
            invalid_ohlc: Number of bars with invalid OHLC
            notes: Additional notes
        """
        # Calculate metrics
        completeness_pct = 100.0 if total_bars == 0 else (
            ((total_bars - missing_bars) / total_bars) * 100.0
        )

        # Quality score (0-100)
        # Deduct points for each type of issue
        quality_score = 100.0
        if total_bars > 0:
            quality_score -= (missing_bars / total_bars) * 30  # 30 points max for missing
            quality_score -= (duplicate_bars / total_bars) * 20  # 20 points max
            quality_score -= (outlier_bars / total_bars) * 20
            quality_score -= (invalid_ohlc / total_bars) * 30
        quality_score = max(0.0, quality_score)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO data_quality (
                    symbol, timeframe, check_date, total_bars,
                    missing_bars, duplicate_bars, outlier_bars,
                    zero_volume_bars, negative_prices, invalid_ohlc,
                    completeness_pct, quality_score, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timeframe, check_date, total_bars,
                  missing_bars, duplicate_bars, outlier_bars,
                  zero_volume_bars, negative_prices, invalid_ohlc,
                  completeness_pct, quality_score, notes))

        logger.info(
            f"Quality check for {symbol} ({timeframe}): "
            f"score={quality_score:.1f}, completeness={completeness_pct:.1f}%"
        )

    def get_quality_metrics(
        self,
        symbol: str,
        timeframe: str,
        latest_only: bool = True
    ) -> Optional[Dict]:
        """Get quality metrics for symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            latest_only: Return only latest check

        Returns:
            Quality metrics dict or None
        """
        if latest_only:
            query = """
                SELECT * FROM data_quality
                WHERE symbol = ? AND timeframe = ?
                ORDER BY check_date DESC
                LIMIT 1
            """
        else:
            query = """
                SELECT * FROM data_quality
                WHERE symbol = ? AND timeframe = ?
                ORDER BY check_date DESC
            """

        with self._get_connection() as conn:
            cursor = conn.execute(query, (symbol, timeframe))
            if latest_only:
                row = cursor.fetchone()
                return dict(row) if row else None
            else:
                return [dict(row) for row in cursor.fetchall()]

    # ===== Corporate Actions =====

    def add_corporate_action(
        self,
        symbol: str,
        action_type: str,
        ex_date: date,
        record_date: Optional[date] = None,
        payment_date: Optional[date] = None,
        split_ratio: Optional[float] = None,
        dividend_amount: Optional[float] = None,
        details: Optional[Dict] = None
    ) -> int:
        """Add corporate action.

        Args:
            symbol: Trading symbol
            action_type: Type ('split', 'dividend', 'merger', etc.)
            ex_date: Ex-dividend/ex-split date
            record_date: Record date
            payment_date: Payment date
            split_ratio: Split ratio (e.g., 2.0 for 2-for-1)
            dividend_amount: Dividend amount per share
            details: Additional details as dict

        Returns:
            action_id
        """
        details_json = json.dumps(details) if details else None

        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO corporate_actions (
                    symbol, action_type, ex_date, record_date, payment_date,
                    split_ratio, dividend_amount, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action_type, ex_date, record_date, payment_date,
                  split_ratio, dividend_amount, details_json))

            action_id = cursor.lastrowid

        logger.info(f"Added {action_type} for {symbol} on {ex_date}")
        return action_id

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        action_type: Optional[str] = None
    ) -> List[Dict]:
        """Get corporate actions for symbol.

        Args:
            symbol: Trading symbol
            start_date: Filter by ex_date >= start_date
            end_date: Filter by ex_date <= end_date
            action_type: Filter by action type

        Returns:
            List of corporate action dicts
        """
        query = "SELECT * FROM corporate_actions WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND ex_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND ex_date <= ?"
            params.append(end_date)
        if action_type:
            query += " AND action_type = ?"
            params.append(action_type)

        query += " ORDER BY ex_date DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                item = dict(row)
                # Parse JSON details
                if item['details']:
                    item['details'] = json.loads(item['details'])
                results.append(item)
            return results

    # ===== Ingestion Log =====

    def log_ingestion(
        self,
        symbol: str,
        timeframe: str,
        source: str,
        start_date: date,
        end_date: date,
        bars_ingested: int,
        status: str,
        error_message: Optional[str] = None,
        ingestion_time_seconds: Optional[float] = None
    ) -> int:
        """Log data ingestion event.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            source: Data source ('yahoo', 'alpaca', etc.)
            start_date: Start date of ingested data
            end_date: End date of ingested data
            bars_ingested: Number of bars ingested
            status: Status ('success', 'partial', 'failed')
            error_message: Error message if failed
            ingestion_time_seconds: Time taken to ingest

        Returns:
            log_id
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO ingestion_log (
                    symbol, timeframe, source, start_date, end_date,
                    bars_ingested, status, error_message, ingestion_time_seconds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timeframe, source, start_date, end_date,
                  bars_ingested, status, error_message, ingestion_time_seconds))

            log_id = cursor.lastrowid

        logger.info(
            f"Logged ingestion for {symbol} ({timeframe}): "
            f"{bars_ingested} bars, status={status}"
        )
        return log_id

    def get_ingestion_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get ingestion history.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of records

        Returns:
            List of ingestion log dicts
        """
        if symbol:
            query = """
                SELECT * FROM ingestion_log
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (symbol, limit)
        else:
            query = """
                SELECT * FROM ingestion_log
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (limit,)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ===== Summary Views =====

    def get_data_summary(self) -> List[Dict]:
        """Get summary of all data availability.

        Returns:
            List of summary dicts from v_data_availability view
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM v_data_availability
                ORDER BY symbol, timeframe
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_quality_summary(self) -> List[Dict]:
        """Get summary of latest quality checks.

        Returns:
            List of quality dicts from v_latest_quality view
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM v_latest_quality
                ORDER BY quality_score ASC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def __repr__(self) -> str:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM symbols")
            num_symbols = cursor.fetchone()['count']

        return f"MarketDatabase(db_path={self.db_path}, symbols={num_symbols})"
