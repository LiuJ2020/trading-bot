"""Data ingestion pipeline orchestration.

Coordinates the full data ingestion workflow:
1. Fetch data from Yahoo Finance
2. Validate data quality
3. Store in database (SQLite + Parquet)
4. Handle incremental updates
5. Log ingestion metrics
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

from data.loaders.yahoo_loader import YahooFinanceLoader, interval_to_timeframe, timeframe_to_interval
from data.storage.market_data_store import MarketDataStore
from data.pipeline.validation import DataValidator, validate_and_clean

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrate market data ingestion from source to storage.

    Features:
    - Batch symbol ingestion with error handling
    - Incremental updates (fetch only new data)
    - Data validation and quality checks
    - Automatic retry on failures
    - Comprehensive logging
    """

    def __init__(
        self,
        data_store: MarketDataStore,
        loader: Optional[YahooFinanceLoader] = None,
        validator: Optional[DataValidator] = None
    ):
        """Initialize ingestion pipeline.

        Args:
            data_store: Market data storage backend
            loader: Data loader (defaults to YahooFinanceLoader)
            validator: Data validator (defaults to DataValidator)
        """
        self.data_store = data_store
        self.loader = loader or YahooFinanceLoader()
        self.validator = validator or DataValidator()

    def ingest_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        update_metadata: bool = True
    ) -> bool:
        """Ingest data for a single symbol.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            interval: Data interval ('1d', '1h', etc.)
            update_metadata: Whether to update symbol metadata

        Returns:
            True if successful
        """
        logger.info(f"Ingesting {symbol} from {start.date()} to {end.date()} ({interval})")

        try:
            # 1. Fetch data
            df = self.loader.fetch_symbol(symbol, start, end, interval)
            if df is None or df.empty:
                logger.warning(f"No data fetched for {symbol}")
                return False

            # 2. Validate and clean data
            df_clean, validation_results = validate_and_clean(df, symbol)

            if not validation_results['valid']:
                logger.error(f"Data validation failed for {symbol}: {validation_results['issues']}")
                return False

            # 3. Update symbol metadata if requested
            if update_metadata:
                symbol_info = self.loader.get_symbol_info(symbol)
                if symbol_info:
                    self.data_store.add_symbol(
                        symbol=symbol,
                        name=symbol_info.get('name'),
                        exchange=symbol_info.get('exchange'),
                        asset_class=symbol_info.get('asset_class'),
                        sector=symbol_info.get('sector'),
                        industry=symbol_info.get('industry'),
                        market_cap=symbol_info.get('market_cap')
                    )

            # 4. Store data (uses the new unified MarketDataStore interface)
            timeframe = interval_to_timeframe(interval)
            result = self.data_store.write_bars(
                symbol=symbol,
                data=df_clean,
                timeframe=timeframe,
                source='yahoo',
                mode='update'  # Merge with existing data
            )

            if result['status'] != 'success':
                logger.error(f"Failed to store data for {symbol}")
                return False

            # Quality score is included in result
            quality_issues = result.get('quality_issues', {})
            logger.info(f"Successfully ingested {symbol}: {len(df_clean)} bars, quality score {validation_results['quality_score']:.1f}")
            return True

        except Exception as e:
            logger.error(f"Error ingesting {symbol}: {e}", exc_info=True)
            return False

    def ingest_symbols(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
        continue_on_error: bool = True
    ) -> Dict[str, bool]:
        """Ingest data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date
            interval: Data interval
            continue_on_error: Continue if one symbol fails

        Returns:
            Dict mapping symbol -> success status
        """
        logger.info(f"Starting batch ingestion of {len(symbols)} symbols")

        results = {}
        successful = 0
        failed = 0

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {i}/{len(symbols)}: {symbol}")

            try:
                success = self.ingest_symbol(symbol, start, end, interval)
                results[symbol] = success

                if success:
                    successful += 1
                else:
                    failed += 1

                    if not continue_on_error:
                        logger.error(f"Stopping batch ingestion due to failure on {symbol}")
                        break

            except Exception as e:
                logger.error(f"Exception processing {symbol}: {e}")
                results[symbol] = False
                failed += 1

                if not continue_on_error:
                    break

        logger.info(f"Batch ingestion complete: {successful} successful, {failed} failed")
        return results

    def ingest_incremental(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: int = 7
    ) -> bool:
        """Perform incremental update for a symbol.

        Fetches only new data since last update.

        Args:
            symbol: Stock symbol
            interval: Data interval
            lookback_days: Days to look back from last available date

        Returns:
            True if successful
        """
        logger.info(f"Incremental update for {symbol} ({interval})")

        try:
            # Get last available data date
            timeframe = interval_to_timeframe(interval)
            date_range = self.data_store.get_date_range(symbol, timeframe)

            if date_range:
                # Start from last available date minus lookback (to handle missing data)
                end_date = date_range[1]
                start = end_date - timedelta(days=lookback_days)
                logger.info(f"{symbol}: Last data on {end_date.date()}, fetching from {start.date()}")
            else:
                # No existing data, fetch last year
                start = datetime.now() - timedelta(days=365)
                logger.info(f"{symbol}: No existing data, fetching from {start.date()}")

            end = datetime.now()
            interval_str = timeframe_to_interval(timeframe)

            return self.ingest_symbol(symbol, start, end, interval_str)

        except Exception as e:
            logger.error(f"Error in incremental update for {symbol}: {e}")
            return False

    def ingest_incremental_batch(
        self,
        symbols: List[str],
        interval: str = "1d",
        lookback_days: int = 7
    ) -> Dict[str, bool]:
        """Perform incremental updates for multiple symbols.

        Args:
            symbols: List of stock symbols
            interval: Data interval
            lookback_days: Days to look back from last available date

        Returns:
            Dict mapping symbol -> success status
        """
        logger.info(f"Starting incremental batch update of {len(symbols)} symbols")

        results = {}
        successful = 0
        failed = 0

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {i}/{len(symbols)}: {symbol}")

            try:
                success = self.ingest_incremental(symbol, interval, lookback_days)
                results[symbol] = success

                if success:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Exception in incremental update for {symbol}: {e}")
                results[symbol] = False
                failed += 1

        logger.info(f"Incremental batch update complete: {successful} successful, {failed} failed")
        return results

    def backfill_symbol(
        self,
        symbol: str,
        years: int = 5,
        interval: str = "1d"
    ) -> bool:
        """Backfill historical data for a symbol.

        Args:
            symbol: Stock symbol
            years: Number of years to backfill
            interval: Data interval

        Returns:
            True if successful
        """
        end = datetime.now()
        start = end - timedelta(days=years * 365)

        logger.info(f"Backfilling {symbol} for {years} years")
        return self.ingest_symbol(symbol, start, end, interval)

    def get_ingestion_summary(self, days: int = 7) -> pd.DataFrame:
        """Get summary of recent ingestion activity.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with ingestion statistics
        """
        history = self.data_store.db.get_ingestion_history(limit=100)
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)

        # Filter by date
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['created_at'] >= cutoff]

        return df

    def validate_existing_data(
        self,
        symbol: str,
        timeframe: str = "1D"
    ) -> Optional[Dict]:
        """Validate existing data in storage.

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe

        Returns:
            Validation results dict
        """
        logger.info(f"Validating existing data for {symbol} {timeframe}")

        try:
            # Load data from storage
            df = self.data_store.get_bars(symbol, timeframe=timeframe)

            if df is None or df.empty:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return None

            # Validate
            validation_results = self.validator.validate(df, symbol, timeframe)

            return validation_results

        except Exception as e:
            logger.error(f"Error validating data for {symbol} {timeframe}: {e}")
            return None
