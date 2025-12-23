#!/usr/bin/env python3
"""Data ingestion CLI script.

Fetch historical OHLCV data from Yahoo Finance and store in database.

Usage:
    # Ingest specific symbols for date range
    python scripts/ingest_data.py --symbols AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-01-01

    # Incremental update (fetch only new data)
    python scripts/ingest_data.py --symbols AAPL,MSFT --incremental

    # Backfill historical data
    python scripts/ingest_data.py --symbols AAPL --backfill --years 5

    # Ingest from file
    python scripts/ingest_data.py --symbol-file symbols.txt --start 2023-01-01 --end 2024-01-01

    # Validate existing data
    python scripts/ingest_data.py --symbols AAPL --validate-only
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders.yahoo_loader import YahooFinanceLoader
from data.storage.market_data_store import MarketDataStore
from data.pipeline.ingestion import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ingest market data from Yahoo Finance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)'
    )
    symbol_group.add_argument(
        '--symbol-file',
        type=str,
        help='File containing symbols (one per line)'
    )

    # Date range
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD, defaults to today)'
    )

    # Interval
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        choices=['1d', '1h', '5m', '15m', '30m', '1wk', '1mo'],
        help='Data interval (default: 1d)'
    )

    # Operation mode
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Incremental update (fetch only new data since last update)'
    )
    parser.add_argument(
        '--backfill',
        action='store_true',
        help='Backfill historical data'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Years to backfill (used with --backfill, default: 5)'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=7,
        help='Days to look back for incremental updates (default: 7)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing data, do not fetch new data'
    )

    # Storage
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data_storage',
        help='Data storage directory (default: ./data_storage)'
    )

    # Concurrency
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum concurrent downloads (default: 5)'
    )

    # Error handling
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop on first error (default: continue)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal logging'
    )

    return parser.parse_args()


def load_symbols_from_file(filepath: str) -> List[str]:
    """Load symbols from file (one per line).

    Args:
        filepath: Path to symbols file

    Returns:
        List of symbols
    """
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols from {filepath}: {e}")
        sys.exit(1)


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime.

    Args:
        date_str: Date string (YYYY-MM-DD)

    Returns:
        datetime object
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Load symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = load_symbols_from_file(args.symbol_file)

    logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")

    # Initialize components
    logger.info(f"Initializing data store at {args.data_dir}")
    data_store = MarketDataStore(args.data_dir)

    loader = YahooFinanceLoader(max_workers=args.max_workers)
    pipeline = IngestionPipeline(data_store, loader)

    # Validate only mode
    if args.validate_only:
        logger.info("Validation mode - checking existing data")
        for symbol in symbols:
            validation_results = pipeline.validate_existing_data(symbol)
            if validation_results:
                quality_score = validation_results['quality_score']
                logger.info(f"{symbol}: Quality score {quality_score:.1f}/100")
                if validation_results['warnings']:
                    logger.warning(f"{symbol}: Warnings: {validation_results['warnings']}")
        return

    # Incremental mode
    if args.incremental:
        logger.info("Incremental update mode")
        results = pipeline.ingest_incremental_batch(
            symbols=symbols,
            interval=args.interval,
            lookback_days=args.lookback_days
        )

    # Backfill mode
    elif args.backfill:
        logger.info(f"Backfill mode - fetching {args.years} years of history")
        results = {}
        for symbol in symbols:
            success = pipeline.backfill_symbol(
                symbol=symbol,
                years=args.years,
                interval=args.interval
            )
            results[symbol] = success
            if not success and args.stop_on_error:
                logger.error(f"Stopping due to error on {symbol}")
                break

    # Full date range mode
    else:
        if not args.start:
            logger.error("--start is required for full date range mode")
            sys.exit(1)

        start = parse_date(args.start)
        end = parse_date(args.end) if args.end else datetime.now()

        logger.info(f"Fetching data from {start.date()} to {end.date()}")

        results = pipeline.ingest_symbols(
            symbols=symbols,
            start=start,
            end=end,
            interval=args.interval,
            continue_on_error=not args.stop_on_error
        )

    # Print summary
    successful = sum(1 for success in results.values() if success)
    failed = sum(1 for success in results.values() if not success)

    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Total symbols:     {len(symbols)}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"Success rate:      {successful / len(symbols) * 100:.1f}%")
    print("="*60)

    # Show failed symbols
    if failed > 0:
        failed_symbols = [sym for sym, success in results.items() if not success]
        print(f"\nFailed symbols: {', '.join(failed_symbols)}")

    # Show ingestion stats
    print("\nRecent ingestion activity:")
    stats = pipeline.get_ingestion_summary(days=1)
    if not stats.empty:
        print(stats.to_string(index=False))

    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
