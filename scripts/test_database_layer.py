#!/usr/bin/env python3
"""Test script for database layer.

Demonstrates:
1. Creating database and parquet storage
2. Writing sample market data
3. Reading data back
4. Quality checks
5. Data availability queries
6. Integration with HistoricalDataSource
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from data.storage.market_data_store import MarketDataStore
from data.sources.database_data_source import DatabaseDataSource

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_ohlcv(
    symbol: str,
    start_date: datetime,
    days: int = 252,
    initial_price: float = 100.0
) -> pd.DataFrame:
    """Generate sample OHLCV data for testing.

    Args:
        symbol: Stock symbol
        start_date: Start date
        days: Number of trading days
        initial_price: Starting price

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start=start_date, periods=days, freq='B')  # Business days

    # Generate random walk
    returns = np.random.normal(0.0005, 0.02, days)  # 0.05% daily return, 2% vol
    closes = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    highs = closes * (1 + np.abs(np.random.normal(0, 0.01, days)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.01, days)))
    opens = closes * (1 + np.random.normal(0, 0.005, days))

    # Ensure OHLC relationships
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    # Volume
    volumes = np.random.uniform(1_000_000, 10_000_000, days)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    return df


def test_database_creation():
    """Test 1: Database creation and initialization."""
    print("\n" + "="*80)
    print("TEST 1: Database Creation and Initialization")
    print("="*80)

    db_path = "/tmp/trading_bot_test_db"

    # Create store
    store = MarketDataStore(db_path)

    # Check schema version
    version = store.get_schema_version()
    print(f"\nDatabase schema version: {version}")

    # Verify empty database
    symbols = store.list_symbols()
    print(f"Initial symbols in database: {len(symbols)}")

    assert version == 1, "Schema version should be 1"
    assert len(symbols) == 0, "Database should be empty initially"

    print("\n✓ Database created successfully")
    return store


def test_symbol_management(store: MarketDataStore):
    """Test 2: Symbol management."""
    print("\n" + "="*80)
    print("TEST 2: Symbol Management")
    print("="*80)

    # Add symbols
    symbols_data = [
        ('AAPL', 'Apple Inc.', 'NASDAQ', 'stock', 'Technology'),
        ('MSFT', 'Microsoft Corp.', 'NASDAQ', 'stock', 'Technology'),
        ('SPY', 'SPDR S&P 500 ETF', 'NYSE', 'etf', None),
    ]

    for symbol, name, exchange, asset_class, sector in symbols_data:
        store.add_symbol(
            symbol=symbol,
            name=name,
            exchange=exchange,
            asset_class=asset_class,
            sector=sector
        )
        print(f"Added symbol: {symbol} ({name})")

    # Verify symbols added
    all_symbols = store.list_symbols()
    print(f"\nTotal symbols: {len(all_symbols)}")
    print(f"Symbols: {', '.join(all_symbols)}")

    # Get specific symbol
    aapl = store.get_symbol('AAPL')
    print(f"\nAAPL metadata:")
    for key, value in aapl.items():
        print(f"  {key}: {value}")

    assert len(all_symbols) == 3, "Should have 3 symbols"
    assert 'AAPL' in all_symbols, "AAPL should be in symbols"

    print("\n✓ Symbol management working correctly")


def test_write_read_data(store: MarketDataStore):
    """Test 3: Writing and reading OHLCV data."""
    print("\n" + "="*80)
    print("TEST 3: Writing and Reading OHLCV Data")
    print("="*80)

    # Generate sample data
    start_date = datetime(2023, 1, 1)

    symbols = ['AAPL', 'MSFT', 'SPY']
    for symbol in symbols:
        print(f"\nGenerating data for {symbol}...")
        data = generate_sample_ohlcv(symbol, start_date, days=252)

        # Write to database
        result = store.write_bars(
            symbol=symbol,
            data=data,
            timeframe='1D',
            source='test_script'
        )

        print(f"  Wrote {result['bars_written']} bars")
        print(f"  File size: {result['file_id']}")
        print(f"  Quality issues: {result['quality_issues']}")
        print(f"  Time: {result['elapsed_seconds']:.3f}s")

    # Read data back
    print("\n\nReading data back...")
    for symbol in symbols:
        df = store.get_bars(symbol, timeframe='1D')
        print(f"\n{symbol}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        assert len(df) == 252, f"Should have 252 bars for {symbol}"
        assert not df.empty, f"Data should not be empty for {symbol}"

    print("\n✓ Data write/read working correctly")


def test_date_range_queries(store: MarketDataStore):
    """Test 4: Date range queries."""
    print("\n" + "="*80)
    print("TEST 4: Date Range Queries")
    print("="*80)

    # Query specific date range
    start = datetime(2023, 6, 1)
    end = datetime(2023, 12, 31)

    print(f"\nQuerying AAPL from {start.date()} to {end.date()}...")
    df = store.get_bars('AAPL', start=start, end=end)

    print(f"  Rows returned: {len(df)}")
    print(f"  Actual date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    # Verify filtering
    assert df['timestamp'].min() >= start, "Start date filter failed"
    assert df['timestamp'].max() <= end, "End date filter failed"

    # Get available date range
    date_range = store.get_date_range('AAPL', '1D')
    print(f"\nAvailable date range for AAPL: {date_range[0].date()} to {date_range[1].date()}")

    print("\n✓ Date range queries working correctly")


def test_quality_metrics(store: MarketDataStore):
    """Test 5: Data quality metrics."""
    print("\n" + "="*80)
    print("TEST 5: Data Quality Metrics")
    print("="*80)

    # Get quality metrics for each symbol
    for symbol in ['AAPL', 'MSFT', 'SPY']:
        metrics = store.get_quality_metrics(symbol, '1D')

        if metrics:
            print(f"\n{symbol} quality metrics:")
            print(f"  Total bars: {metrics['total_bars']}")
            print(f"  Missing bars: {metrics['missing_bars']}")
            print(f"  Duplicate bars: {metrics['duplicate_bars']}")
            print(f"  Outlier bars: {metrics['outlier_bars']}")
            print(f"  Completeness: {metrics['completeness_pct']:.1f}%")
            print(f"  Quality score: {metrics['quality_score']:.1f}/100")

    # Get overall quality summary
    print("\n\nOverall quality summary:")
    quality_df = store.get_quality_summary()
    print(quality_df.to_string(index=False))

    print("\n✓ Quality metrics working correctly")


def test_data_summary(store: MarketDataStore):
    """Test 6: Data availability summary."""
    print("\n" + "="*80)
    print("TEST 6: Data Availability Summary")
    print("="*80)

    # Get data summary
    summary = store.get_data_summary()

    print("\nData availability:")
    print(summary[['symbol', 'timeframe', 'earliest_date', 'latest_date', 'total_bars']].to_string(index=False))

    # Get storage stats
    stats = store.get_storage_stats()
    print("\n\nStorage statistics:")
    print(f"  Symbols: {stats['num_symbols']}")
    print(f"  Total bars: {stats['total_bars']:,}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    print(f"  Avg bars/symbol: {stats['avg_bars_per_symbol']:.0f}")

    print("\n✓ Data summary working correctly")


def test_database_data_source(store: MarketDataStore):
    """Test 7: Integration with DatabaseDataSource."""
    print("\n" + "="*80)
    print("TEST 7: DatabaseDataSource Integration")
    print("="*80)

    # Create data source
    db_path = "/tmp/trading_bot_test_db"
    data_source = DatabaseDataSource(
        db_path=db_path,
        symbols=['AAPL', 'MSFT'],
        timeframe='1D'
    )

    print(f"\nData source: {data_source}")

    # Get date range
    start, end = data_source.get_date_range()
    print(f"Date range: {start.date()} to {end.date()}")

    # Get events for a specific date
    test_date = datetime(2023, 3, 15)
    events = data_source.get_events(test_date)

    print(f"\nEvents on {test_date.date()}:")
    for event in events:
        print(f"  {event.symbol}: O={event.open:.2f} H={event.high:.2f} "
              f"L={event.low:.2f} C={event.close:.2f} V={event.volume:,.0f}")

    assert len(events) > 0, "Should have events for test date"

    # Get bars for symbols
    bars = data_source.get_bars(['AAPL', 'MSFT'], limit=5)
    print(f"\nLast 5 bars for each symbol:")
    print(bars[['timestamp', 'symbol', 'close']].to_string(index=False))

    print("\n✓ DatabaseDataSource integration working correctly")


def test_update_mode(store: MarketDataStore):
    """Test 8: Update mode (append new data)."""
    print("\n" + "="*80)
    print("TEST 8: Update Mode (Append New Data)")
    print("="*80)

    # Get current bar count
    initial_count = store.get_bar_count('AAPL', '1D')
    print(f"Initial bar count for AAPL: {initial_count}")

    # Generate new data (next 30 days)
    last_date = datetime(2023, 12, 31)
    new_data = generate_sample_ohlcv('AAPL', last_date + timedelta(days=1), days=30)

    # Write with update mode
    result = store.write_bars(
        symbol='AAPL',
        data=new_data,
        timeframe='1D',
        source='test_update',
        mode='update'
    )

    print(f"\nAppended {result['bars_written']} bars")

    # Verify new count
    new_count = store.get_bar_count('AAPL', '1D')
    print(f"New bar count for AAPL: {new_count}")

    assert new_count > initial_count, "Bar count should have increased"

    print("\n✓ Update mode working correctly")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DATABASE LAYER TEST SUITE")
    print("="*80)

    try:
        # Test 1: Database creation
        store = test_database_creation()

        # Test 2: Symbol management
        test_symbol_management(store)

        # Test 3: Write and read data
        test_write_read_data(store)

        # Test 4: Date range queries
        test_date_range_queries(store)

        # Test 5: Quality metrics
        test_quality_metrics(store)

        # Test 6: Data summary
        test_data_summary(store)

        # Test 7: DatabaseDataSource integration
        test_database_data_source(store)

        # Test 8: Update mode
        test_update_mode(store)

        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nDatabase location: /tmp/trading_bot_test_db")
        print("  - market_data.db (SQLite metadata)")
        print("  - parquet/ (OHLCV timeseries)")
        print("\nYou can inspect the database with:")
        print("  sqlite3 /tmp/trading_bot_test_db/market_data.db")
        print("\n" + "="*80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
