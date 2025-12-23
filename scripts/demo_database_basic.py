#!/usr/bin/env python3
"""
Basic demonstration of the database layer.

This script shows how to:
1. Initialize the database
2. Add symbols
3. Write sample data
4. Query the data back
5. Check data quality

Run: python3 scripts/demo_database_basic.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("DATABASE LAYER BASIC DEMO")
print("="*80)

# Check dependencies
try:
    import pandas as pd
    import numpy as np
    print("\n✓ Dependencies loaded (pandas, numpy)")
except ImportError as e:
    print(f"\n✗ Missing dependency: {e}")
    print("\nPlease install: pip install pandas numpy pyarrow")
    sys.exit(1)

# Import database layer
try:
    from data.storage import MarketDataStore
    print("✓ Database layer imported")
except ImportError as e:
    print(f"\n✗ Could not import database layer: {e}")
    sys.exit(1)

# Setup
db_path = "/tmp/trading_bot_demo"
print(f"\nDatabase location: {db_path}")

# Step 1: Initialize database
print("\n" + "-"*80)
print("STEP 1: Initialize Database")
print("-"*80)

try:
    store = MarketDataStore(db_path)
    print(f"✓ Database initialized")
    print(f"  Schema version: {store.get_schema_version()}")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

# Step 2: Add symbols
print("\n" + "-"*80)
print("STEP 2: Add Symbols")
print("-"*80)

symbols = [
    ('AAPL', 'Apple Inc.', 'NASDAQ', 'stock', 'Technology'),
    ('MSFT', 'Microsoft Corp.', 'NASDAQ', 'stock', 'Technology'),
]

for symbol, name, exchange, asset_class, sector in symbols:
    store.add_symbol(
        symbol=symbol,
        name=name,
        exchange=exchange,
        asset_class=asset_class,
        sector=sector
    )
    print(f"✓ Added {symbol} ({name})")

# Step 3: Generate and write sample data
print("\n" + "-"*80)
print("STEP 3: Write Sample Data")
print("-"*80)

# Generate simple sample data (30 days)
dates = pd.date_range('2024-01-01', periods=30, freq='D')

for symbol in ['AAPL', 'MSFT']:
    # Simple random walk
    prices = 100 + np.cumsum(np.random.randn(30) * 2)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(30) * 0.5,
        'high': prices + np.abs(np.random.randn(30)) * 1.0,
        'low': prices - np.abs(np.random.randn(30)) * 1.0,
        'close': prices,
        'volume': np.random.uniform(1e6, 1e7, 30)
    })

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    result = store.write_bars(
        symbol=symbol,
        data=df,
        timeframe='1D',
        source='demo_script'
    )

    print(f"\n{symbol}:")
    print(f"  Bars written: {result['bars_written']}")
    print(f"  Time taken: {result['elapsed_seconds']:.3f}s")
    print(f"  Quality issues: {result['quality_issues']}")

# Step 4: Read data back
print("\n" + "-"*80)
print("STEP 4: Read Data Back")
print("-"*80)

for symbol in ['AAPL', 'MSFT']:
    df = store.get_bars(symbol, timeframe='1D')

    print(f"\n{symbol}:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if not df.empty:
        print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"\nFirst 3 rows:")
        print(df.head(3)[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False))

# Step 5: Query with filters
print("\n" + "-"*80)
print("STEP 5: Query with Date Filter")
print("-"*80)

start = pd.Timestamp('2024-01-15')
end = pd.Timestamp('2024-01-20')

df = store.get_bars('AAPL', start=start, end=end, timeframe='1D')
print(f"\nAAPL from {start.date()} to {end.date()}:")
print(f"  Rows returned: {len(df)}")
if not df.empty:
    print(f"\n{df[['timestamp', 'close']].to_string(index=False)}")

# Step 6: Check data quality
print("\n" + "-"*80)
print("STEP 6: Data Quality Metrics")
print("-"*80)

for symbol in ['AAPL', 'MSFT']:
    metrics = store.get_quality_metrics(symbol, '1D')

    if metrics:
        print(f"\n{symbol}:")
        print(f"  Total bars: {metrics['total_bars']}")
        print(f"  Missing: {metrics['missing_bars']}")
        print(f"  Duplicates: {metrics['duplicate_bars']}")
        print(f"  Quality score: {metrics['quality_score']:.1f}/100")

# Step 7: Data summary
print("\n" + "-"*80)
print("STEP 7: Data Summary")
print("-"*80)

summary = store.get_data_summary()
if not summary.empty:
    print("\nData availability:")
    print(summary[['symbol', 'timeframe', 'earliest_date', 'latest_date', 'total_bars']].to_string(index=False))

# Step 8: Storage stats
print("\n" + "-"*80)
print("STEP 8: Storage Statistics")
print("-"*80)

stats = store.get_storage_stats()
print(f"\nSymbols: {stats['num_symbols']}")
print(f"Total bars: {stats['total_bars']:,}")
print(f"Storage size: {stats['total_size_mb']:.2f} MB")
print(f"Average bars/symbol: {stats['avg_bars_per_symbol']:.0f}")

# Summary
print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print(f"\nDatabase files created at: {db_path}")
print("  - market_data.db (SQLite metadata)")
print("  - parquet/ (OHLCV timeseries)")
print("\nTo inspect the database:")
print(f"  sqlite3 {db_path}/market_data.db")
print("\nSQL queries you can try:")
print("  SELECT * FROM symbols;")
print("  SELECT * FROM v_data_availability;")
print("  SELECT * FROM v_latest_quality;")
print("  SELECT * FROM ingestion_log ORDER BY created_at DESC;")
print("\n" + "="*80)
