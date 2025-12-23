# Data Ingestion Pipeline

## Overview

Comprehensive data ingestion pipeline for fetching historical OHLCV data from Yahoo Finance and storing it in a hybrid SQLite + Parquet database.

## Architecture

```
Yahoo Finance → Yahoo Loader → Data Validation → Database Storage
                                                      ├─ SQLite (metadata)
                                                      └─ Parquet (OHLCV data)
```

### Components

1. **Yahoo Loader** (`data/loaders/yahoo_loader.py`)
   - Fetches historical OHLCV data using yfinance library
   - Concurrent fetching with rate limiting
   - Automatic retry on failures
   - Symbol metadata extraction

2. **Data Validation** (`data/pipeline/validation.py`)
   - Comprehensive quality checks
   - Detects gaps, outliers, invalid OHLC relationships
   - Calculates quality score (0-100)
   - Auto-cleaning of common issues

3. **Database Storage** (`data/storage/`)
   - **SQLite**: Symbol metadata, data availability, quality metrics, ingestion logs
   - **Parquet**: Compressed OHLCV time series data
   - Automatic deduplication and merging
   - Indexed queries for fast access

4. **Ingestion Pipeline** (`data/pipeline/ingestion.py`)
   - Orchestrates the full workflow
   - Handles incremental updates
   - Batch processing with error isolation
   - Comprehensive logging

## Storage Structure

```
data_storage/
├── market_data.db           # SQLite database (metadata)
└── parquet/
    ├── AAPL/
    │   └── 1D/
    │       └── data.parquet
    ├── MSFT/
    │   └── 1D/
    │       └── data.parquet
    └── ...
```

## Quick Start

### 1. Install Dependencies

```bash
source .venv/bin/activate
pip install yfinance pyarrow
```

### 2. Ingest Historical Data

```bash
# Ingest specific symbols for date range
python scripts/ingest_data.py \
    --symbols AAPL,MSFT,GOOGL \
    --start 2023-01-01 \
    --end 2024-01-01

# Output:
# ============================================================
# INGESTION SUMMARY
# ============================================================
# Total symbols:     3
# Successful:        3
# Failed:            0
# Success rate:      100.0%
# ============================================================
```

### 3. Incremental Updates

```bash
# Fetch only new data since last update
python scripts/ingest_data.py \
    --symbols AAPL,MSFT,GOOGL \
    --incremental

# Automatically determines last available date and fetches new data
```

### 4. Backfill Historical Data

```bash
# Backfill 5 years of historical data
python scripts/ingest_data.py \
    --symbols AAPL \
    --backfill \
    --years 5
```

### 5. Batch Ingestion from File

```bash
# Create symbols file (one per line)
cat > symbols.txt <<EOF
AAPL
MSFT
GOOGL
AMZN
META
EOF

# Ingest all symbols
python scripts/ingest_data.py \
    --symbol-file symbols.txt \
    --start 2023-01-01 \
    --end 2024-01-01
```

## CLI Reference

```bash
python scripts/ingest_data.py [OPTIONS]
```

### Required Options

- `--symbols SYMBOLS`: Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)
- `--symbol-file FILE`: File containing symbols (one per line)

### Date Range Options

- `--start YYYY-MM-DD`: Start date (required for full date range mode)
- `--end YYYY-MM-DD`: End date (defaults to today)

### Interval Options

- `--interval {1d,1h,5m,15m,30m,1wk,1mo}`: Data interval (default: 1d)

### Operation Modes

- `--incremental`: Incremental update (fetch only new data)
- `--backfill`: Backfill historical data
- `--validate-only`: Only validate existing data
- `--years N`: Years to backfill (default: 5)
- `--lookback-days N`: Days to look back for incremental updates (default: 7)

### Storage Options

- `--data-dir PATH`: Data storage directory (default: ./data_storage)

### Performance Options

- `--max-workers N`: Maximum concurrent downloads (default: 5)
- `--stop-on-error`: Stop on first error (default: continue)

### Logging Options

- `--verbose`: Verbose logging
- `--quiet`: Minimal logging

## Usage Examples

### Example 1: Daily Data Ingestion

```bash
# Fetch daily data for major tech stocks
python scripts/ingest_data.py \
    --symbols AAPL,MSFT,GOOGL,AMZN,META \
    --start 2020-01-01 \
    --end 2024-01-01
```

### Example 2: Intraday Data

```bash
# Fetch hourly data for the last 30 days
python scripts/ingest_data.py \
    --symbols SPY,QQQ \
    --start 2024-11-01 \
    --end 2024-12-01 \
    --interval 1h
```

### Example 3: Automated Daily Updates

```bash
#!/bin/bash
# daily_update.sh - Add to cron for daily execution

python scripts/ingest_data.py \
    --symbol-file production_symbols.txt \
    --incremental \
    --verbose
```

### Example 4: Validate Data Quality

```bash
# Check quality of existing data
python scripts/ingest_data.py \
    --symbols AAPL,MSFT,GOOGL \
    --validate-only
```

## Data Quality Checks

The pipeline performs comprehensive quality validation:

### Completeness
- Missing bars detection
- Gap identification in time series
- Completeness percentage calculation

### Consistency
- OHLC relationship validation (high >= low, etc.)
- Price continuity checks
- Volume validation

### Accuracy
- Negative price detection
- Outlier detection (5-sigma threshold)
- Duplicate timestamp detection
- Zero volume identification

### Quality Score

Quality score (0-100) is calculated based on:
- Missing bars: -30 points max
- Invalid OHLC: -30 points max
- Duplicates: -20 points max
- Outliers: -20 points max

## Accessing Data Programmatically

```python
from data.storage.market_data_store import MarketDataStore
from datetime import datetime

# Initialize store
store = MarketDataStore('data_storage')

# Read OHLCV data
df = store.get_bars(
    symbol='AAPL',
    start=datetime(2023, 1, 1),
    end=datetime(2024, 1, 1),
    timeframe='1D'
)

print(f"Retrieved {len(df)} bars")
print(df.head())

# Check data availability
date_range = store.get_date_range('AAPL', '1D')
print(f"Available data: {date_range[0]} to {date_range[1]}")

# Get quality metrics
quality = store.get_quality_metrics('AAPL', '1D')
print(f"Quality score: {quality['quality_score']}/100")

# List all symbols
symbols = store.list_symbols()
print(f"Available symbols: {symbols}")

# Get storage statistics
stats = store.get_storage_stats()
print(f"Total bars: {stats['total_bars']:,}")
print(f"Storage size: {stats['total_size_mb']:.2f} MB")
```

## Database Schema

### SQLite Tables

- **symbols**: Symbol universe and metadata
- **data_files**: Parquet file tracking and availability
- **data_quality**: Quality metrics per symbol/timeframe
- **corporate_actions**: Splits, dividends, etc.
- **ingestion_log**: Ingestion history and errors

### Views

- **v_data_availability**: Data availability summary
- **v_latest_quality**: Latest quality scores

See `data/storage/schema.sql` for complete schema.

## Error Handling

### Partial Failures

By default, the pipeline continues processing even if some symbols fail:

```bash
# Continue on errors (default)
python scripts/ingest_data.py --symbols AAPL,INVALID,MSFT

# Stop on first error
python scripts/ingest_data.py --symbols AAPL,INVALID,MSFT --stop-on-error
```

### Failed Symbols

All failures are logged in the ingestion_log table:

```python
from data.storage.market_data_store import MarketDataStore

store = MarketDataStore('data_storage')
history = store.db.get_ingestion_history(limit=100)

# Filter failed ingestions
failed = [h for h in history if h['status'] == 'failed']
for f in failed:
    print(f"{f['symbol']}: {f['error_message']}")
```

### Retry Logic

The Yahoo loader automatically retries failed fetches:
- 3 retry attempts by default
- Exponential backoff (1s, 2s, 4s)
- Rate limiting between requests (0.5s default)

## Performance

### Benchmarks

- ~250 bars/symbol/year for daily data
- ~0.01s write time per symbol
- ~16 KB Parquet file per 250 bars
- ~100 symbols/minute throughput

### Optimization Tips

1. **Concurrent Downloads**
   ```bash
   # Increase workers for large batches
   python scripts/ingest_data.py \
       --symbols AAPL,MSFT,... \
       --max-workers 10
   ```

2. **Incremental Updates**
   - Use `--incremental` for daily updates
   - Only fetches new data since last update
   - Much faster than full re-download

3. **Storage Efficiency**
   - Parquet compression saves ~80% space vs CSV
   - SQLite indexes speed up metadata queries
   - Automatic deduplication prevents bloat

## Monitoring

### Ingestion Logs

```python
from data.storage.market_data_store import MarketDataStore
import pandas as pd

store = MarketDataStore('data_storage')

# Get recent ingestion activity
history = store.db.get_ingestion_history(limit=50)
df = pd.DataFrame(history)

# Success rate
success_rate = (df['status'] == 'success').mean() * 100
print(f"Success rate: {success_rate:.1f}%")

# Average ingestion time
avg_time = df[df['status'] == 'success']['ingestion_time_seconds'].mean()
print(f"Average ingestion time: {avg_time:.2f}s")
```

### Quality Monitoring

```python
# Get quality summary for all symbols
quality_summary = store.get_quality_summary()
df = pd.DataFrame(quality_summary)

# Symbols with low quality
low_quality = df[df['quality_score'] < 80]
print(f"Low quality symbols: {len(low_quality)}")
print(low_quality[['symbol', 'quality_score', 'notes']])
```

## Troubleshooting

### Issue: "No data returned for symbol"

**Cause**: Symbol may be delisted or invalid

**Solution**:
- Verify symbol exists on Yahoo Finance
- Check if symbol is still trading
- Try different date range

### Issue: "Cannot compare tz-naive and tz-aware timestamps"

**Cause**: Timezone mismatch between existing and new data

**Solution**: Fixed in pipeline - all timestamps converted to UTC without timezone

### Issue: "Rate limit exceeded"

**Cause**: Too many requests to Yahoo Finance

**Solution**:
```bash
# Reduce workers and add delay
python scripts/ingest_data.py \
    --symbols AAPL,MSFT,... \
    --max-workers 3
```

### Issue: Low quality score

**Cause**: Data has gaps, outliers, or invalid values

**Solution**:
- Check validation warnings in logs
- Re-fetch data with `--backfill`
- Validate with `--validate-only`

## Best Practices

1. **Daily Updates**: Use `--incremental` for daily updates
2. **Batch Processing**: Use symbol files for large batches
3. **Error Handling**: Don't use `--stop-on-error` for production
4. **Monitoring**: Check quality scores regularly
5. **Backups**: Backup `data_storage/` directory regularly
6. **Logging**: Use `--verbose` for debugging, default for production

## Integration with Trading Framework

The ingested data integrates seamlessly with the trading framework:

```python
from data.sources.historical_data import HistoricalDataSource
from data.storage.market_data_store import MarketDataStore

# Load data from storage
store = MarketDataStore('data_storage')
df = store.get_bars('AAPL', timeframe='1D')

# Create data source for backtesting
data_source = HistoricalDataSource(df, timeframe='1D')

# Use in engine
from engine.core.simulation_engine import SimulationEngine
# ... rest of backtesting code
```

## Future Enhancements

- [ ] Support for additional data sources (Alpaca, Alpha Vantage)
- [ ] Real-time data ingestion
- [ ] Corporate actions tracking
- [ ] Data versioning and snapshots
- [ ] Distributed ingestion for large-scale operations
- [ ] Data quality alerting
- [ ] Automatic data repair

## Support

For issues or questions:
1. Check logs in `data_ingestion.log`
2. Verify data with `--validate-only`
3. Review ingestion history in database
4. Check quality scores

## License

MIT License - See LICENSE file
