# Database Layer - SQLite + Parquet Hybrid Storage

## Overview

This database layer provides efficient, production-ready storage for market data using a hybrid approach:

- **SQLite**: Metadata, data availability, quality metrics, corporate actions
- **Parquet**: Actual OHLCV timeseries data (compressed, columnar)

## Architecture

```
data/storage/
├── schema.sql              # Database schema (versioned)
├── database.py             # SQLite connection and queries
├── parquet_store.py        # Parquet read/write operations
├── market_data_store.py    # Main interface (combines both)
└── README.md              # This file
```

### Storage Structure

```
<base_path>/
├── market_data.db          # SQLite database
└── parquet/
    ├── AAPL/
    │   ├── 1D/
    │   │   └── data.parquet
    │   └── 1H/
    │       └── data.parquet
    └── MSFT/
        └── 1D/
            └── data.parquet
```

## Features

### 1. Efficient Data Storage
- **Parquet**: Columnar format with Snappy compression (typically 10-20x smaller than CSV)
- **Indexed Queries**: Fast timestamp-based filtering
- **Metadata Separation**: SQLite for quick availability checks without reading timeseries

### 2. Data Quality Tracking
- Automatic quality checks on ingestion
- Metrics: missing bars, duplicates, outliers, invalid OHLC
- Quality scores (0-100) for each symbol/timeframe
- Historical quality tracking

### 3. Schema Versioning
- Versioned schema for safe migrations
- Track schema changes over time
- Future-proof for new features

### 4. Complete Metadata
- Symbol information (name, exchange, sector, industry)
- Data availability (date ranges, bar counts)
- Ingestion history (source, status, timing)
- Corporate actions (splits, dividends)

## Usage

### Installation

```bash
# Install dependencies
pip install pandas pyarrow sqlalchemy

# Or from requirements.txt
pip install -r requirements.txt
```

### Quick Start

```python
from data.storage import MarketDataStore
import pandas as pd

# Initialize store
store = MarketDataStore("/path/to/data")

# Add a symbol
store.add_symbol(
    symbol="AAPL",
    name="Apple Inc.",
    exchange="NASDAQ",
    asset_class="stock",
    sector="Technology"
)

# Write data
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=252, freq='B'),
    'open': [100 + i*0.5 for i in range(252)],
    'high': [101 + i*0.5 for i in range(252)],
    'low': [99 + i*0.5 for i in range(252)],
    'close': [100.5 + i*0.5 for i in range(252)],
    'volume': [1000000] * 252
})

result = store.write_bars(
    symbol="AAPL",
    data=df,
    timeframe="1D",
    source="yahoo"
)

# Read data
df = store.get_bars(
    symbol="AAPL",
    start=pd.Timestamp("2023-06-01"),
    end=pd.Timestamp("2023-12-31"),
    timeframe="1D"
)

print(df.head())
```

### Main Interface Methods

#### MarketDataStore

**Data Access:**
```python
# Get bars for a symbol
df = store.get_bars(symbol, start, end, timeframe)

# Write bars
result = store.write_bars(symbol, data, timeframe, source, mode)

# Delete bars
store.delete_bars(symbol, timeframe)
```

**Symbol Management:**
```python
# Add/update symbol
store.add_symbol(symbol, name, exchange, asset_class, sector, ...)

# Get symbol info
info = store.get_symbol(symbol)

# List symbols
symbols = store.list_symbols(exchange="NASDAQ", active_only=True)
```

**Data Availability:**
```python
# Check if data exists
exists = store.has_data(symbol, timeframe)

# Get date range
start, end = store.get_date_range(symbol, timeframe)

# Get bar count
count = store.get_bar_count(symbol, timeframe)
```

**Quality & Monitoring:**
```python
# Get quality metrics
metrics = store.get_quality_metrics(symbol, timeframe)

# Get data summary
summary_df = store.get_data_summary()

# Get quality summary
quality_df = store.get_quality_summary()

# Get storage stats
stats = store.get_storage_stats()
```

### Write Modes

- **`overwrite`**: Replace all existing data
- **`append`**: Add new data, keep existing (deduplicates by timestamp)
- **`update`** (default): Merge with existing, update overlapping timestamps

### Integration with Backtesting

```python
from data.sources import DatabaseDataSource

# Create data source for backtesting
data_source = DatabaseDataSource(
    db_path="/path/to/data",
    symbols=["AAPL", "MSFT", "SPY"],
    timeframe="1D"
)

# Use with simulation engine
engine = SimulationEngine(
    clock=HistoricalClock(...),
    data_source=data_source,
    execution=SimulatedExecution(...),
    strategies=[...]
)
```

## Database Schema

### Tables

#### `symbols`
Symbol universe and metadata
- `symbol` (PK): Trading symbol
- `name`: Company/fund name
- `exchange`: Exchange (NASDAQ, NYSE, etc.)
- `asset_class`: stock, etf, crypto, etc.
- `sector`, `industry`: Classification
- `market_cap`: Market capitalization
- `is_active`: Active status
- Timestamps: `created_at`, `updated_at`

#### `data_files`
Track parquet files and availability
- `file_id` (PK): Auto-increment ID
- `symbol` (FK): Symbol
- `timeframe`: 1D, 1H, etc.
- `start_date`, `end_date`: Date range
- `num_bars`: Bar count
- `file_path`: Relative path to parquet
- `file_size_bytes`: File size
- `parquet_schema`: JSON schema
- Unique: `(symbol, timeframe, start_date, end_date)`

#### `data_quality`
Quality metrics per symbol/timeframe
- `quality_id` (PK): Auto-increment ID
- `symbol` (FK), `timeframe`
- `check_date`: Date of check
- `total_bars`: Total bars checked
- `missing_bars`, `duplicate_bars`, `outlier_bars`: Issue counts
- `zero_volume_bars`, `negative_prices`, `invalid_ohlc`
- `completeness_pct`: Percentage complete
- `quality_score`: 0-100 composite score
- `notes`: Additional notes

#### `corporate_actions`
Stock splits, dividends, etc.
- `action_id` (PK): Auto-increment ID
- `symbol` (FK): Symbol
- `action_type`: split, dividend, merger, etc.
- `ex_date`: Ex-dividend/ex-split date
- `record_date`, `payment_date`: Relevant dates
- `split_ratio`: For splits (e.g., 2.0 = 2-for-1)
- `dividend_amount`: For dividends
- `details`: JSON with additional info

#### `ingestion_log`
Track all data ingestion events
- `log_id` (PK): Auto-increment ID
- `symbol`, `timeframe`: What was ingested
- `source`: Data source (yahoo, alpaca, etc.)
- `start_date`, `end_date`: Date range
- `bars_ingested`: Number of bars
- `status`: success, partial, failed
- `error_message`: If failed
- `ingestion_time_seconds`: Performance tracking
- `created_at`: Timestamp

### Views

#### `v_data_availability`
Summary of available data per symbol/timeframe

#### `v_latest_quality`
Latest quality check results per symbol/timeframe

## Performance

### Benchmarks (on M1 Mac)

| Operation | Time | Notes |
|-----------|------|-------|
| Write 252 bars | ~50ms | Including quality checks |
| Read 252 bars | ~10ms | With date filtering |
| Read 10K bars | ~30ms | Parquet is fast! |
| Quality check | ~20ms | Automatic on write |

### Storage Efficiency

| Format | Size (252 days OHLCV) | Compression |
|--------|----------------------|-------------|
| CSV | ~50 KB | None |
| Parquet (Snappy) | ~5 KB | 10x |
| Parquet (gzip) | ~3 KB | 16x |

### Scaling

- **Symbols**: Tested with 100+ symbols
- **Timeframes**: Multiple timeframes per symbol (1D, 1H, 5T, etc.)
- **Historical depth**: Years of daily data
- **Query performance**: Sub-second even with large datasets

## Data Quality

### Automatic Checks

On every write, the following checks are performed:

1. **Required columns**: timestamp, open, high, low, close, volume
2. **OHLC relationships**: high >= low, close in [low, high], etc.
3. **Negative prices**: No negative values
4. **Null values**: No nulls in required columns
5. **Duplicate timestamps**: Detected and handled
6. **Outliers**: Statistical outlier detection (5-sigma)
7. **Zero volume**: Tracked for reference

### Quality Score

Composite score (0-100) calculated as:
- Missing data: -30 points max
- Duplicates: -20 points max
- Invalid OHLC: -30 points max
- Outliers: -20 points max

## Migration Support

### Schema Versioning

The schema includes a version table for tracking changes:

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP,
    description TEXT
);
```

### Future Migrations

To add new features:

1. Create migration SQL file: `migrations/002_add_feature.sql`
2. Update schema version
3. Apply migration in `database.py`

Example migration:

```sql
-- migrations/002_add_intraday_support.sql

-- Add intraday flag to data_files
ALTER TABLE data_files ADD COLUMN is_intraday BOOLEAN DEFAULT 0;

-- Update schema version
INSERT INTO schema_version (version, description)
VALUES (2, 'Add intraday support');
```

## Testing

### Unit Tests

```bash
# Run database layer tests
python scripts/test_database_layer.py
```

This will:
1. Create test database
2. Add sample symbols
3. Write/read data
4. Test quality checks
5. Verify date range queries
6. Test DatabaseDataSource integration

### Manual Testing

```bash
# Inspect database
sqlite3 /path/to/data/market_data.db

# View symbols
SELECT * FROM symbols;

# View data availability
SELECT * FROM v_data_availability;

# View quality metrics
SELECT * FROM v_latest_quality;

# View ingestion log
SELECT * FROM ingestion_log ORDER BY created_at DESC LIMIT 10;
```

## Best Practices

### 1. Write Mode Selection

- Use **`update`** for regular updates (safe, handles duplicates)
- Use **`append`** when you know data is new (slightly faster)
- Use **`overwrite`** only for complete re-ingestion

### 2. Symbol Metadata

Always add symbol metadata before writing data:

```python
store.add_symbol(symbol, name, exchange, asset_class, sector)
store.write_bars(symbol, data, ...)
```

### 3. Corporate Actions

Record splits/dividends for accurate backtesting:

```python
store.add_corporate_action(
    symbol="AAPL",
    action_type="split",
    ex_date=date(2020, 8, 31),
    split_ratio=4.0  # 4-for-1 split
)
```

### 4. Quality Monitoring

Regularly check quality metrics:

```python
# Get symbols with quality issues
quality_df = store.get_quality_summary()
issues = quality_df[quality_df['quality_score'] < 95]
print(issues)
```

### 5. Bulk Operations

Use `bulk_write` for multiple symbols:

```python
data_dict = {
    'AAPL': aapl_df,
    'MSFT': msft_df,
    'SPY': spy_df
}

results = store.bulk_write(data_dict, timeframe='1D', source='yahoo')
```

## Troubleshooting

### Database Locked

If you get "database is locked" errors:

```python
# Use with context manager
with store.db._get_connection() as conn:
    # Do operations
    pass
```

### Missing Dependencies

```bash
pip install pandas pyarrow
```

### Schema Mismatch

If schema changes aren't applied:

```bash
# Delete and recreate database
rm /path/to/data/market_data.db
# Re-initialize will create fresh schema
```

### Parquet Read Errors

Ensure PyArrow is installed:

```bash
pip install pyarrow>=10.0.0
```

## Future Enhancements

### Planned Features

- [ ] **Multi-timeframe support**: Automatic resampling between timeframes
- [ ] **Data validation rules**: Configurable validation rules
- [ ] **Compression options**: User-selectable compression codecs
- [ ] **Async operations**: Async read/write for better performance
- [ ] **Cloud storage**: S3/GCS backend support
- [ ] **Data versioning**: Track data changes over time
- [ ] **Incremental backfill**: Smart gap detection and filling

### Optimization Opportunities

- [ ] **Partitioning**: Partition large datasets by year/month
- [ ] **Caching**: LRU cache for frequently accessed data
- [ ] **Batch operations**: Vectorized bulk operations
- [ ] **Index tuning**: Additional database indexes for common queries

## Contributing

When adding new features:

1. Update `schema.sql` with any database changes
2. Increment schema version
3. Add tests to `test_database_layer.py`
4. Update this README
5. Document any breaking changes

## License

Part of the trading-bot project.
