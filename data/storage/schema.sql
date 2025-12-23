-- Trading Bot Database Schema
-- Version: 1.0
-- Purpose: Metadata storage for market data (actual OHLCV stored in Parquet)

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT NOT NULL
);

-- Insert initial version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema with symbols, data_files, and data_quality tables');

-- Symbol universe and metadata
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    asset_class TEXT,  -- 'stock', 'etf', 'crypto', etc.
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    is_active BOOLEAN DEFAULT 1,
    first_traded_date DATE,
    last_traded_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_symbols_exchange ON symbols(exchange);
CREATE INDEX IF NOT EXISTS idx_symbols_asset_class ON symbols(asset_class);
CREATE INDEX IF NOT EXISTS idx_symbols_sector ON symbols(sector);
CREATE INDEX IF NOT EXISTS idx_symbols_active ON symbols(is_active);

-- Track parquet files and data availability
CREATE TABLE IF NOT EXISTS data_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,  -- '1D', '1H', '5T', etc.
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    num_bars INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    parquet_schema TEXT,  -- JSON representation of schema
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_data_files_unique
    ON data_files(symbol, timeframe, start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_data_files_symbol ON data_files(symbol);
CREATE INDEX IF NOT EXISTS idx_data_files_timeframe ON data_files(timeframe);
CREATE INDEX IF NOT EXISTS idx_data_files_dates ON data_files(start_date, end_date);

-- Data quality metrics per symbol/timeframe
CREATE TABLE IF NOT EXISTS data_quality (
    quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    check_date DATE NOT NULL,
    total_bars INTEGER NOT NULL,
    missing_bars INTEGER DEFAULT 0,
    duplicate_bars INTEGER DEFAULT 0,
    outlier_bars INTEGER DEFAULT 0,
    zero_volume_bars INTEGER DEFAULT 0,
    negative_prices INTEGER DEFAULT 0,
    invalid_ohlc INTEGER DEFAULT 0,  -- high < low, close > high, etc.
    completeness_pct REAL,  -- percentage of expected bars present
    quality_score REAL,  -- 0-100 composite quality score
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_quality_symbol ON data_quality(symbol);
CREATE INDEX IF NOT EXISTS idx_quality_timeframe ON data_quality(timeframe);
CREATE INDEX IF NOT EXISTS idx_quality_date ON data_quality(check_date);
CREATE INDEX IF NOT EXISTS idx_quality_score ON data_quality(quality_score);

-- Corporate actions (splits, dividends, etc.)
CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    action_type TEXT NOT NULL,  -- 'split', 'dividend', 'merger', etc.
    ex_date DATE NOT NULL,
    record_date DATE,
    payment_date DATE,
    split_ratio REAL,  -- for splits (e.g., 2.0 for 2-for-1)
    dividend_amount REAL,  -- for dividends
    details TEXT,  -- JSON with additional info
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol ON corporate_actions(symbol);
CREATE INDEX IF NOT EXISTS idx_corp_actions_type ON corporate_actions(action_type);
CREATE INDEX IF NOT EXISTS idx_corp_actions_ex_date ON corporate_actions(ex_date);

-- Data ingestion log for tracking updates
CREATE TABLE IF NOT EXISTS ingestion_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'yahoo', 'alpaca', 'manual', etc.
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    bars_ingested INTEGER NOT NULL,
    status TEXT NOT NULL,  -- 'success', 'partial', 'failed'
    error_message TEXT,
    ingestion_time_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ingestion_symbol ON ingestion_log(symbol);
CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_log(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_created ON ingestion_log(created_at);

-- View: Data availability summary per symbol
CREATE VIEW IF NOT EXISTS v_data_availability AS
SELECT
    s.symbol,
    s.name,
    s.exchange,
    s.asset_class,
    s.is_active,
    df.timeframe,
    MIN(df.start_date) as earliest_date,
    MAX(df.end_date) as latest_date,
    SUM(df.num_bars) as total_bars,
    COUNT(df.file_id) as num_files,
    SUM(df.file_size_bytes) as total_size_bytes
FROM symbols s
LEFT JOIN data_files df ON s.symbol = df.symbol
GROUP BY s.symbol, s.name, s.exchange, s.asset_class, s.is_active, df.timeframe;

-- View: Latest data quality scores
CREATE VIEW IF NOT EXISTS v_latest_quality AS
SELECT
    dq.symbol,
    dq.timeframe,
    dq.check_date,
    dq.total_bars,
    dq.missing_bars,
    dq.completeness_pct,
    dq.quality_score,
    dq.notes
FROM data_quality dq
INNER JOIN (
    SELECT symbol, timeframe, MAX(check_date) as max_date
    FROM data_quality
    GROUP BY symbol, timeframe
) latest ON dq.symbol = latest.symbol
    AND dq.timeframe = latest.timeframe
    AND dq.check_date = latest.max_date;
