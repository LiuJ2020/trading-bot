# Feature Catalog

Complete reference for all available features in the feature store.

## Table of Contents

1. [Trend Indicators](#trend-indicators)
2. [Momentum Indicators](#momentum-indicators)
3. [Volatility Indicators](#volatility-indicators)
4. [Volume Indicators](#volume-indicators)
5. [Price Transforms](#price-transforms)

---

## Trend Indicators

Features that identify and measure price trends.

### SMA - Simple Moving Average

**Class:** `SMA(period: int = 20, column: str = "close")`

**Description:** Arithmetic mean of prices over a specified period.

**Parameters:**
- `period`: Number of periods for averaging (default: 20)
- `column`: Price column to use (default: "close")

**Output Name:** `sma_{column}_{period}` (e.g., `sma_20`, `sma_close_50`)

**Window:** Equal to `period`

**Common Usage:**
```python
SMA(period=20)   # 20-day moving average
SMA(period=50)   # 50-day moving average
SMA(period=200)  # 200-day moving average
```

**Interpretation:**
- Price > SMA: Potential uptrend
- Price < SMA: Potential downtrend
- SMA slope indicates trend strength

---

### EMA - Exponential Moving Average

**Class:** `EMA(period: int = 20, column: str = "close")`

**Description:** Exponentially weighted moving average that gives more weight to recent prices.

**Parameters:**
- `period`: Number of periods (default: 20)
- `column`: Price column to use (default: "close")

**Output Name:** `ema_{column}_{period}` (e.g., `ema_12`, `ema_26`)

**Window:** Equal to `period`

**Common Usage:**
```python
EMA(period=12)  # Fast EMA for MACD
EMA(period=26)  # Slow EMA for MACD
EMA(period=50)  # Medium-term trend
```

**Interpretation:**
- More responsive to recent price changes than SMA
- Commonly used in MACD calculation
- Crossovers signal potential trend changes

---

### ADX - Average Directional Index

**Class:** `ADX(period: int = 14)`

**Description:** Measures trend strength (not direction).

**Parameters:**
- `period`: Number of periods (default: 14)

**Output Name:** `adx_{period}` (e.g., `adx_14`)

**Window:** `period * 2` (28 for default)

**Common Usage:**
```python
ADX(period=14)  # Standard ADX
```

**Interpretation:**
- < 20: Weak trend (ranging market)
- 20-25: Emerging trend
- 25-50: Strong trend
- > 50: Very strong trend
- Does not indicate trend direction, only strength

---

## Momentum Indicators

Features that measure the speed and magnitude of price changes.

### RSI - Relative Strength Index

**Class:** `RSI(period: int = 14, column: str = "close")`

**Description:** Momentum oscillator measuring overbought/oversold conditions.

**Parameters:**
- `period`: Number of periods (default: 14)
- `column`: Price column to use (default: "close")

**Output Name:** `rsi_{column}_{period}` (e.g., `rsi_14`)

**Window:** `period + 1`

**Common Usage:**
```python
RSI(period=14)  # Standard RSI
RSI(period=7)   # Faster, more sensitive
RSI(period=21)  # Slower, smoother
```

**Interpretation:**
- 0-30: Oversold (potential buy signal)
- 30-70: Neutral zone
- 70-100: Overbought (potential sell signal)
- Divergences indicate potential reversals

---

### MACD - Moving Average Convergence Divergence

**Class:** `MACD(fast_period: int = 12, slow_period: int = 26, column: str = "close")`

**Description:** Trend-following momentum indicator showing relationship between two EMAs.

**Parameters:**
- `fast_period`: Fast EMA period (default: 12)
- `slow_period`: Slow EMA period (default: 26)
- `column`: Price column to use (default: "close")

**Output Name:** `macd_{fast_period}_{slow_period}` (e.g., `macd_12_26`)

**Window:** `slow_period`

**Common Usage:**
```python
MACD()  # Standard 12/26 MACD
```

**Interpretation:**
- Positive values: Fast EMA above slow EMA (bullish)
- Negative values: Fast EMA below slow EMA (bearish)
- Use with MACDSignal for crossover signals

---

### MACDSignal - MACD Signal Line

**Class:** `MACDSignal(signal_period: int = 9)`

**Description:** 9-period EMA of the MACD line (composite feature).

**Parameters:**
- `signal_period`: Signal line EMA period (default: 9)

**Dependencies:** Requires `macd_12_26`

**Output Name:** `macd_signal_{signal_period}` (e.g., `macd_signal_9`)

**Window:** `26 + signal_period`

**Common Usage:**
```python
# Register both MACD and signal
store.register_batch([MACD(), MACDSignal()])
```

**Interpretation:**
- MACD crossing above signal: Bullish crossover
- MACD crossing below signal: Bearish crossover

---

### MACDHistogram - MACD Histogram

**Class:** `MACDHistogram()`

**Description:** Difference between MACD line and signal line (composite feature).

**Dependencies:** Requires `macd_12_26` and `macd_signal_9`

**Output Name:** `macd_histogram`

**Window:** `26 + 9`

**Common Usage:**
```python
store.register_batch([MACD(), MACDSignal(), MACDHistogram()])
```

**Interpretation:**
- Positive bars: MACD above signal (bullish momentum)
- Negative bars: MACD below signal (bearish momentum)
- Increasing bars: Strengthening momentum
- Decreasing bars: Weakening momentum

---

### Stochastic - Stochastic Oscillator %K

**Class:** `Stochastic(period: int = 14)`

**Description:** Momentum indicator comparing closing price to price range.

**Parameters:**
- `period`: Lookback period (default: 14)

**Output Name:** `stoch_{period}` (e.g., `stoch_14`)

**Window:** `period`

**Common Usage:**
```python
Stochastic(period=14)  # Standard stochastic
```

**Interpretation:**
- 0-20: Oversold
- 20-80: Neutral
- 80-100: Overbought
- Use with StochasticD for crossover signals

---

### StochasticD - Stochastic %D Signal

**Class:** `StochasticD(smooth_period: int = 3)`

**Description:** 3-period SMA of Stochastic %K (composite feature).

**Parameters:**
- `smooth_period`: Smoothing period (default: 3)

**Dependencies:** Requires `stoch_14`

**Output Name:** `stoch_d_{smooth_period}` (e.g., `stoch_d_3`)

**Window:** `14 + smooth_period`

**Common Usage:**
```python
store.register_batch([Stochastic(), StochasticD()])
```

**Interpretation:**
- %K crossing above %D in oversold zone: Buy signal
- %K crossing below %D in overbought zone: Sell signal

---

## Volatility Indicators

Features that measure price volatility and range.

### ATR - Average True Range

**Class:** `ATR(period: int = 14)`

**Description:** Measures market volatility by decomposing the entire range of an asset.

**Parameters:**
- `period`: Number of periods (default: 14)

**Output Name:** `atr_{period}` (e.g., `atr_14`)

**Window:** `period + 1`

**Common Usage:**
```python
ATR(period=14)  # Standard ATR
ATR(period=7)   # Faster, more sensitive
ATR(period=21)  # Slower, smoother
```

**Interpretation:**
- Higher ATR: Higher volatility
- Lower ATR: Lower volatility
- Use for position sizing and stop-loss placement
- Compare to historical ATR for relative volatility

---

### BollingerUpper - Bollinger Band Upper

**Class:** `BollingerUpper(period: int = 20, num_std: float = 2.0, column: str = "close")`

**Description:** Upper Bollinger Band (SMA + N standard deviations).

**Parameters:**
- `period`: Moving average period (default: 20)
- `num_std`: Number of standard deviations (default: 2.0)
- `column`: Price column to use (default: "close")

**Output Name:** `bb_upper_{period}_{num_std}` (e.g., `bb_upper_20_2.0`)

**Window:** `period`

**Common Usage:**
```python
BollingerUpper(period=20, num_std=2.0)  # Standard BB
BollingerUpper(period=20, num_std=1.5)  # Tighter bands
```

**Interpretation:**
- Price touching upper band: Potentially overbought
- Price consistently above band: Strong uptrend
- Band width indicates volatility

---

### BollingerLower - Bollinger Band Lower

**Class:** `BollingerLower(period: int = 20, num_std: float = 2.0, column: str = "close")`

**Description:** Lower Bollinger Band (SMA - N standard deviations).

**Parameters:**
- `period`: Moving average period (default: 20)
- `num_std`: Number of standard deviations (default: 2.0)
- `column`: Price column to use (default: "close")

**Output Name:** `bb_lower_{period}_{num_std}` (e.g., `bb_lower_20_2.0`)

**Window:** `period`

**Common Usage:**
```python
BollingerLower(period=20, num_std=2.0)  # Standard BB
```

**Interpretation:**
- Price touching lower band: Potentially oversold
- Price consistently below band: Strong downtrend
- Use with upper band to identify squeeze (low volatility)

---

## Volume Indicators

Features that incorporate volume information.

### OBV - On-Balance Volume

**Class:** `OBV()`

**Description:** Cumulative volume-based indicator measuring buying/selling pressure.

**Output Name:** `obv`

**Window:** 2

**Common Usage:**
```python
OBV()  # Standard OBV
```

**Interpretation:**
- Rising OBV: Accumulation (buying pressure)
- Falling OBV: Distribution (selling pressure)
- Divergences with price suggest potential reversals
- Confirm price trends with OBV direction

---

### VWAP - Volume Weighted Average Price

**Class:** `VWAP(window: int = 20)`

**Description:** Average price weighted by volume over a rolling window.

**Parameters:**
- `window`: Rolling window period (default: 20)

**Output Name:** `vwap_{window}` (e.g., `vwap_20`)

**Window:** `window`

**Common Usage:**
```python
VWAP(window=20)   # 20-period VWAP
VWAP(window=100)  # Longer-term VWAP
```

**Interpretation:**
- Price > VWAP: Bullish (buyers in control)
- Price < VWAP: Bearish (sellers in control)
- Institutional traders often use as benchmark

---

### VolumeRatio - Volume Ratio

**Class:** `VolumeRatio(window: int = 20)`

**Description:** Current volume as ratio of average volume.

**Parameters:**
- `window`: Window for average volume (default: 20)

**Output Name:** `volume_ratio_{window}` (e.g., `volume_ratio_20`)

**Window:** `window`

**Common Usage:**
```python
VolumeRatio(window=20)  # 20-day average
```

**Interpretation:**
- > 1.0: Above-average volume
- < 1.0: Below-average volume
- Spikes confirm breakouts
- Low values suggest weak trend

---

## Price Transforms

Basic price transformations and calculations.

### SimpleReturns - Simple Returns

**Class:** `SimpleReturns(column: str = "close", periods: int = 1)`

**Description:** Arithmetic returns (percentage change).

**Parameters:**
- `column`: Price column (default: "close")
- `periods`: Number of periods (default: 1)

**Output Name:** `returns_{column}_{periods}` (e.g., `returns_close`)

**Window:** `periods + 1`

**Common Usage:**
```python
SimpleReturns()                    # 1-day returns
SimpleReturns(periods=5)           # 5-day returns
SimpleReturns(column="open")       # Returns on open
```

**Interpretation:**
- Positive: Price increased
- Negative: Price decreased
- Use for momentum and mean reversion strategies

---

### LogReturns - Logarithmic Returns

**Class:** `LogReturns(column: str = "close", periods: int = 1)`

**Description:** Natural log of price ratios.

**Parameters:**
- `column`: Price column (default: "close")
- `periods`: Number of periods (default: 1)

**Output Name:** `log_returns_{column}_{periods}` (e.g., `log_returns_close`)

**Window:** `periods + 1`

**Common Usage:**
```python
LogReturns()           # 1-day log returns
LogReturns(periods=5)  # 5-day log returns
```

**Interpretation:**
- Preferred for statistical analysis (more normal distribution)
- Can be summed for multi-period returns
- Symmetric (log return of -10% ≈ log return of +10%)

---

### StandardizedPrice - Z-Score

**Class:** `StandardizedPrice(column: str = "close", window: int = 20)`

**Description:** Price normalized by rolling mean and standard deviation.

**Parameters:**
- `column`: Price column (default: "close")
- `window`: Rolling window (default: 20)

**Output Name:** `zscore_{column}_{window}` (e.g., `zscore_close_20`)

**Window:** `window`

**Common Usage:**
```python
StandardizedPrice(window=20)  # 20-period z-score
StandardizedPrice(window=50)  # Longer-term normalization
```

**Interpretation:**
- > +2: Price 2 std deviations above mean (expensive)
- -2 to +2: Normal range
- < -2: Price 2 std deviations below mean (cheap)
- Useful for mean reversion strategies

---

### PriceRank - Percentile Rank

**Class:** `PriceRank(column: str = "close", window: int = 20)`

**Description:** Percentile rank of current price within rolling window.

**Parameters:**
- `column`: Price column (default: "close")
- `window`: Rolling window (default: 20)

**Output Name:** `price_rank_{column}_{window}` (e.g., `price_rank_close_20`)

**Window:** `window`

**Common Usage:**
```python
PriceRank(window=20)  # 20-day rank
```

**Interpretation:**
- 1.0: Highest price in window
- 0.5: Median price
- 0.0: Lowest price in window
- Non-parametric alternative to z-score

---

### HighLowRange - Daily Range

**Class:** `HighLowRange()`

**Description:** (High - Low) / Close as percentage.

**Output Name:** `high_low_range`

**Window:** 1

**Common Usage:**
```python
HighLowRange()  # Daily range percentage
```

**Interpretation:**
- Higher values: More intraday volatility
- Lower values: Tight trading range
- Use for volatility breakout strategies

---

### TypicalPrice - Typical Price

**Class:** `TypicalPrice()`

**Description:** (High + Low + Close) / 3.

**Output Name:** `typical_price`

**Window:** 1

**Common Usage:**
```python
TypicalPrice()  # Central price measure
```

**Interpretation:**
- More stable than close alone
- Often used in volume indicators
- Represents average price for the period

---

## Feature Combinations

### Common Strategy Patterns

#### Trend Following
```python
store.register_batch([
    SMA(period=50),
    SMA(period=200),
    EMA(period=12),
    EMA(period=26),
    ADX(period=14)
])
```

#### Mean Reversion
```python
store.register_batch([
    RSI(period=14),
    BollingerUpper(period=20),
    BollingerLower(period=20),
    StandardizedPrice(window=20)
])
```

#### Momentum
```python
store.register_batch([
    RSI(period=14),
    MACD(),
    MACDSignal(),
    Stochastic(period=14),
    SimpleReturns()
])
```

#### Volatility Breakout
```python
store.register_batch([
    ATR(period=14),
    BollingerUpper(period=20),
    BollingerLower(period=20),
    HighLowRange()
])
```

---

## Notes

### Window Sizes

The `window` parameter indicates the minimum number of rows needed. Features will return NaN for the first `window - 1` rows.

### Dependencies

Composite features automatically compute their dependencies. For example:
- `MACDSignal` requires `MACD`
- `StochasticD` requires `Stochastic`

The FeatureStore handles this automatically when you call `compute()`.

### Naming Conventions

Feature names include parameters for clarity:
- `sma_20` - SMA with 20-period window
- `rsi_14` - RSI with 14-period window
- `bb_upper_20_2.0` - Bollinger upper with 20 period and 2.0 std

### Performance Tips

1. Register all features once at initialization
2. Enable caching for repeated computations
3. Batch compute multiple features together
4. Use vectorized pandas/numpy operations
