# Feature Store

A production-ready feature store for trading signals and technical indicators. Features defined here work seamlessly in both research (Jupyter notebooks) and production (backtests/live trading).

## Overview

The feature store provides:

- **Reusable feature definitions** - Define once, use everywhere
- **Automatic dependency resolution** - Composite features automatically compute their dependencies
- **Caching** - Avoid redundant computations
- **Versioning** - Reproducible research with feature version tracking
- **Type safety** - Consistent data types and validation
- **Easy extension** - Create custom features with minimal code

## Quick Start

```python
from features import FeatureStore
from features.definitions.technical import RSI, SMA, MACD
from features.definitions.transforms import SimpleReturns

# Create store and register features
store = FeatureStore()
store.register_batch([
    RSI(period=14),
    SMA(period=50),
    MACD(),
    SimpleReturns()
])

# Compute features on OHLCV data
features = store.compute(["rsi_14", "sma_50"], data)
# Returns DataFrame with columns: rsi_14, sma_50
```

## Architecture

### Core Components

1. **Feature** - Base class for all feature definitions
2. **FeatureStore** - Central registry and computation engine
3. **FeatureMetadata** - Versioning and documentation
4. **CompositeFeature** - Features that depend on other features

### File Structure

```
features/
├── store/
│   ├── base.py              # Feature base class
│   └── feature_store.py     # FeatureStore implementation
├── definitions/
│   ├── technical.py         # Technical indicators (RSI, MACD, etc.)
│   └── transforms.py        # Price transforms (returns, z-score, etc.)
├── utils/
│   ├── validation.py        # Validation utilities
│   └── registry_helpers.py  # Helper functions for registration
└── examples/
    └── feature_demo.ipynb   # Example notebook
```

## Available Features

### Technical Indicators

#### Trend Following
- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average
- **ADX** - Average Directional Index (trend strength)

#### Momentum
- **RSI** - Relative Strength Index
- **MACD** - Moving Average Convergence Divergence
- **MACDSignal** - MACD signal line
- **MACDHistogram** - MACD histogram
- **Stochastic** - Stochastic oscillator %K
- **StochasticD** - Stochastic %D (signal line)

#### Volatility
- **ATR** - Average True Range
- **BollingerUpper** - Bollinger Band upper band
- **BollingerLower** - Bollinger Band lower band

#### Volume
- **OBV** - On-Balance Volume
- **VWAP** - Volume-Weighted Average Price
- **VolumeRatio** - Current volume / average volume

### Price Transforms

- **SimpleReturns** - Arithmetic returns
- **LogReturns** - Logarithmic returns
- **PercentChange** - Percentage change
- **StandardizedPrice** - Z-score normalized price
- **PriceRank** - Percentile rank within window
- **HighLowRange** - (High - Low) / Close as percentage
- **TypicalPrice** - (High + Low + Close) / 3

## Usage Examples

### Basic Usage

```python
from features import FeatureStore
from features.definitions.technical import RSI, SMA

# Create store
store = FeatureStore(enable_cache=True)

# Register features
store.register(RSI(period=14))
store.register(SMA(period=50))

# Compute single feature
rsi = store.compute_single("rsi_14", data)

# Compute multiple features
features = store.compute(["rsi_14", "sma_50"], data)
```

### Composite Features

Features can depend on other features. The store automatically resolves dependencies:

```python
from features.definitions.technical import MACD, MACDSignal

store.register(MACD())
store.register(MACDSignal())  # Depends on MACD

# This will compute MACD first, then MACDSignal
result = store.compute(["macd_signal_9"], data)
```

### Creating Custom Features

```python
from features.store.base import Feature
import pandas as pd

class MyCustomFeature(Feature):
    """Custom feature example."""

    def __init__(self, window: int = 20):
        super().__init__(
            name=f"my_feature_{window}",
            version="1.0.0",
            description=f"My custom feature with {window} window",
            dependencies=["close"],
            window=window,
            parameters={"window": window},
            tags={"custom"}
        )
        self.window_size = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute the feature."""
        self.validate_data(data)

        # Your computation logic here
        result = data["close"].rolling(window=self.window_size).mean()

        return result

# Register and use
store.register(MyCustomFeature(window=20))
result = store.compute_single("my_feature_20", data)
```

### Batch Registration

Use helper functions to register common feature sets:

```python
from features.utils.registry_helpers import (
    create_default_store,
    register_momentum_features,
    register_trend_features
)

# Option 1: Create store with all features
store = create_default_store(include_all=True)

# Option 2: Register specific feature groups
store = FeatureStore()
register_momentum_features(store)
register_trend_features(store)
```

### Feature Validation

```python
# Validate that features can be computed on your data
validation = store.validate_features(["rsi_14", "sma_50"], data)
# Returns: {"rsi_14": True, "sma_50": True}

# Get feature information
info = store.get_info("rsi_14")
# Returns: {
#     "name": "rsi_14",
#     "version": "1.0.0",
#     "description": "...",
#     "window": 15,
#     "dependencies": ["close"],
#     ...
# }
```

### Caching

The feature store automatically caches computed features:

```python
store = FeatureStore(enable_cache=True)

# First computation - calculates from scratch
result1 = store.compute(["rsi_14"], data)

# Second computation - uses cached result (faster)
result2 = store.compute(["rsi_14"], data)

# Clear cache if needed
store.clear_cache()

# Get computation statistics
stats = store.get_stats()
# Returns: {"rsi_14": 2, "sma_50": 1, ...}
```

## Best Practices

### Feature Design

1. **Use descriptive names** - Include parameters in the name (e.g., `rsi_14`, not just `rsi`)
2. **Document thoroughly** - Provide clear descriptions and parameter explanations
3. **Handle edge cases** - Return NaN for periods with insufficient data
4. **Validate inputs** - Use `self.validate_data(data)` to check dependencies
5. **Version your features** - Change version when computation logic changes

### Performance

1. **Enable caching** - Set `enable_cache=True` for repeated computations
2. **Vectorize operations** - Use pandas/numpy operations instead of loops
3. **Batch compute** - Compute multiple features at once to share computations
4. **Clear cache** - Clear cache between different datasets to avoid memory issues

### Testing

1. **Validate outputs** - Check for NaN handling and value ranges
2. **Test edge cases** - Empty data, insufficient data, missing columns
3. **Verify consistency** - Same input should produce same output
4. **Check dependencies** - Ensure composite features compute dependencies correctly

## Feature Metadata

Every feature includes metadata for versioning and reproducibility:

```python
feature = RSI(period=14)

metadata = feature.metadata
# FeatureMetadata(
#     name="rsi_14",
#     version="1.0.0",
#     description="Relative Strength Index...",
#     dependencies=["close"],
#     parameters={"period": 14, "column": "close"},
#     tags={"technical", "momentum", "oscillator"},
#     created_at=datetime(...),
#     author=None
# )
```

## Integration with Strategies

Features integrate seamlessly with the trading strategy SDK:

```python
from strategies.sdk.base import BaseStrategy
from strategies.sdk.context import Context

class MyStrategy(BaseStrategy):
    def on_start(self, context: Context):
        # Features are available through context
        rsi = context.features.compute_single("rsi_14", context.historical_data)

    def generate_orders(self, context: Context):
        # Use features to generate signals
        features = context.features.compute(
            ["rsi_14", "sma_50"],
            context.historical_data
        )
        # ... trading logic
```

## Testing

Run the test suite to verify all features work correctly:

```bash
# From project root
python scripts/test_features.py
```

The test suite validates:
- Individual feature computations
- Composite features and dependency resolution
- Feature store operations (registration, caching, validation)
- Edge cases and error handling

## Example Notebook

See `features/examples/feature_demo.ipynb` for a comprehensive demonstration including:
- Feature computation and visualization
- MACD, RSI, Bollinger Bands examples
- Performance and caching
- Creating custom features
- Feature correlation analysis

## Common Issues

### Issue: "Feature requires columns [...] but missing"

**Solution**: Ensure your DataFrame has the required columns. Use OHLCV column names: `open`, `high`, `low`, `close`, `volume`.

### Issue: "Insufficient data" validation fails

**Solution**: Some features require a minimum number of rows (the `window` size). Check `feature.window` and ensure your data has enough rows.

### Issue: All values are NaN

**Solution**: This is normal for the first `window` rows. Features that need lookback data will return NaN until sufficient data is available.

### Issue: Cached results seem stale

**Solution**: Call `store.clear_cache()` when switching to a new dataset or if the underlying data has changed.

## Contributing

To add new features:

1. Create a new class inheriting from `Feature` or `CompositeFeature`
2. Implement the `compute()` method
3. Set appropriate metadata (name, version, dependencies, window)
4. Add to the appropriate file in `features/definitions/`
5. Update `__init__.py` to export the feature
6. Add tests to `scripts/test_features.py`

Example:

```python
class MyIndicator(Feature):
    """Your new indicator."""

    def __init__(self, period: int = 20):
        super().__init__(
            name=f"my_indicator_{period}",
            version="1.0.0",
            description="Description of what it does",
            dependencies=["close"],
            window=period,
            parameters={"period": period},
            tags={"technical", "custom"}
        )
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        # Your computation here
        return result
```

## References

- Architecture: See `ARCHITECTURE.md` section on Feature Store (FEAT-001 to FEAT-008)
- Strategy SDK: See `strategies/sdk/` for integration points
- Examples: See `features/examples/feature_demo.ipynb`
