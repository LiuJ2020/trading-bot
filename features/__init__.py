"""Feature store module for reusable signal/indicator definitions.

This module provides:
- Feature base classes for creating custom indicators
- FeatureStore for registration, caching, and dependency resolution
- Common technical indicators (RSI, MACD, SMA, etc.)
- Price transforms (returns, z-score, etc.)
- Validation utilities

Quick Start:
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
    ```
"""

from features.store.base import Feature, CompositeFeature, FeatureMetadata
from features.store.feature_store import FeatureStore

__all__ = [
    "Feature",
    "CompositeFeature",
    "FeatureMetadata",
    "FeatureStore",
]
