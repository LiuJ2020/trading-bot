"""Helper functions for registering common feature sets.

Provides convenient functions to register groups of related features.
"""

from typing import List

from features.store.base import Feature
from features.store.feature_store import FeatureStore
from features.definitions.technical import (
    SMA, EMA, RSI, MACD, MACDSignal, MACDHistogram,
    BollingerUpper, BollingerLower, ATR, Stochastic, ADX
)
from features.definitions.transforms import (
    SimpleReturns, LogReturns, StandardizedPrice,
    HighLowRange, VWAP, VolumeRatio
)


def register_all_technical_indicators(
    store: FeatureStore,
    include_variants: bool = True
) -> List[str]:
    """Register all common technical indicators.

    Args:
        store: FeatureStore instance
        include_variants: Include different parameter variants

    Returns:
        List of registered feature names
    """
    features = []

    # Moving averages - common periods
    for period in [10, 20, 50, 100, 200]:
        features.extend([
            SMA(period=period),
            EMA(period=period)
        ])

    # RSI
    features.append(RSI(period=14))
    if include_variants:
        features.extend([RSI(period=7), RSI(period=21)])

    # MACD
    features.extend([
        MACD(),
        MACDSignal(),
        MACDHistogram()
    ])

    # Bollinger Bands
    features.extend([
        BollingerUpper(period=20, num_std=2.0),
        BollingerLower(period=20, num_std=2.0)
    ])

    # Volatility
    features.append(ATR(period=14))
    if include_variants:
        features.extend([ATR(period=7), ATR(period=21)])

    # Stochastic
    features.append(Stochastic(period=14))

    # Trend strength
    features.append(ADX(period=14))

    store.register_batch(features)
    return [f.name for f in features]


def register_all_transforms(store: FeatureStore) -> List[str]:
    """Register all price transforms.

    Args:
        store: FeatureStore instance

    Returns:
        List of registered feature names
    """
    features = [
        SimpleReturns(column="close", periods=1),
        LogReturns(column="close", periods=1),
        StandardizedPrice(column="close", window=20),
        HighLowRange(),
        VWAP(window=20),
        VolumeRatio(window=20)
    ]

    store.register_batch(features)
    return [f.name for f in features]


def register_momentum_features(store: FeatureStore) -> List[str]:
    """Register momentum-focused features.

    Args:
        store: FeatureStore instance

    Returns:
        List of registered feature names
    """
    features = [
        RSI(period=14),
        RSI(period=7),
        MACD(),
        MACDSignal(),
        MACDHistogram(),
        Stochastic(period=14),
        SimpleReturns(column="close", periods=1),
        SimpleReturns(column="close", periods=5),
        SimpleReturns(column="close", periods=20)
    ]

    store.register_batch(features)
    return [f.name for f in features]


def register_trend_features(store: FeatureStore) -> List[str]:
    """Register trend-focused features.

    Args:
        store: FeatureStore instance

    Returns:
        List of registered feature names
    """
    features = [
        SMA(period=10),
        SMA(period=20),
        SMA(period=50),
        SMA(period=200),
        EMA(period=12),
        EMA(period=26),
        ADX(period=14)
    ]

    store.register_batch(features)
    return [f.name for f in features]


def register_volatility_features(store: FeatureStore) -> List[str]:
    """Register volatility-focused features.

    Args:
        store: FeatureStore instance

    Returns:
        List of registered feature names
    """
    features = [
        ATR(period=14),
        ATR(period=21),
        BollingerUpper(period=20, num_std=2.0),
        BollingerLower(period=20, num_std=2.0),
        StandardizedPrice(column="close", window=20),
        HighLowRange()
    ]

    store.register_batch(features)
    return [f.name for f in features]


def create_default_store(include_all: bool = True) -> FeatureStore:
    """Create a FeatureStore with common features pre-registered.

    Args:
        include_all: Register all available features

    Returns:
        FeatureStore with features registered
    """
    store = FeatureStore(enable_cache=True)

    if include_all:
        register_all_technical_indicators(store)
        register_all_transforms(store)
    else:
        # Register just the most common features
        register_trend_features(store)
        register_momentum_features(store)

    return store
