#!/usr/bin/env python3
"""Test script for feature store functionality.

Verifies that all features compute correctly and the feature store
works as expected.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from features import FeatureStore
from features.definitions.technical import (
    SMA, EMA, RSI, MACD, MACDSignal, MACDHistogram,
    BollingerUpper, BollingerLower, ATR, Stochastic, StochasticD, OBV, ADX
)
from features.definitions.transforms import (
    SimpleReturns, LogReturns, PercentChange, StandardizedPrice,
    PriceRank, HighLowRange, TypicalPrice, VWAP, VolumeRatio
)
from features.utils.registry_helpers import create_default_store
from features.utils.validation import validate_ohlcv_data, FeatureValidator


def generate_sample_data(periods: int = 252) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')

    # Simulate realistic price movement
    returns = np.random.randn(periods) * 0.02  # 2% daily volatility
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(periods)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(periods)) * 0.01)
    open_price = close * (1 + np.random.randn(periods) * 0.005)
    volume = np.random.randint(1_000_000, 5_000_000, periods)

    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return data


def test_data_validation():
    """Test data validation utilities."""
    print("\n=== Testing Data Validation ===")

    data = generate_sample_data(100)

    # Test OHLCV validation
    is_valid, msg = validate_ohlcv_data(data)
    assert is_valid, f"OHLCV validation failed: {msg}"
    print("OHLCV validation: PASSED")

    # Test feature validator
    validator = FeatureValidator()

    # Create a test series
    test_series = pd.Series(np.random.randn(100), index=data.index)

    is_valid, msg = validator.validate_series(
        test_series, "test_feature", expected_length=100
    )
    assert is_valid, f"Series validation failed: {msg}"
    print("Series validation: PASSED")

    # Test with NaN values
    test_series_nan = test_series.copy()
    test_series_nan.iloc[:10] = np.nan

    is_valid, msg = validator.validate_series(
        test_series_nan, "test_nan", allow_nan=True, min_valid_ratio=0.8
    )
    assert is_valid, f"NaN validation failed: {msg}"
    print("NaN handling: PASSED")


def test_individual_features():
    """Test individual feature computations."""
    print("\n=== Testing Individual Features ===")

    data = generate_sample_data(252)
    store = FeatureStore()

    # Test each feature type
    features_to_test = [
        # Trend
        (SMA(period=20), "sma_20"),
        (EMA(period=20), "ema_20"),

        # Momentum
        (RSI(period=14), "rsi_14"),
        (MACD(), "macd_12_26"),
        (Stochastic(period=14), "stoch_14"),

        # Volatility
        (ATR(period=14), "atr_14"),
        (BollingerUpper(period=20), "bb_upper_20_2.0"),
        (BollingerLower(period=20), "bb_lower_20_2.0"),

        # Volume
        (OBV(), "obv"),

        # Transforms
        (SimpleReturns(), "returns_close"),
        (LogReturns(), "log_returns_close"),
        (HighLowRange(), "high_low_range"),
        (TypicalPrice(), "typical_price"),
        (VWAP(window=20), "vwap_20"),
        (VolumeRatio(window=20), "volume_ratio_20"),
    ]

    for feature, expected_name in features_to_test:
        store.register(feature)
        assert feature.name == expected_name, f"Name mismatch: {feature.name} != {expected_name}"

        result = store.compute_single(feature.name, data)

        # Validate result
        assert isinstance(result, pd.Series), f"{feature.name} should return Series"
        assert len(result) == len(data), f"{feature.name} length mismatch"
        assert result.index.equals(data.index), f"{feature.name} index mismatch"

        # Check for reasonable values (not all NaN)
        valid_count = result.notna().sum()
        assert valid_count > 0, f"{feature.name} produced all NaN values"

        print(f"  {feature.name:30s}: PASSED (window={feature.window:3d}, valid={valid_count}/{len(result)})")


def test_composite_features():
    """Test composite features with dependencies."""
    print("\n=== Testing Composite Features ===")

    data = generate_sample_data(252)
    store = FeatureStore()

    # Register MACD and signal (signal depends on MACD)
    store.register(MACD())
    store.register(MACDSignal())
    store.register(MACDHistogram())

    # Compute signal (should automatically compute MACD first)
    result = store.compute(['macd_signal_9', 'macd_histogram'], data)

    assert 'macd_signal_9' in result.columns
    assert 'macd_histogram' in result.columns
    assert result['macd_signal_9'].notna().sum() > 0
    assert result['macd_histogram'].notna().sum() > 0

    print("  Composite feature dependency resolution: PASSED")

    # Test Stochastic D (depends on Stochastic K)
    store.register(Stochastic(period=14))
    store.register(StochasticD())

    result = store.compute(['stoch_d_3'], data)
    assert 'stoch_d_3' in result.columns
    print("  Stochastic D composite: PASSED")


def test_feature_store_operations():
    """Test feature store operations."""
    print("\n=== Testing Feature Store Operations ===")

    data = generate_sample_data(252)
    store = FeatureStore()

    # Test registration
    rsi = RSI(period=14)
    store.register(rsi)
    assert 'rsi_14' in store.list_features()
    print("  Feature registration: PASSED")

    # Test batch registration
    store.register_batch([
        SMA(period=20),
        SMA(period=50),
        EMA(period=12)
    ])
    assert len(store.list_features()) == 4
    print("  Batch registration: PASSED")

    # Test get_feature
    retrieved = store.get_feature('rsi_14')
    assert retrieved is not None
    assert retrieved.name == 'rsi_14'
    print("  Feature retrieval: PASSED")

    # Test get_info
    info = store.get_info('rsi_14')
    assert info['name'] == 'rsi_14'
    assert info['window'] == 15  # RSI needs period + 1
    assert 'description' in info
    print("  Feature info: PASSED")

    # Test compute multiple
    result = store.compute(['rsi_14', 'sma_20', 'ema_12'], data)
    assert result.shape == (len(data), 3)
    assert all(col in result.columns for col in ['rsi_14', 'sma_20', 'ema_12'])
    print("  Multiple feature computation: PASSED")


def test_caching():
    """Test feature caching."""
    print("\n=== Testing Feature Caching ===")

    data = generate_sample_data(252)
    store = FeatureStore(enable_cache=True)

    store.register_batch([
        RSI(period=14),
        SMA(period=20),
        MACD()
    ])

    # First computation
    store.clear_cache()
    result1 = store.compute(['rsi_14', 'sma_20', 'macd_12_26'], data)

    # Second computation (should use cache)
    result2 = store.compute(['rsi_14', 'sma_20', 'macd_12_26'], data)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)
    print("  Cache consistency: PASSED")

    # Stats should show 2 computations (one cached)
    stats = store.get_stats()
    assert stats['rsi_14'] > 0
    print("  Computation tracking: PASSED")


def test_validation():
    """Test feature validation."""
    print("\n=== Testing Feature Validation ===")

    data = generate_sample_data(100)
    store = FeatureStore()

    store.register_batch([
        SMA(period=20),
        RSI(period=14),
        ATR(period=14)
    ])

    # Validate features can be computed
    validation = store.validate_features(['sma_20', 'rsi_14', 'atr_14'], data)

    assert all(validation.values()), f"Feature validation failed: {validation}"
    print("  Feature validation: PASSED")

    # Test with insufficient data
    short_data = data.iloc[:10]  # Only 10 rows
    validation = store.validate_features(['sma_20'], short_data)
    assert not validation['sma_20'], "Should fail with insufficient data"
    print("  Insufficient data detection: PASSED")


def test_default_store():
    """Test default store creation."""
    print("\n=== Testing Default Store ===")

    store = create_default_store(include_all=True)

    features = store.list_features()
    assert len(features) > 20, f"Expected many features, got {len(features)}"
    print(f"  Default store created with {len(features)} features: PASSED")

    # Test that we can compute some features
    data = generate_sample_data(252)
    result = store.compute(['rsi_14', 'sma_50', 'macd_12_26'], data)
    assert result.shape[0] == len(data)
    print("  Default features computation: PASSED")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")

    data = generate_sample_data(100)
    store = FeatureStore()

    # Test computing unregistered feature
    try:
        store.compute_single('nonexistent_feature', data)
        assert False, "Should raise error for unregistered feature"
    except ValueError as e:
        assert 'not registered' in str(e)
        print("  Unregistered feature error: PASSED")

    # Test missing dependencies
    store.register(SMA(period=20))
    try:
        bad_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        store.compute_single('sma_20', bad_data)
        assert False, "Should raise error for missing columns"
    except ValueError as e:
        assert 'missing' in str(e).lower()
        print("  Missing dependency error: PASSED")

    # Test empty data
    empty_data = pd.DataFrame({'close': []})
    result = store.compute_single('sma_20', empty_data)
    assert len(result) == 0
    print("  Empty data handling: PASSED")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("FEATURE STORE TEST SUITE")
    print("=" * 60)

    try:
        test_data_validation()
        test_individual_features()
        test_composite_features()
        test_feature_store_operations()
        test_caching()
        test_validation()
        test_default_store()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

        return True

    except AssertionError as e:
        print(f"\n!!! TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:
        print(f"\n!!! UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
