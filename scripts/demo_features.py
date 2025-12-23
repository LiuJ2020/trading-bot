#!/usr/bin/env python3
"""Simple demonstration of the feature store.

Shows how to compute and visualize technical indicators.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from features import FeatureStore
from features.definitions.technical import (
    RSI, SMA, EMA, MACD, MACDSignal, BollingerUpper, BollingerLower, ATR
)
from features.definitions.transforms import SimpleReturns, LogReturns


def generate_sample_data(periods: int = 252) -> pd.DataFrame:
    """Generate sample OHLCV data."""
    print(f"Generating {periods} days of sample data...")

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')

    # Simulate realistic price movement with trend
    returns = np.random.randn(periods) * 0.015  # 1.5% daily volatility
    trend = np.linspace(0, 0.3, periods)  # 30% upward trend over the period
    close = 100 * np.exp(np.cumsum(returns) + trend)

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


def main():
    """Run feature store demonstration."""
    print("=" * 70)
    print("FEATURE STORE DEMONSTRATION")
    print("=" * 70)

    # Generate sample data
    data = generate_sample_data(252)

    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")

    # Create feature store
    print("\n" + "-" * 70)
    print("Creating Feature Store and Registering Features")
    print("-" * 70)

    store = FeatureStore(enable_cache=True)

    # Register features
    features_to_register = [
        # Trend
        SMA(period=20),
        SMA(period=50),
        SMA(period=200),
        EMA(period=12),
        EMA(period=26),

        # Momentum
        RSI(period=14),
        MACD(),
        MACDSignal(),

        # Volatility
        BollingerUpper(period=20, num_std=2.0),
        BollingerLower(period=20, num_std=2.0),
        ATR(period=14),

        # Transforms
        SimpleReturns(),
        LogReturns()
    ]

    store.register_batch(features_to_register)

    print(f"Registered {len(store.list_features())} features")
    print("\nAvailable features:")
    for name in sorted(store.list_features()):
        info = store.get_info(name)
        print(f"  - {name:30s} | window={info['window']:3d} | {info['description'][:50]}")

    # Compute individual features
    print("\n" + "-" * 70)
    print("Computing Individual Features")
    print("-" * 70)

    rsi = store.compute_single('rsi_14', data)
    print(f"\nRSI-14:")
    print(f"  Min: {rsi.min():.2f}")
    print(f"  Max: {rsi.max():.2f}")
    print(f"  Mean: {rsi.mean():.2f}")
    print(f"  Current: {rsi.iloc[-1]:.2f}")

    # Identify overbought/oversold conditions
    overbought = (rsi > 70).sum()
    oversold = (rsi < 30).sum()
    print(f"  Overbought periods (>70): {overbought}")
    print(f"  Oversold periods (<30): {oversold}")

    # Compute multiple features
    print("\n" + "-" * 70)
    print("Computing Multiple Features")
    print("-" * 70)

    features = store.compute([
        'sma_20', 'sma_50',
        'rsi_14',
        'macd_12_26', 'macd_signal_9',
        'atr_14'
    ], data)

    print(f"\nComputed features shape: {features.shape}")
    print(f"\nLatest values:")
    latest = features.iloc[-1]
    for col in features.columns:
        print(f"  {col:20s}: {latest[col]:10.4f}")

    # Analyze trend
    print("\n" + "-" * 70)
    print("Trend Analysis")
    print("-" * 70)

    current_price = data['close'].iloc[-1]
    sma_20 = features['sma_20'].iloc[-1]
    sma_50 = features['sma_50'].iloc[-1]

    print(f"\nCurrent price: ${current_price:.2f}")
    print(f"SMA(20): ${sma_20:.2f}")
    print(f"SMA(50): ${sma_50:.2f}")

    if current_price > sma_20 > sma_50:
        print("Trend: BULLISH (price > SMA20 > SMA50)")
    elif current_price < sma_20 < sma_50:
        print("Trend: BEARISH (price < SMA20 < SMA50)")
    else:
        print("Trend: NEUTRAL")

    # MACD analysis
    print("\n" + "-" * 70)
    print("MACD Analysis")
    print("-" * 70)

    macd = features['macd_12_26'].iloc[-1]
    macd_signal = features['macd_signal_9'].iloc[-1]
    macd_histogram = macd - macd_signal

    print(f"\nMACD: {macd:.4f}")
    print(f"Signal: {macd_signal:.4f}")
    print(f"Histogram: {macd_histogram:.4f}")

    if macd > macd_signal:
        print("Signal: BULLISH (MACD above signal)")
    else:
        print("Signal: BEARISH (MACD below signal)")

    # Performance testing
    print("\n" + "-" * 70)
    print("Performance Testing (Caching)")
    print("-" * 70)

    import time

    # First computation (no cache)
    store.clear_cache()
    start = time.time()
    result1 = store.compute(['rsi_14', 'sma_50', 'macd_12_26'], data)
    time1 = time.time() - start

    # Second computation (with cache)
    start = time.time()
    result2 = store.compute(['rsi_14', 'sma_50', 'macd_12_26'], data)
    time2 = time.time() - start

    print(f"\nFirst computation:  {time1*1000:.2f} ms")
    print(f"Second computation: {time2*1000:.2f} ms (cached)")
    print(f"Speedup: {time1/time2:.1f}x")

    # Computation statistics
    print("\n" + "-" * 70)
    print("Computation Statistics")
    print("-" * 70)

    stats = store.get_stats()
    print("\nFeature computation counts:")
    for name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:30s}: {count:3d} times")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal features registered: {len(store.list_features())}")
    print(f"Total features computed: {len(stats)}")
    print(f"Data points processed: {len(data)}")
    print("\nFeature store is ready for use in backtests and live trading!")
    print("=" * 70)


if __name__ == "__main__":
    main()
