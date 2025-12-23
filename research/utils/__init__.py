"""Research workspace utilities for rapid strategy development.

This module provides tools optimized for notebook workflows:
- QuickBacktest: Fast, lightweight backtesting
- Data loaders: Easy access to market data
- Visualization: Rich plotting utilities
- Analysis: Statistical analysis tools
"""

from research.utils.quick_backtest import QuickBacktest, BacktestResult
from research.utils.data_loaders import (
    load_bars,
    load_sample_data,
    generate_random_walk,
    DataLoader,
)
from research.utils.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_signals,
)
from research.utils.analysis import (
    calculate_metrics,
    correlation_analysis,
    stationarity_test,
    distribution_analysis,
    feature_importance,
)

__all__ = [
    # Quick backtesting
    "QuickBacktest",
    "BacktestResult",
    # Data loading
    "load_bars",
    "load_sample_data",
    "generate_random_walk",
    "DataLoader",
    # Visualization
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
    "plot_correlation_matrix",
    "plot_feature_distributions",
    "plot_signals",
    # Analysis
    "calculate_metrics",
    "correlation_analysis",
    "stationarity_test",
    "distribution_analysis",
    "feature_importance",
]
