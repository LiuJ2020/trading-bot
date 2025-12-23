# Research Workspace

Fast iteration tools for strategy development in Jupyter notebooks.

## Overview

The research workspace provides lightweight utilities optimized for rapid prototyping and idea validation. These tools prioritize speed and ease-of-use over production accuracy.

**Important**: Research tools are NOT production-quality. Use the full simulation engine for accurate backtesting.

## Quick Start

```python
import sys
sys.path.append('/Users/jacobliu/repos/projects/trading-bot')

from research.utils import (
    QuickBacktest,
    load_sample_data,
    plot_equity_curve,
    calculate_metrics,
)

# Load data
data = load_sample_data('SPY', days=500)

# Backtest a strategy
bt = QuickBacktest(initial_capital=100000)
result = bt.run_signals(data, signals)

# Visualize results
result.plot()
```

## Components

### 1. QuickBacktest

Lightweight backtesting engine for notebooks.

```python
from research.utils import QuickBacktest

# Initialize
bt = QuickBacktest(
    initial_capital=100000,
    commission=0.001,  # 10 bps
    slippage=0.0005,   # 5 bps
)

# Option A: Use with strategy function
def my_strategy(data, i):
    # Return position size (-1 to 1)
    if data['rsi'].iloc[i] < 30:
        return 1.0
    return 0.0

result = bt.run(data, my_strategy)

# Option B: Use with pre-computed signals
result = bt.run_signals(data, signals)

# View results
print(result)
result.plot()
```

### 2. Data Loaders

Easy access to market data.

```python
from research.utils import load_bars, load_sample_data
from research.utils.data_loaders import add_features

# Load sample data
data = load_sample_data('SPY', days=252)

# Load real data (if available)
data = load_bars('AAPL', '2023-01-01', '2024-01-01')

# Add technical features
data = add_features(data, features=['returns', 'sma', 'ema', 'rsi', 'bbands'])
```

### 3. Visualization

Rich plotting utilities.

```python
from research.utils import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_signals,
)

# Plot equity curve
plot_equity_curve(result.equity_curve, benchmark=spy)

# Plot trading signals
plot_signals(
    data,
    buy_signals=buy_signals,
    sell_signals=sell_signals,
    show_features=['rsi_14', 'volume_ratio']
)

# Plot feature distributions
plot_feature_distributions(data, features=['returns', 'rsi', 'momentum'])

# Correlation heatmap
plot_correlation_matrix(data[features])
```

### 4. Analysis Tools

Statistical analysis utilities.

```python
from research.utils import (
    calculate_metrics,
    correlation_analysis,
    stationarity_test,
    distribution_analysis,
    feature_importance,
    rolling_metrics,
)

# Performance metrics
metrics = calculate_metrics(returns)
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")

# Feature analysis
corr = correlation_analysis(data, target='returns', threshold=0.3)
importance = feature_importance(X, y, method='mutual_info')

# Stationarity test
result = stationarity_test(data['returns'])
if result['is_stationary']:
    print("Series is stationary")

# Rolling metrics
rolling = rolling_metrics(returns, window=60, metrics=['sharpe', 'vol'])
```

## Notebook Templates

Pre-built templates for common workflows:

### Signal Discovery (`signal_discovery.ipynb`)

Complete workflow for discovering trading signals:
1. Load and prepare data
2. Engineer features
3. Explore correlations
4. Generate candidate signals
5. Backtest and compare
6. Document findings

### Strategy Validation (`strategy_validation.ipynb`)

Rigorous testing before production:
1. In-sample vs out-of-sample testing
2. Walk-forward analysis
3. Parameter sensitivity analysis
4. Transaction cost sensitivity
5. Rolling performance metrics
6. Final validation checklist

### Feature Engineering (`feature_engineering.ipynb`)

Develop and test new features:
1. Create candidate features
2. Test stationarity
3. Analyze distributions
4. Calculate feature importance
5. Test predictive power
6. Export feature definitions

### Demo (`demo_research_workflow.ipynb`)

Working example showing:
- Load data → compute features → backtest → visualize

## File Structure

```
research/
├── README.md                    # This file
├── utils/
│   ├── __init__.py             # Main exports
│   ├── quick_backtest.py       # QuickBacktest class
│   ├── data_loaders.py         # Data loading utilities
│   ├── visualization.py        # Plotting functions
│   └── analysis.py             # Statistical analysis
└── notebooks/
    ├── demo_research_workflow.ipynb
    └── templates/
        ├── signal_discovery.ipynb
        ├── strategy_validation.ipynb
        └── feature_engineering.ipynb
```

## Key Features

### QuickBacktest Features

- Simple API: strategy as a function
- Pre-computed signals support
- Basic slippage/commission modeling
- Comprehensive metrics calculation
- Rich visualization
- Fast execution

### Analysis Features

- Performance metrics (Sharpe, Sortino, Calmar)
- Risk metrics (VaR, drawdown, volatility)
- Statistical tests (stationarity, normality)
- Feature importance (mutual info, correlation, RF)
- Rolling metrics
- Distribution analysis

### Visualization Features

- Equity curves with drawdown
- Returns distribution with Q-Q plot
- Correlation heatmaps
- Feature distributions
- Trading signals on price chart
- Feature importance charts

## Best Practices

### 1. Research Workflow

```
Idea → Quick Test (QuickBacktest) → Validation → Production
  ↓         ↓                          ↓              ↓
 Fast    Rough metrics         Rigorous testing   Full engine
```

### 2. Feature Development

1. Create features in notebook
2. Test stationarity and distributions
3. Check predictive power
4. Document in feature definitions
5. Implement in feature store
6. Use in production strategies

### 3. Strategy Validation

Before moving to production:
- [ ] Test on out-of-sample data
- [ ] Run walk-forward analysis
- [ ] Check parameter sensitivity
- [ ] Verify at high transaction costs
- [ ] Compare to simple benchmarks
- [ ] Document all findings

## Examples

### Example 1: Quick RSI Strategy Test

```python
from research.utils import QuickBacktest, load_sample_data
from research.utils.data_loaders import add_features

# Load and prepare data
data = load_sample_data('SPY', days=500)
data = data.set_index('timestamp')
data = add_features(data, features=['rsi'])

# Generate signals
data['signal'] = 0.0
data.loc[data['rsi_14'] < 30, 'signal'] = 1.0
data.loc[data['rsi_14'] > 70, 'signal'] = 0.0
data['signal'] = data['signal'].fillna(method='ffill')

# Backtest
bt = QuickBacktest(initial_capital=100000)
result = bt.run_signals(data, data['signal'])

# Results
print(result)
result.plot()
```

### Example 2: Feature Correlation Analysis

```python
from research.utils import load_sample_data, correlation_analysis, plot_correlation_matrix
from research.utils.data_loaders import add_features

# Load data
data = load_sample_data('SPY', days=1000)
data = add_features(data, features=['returns', 'sma', 'rsi', 'bbands'])

# Create target
data['future_returns'] = data['returns'].shift(-1)

# Find correlated features
features = ['rsi_14', 'bb_width', 'sma_10', 'sma_20']
corr = correlation_analysis(
    data[features + ['future_returns']],
    target='future_returns',
    threshold=0.1
)
print(corr)

# Visualize
plot_correlation_matrix(data[features])
```

### Example 3: Walk-Forward Testing

```python
from research.utils import QuickBacktest, load_sample_data

data = load_sample_data('SPY', days=1000)
bt = QuickBacktest()

window_size = 126  # 6 months
step_size = 21     # 1 month

results = []
for start in range(0, len(data) - window_size, step_size):
    window_data = data.iloc[start:start+window_size]
    result = bt.run_signals(window_data, signals)
    results.append({
        'date': window_data['timestamp'].iloc[-1],
        'sharpe': result.metrics['sharpe_ratio'],
        'return': result.metrics['total_return'],
    })

wf_df = pd.DataFrame(results)
wf_df['sharpe'].plot(title='Walk-Forward Sharpe Ratio')
```

## Limitations

QuickBacktest is NOT suitable for:
- Production backtesting (use simulation engine)
- Accurate fill modeling (use execution engine)
- Multiple strategies
- Complex order types
- Realistic slippage modeling
- Broker integration

QuickBacktest IS suitable for:
- Rapid idea testing
- Feature validation
- Signal discovery
- Parameter exploration
- Educational purposes

## Performance

QuickBacktest is optimized for speed:
- 1 year daily data: < 1 second
- 5 years daily data: < 5 seconds
- Simple vectorized operations
- Minimal overhead

For comparison, production backtesting should be slower but more accurate.

## Integration with Production

### Moving to Production

1. **Validate in Research**
   - Use QuickBacktest for initial testing
   - Run validation notebook
   - Document performance expectations

2. **Implement in SDK**
   - Convert notebook logic to Strategy class
   - Use proper Context objects
   - Follow SDK constraints

3. **Test in Simulation Engine**
   - Run with realistic execution
   - Compare to research results
   - Document any differences

4. **Paper Trading**
   - Deploy with simulated execution
   - Monitor for 5+ days
   - Verify consistency

5. **Live Trading**
   - Start with small capital
   - Gradual scale-up
   - Continuous monitoring

### Exporting Features

Document features for production:

```python
feature_config = {
    'name': 'rsi_mean_reversion_v1',
    'version': '1.0',
    'features': ['rsi_14', 'bb_width', 'volume_ratio'],
    'parameters': {
        'rsi_threshold': 30,
        'bb_threshold': 0.5,
    },
    'performance': {
        'sharpe': 2.1,
        'max_dd': -0.15,
    },
}
```

## Tips and Tricks

### Speed Up Backtests

```python
# Use smaller date ranges
data = data.iloc[-252:]  # Last year only

# Reduce features
data = data[['close', 'rsi_14']]  # Only what you need

# Pre-compute signals
signals = compute_signals(data)  # Once
result = bt.run_signals(data, signals)  # Fast
```

### Debug Strategies

```python
# Use verbose mode
result = bt.run(data, strategy_func, verbose=True)

# Check trade log
print(result.trades)

# Plot positions
result.positions['position'].plot()
```

### Compare Strategies

```python
results = {
    'Strategy A': bt.run_signals(data, signals_a),
    'Strategy B': bt.run_signals(data, signals_b),
    'Buy & Hold': bt.run_signals(data, pd.Series(1.0, index=data.index)),
}

# Compare equity curves
equity_df = pd.DataFrame({
    name: res.equity_curve for name, res in results.items()
})
plot_equity_curve(equity_df)

# Compare metrics
metrics_df = pd.DataFrame({
    name: res.metrics for name, res in results.items()
}).T
print(metrics_df)
```

## Support

For questions or issues:
1. Check the demo notebook
2. Review template notebooks
3. Consult the architecture document
4. Ask the team

## Next Steps

1. **Start with the demo**: `demo_research_workflow.ipynb`
2. **Try a template**: Pick signal discovery, validation, or feature engineering
3. **Build your strategy**: Use the utilities to test your ideas
4. **Validate rigorously**: Use the validation template
5. **Move to production**: Implement in the strategy SDK

Happy researching!
